"""Small synthetic checks: no dataset download or full training required."""

import sys

import numpy as np
import pytest
import torch

from dual_conformer_gate import config
from dual_conformer_gate.cli import parse_args
from dual_conformer_gate.dataset import stratified_split_64_16_20
from dual_conformer_gate.features import extract_acoustic156_from_wav, extract_iapd26
from dual_conformer_gate.metrics import compute_eer, compute_min_tDCF, evaluate_file_level
from dual_conformer_gate.model import ConformerDualEncoderClassifier
from dual_conformer_gate.trainer import EarlyStoppingMin


def make_model():
    return ConformerDualEncoderClassifier(
        acoustic_input_dim=config.ACOUSTIC_INPUT_DIM,
        industrial_input_dim=config.INDUSTRIAL_INPUT_DIM,
        d_model=config.BEST_D_MODEL,
        num_blocks=config.BEST_NUM_BLOCKS,
        n_heads=config.BEST_N_HEADS,
        ff_mult=config.BEST_FF_MULT,
        dropout=config.BEST_DROPOUT,
    )


def test_cli_preserves_experiment_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["conformer-train", "--data_root", "data"])
    args = parse_args()
    assert (args.machine, args.noise, args.seed) == ("all", "all", 42)
    assert args.machine_ids == "id_00,id_02,id_04,id_06"
    assert (args.max_epochs, args.patience, args.grad_clip) == (50, 10, 3.0)
    assert (args.batch_size, args.lr) == (64, 0.0005184655055267425)
    assert (args.num_workers, args.cmvn_eps) == (0, 1e-5)
    assert not args.use_cache and not args.force and not args.no_plot
    assert (args.feature_root, args.output_root) == ("", "./results")


def test_cli_requires_data_path(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["conformer-train"])
    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 2


def test_file_splits_are_disjoint_stratified_and_repeatable():
    items = [(f"file-{i}.wav", i % 2, str(i)) for i in range(100)]
    splits = stratified_split_64_16_20(items, 42)
    assert [len(part) for part in splits] == [64, 16, 20]
    ids = [{item[2] for item in part} for part in splits]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set.union(*ids) == {str(i) for i in range(100)}
    assert all(sum(item[1] for item in part) == len(part) // 2 for part in splits)
    assert splits == stratified_split_64_16_20(items, 42)


def test_model_gate_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(42)
    model = make_model().eval()
    acoustic = torch.randn(2, 16, 156)
    industrial = torch.randn(2, 16, 26)
    with torch.no_grad():
        logits, gate = model(acoustic, industrial, return_gate=True)
    assert logits.shape == (2, 1)
    assert gate.shape == (2, 2)
    assert torch.isfinite(logits).all()
    assert (gate >= 0).all() and (gate <= 1).all()
    torch.testing.assert_close(gate.sum(dim=1), torch.ones(2))
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = make_model().eval()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(restored(acoustic, industrial), logits)


def test_synthetic_audio_features_have_original_dimensions():
    rng = np.random.default_rng(42)
    audio = rng.normal(0, 0.05, size=config.TARGET_SR * 2).astype(np.float32)
    acoustic = extract_acoustic156_from_wav(audio, config.TARGET_SR)
    industrial = extract_iapd26(audio, config.TARGET_SR)
    assert acoustic.ndim == industrial.ndim == 2
    assert acoustic.shape[1] == 156 and industrial.shape[1] == 26
    assert np.isfinite(acoustic).all() and np.isfinite(industrial).all()


def test_metrics_keep_original_formula_and_single_class_behavior():
    labels = np.array([0, 0, 1, 1])
    assert compute_eer(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 0
    assert compute_min_tDCF(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 0
    reversed_scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert compute_min_tDCF(labels, reversed_scores) == pytest.approx(0.25)
    assert compute_eer(np.zeros(3), np.zeros(3)) == 1
    assert compute_min_tDCF(np.zeros(3), np.zeros(3)) == 1


def test_file_level_evaluation_pools_segment_probabilities(tmp_path):
    class FixedModel(torch.nn.Module):
        def forward(self, acoustic, industrial, return_gate=False):
            logits = acoustic[:, :1]
            gate = torch.full((len(acoustic), 2), 0.5)
            return (logits, gate) if return_gate else logits

    # File A averages to 0.4; file B averages to 0.8.
    items = [
        (torch.logit(torch.tensor([p])), torch.zeros(1), y, fid)
        for p, y, fid in [(0.1, 0, "a"), (0.7, 0, "a"), (0.8, 1, "b")]
    ]
    loader = torch.utils.data.DataLoader(items, batch_size=2)
    path = tmp_path / "scores.npz"
    result = evaluate_file_level(FixedModel(), loader, torch.device("cpu"), "test", str(path))
    assert result["tn"] == result["tp"] == 1
    assert result["fp"] == result["fn"] == 0
    assert result["f1"] == 1
    with np.load(path) as data:
        np.testing.assert_allclose(data["scores"], [0.4, 0.8], rtol=1e-6)
        np.testing.assert_array_equal(data["labels"], [0, 1])


def test_early_stopping_keeps_an_independent_best_state():
    model = torch.nn.Linear(1, 1)
    stopper = EarlyStoppingMin(patience=2)
    assert stopper.step(0.2, model, 1) == (False, True)
    saved = stopper.best_state["weight"].clone()
    with torch.no_grad():
        model.weight.add_(1)
    torch.testing.assert_close(stopper.best_state["weight"], saved)
    assert stopper.step(0.3, model, 2) == (False, False)
    assert stopper.step(0.4, model, 3) == (True, False)
    assert stopper.best_epoch == 1
