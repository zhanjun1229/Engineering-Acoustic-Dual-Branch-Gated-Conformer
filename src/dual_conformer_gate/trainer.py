"""Training, early stopping, checkpoints, and sequential machine-noise experiments."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .config import (
    ACOUSTIC_INPUT_DIM,
    BEST_D_MODEL,
    BEST_DROPOUT,
    BEST_FF_MULT,
    BEST_N_HEADS,
    BEST_NUM_BLOCKS,
    FEATURE_VERSION,
    FIXED_THRESHOLD,
    INDUSTRIAL_INPUT_DIM,
    MACHINE_CONFIGS,
    N_CEPS,
    NOISE_KEYS,
    NOISE_READABLE,
    SEGMENT_STRIDE,
    SEGMENT_T,
    TARGET_SR,
)
from .dataset import (
    build_dataloaders,
)
from .metrics import evaluate_file_level
from .model import ConformerDualEncoderClassifier
from .utils import (
    Tee,
    append_or_replace_summary,
    count_params,
    cuda_sync,
    save_json,
    set_seed,
)


class EarlyStoppingMin:
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.counter = 0
        self.best = float("inf")
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_epoch = 0

    def step(self, metric_value: float, model: nn.Module, epoch: int) -> Tuple[bool, bool]:
        if metric_value < self.best:
            self.best = metric_value
            self.best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            self.best_epoch = epoch
            self.counter = 0
            improved = True
        else:
            self.counter += 1
            improved = False
        return self.counter >= self.patience, improved


def save_checkpoint(
    path: str,
    model: nn.Module,
    args: argparse.Namespace,
    machine: str,
    noise_key: str,
    train_files: Sequence,
    dev_files: Sequence,
    test_files: Sequence,
) -> None:
    payload = {
        "model_name": "ConformerDualEncoderClassifier",
        "state_dict": model.state_dict(),
        "config": {
            "acoustic_input_dim": ACOUSTIC_INPUT_DIM,
            "industrial_input_dim": INDUSTRIAL_INPUT_DIM,
            "d_model": BEST_D_MODEL,
            "num_blocks": BEST_NUM_BLOCKS,
            "n_heads": BEST_N_HEADS,
            "ff_mult": BEST_FF_MULT,
            "dropout": BEST_DROPOUT,
            "fusion": "dynamic_gate_weighted_concat",
            "cross_attention": False,
            "dynamic_gate": True,
            "gate_normalization": "softmax",
        },
        "feature_cfg": {
            "acoustic": "MFCC39(spafe)+LFCC39(spafe)+CQCC39(CQT-log1p-DCT)+GFCC39(spafe)",
            "feature_version": FEATURE_VERSION,
            "industrial": "IAPD26",
            "sr": TARGET_SR,
            "win_ms": 25,
            "hop_ms": 10,
            "n_ceps": N_CEPS,
            "segment_T": SEGMENT_T,
            "segment_stride": SEGMENT_STRIDE,
            "iapd_cmvn_eps": args.cmvn_eps,
        },
        "experiment": {
            "machine": machine,
            "machine_cn": MACHINE_CONFIGS[machine]["cn"],
            "noise_key": noise_key,
            "noise_db": NOISE_READABLE[noise_key],
            "protocol": "single-noise in-domain",
            "split": "stratified Train/Dev/Test=64/16/20",
            "seed": int(args.seed),
            "threshold": FIXED_THRESHOLD,
            "train_files": len(train_files),
            "dev_files": len(dev_files),
            "test_files": len(test_files),
        },
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    print(f"[CKPT] Saved to: {path}")


def plot_training_curve(
    train_losses: Sequence[float],
    dev_eers: Sequence[float],
    output_path: str,
    title: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(dev_eers) + 1), dev_eers, label="Dev EER (file-level)")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=180)
    print(f"[Curve] saved to: {output_path}")
    # 使用 Agg 后端保存图片，并关闭图形以释放资源。
    plt.close()


def train_one_experiment(
    args: argparse.Namespace,
    machine: str,
    noise_key: str,
    output_dir: str,
    device: torch.device,
) -> Dict:
    set_seed(args.seed)

    train_loader, dev_loader, test_loader, train_files, dev_files, test_files = build_dataloaders(
        args=args,
        machine=machine,
        noise_key=noise_key,
        output_dir=output_dir,
        device=device,
    )

    model = ConformerDualEncoderClassifier(
        acoustic_input_dim=ACOUSTIC_INPUT_DIM,
        industrial_input_dim=INDUSTRIAL_INPUT_DIM,
        d_model=BEST_D_MODEL,
        num_blocks=BEST_NUM_BLOCKS,
        n_heads=BEST_N_HEADS,
        ff_mult=BEST_FF_MULT,
        dropout=BEST_DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    stopper = EarlyStoppingMin(patience=args.patience)

    print("\n================= CONFIG =================")
    print(f"Machine={machine} ({MACHINE_CONFIGS[machine]['cn']})")
    print(f"Noise={NOISE_READABLE[noise_key]} | protocol=single-noise in-domain")
    print("Branch-1=MFCC39+LFCC39+CQCC39+GFCC39 (156D)")
    print("Branch-2=IAPD26 (26D)")
    print("Fusion=dynamic Gate weighted concat | Cross-Attention=False")
    print(
        f"d_model={BEST_D_MODEL}, n_heads={BEST_N_HEADS}, "
        f"num_blocks={BEST_NUM_BLOCKS}, ff_mult={BEST_FF_MULT}, "
        f"dropout={BEST_DROPOUT}"
    )
    print(f"batch_size={args.batch_size}, lr={args.lr}, threshold={FIXED_THRESHOLD}")
    print(
        "Feature extractor=baseline-consistent: MFCC/LFCC/GFCC=spafe; CQCC=preemphasis+CQT+log1p+DCT"
    )
    print(f"Feature version={FEATURE_VERSION}")
    print("==========================================\n")
    print(f"[模型复杂度] Trainable Params: {count_params(model):,}\n")

    train_losses: List[float] = []
    dev_eers: List[float] = []

    cuda_sync(device)
    train_start = time.time()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        cuda_sync(device)
        epoch_start = time.time()

        for acoustic, industrial, labels, _file_ids in train_loader:
            acoustic = acoustic.to(device, non_blocking=True)
            industrial = industrial.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            logits = model(acoustic, industrial)
            loss = criterion(logits, labels)
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_loss += float(loss.item())

        cuda_sync(device)
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(1, len(train_loader))
        train_losses.append(avg_loss)
        print(f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f} | EpochTime: {epoch_time:.2f}s")

        dev_metrics = evaluate_file_level(
            model=model,
            loader=dev_loader,
            device=device,
            name="Dev",
            save_path=None,
            threshold=FIXED_THRESHOLD,
        )
        dev_eer = float(dev_metrics["eer"])
        dev_eers.append(dev_eer)

        should_stop, improved = stopper.step(dev_eer, model, epoch)
        if improved:
            print(f"[早停] Dev EER improved to {dev_eer:.6f} at epoch {epoch}.")
        if should_stop:
            print("[早停] Dev EER 无下降，停止训练。")
            break

    cuda_sync(device)
    train_time = time.time() - train_start

    if stopper.best_state is None:
        raise RuntimeError("训练过程中没有保存到最佳模型状态。")
    model.load_state_dict(stopper.best_state)

    checkpoint_path = os.path.join(output_dir, "best_model.pt")
    save_checkpoint(
        checkpoint_path,
        model,
        args,
        machine,
        noise_key,
        train_files,
        dev_files,
        test_files,
    )

    eval_start = time.time()
    outputs_dir = os.path.join(output_dir, "outputs_significance")
    dev_best = evaluate_file_level(
        model=model,
        loader=dev_loader,
        device=device,
        name="Dev(best)",
        save_path=os.path.join(outputs_dir, "dev_best.npz"),
        threshold=FIXED_THRESHOLD,
    )
    test_metrics = evaluate_file_level(
        model=model,
        loader=test_loader,
        device=device,
        name="Test",
        save_path=os.path.join(outputs_dir, "test.npz"),
        threshold=FIXED_THRESHOLD,
    )
    eval_time = time.time() - eval_start

    curve_path = os.path.join(output_dir, "training_curve.png")
    plot_training_curve(
        train_losses=train_losses,
        dev_eers=dev_eers,
        output_path=curve_path,
        title=f"{machine} {NOISE_READABLE[noise_key]} Acoustic156 + IAPD26 + Gate",
    )

    total_time = train_time + eval_time
    print("\n================= SUMMARY =================")
    print(f"Best Epoch: {stopper.best_epoch} | Best Dev EER: {stopper.best:.6f}")
    print(f"TrainTime: {train_time:.2f}s | EvalTime: {eval_time:.2f}s | Total: {total_time:.2f}s")
    print("===========================================\n")

    result = {
        "machine": machine,
        "machine_cn": MACHINE_CONFIGS[machine]["cn"],
        "noise_key": noise_key,
        "noise_db": NOISE_READABLE[noise_key],
        "status": "completed",
        "train_files": len(train_files),
        "dev_files": len(dev_files),
        "test_files": len(test_files),
        "dev_f1": dev_best["f1"],
        "dev_auc": dev_best["auc_roc"],
        "dev_eer": dev_best["eer"],
        "dev_min_tDCF": dev_best["min_tDCF"],
        "test_f1": test_metrics["f1"],
        "test_auc": test_metrics["auc_roc"],
        "test_eer": test_metrics["eer"],
        "test_min_tDCF": test_metrics["min_tDCF"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "fpr": test_metrics["fpr"],
        "tn": test_metrics["tn"],
        "fp": test_metrics["fp"],
        "fn": test_metrics["fn"],
        "tp": test_metrics["tp"],
        "threshold": FIXED_THRESHOLD,
        "train_time_sec": train_time,
        "eval_time_sec": eval_time,
        "total_time_sec": total_time,
        "trainable_params": count_params(model),
        "gate_acoustic_mean": test_metrics["gate_acoustic_mean"],
        "gate_iapd_mean": test_metrics["gate_iapd_mean"],
        "gate_acoustic_std": test_metrics["gate_acoustic_std"],
        "gate_iapd_std": test_metrics["gate_iapd_std"],
        "output_dir": output_dir,
        "error": "",
        "best_epoch": stopper.best_epoch,
        "best_dev_eer": stopper.best,
    }

    save_json(os.path.join(output_dir, "completed.json"), result)
    return result


def experiment_output_dir(
    output_root: str,
    machine: str,
    noise_key: str,
    seed: int,
) -> str:
    return os.path.join(
        output_root,
        f"{machine}_{noise_key}_mfcc_lfcc_cqcc_gfcc_iapd26_gate_s{seed}",
    )


def load_completed_result(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def run_all(args: argparse.Namespace) -> None:
    os.makedirs(args.output_root, exist_ok=True)
    summary_path = os.path.join(args.output_root, "summary_4machines_3noises.csv")

    machines = list(MACHINE_CONFIGS.keys()) if args.machine == "all" else [args.machine]
    noises = list(NOISE_KEYS) if args.noise == "all" else [args.noise]
    jobs = [(machine, noise) for machine in machines for noise in noises]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"计划运行 {len(jobs)} 组实验：{jobs}")

    for job_index, (machine, noise_key) in enumerate(jobs, start=1):
        output_dir = experiment_output_dir(args.output_root, machine, noise_key, args.seed)
        completed_path = os.path.join(output_dir, "completed.json")
        existing = load_completed_result(completed_path)

        print("\n" + "=" * 90)
        print(
            f"[{job_index}/{len(jobs)}] {machine} ({MACHINE_CONFIGS[machine]['cn']}) "
            f"{NOISE_READABLE[noise_key]} 单噪声域内实验"
        )
        print("=" * 90)

        if existing is not None and not args.force:
            print(f"[跳过] 已完成：{completed_path}")
            append_or_replace_summary(summary_path, existing)
            continue

        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "run.log")
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        with open(log_path, "a", encoding="utf-8", buffering=1) as log_file:
            sys.stdout = Tee(original_stdout, log_file)
            sys.stderr = Tee(original_stderr, log_file)
            try:
                print(f"\n[Run started] {datetime.now().isoformat(timespec='seconds')}")
                result = train_one_experiment(
                    args=args,
                    machine=machine,
                    noise_key=noise_key,
                    output_dir=output_dir,
                    device=device,
                )
            except Exception as exc:
                error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                print("[ERROR] 当前实验失败，但会继续运行后续实验。")
                print(error_text)
                result = {
                    "machine": machine,
                    "machine_cn": MACHINE_CONFIGS[machine]["cn"],
                    "noise_key": noise_key,
                    "noise_db": NOISE_READABLE[noise_key],
                    "status": "failed",
                    "output_dir": output_dir,
                    "error": str(exc),
                }
                save_json(
                    os.path.join(output_dir, "failed.json"), {**result, "traceback": error_text}
                )
            finally:
                print(f"[Run ended] {datetime.now().isoformat(timespec='seconds')}")
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        append_or_replace_summary(summary_path, result)
        print(f"[总表] 已更新：{summary_path}")

        # 每组结束后执行垃圾回收，并在可用时清理 CUDA 缓存。
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n全部指定实验已处理完成。")
    print(f"结果总表：{summary_path}")
