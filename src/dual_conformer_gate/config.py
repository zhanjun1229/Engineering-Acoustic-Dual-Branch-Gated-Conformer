"""Feature dimensions, dataset identifiers, and fixed experiment settings."""

from __future__ import annotations

from typing import Dict, Sequence

FEATURE_VERSION = "mimii_acoustic156_baseline_encoding_v1"

MACHINE_CONFIGS: Dict[str, Dict[str, Sequence[str]]] = {
    "fan": {"cn": "风扇", "aliases": ["fan", "风扇"]},
    "pump": {"cn": "泵", "aliases": ["pump", "泵", "水泵"]},
    "slider": {"cn": "滑轨", "aliases": ["slider", "slide_rail", "滑轨", "滑块"]},
    "valve": {"cn": "阀门", "aliases": ["valve", "阀门", "阀"]},
}

NOISE_KEYS = ("m6", "0", "6")

NOISE_TEXT = {"m6": "-6", "0": "0", "6": "6"}

NOISE_READABLE = {"m6": "-6 dB", "0": "0 dB", "6": "+6 dB"}

ACOUSTIC_INPUT_DIM = 156

INDUSTRIAL_INPUT_DIM = 26

TARGET_SR = 16000

N_FFT = 512

WIN_LENGTH = 400

HOP_LENGTH = 160

N_CEPS = 13

SEGMENT_T = 128

SEGMENT_STRIDE = 64

FIXED_THRESHOLD = 0.5

FILE_POOL = "mean"

BEST_D_MODEL = 64

BEST_N_HEADS = 4

BEST_NUM_BLOCKS = 2

BEST_FF_MULT = 2

BEST_DROPOUT = 0.10654172080387744

BEST_BATCH_SIZE = 64

BEST_LR = 0.0005184655055267425

SUMMARY_FIELDS = [
    "machine",
    "machine_cn",
    "noise_key",
    "noise_db",
    "status",
    "train_files",
    "dev_files",
    "test_files",
    "dev_f1",
    "dev_auc",
    "dev_eer",
    "dev_min_tDCF",
    "test_f1",
    "test_auc",
    "test_eer",
    "test_min_tDCF",
    "precision",
    "recall",
    "fpr",
    "tn",
    "fp",
    "fn",
    "tp",
    "threshold",
    "train_time_sec",
    "eval_time_sec",
    "total_time_sec",
    "trainable_params",
    "gate_acoustic_mean",
    "gate_iapd_mean",
    "gate_acoustic_std",
    "gate_iapd_std",
    "output_dir",
    "error",
]

IAPD_FEATURE_NAMES = [
    "mean",
    "std",
    "rms",
    "abs_peak",
    "peak_to_peak",
    "skewness",
    "kurtosis",
    "crest_factor",
    "impulse_factor",
    "clearance_factor",
    "shape_factor",
    "short_time_energy",
    "zero_crossing_rate",
    "dominant_frequency",
    "dominant_amplitude",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff_85",
    "spectral_entropy",
    "spectral_flatness",
    "low_band_energy_ratio",
    "mid_band_energy_ratio",
    "high_band_energy_ratio",
    "harmonic_energy_ratio",
    "autocorrelation_peak",
    "envelope_spectrum_peak",
]
