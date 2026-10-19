"""Acoustic156 and IAPD26 extraction, normalization, and aligned segmentation."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from scipy.fft import dct
from scipy.signal import hilbert

from .config import (
    ACOUSTIC_INPUT_DIM,
    HOP_LENGTH,
    INDUSTRIAL_INPUT_DIM,
    N_CEPS,
    N_FFT,
    SEGMENT_STRIDE,
    SEGMENT_T,
    TARGET_SR,
    WIN_LENGTH,
)

try:
    from spafe.features.gfcc import gfcc as spafe_gfcc
    from spafe.features.lfcc import lfcc as spafe_lfcc
    from spafe.features.mfcc import mfcc as spafe_mfcc

    SPAFE_AVAILABLE = True
    SPAFE_IMPORT_ERROR = None
except ImportError as exc:
    spafe_mfcc = spafe_lfcc = spafe_gfcc = None
    SPAFE_AVAILABLE = False
    SPAFE_IMPORT_ERROR = exc


def read_audio_mono(wav_path: str) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(wav_path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
        sr = TARGET_SR
    return audio, sr


def add_deltas(arr: np.ndarray) -> np.ndarray:
    """13维静态 + 13维一阶差分 + 13维二阶差分 = 39维。"""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != N_CEPS:
        raise ValueError(f"基础倒谱特征应为[T,13]，实际为{arr.shape}")
    delta = np.diff(arr, axis=0, prepend=arr[:1])
    delta2 = np.diff(delta, axis=0, prepend=delta[:1])
    return np.concatenate([arr, delta, delta2], axis=1).astype(np.float32)


def _require_spafe() -> None:
    """检查 MFCC、LFCC 和 GFCC 提取所需的 spafe 依赖。"""
    if spafe_mfcc is None or spafe_lfcc is None or spafe_gfcc is None:
        raise ImportError(
            "当前 Acoustic156 基准编码需要 spafe。请先安装：pip install spafe\n"
            f"原始导入错误：{SPAFE_IMPORT_ERROR}"
        )


def _validate_13d(feat: np.ndarray, name: str) -> np.ndarray:
    feat = np.asarray(feat, dtype=np.float32)
    if feat.ndim != 2 or feat.shape[1] != N_CEPS:
        raise RuntimeError(f"{name} 静态特征应为[T,13]，实际为{feat.shape}")
    return feat


def extract_mfcc13(audio: np.ndarray, sr: int) -> np.ndarray:
    """使用 spafe 提取13维静态 MFCC 系数。"""
    _require_spafe()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        audio = np.zeros(WIN_LENGTH, dtype=np.float32)
    feat = spafe_mfcc(
        audio,
        fs=sr,
        nfft=N_FFT,
        nfilts=26,
        num_ceps=N_CEPS,
    )
    return _validate_13d(feat, "MFCC")


def extract_lfcc13(audio: np.ndarray, sr: int) -> np.ndarray:
    """使用 spafe 提取13维静态 LFCC 系数。"""
    _require_spafe()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        audio = np.zeros(WIN_LENGTH, dtype=np.float32)
    feat = spafe_lfcc(
        audio,
        fs=sr,
        nfft=N_FFT,
        nfilts=26,
        num_ceps=N_CEPS,
    )
    return _validate_13d(feat, "LFCC")


def extract_cqcc13(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    提取本项目使用的13维 CQT 倒谱系数。

    处理顺序：0.97预加重 -> librosa.cqt -> log1p(|CQT|) -> DCT-II。
    返回每帧前13个系数；具体参数以本函数实现为准。
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        audio = np.zeros(WIN_LENGTH, dtype=np.float32)

    audio_pe = np.append(
        audio[0],
        audio[1:] - 0.97 * audio[:-1],
    ).astype(np.float32)

    cqt = librosa.cqt(
        audio_pe,
        sr=sr,
        hop_length=HOP_LENGTH,
        fmin=20.0,
        n_bins=96,
        bins_per_octave=12,
    )
    cqt_log = np.log1p(np.abs(cqt))
    cqcc13 = dct(
        cqt_log,
        type=2,
        axis=0,
        norm="ortho",
    )[:N_CEPS].T
    return _validate_13d(cqcc13, "CQCC")


def extract_gfcc13(audio: np.ndarray, sr: int) -> np.ndarray:
    """使用 spafe 的 Gammatone 滤波器组提取13维静态 GFCC 系数。"""
    _require_spafe()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        audio = np.zeros(WIN_LENGTH, dtype=np.float32)
    feat = spafe_gfcc(
        audio,
        fs=sr,
        nfft=N_FFT,
        num_ceps=N_CEPS,
    )
    return _validate_13d(feat, "GFCC")


def extract_acoustic156_from_wav(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    拼接 MFCC39、LFCC39、CQCC39 和 GFCC39，返回156维声学特征。

    每类特征包含13维静态系数及其一阶、二阶差分，按最短帧数对齐。
    """
    mfcc39 = add_deltas(extract_mfcc13(audio, sr))
    lfcc39 = add_deltas(extract_lfcc13(audio, sr))
    cqcc39 = add_deltas(extract_cqcc13(audio, sr))
    gfcc39 = add_deltas(extract_gfcc13(audio, sr))

    min_len = min(len(mfcc39), len(lfcc39), len(cqcc39), len(gfcc39))
    acoustic = np.concatenate(
        [
            mfcc39[:min_len],
            lfcc39[:min_len],
            cqcc39[:min_len],
            gfcc39[:min_len],
        ],
        axis=1,
    )
    if acoustic.shape[1] != ACOUSTIC_INPUT_DIM:
        raise RuntimeError(f"声学分支维度错误：{acoustic.shape}")
    return acoustic.astype(np.float32)


def extract_acoustic156_from_npz(npz_path: str) -> np.ndarray:
    """
    从 NPZ 读取 mfcc39、lfcc39、cqcc39 和 gfcc39，拼接为156维特征。

    每个数组须为[T,39]，各数组按最短帧数对齐。预提取文件的特征定义
    应与 WAV 提取路径一致；段级缓存由 Dataset 单独管理。
    """
    required = ("mfcc39", "lfcc39", "cqcc39", "gfcc39")
    with np.load(npz_path) as data:
        missing = [name for name in required if name not in data]
        if missing:
            raise KeyError(f"{npz_path}缺少特征：{missing}")
        mfcc39 = np.asarray(data["mfcc39"], dtype=np.float32)
        lfcc39 = np.asarray(data["lfcc39"], dtype=np.float32)
        cqcc39 = np.asarray(data["cqcc39"], dtype=np.float32)
        gfcc39 = np.asarray(data["gfcc39"], dtype=np.float32)

    for name, feat in (
        ("mfcc39", mfcc39),
        ("lfcc39", lfcc39),
        ("cqcc39", cqcc39),
        ("gfcc39", gfcc39),
    ):
        if feat.ndim != 2 or feat.shape[1] != 39:
            raise RuntimeError(f"{npz_path}中的{name}形状异常：{feat.shape}")

    min_len = min(len(mfcc39), len(lfcc39), len(cqcc39), len(gfcc39))
    acoustic = np.concatenate(
        [
            mfcc39[:min_len],
            lfcc39[:min_len],
            cqcc39[:min_len],
            gfcc39[:min_len],
        ],
        axis=1,
    )
    if acoustic.shape[1] != ACOUSTIC_INPUT_DIM:
        raise RuntimeError(f"预提取声学分支维度错误：{acoustic.shape}")
    return acoustic.astype(np.float32)


def frame_audio_centered(
    audio: np.ndarray,
    frame_length: int = WIN_LENGTH,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        audio = np.zeros(1, dtype=np.float32)

    pad_mode = "reflect" if audio.size > 1 else "edge"
    padded = np.pad(audio, (frame_length // 2, frame_length // 2), mode=pad_mode)
    if padded.size < frame_length:
        padded = np.pad(padded, (0, frame_length - padded.size), mode="constant")

    n_frames = 1 + (padded.size - frame_length) // hop_length
    starts = np.arange(n_frames, dtype=np.int64) * hop_length
    frames = np.stack([padded[s : s + frame_length] for s in starts], axis=0)
    return frames.astype(np.float32)


def apply_cmvn(feat: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    feat = np.asarray(feat, dtype=np.float32)
    mean = feat.mean(axis=0, keepdims=True)
    std = feat.std(axis=0, keepdims=True)
    return ((feat - mean) / (std + eps)).astype(np.float32)


def extract_iapd26(
    audio: np.ndarray,
    sr: int,
    frame_length: int = WIN_LENGTH,
    hop_length: int = HOP_LENGTH,
    cmvn_eps: float = 1e-5,
) -> np.ndarray:
    """逐帧计算13个时域指标和13个频域/周期指标。"""
    eps = 1e-8
    frames = frame_audio_centered(audio, frame_length, hop_length)
    n_frames = frames.shape[0]
    out = np.zeros((n_frames, INDUSTRIAL_INPUT_DIM), dtype=np.float32)

    mean = frames.mean(axis=1)
    centered = frames - mean[:, None]
    std = np.sqrt(np.mean(centered**2, axis=1) + eps)
    rms = np.sqrt(np.mean(frames**2, axis=1) + eps)
    abs_frame = np.abs(frames)
    abs_peak = abs_frame.max(axis=1)
    peak_to_peak = frames.max(axis=1) - frames.min(axis=1)
    skewness = np.mean(centered**3, axis=1) / (std**3 + eps)
    kurtosis = np.mean(centered**4, axis=1) / (std**4 + eps)
    mean_abs = abs_frame.mean(axis=1)
    root_amplitude = np.mean(np.sqrt(abs_frame + eps), axis=1) ** 2
    crest_factor = abs_peak / (rms + eps)
    impulse_factor = abs_peak / (mean_abs + eps)
    clearance_factor = abs_peak / (root_amplitude + eps)
    shape_factor = rms / (mean_abs + eps)
    short_time_energy = np.sum(frames**2, axis=1)
    zero_crossing_rate = np.mean(frames[:, 1:] * frames[:, :-1] < 0.0, axis=1)

    out[:, :13] = np.column_stack(
        [
            mean,
            std,
            rms,
            abs_peak,
            peak_to_peak,
            skewness,
            kurtosis,
            crest_factor,
            impulse_factor,
            clearance_factor,
            shape_factor,
            short_time_energy,
            zero_crossing_rate,
        ]
    )

    window = np.hanning(frame_length).astype(np.float32)
    spectrum = np.fft.rfft(frames * window[None, :], axis=1)
    magnitude = np.abs(spectrum).astype(np.float64)
    power = magnitude**2
    freqs = np.fft.rfftfreq(frame_length, d=1.0 / float(sr))

    dominant_bin = np.argmax(power[:, 1:], axis=1) + 1
    row_idx = np.arange(n_frames)
    dominant_frequency = freqs[dominant_bin]
    dominant_amplitude = 2.0 * magnitude[row_idx, dominant_bin] / float(frame_length)
    power_sum = power.sum(axis=1) + eps
    spectral_centroid = (power * freqs[None, :]).sum(axis=1) / power_sum
    spectral_bandwidth = np.sqrt(
        (power * (freqs[None, :] - spectral_centroid[:, None]) ** 2).sum(axis=1) / power_sum
    )
    cumulative_power = np.cumsum(power, axis=1)
    rolloff_bin = np.argmax(cumulative_power >= (0.85 * power_sum)[:, None], axis=1)
    spectral_rolloff = freqs[rolloff_bin]
    probability = power / power_sum[:, None]
    spectral_entropy = -np.sum(probability * np.log(probability + eps), axis=1)
    spectral_entropy /= np.log(power.shape[1] + eps)
    spectral_flatness = np.exp(np.mean(np.log(power + eps), axis=1)) / (
        np.mean(power, axis=1) + eps
    )

    low_mask = freqs <= 1000.0
    mid_mask = (freqs > 1000.0) & (freqs <= 4000.0)
    high_mask = freqs > 4000.0
    low_ratio = power[:, low_mask].sum(axis=1) / power_sum
    mid_ratio = power[:, mid_mask].sum(axis=1) / power_sum
    high_ratio = power[:, high_mask].sum(axis=1) / power_sum

    harmonic_ratio = np.zeros(n_frames, dtype=np.float64)
    autocorrelation_peak = np.zeros(n_frames, dtype=np.float64)
    envelope_spectrum_peak = np.zeros(n_frames, dtype=np.float64)

    for i in range(n_frames):
        harmonic_bins = set()
        k0 = int(dominant_bin[i])
        for harmonic in range(1, 6):
            kh = k0 * harmonic
            if kh >= power.shape[1]:
                break
            harmonic_bins.update(range(max(1, kh - 1), min(power.shape[1], kh + 2)))
        if harmonic_bins:
            harmonic_ratio[i] = power[i, list(harmonic_bins)].sum() / power_sum[i]

        x = centered[i].astype(np.float64)
        ac = np.correlate(x, x, mode="full")[frame_length - 1 :]
        if ac.size > 2 and ac[0] > eps:
            autocorrelation_peak[i] = np.max(ac[2:]) / (ac[0] + eps)

        envelope = np.abs(hilbert(frames[i].astype(np.float64)))
        envelope -= envelope.mean()
        envelope_mag = np.abs(np.fft.rfft(envelope))
        if envelope_mag.size > 1:
            envelope_spectrum_peak[i] = envelope_mag[1:].max() / (envelope_mag[1:].sum() + eps)

    out[:, 13:] = np.column_stack(
        [
            dominant_frequency,
            dominant_amplitude,
            spectral_centroid,
            spectral_bandwidth,
            spectral_rolloff,
            spectral_entropy,
            spectral_flatness,
            low_ratio,
            mid_ratio,
            high_ratio,
            harmonic_ratio,
            autocorrelation_peak,
            envelope_spectrum_peak,
        ]
    )

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return apply_cmvn(out, eps=cmvn_eps)


def make_segments(
    acoustic: np.ndarray,
    industrial: np.ndarray,
    segment_t: int = SEGMENT_T,
    stride: int = SEGMENT_STRIDE,
) -> Tuple[np.ndarray, np.ndarray]:
    min_len = min(acoustic.shape[0], industrial.shape[0])
    acoustic = acoustic[:min_len]
    industrial = industrial[:min_len]

    if min_len < segment_t:
        pad_total = segment_t - min_len
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        acoustic = np.pad(acoustic, ((pad_left, pad_right), (0, 0)), mode="constant")
        industrial = np.pad(industrial, ((pad_left, pad_right), (0, 0)), mode="constant")

    acoustic_segments: List[np.ndarray] = []
    industrial_segments: List[np.ndarray] = []
    for start in range(0, acoustic.shape[0] - segment_t + 1, stride):
        acoustic_segments.append(acoustic[start : start + segment_t])
        industrial_segments.append(industrial[start : start + segment_t])

    if not acoustic_segments:
        raise RuntimeError("未能从文件中切出任何128帧片段。")

    return (
        np.asarray(acoustic_segments, dtype=np.float32),
        np.asarray(industrial_segments, dtype=np.float32),
    )


def extract_acoustic_iapd_segments(
    wav_path: str,
    precomputed_path: Optional[str],
    cmvn_eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    audio, sr = read_audio_mono(wav_path)

    if precomputed_path and os.path.exists(precomputed_path):
        try:
            acoustic = extract_acoustic156_from_npz(precomputed_path)
        except Exception as exc:
            print(f"[特征回退] 预提取声学特征不可用，从WAV重提取：{exc}")
            acoustic = extract_acoustic156_from_wav(audio, sr)
    else:
        acoustic = extract_acoustic156_from_wav(audio, sr)

    industrial = extract_iapd26(audio, sr, cmvn_eps=cmvn_eps)
    return make_segments(acoustic, industrial)
