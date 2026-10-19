"""File-level score aggregation, evaluation metrics, and prediction export."""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from .config import FILE_POOL, FIXED_THRESHOLD
from .utils import cuda_sync


def compute_min_tDCF(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 1.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    return float(np.min(0.75 * fnr + 0.25 * fpr))


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 1.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def compute_rrmse(rmse: float, y_true: np.ndarray, eps: float = 1e-12) -> Tuple[float, float]:
    pos_rate = float(np.mean(y_true))
    return float(rmse / (pos_rate + eps)), pos_rate


@torch.no_grad()
def evaluate_file_level(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    name: str,
    save_path: Optional[str],
    threshold: float = FIXED_THRESHOLD,
    pool: str = FILE_POOL,
) -> Dict[str, float]:
    model.eval()
    segment_probs: List[float] = []
    segment_labels: List[float] = []
    segment_file_ids: List[str] = []
    segment_gate_acoustic: List[float] = []
    segment_gate_iapd: List[float] = []

    cuda_sync(device)
    t0 = time.time()

    for acoustic, industrial, labels, file_ids in loader:
        acoustic = acoustic.to(device, non_blocking=True)
        industrial = industrial.to(device, non_blocking=True)
        logits, gate_weights = model(acoustic, industrial, return_gate=True)
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        gate_np = gate_weights.detach().cpu().numpy()

        segment_probs.extend(probs.tolist())
        segment_gate_acoustic.extend(gate_np[:, 0].tolist())
        segment_gate_iapd.extend(gate_np[:, 1].tolist())
        segment_labels.extend(labels.numpy().reshape(-1).tolist())
        segment_file_ids.extend(list(file_ids))

    cuda_sync(device)
    t1 = time.time()

    file2scores: Dict[str, List[float]] = {}
    file2label: Dict[str, int] = {}
    for prob, label, file_id in zip(segment_probs, segment_labels, segment_file_ids):
        file2scores.setdefault(file_id, []).append(float(prob))
        file2label.setdefault(file_id, int(label))

    file_ids_sorted = sorted(file2scores.keys())
    file_scores: List[float] = []
    file_labels: List[int] = []
    for file_id in file_ids_sorted:
        scores = np.asarray(file2scores[file_id], dtype=np.float64)
        if pool == "mean":
            pooled = float(np.mean(scores))
        elif pool == "median":
            pooled = float(np.median(scores))
        else:
            raise ValueError("pool只能是mean或median")
        file_scores.append(pooled)
        file_labels.append(file2label[file_id])

    y_score = np.asarray(file_scores, dtype=np.float64)
    y_true = np.asarray(file_labels, dtype=np.int64)
    y_pred = (y_score >= threshold).astype(np.int64)

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    eer = compute_eer(y_true, y_score)
    min_tdcf = compute_min_tDCF(y_true, y_score)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_score)))
    rrmse, pos_rate = compute_rrmse(rmse, y_true)
    auc_roc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr_value = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    eval_time = t1 - t0
    n_files = len(y_true)
    throughput = float(n_files / eval_time) if eval_time > 0 else float("inf")
    gate_acoustic_arr = np.asarray(segment_gate_acoustic, dtype=np.float64)
    gate_iapd_arr = np.asarray(segment_gate_iapd, dtype=np.float64)
    gate_acoustic_mean = (
        float(np.mean(gate_acoustic_arr)) if gate_acoustic_arr.size else float("nan")
    )
    gate_iapd_mean = float(np.mean(gate_iapd_arr)) if gate_iapd_arr.size else float("nan")
    gate_acoustic_std = float(np.std(gate_acoustic_arr)) if gate_acoustic_arr.size else float("nan")
    gate_iapd_std = float(np.std(gate_iapd_arr)) if gate_iapd_arr.size else float("nan")

    print(
        f"[{name}][FILE] threshold={threshold:.4f} | "
        f"F1: {f1:.4f} | EER: {eer:.4f} | min-tDCF: {min_tdcf:.4f} | "
        f"RMSE: {rmse:.4f} | RRMSE: {rrmse:.4f} (pos_rate={pos_rate:.4f}) | "
        f"AUC-ROC: {auc_roc:.4f} | EvalTime: {eval_time:.2f}s | "
        f"Throughput: {throughput:.2f} files/s"
    )
    print(
        f"[{name}][CONFUSION] TN={tn}, FP={fp}, FN={fn}, TP={tp} | "
        f"Precision={precision:.4f}, Recall={recall:.4f}, FPR={fpr_value:.4f}"
    )
    print(
        f"[{name}][GATE] acoustic_mean={gate_acoustic_mean:.4f}, "
        f"iapd_mean={gate_iapd_mean:.4f}, "
        f"acoustic_std={gate_acoustic_std:.4f}, iapd_std={gate_iapd_std:.4f}"
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(
            save_path,
            file_ids=np.asarray(file_ids_sorted, dtype=object),
            labels=y_true,
            scores=y_score,
            pred=y_pred,
            threshold=np.asarray([threshold], dtype=np.float64),
            tn=np.asarray([tn], dtype=np.int64),
            fp=np.asarray([fp], dtype=np.int64),
            fn=np.asarray([fn], dtype=np.int64),
            tp=np.asarray([tp], dtype=np.int64),
            precision=np.asarray([precision], dtype=np.float64),
            recall=np.asarray([recall], dtype=np.float64),
            fpr=np.asarray([fpr_value], dtype=np.float64),
            gate_acoustic_mean=np.asarray([gate_acoustic_mean], dtype=np.float64),
            gate_iapd_mean=np.asarray([gate_iapd_mean], dtype=np.float64),
            gate_acoustic_std=np.asarray([gate_acoustic_std], dtype=np.float64),
            gate_iapd_std=np.asarray([gate_iapd_std], dtype=np.float64),
        )
        print(f"[SAVE] file-level outputs saved to: {save_path}")

    return {
        "f1": f1,
        "eer": eer,
        "min_tDCF": min_tdcf,
        "rmse": rmse,
        "rrmse": rrmse,
        "pos_rate": pos_rate,
        "auc_roc": auc_roc,
        "eval_time_sec": eval_time,
        "throughput_filesps": throughput,
        "n_files": int(n_files),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": precision,
        "recall": recall,
        "fpr": fpr_value,
        "threshold": float(threshold),
        "gate_acoustic_mean": gate_acoustic_mean,
        "gate_iapd_mean": gate_iapd_mean,
        "gate_acoustic_std": gate_acoustic_std,
        "gate_iapd_std": gate_iapd_std,
    }
