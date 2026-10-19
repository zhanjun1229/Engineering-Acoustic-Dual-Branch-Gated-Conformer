"""Random seeds, filesystem output, logging, and experiment summaries."""

from __future__ import annotations

import csv
import json
import os
import random
import re
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .config import (
    SUMMARY_FIELDS,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def safe_name(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_\-.]+", "_", str(text))


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def normalize_name(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(text).lower())


def save_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_or_replace_summary(summary_path: str, row: Dict) -> None:
    rows: List[Dict[str, str]] = []
    key = (str(row.get("machine")), str(row.get("noise_key")))

    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8-sig", newline="") as f:
            for old in csv.DictReader(f):
                old_key = (str(old.get("machine")), str(old.get("noise_key")))
                if old_key != key:
                    rows.append(old)

    normalized = {field: row.get(field, "") for field in SUMMARY_FIELDS}
    rows.append(normalized)

    machine_order = {"fan": 0, "pump": 1, "slider": 2, "valve": 3}
    noise_order = {"m6": 0, "0": 1, "6": 2}
    rows.sort(
        key=lambda r: (
            machine_order.get(str(r.get("machine")), 99),
            noise_order.get(str(r.get("noise_key")), 99),
        )
    )

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class Tee:
    """同时把单组实验日志写入控制台和run.log。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()
