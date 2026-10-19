"""Command-line interface for MIMII single-noise classification experiments."""

from __future__ import annotations

import argparse

from .config import BEST_BATCH_SIZE, BEST_LR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acoustic156 / IAPD26 dual-Conformer + dynamic Gate: 4 machines x 3 single-noise experiments"
    )
    parser.add_argument(
        "--machine",
        type=str,
        default="all",
        choices=["fan", "pump", "slider", "valve", "all"],
        help="默认all，依次运行四类机器；也可只运行指定机器。",
    )
    parser.add_argument(
        "--noise",
        type=str,
        default="all",
        choices=["m6", "0", "6", "all"],
        help="默认all，依次运行-6、0、+6 dB；也可只运行一个噪声。",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="包含所有MIMII机器与噪声文件夹的根目录。",
    )
    parser.add_argument(
        "--feature_root",
        type=str,
        default="",
        help=(
            "可选预提取声学特征目录。期望结构："
            "feature_root/machine/noise/id_xx/normal|abnormal/file.npz；"
            "至少含mfcc39、lfcc39、cqcc39和gfcc39。缺失时自动从WAV提取。"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./results",
        help="12组实验统一输出根目录。",
    )
    parser.add_argument(
        "--machine_ids",
        type=str,
        default="id_00,id_02,id_04,id_06",
        help="使用的机器ID，多个用逗号分隔。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=BEST_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=BEST_LR)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cmvn_eps", type=float, default=1e-5)
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="缓存每个WAV的MFCC+LFCC+CQCC+GFCC和IAPD26分段，后续运行直接复用。",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="保留的兼容选项；无论是否传入，均保存训练曲线且不弹窗。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使某组已有completed.json，也强制重新运行。",
    )
    return parser.parse_args()


def main() -> None:
    """Parse filesystem and experiment arguments, then run selected experiments."""
    args = parse_args()

    from .trainer import run_all

    run_all(args)
