"""MIMII discovery, file-level splitting, feature caching, and data loaders."""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .config import (
    ACOUSTIC_INPUT_DIM,
    FEATURE_VERSION,
    INDUSTRIAL_INPUT_DIM,
    MACHINE_CONFIGS,
    NOISE_READABLE,
    NOISE_TEXT,
)
from .features import (
    extract_acoustic_iapd_segments,
)
from .utils import normalize_name, parse_csv, safe_name


def domain_candidates(machine: str, noise_key: str) -> List[str]:
    noise = NOISE_TEXT[noise_key]
    candidates: List[str] = []
    for alias in MACHINE_CONFIGS[machine]["aliases"]:
        candidates.extend(
            [
                f"{noise}_dB_{alias}",
                f"{noise}dB_{alias}",
                f"MIMII-{noise}_dB_{alias}",
                f"MIMII-{noise}dB_{alias}",
                f"{alias}_{noise}_dB",
                f"{alias}_{noise}dB",
            ]
        )
    return list(dict.fromkeys(candidates))


def resolve_domain_dir(data_root: str, machine: str, noise_key: str) -> str:
    candidates = domain_candidates(machine, noise_key)
    for name in candidates:
        path = os.path.join(data_root, name)
        if os.path.isdir(path):
            return path

    # 大小写和符号不一致时，扫描第一层目录进行宽松匹配。
    wanted = {normalize_name(name) for name in candidates}
    if os.path.isdir(data_root):
        for entry in os.scandir(data_root):
            if entry.is_dir() and normalize_name(entry.name) in wanted:
                return entry.path

    raise FileNotFoundError(
        f"未找到 {machine} {NOISE_READABLE[noise_key]} 数据目录。\n"
        f"数据根目录：{data_root}\n"
        f"尝试过：{candidates}"
    )


def find_subdir_case_insensitive(parent: str, target_name: str) -> Optional[str]:
    direct = os.path.join(parent, target_name)
    if os.path.isdir(direct):
        return direct

    target_norm = normalize_name(target_name)
    if os.path.isdir(parent):
        for root, dirs, _files in os.walk(parent):
            for dirname in dirs:
                if normalize_name(dirname) == target_norm:
                    return os.path.join(root, dirname)
    return None


def collect_noise_files(
    data_root: str,
    machine: str,
    noise_key: str,
    machine_ids: Sequence[str],
) -> List[Tuple[str, int, str]]:
    domain_dir = resolve_domain_dir(data_root, machine, noise_key)
    all_items: List[Tuple[str, int, str]] = []

    print(f"[Domain] {machine} {NOISE_READABLE[noise_key]} -> {domain_dir}")

    for machine_id in machine_ids:
        id_dir = find_subdir_case_insensitive(domain_dir, machine_id)
        if id_dir is None:
            print(f"[警告] 未找到机器ID目录：{machine_id}，跳过。")
            continue

        for class_name, label in (("normal", 0), ("abnormal", 1)):
            class_dir = find_subdir_case_insensitive(id_dir, class_name)
            if class_dir is None:
                print(f"[警告] 未找到目录：{id_dir}/{class_name}，跳过。")
                continue

            wav_files: List[str] = []
            for root, _dirs, files in os.walk(class_dir):
                for fname in files:
                    if fname.lower().endswith(".wav"):
                        wav_files.append(os.path.join(root, fname))
            wav_files.sort()

            for wav_path in wav_files:
                fname = os.path.basename(wav_path)
                file_id = f"{machine}_{noise_key}_{machine_id}_{class_name}_{fname}"
                all_items.append((wav_path, label, file_id))

    normal_n = sum(1 for _p, y, _fid in all_items if y == 0)
    abnormal_n = sum(1 for _p, y, _fid in all_items if y == 1)
    print(
        f"[Collect] {MACHINE_CONFIGS[machine]['cn']} {NOISE_READABLE[noise_key]} | "
        f"normal={normal_n}, abnormal={abnormal_n}, total={len(all_items)}"
    )

    if normal_n == 0 or abnormal_n == 0:
        raise RuntimeError(f"{machine} {noise_key} 缺少normal或abnormal文件，无法进行二分类实验。")
    return all_items


def stratified_split_64_16_20(
    items: Sequence[Tuple[str, int, str]],
    seed: int,
) -> Tuple[List[Tuple[str, int, str]], List[Tuple[str, int, str]], List[Tuple[str, int, str]]]:
    items = list(items)
    labels = [x[1] for x in items]

    try:
        train_dev, test = train_test_split(
            items,
            test_size=0.20,
            random_state=seed,
            stratify=labels,
            shuffle=True,
        )
        train_dev_labels = [x[1] for x in train_dev]
        train, dev = train_test_split(
            train_dev,
            test_size=0.20,
            random_state=seed,
            stratify=train_dev_labels,
            shuffle=True,
        )
    except ValueError as exc:
        raise RuntimeError(f"分层划分失败，请检查每类文件数量：{exc}") from exc

    def describe(name: str, subset: Sequence[Tuple[str, int, str]]) -> None:
        normal = sum(1 for _p, y, _fid in subset if y == 0)
        abnormal = sum(1 for _p, y, _fid in subset if y == 1)
        print(f"[Split] {name}: normal={normal}, abnormal={abnormal}, total={len(subset)}")

    describe("Train", train)
    describe("Dev", dev)
    describe("Test", test)
    return list(train), list(dev), list(test)


def feature_path_from_file_id(file_id: str, feature_root: str) -> str:
    parts = str(file_id).split("_")
    if len(parts) < 6:
        raise RuntimeError(f"file_id格式不符合预期：{file_id}")

    machine = parts[0]
    noise = parts[1]
    machine_id = parts[2] + "_" + parts[3]
    class_name = parts[4]
    fname = "_".join(parts[5:])
    stem = os.path.splitext(fname)[0]
    return os.path.join(feature_root, machine, noise, machine_id, class_name, stem + ".npz")


class SegmentDataset(Dataset):
    def __init__(
        self,
        file_items: Sequence[Tuple[str, int, str]],
        cache_dir: Optional[str],
        use_cache: bool,
        feature_root: str,
        cmvn_eps: float,
    ):
        self.samples: List[Tuple[np.ndarray, np.ndarray, int, str]] = []
        self.cache_dir = cache_dir
        self.use_cache = bool(use_cache)
        self.feature_root = feature_root
        self.cmvn_eps = cmvn_eps

        if self.use_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        fallback_count = 0
        for index, (wav_path, label, file_id) in enumerate(file_items, start=1):
            try:
                acoustic_segments, industrial_segments, used_fallback = self._load_or_extract(
                    wav_path,
                    file_id,
                )
                fallback_count += int(used_fallback)
                for acoustic, industrial in zip(acoustic_segments, industrial_segments):
                    self.samples.append((acoustic, industrial, int(label), file_id))
            except Exception as exc:
                print(f"[警告] 特征读取失败：{wav_path} | {exc}")

            if index % 100 == 0 or index == len(file_items):
                print(
                    f"[Dataset进度] files={index}/{len(file_items)}, segments={len(self.samples)}"
                )

        print(
            f"[Dataset完成] files={len(file_items)}, segments={len(self.samples)}, "
            f"raw_wav_fallback={fallback_count}"
        )
        if len(self.samples) == 0:
            raise RuntimeError("Dataset中没有可用片段，请检查数据路径、WAV文件或特征提取依赖。")

    def _cache_path(self, file_id: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, safe_name(file_id) + ".npz")

    def _load_or_extract(
        self,
        wav_path: str,
        file_id: str,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        cache_path = self._cache_path(file_id)
        if self.use_cache and cache_path and os.path.exists(cache_path):
            try:
                with np.load(cache_path, allow_pickle=True) as data:
                    if "feature_version" not in data:
                        raise RuntimeError("缓存缺少feature_version，拒绝复用旧段级缓存")
                    saved_version = str(np.asarray(data["feature_version"]).reshape(-1)[0])
                    if saved_version != FEATURE_VERSION:
                        raise RuntimeError(f"缓存版本={saved_version}，当前要求={FEATURE_VERSION}")
                    acoustic = np.asarray(data["acoustic"], dtype=np.float32)
                    industrial = np.asarray(data["industrial"], dtype=np.float32)
                if acoustic.ndim != 3 or acoustic.shape[-1] != ACOUSTIC_INPUT_DIM:
                    raise RuntimeError(f"声学缓存形状错误：{acoustic.shape}")
                if industrial.ndim != 3 or industrial.shape[-1] != INDUSTRIAL_INPUT_DIM:
                    raise RuntimeError(f"IAPD缓存形状错误：{industrial.shape}")
                return acoustic, industrial, False
            except Exception as exc:
                print(f"[Cache警告] 缓存损坏，将重提取：{cache_path} | {exc}")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

        precomputed_path = None
        used_fallback = True
        if self.feature_root:
            candidate = feature_path_from_file_id(file_id, self.feature_root)
            if os.path.exists(candidate):
                precomputed_path = candidate
                used_fallback = False

        acoustic, industrial = extract_acoustic_iapd_segments(
            wav_path=wav_path,
            precomputed_path=precomputed_path,
            cmvn_eps=self.cmvn_eps,
        )

        if self.use_cache and cache_path:
            np.savez_compressed(
                cache_path,
                acoustic=acoustic.astype(np.float32),
                industrial=industrial.astype(np.float32),
                feature_version=np.asarray([FEATURE_VERSION]),
                acoustic_definition=np.asarray(
                    ["MFCC39(spafe)+LFCC39(spafe)+CQCC39(librosa.cqt-log1p-DCT)+GFCC39(spafe)"]
                ),
            )
        return acoustic, industrial, used_fallback

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        acoustic, industrial, label, file_id = self.samples[idx]
        return (
            torch.tensor(acoustic, dtype=torch.float32),
            torch.tensor(industrial, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            file_id,
        )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


def build_dataloaders(
    args: argparse.Namespace,
    machine: str,
    noise_key: str,
    output_dir: str,
    device: torch.device,
):
    all_items = collect_noise_files(
        data_root=args.data_root,
        machine=machine,
        noise_key=noise_key,
        machine_ids=parse_csv(args.machine_ids),
    )
    train_files, dev_files, test_files = stratified_split_64_16_20(all_items, args.seed)

    split_path = os.path.join(output_dir, "splits", "split.npz")
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    np.savez(
        split_path,
        train=np.asarray(train_files, dtype=object),
        dev=np.asarray(dev_files, dtype=object),
        test=np.asarray(test_files, dtype=object),
    )
    print(f"[Split] saved to: {split_path}")

    cache_dir = os.path.join(
        args.output_root,
        "_shared_feature_cache",
        FEATURE_VERSION,
        machine,
        noise_key,
    )

    train_dataset = SegmentDataset(
        train_files,
        cache_dir=cache_dir,
        use_cache=args.use_cache,
        feature_root=args.feature_root,
        cmvn_eps=args.cmvn_eps,
    )
    dev_dataset = SegmentDataset(
        dev_files,
        cache_dir=cache_dir,
        use_cache=args.use_cache,
        feature_root=args.feature_root,
        cmvn_eps=args.cmvn_eps,
    )
    test_dataset = SegmentDataset(
        test_files,
        cache_dir=cache_dir,
        use_cache=args.use_cache,
        feature_root=args.feature_root,
        cmvn_eps=args.cmvn_eps,
    )

    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        device=device,
        num_workers=args.num_workers,
    )
    dev_loader = make_loader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        device=device,
        num_workers=args.num_workers,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        device=device,
        num_workers=args.num_workers,
    )

    return train_loader, dev_loader, test_loader, train_files, dev_files, test_files
