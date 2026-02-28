import csv
import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

Layout = Literal["auto", "imagefolder", "manifest_csv"]
Sample = Tuple[str, str]
IndexedSample = Tuple[str, int]

_VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ClearDataConfig:
    root: str
    layout: Layout = "auto"
    bucket_glob: str = "bucket_*"
    manifest_name: str = "labels.csv"
    image_size: int = 224
    num_workers: int = 4
    batch_size: int = 64


@dataclass
class BucketInfo:
    name: str
    path: str
    layout: Literal["imagefolder", "manifest_csv"]


class ImageListDataset(Dataset):
    def __init__(self, samples: Sequence[IndexedSample], image_size: int):
        self.samples = list(samples)
        self.transform = transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, y = self.samples[idx]
        with Image.open(img_path) as img:
            x = self.transform(img.convert("RGB"))
        return x, y


def _discover_bucket_dirs(root: str, bucket_glob: str) -> List[str]:
    pattern = os.path.join(root, bucket_glob)
    dirs = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    return sorted(dirs)


def _has_imagefolder_layout(bucket_dir: str) -> bool:
    class_dirs = [
        p
        for p in sorted(glob.glob(os.path.join(bucket_dir, "*")))
        if os.path.isdir(p)
    ]
    if not class_dirs:
        return False
    for class_dir in class_dirs:
        for fn in os.listdir(class_dir):
            ext = os.path.splitext(fn)[1].lower()
            if ext in _VALID_EXTS:
                return True
    return False


def _detect_layout(
    bucket_dir: str, requested_layout: Layout, manifest_name: str
) -> Literal["imagefolder", "manifest_csv"]:
    if requested_layout in ("imagefolder", "manifest_csv"):
        return requested_layout

    manifest_path = os.path.join(bucket_dir, manifest_name)
    if os.path.isfile(manifest_path):
        return "manifest_csv"
    if _has_imagefolder_layout(bucket_dir):
        return "imagefolder"
    raise ValueError(
        f"Could not auto-detect layout for bucket {bucket_dir}. "
        f"Expected either {manifest_name} or class subdirectories with images."
    )


def _read_manifest_samples(bucket_dir: str, manifest_name: str) -> List[Sample]:
    manifest_path = os.path.join(bucket_dir, manifest_name)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    rows: List[Sample] = []
    with open(manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "filepath" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(
                f"Manifest {manifest_path} must contain filepath,label columns."
            )
        for row in reader:
            rel = row["filepath"]
            label = row["label"]
            abs_path = os.path.join(bucket_dir, rel)
            if os.path.isfile(abs_path):
                rows.append((abs_path, label))
    if not rows:
        raise ValueError(f"No valid rows found in manifest: {manifest_path}")
    return rows


def _read_imagefolder_samples(bucket_dir: str) -> List[Sample]:
    rows: List[Sample] = []
    class_dirs = [
        p
        for p in sorted(glob.glob(os.path.join(bucket_dir, "*")))
        if os.path.isdir(p)
    ]
    for class_dir in class_dirs:
        label = os.path.basename(class_dir)
        for fn in sorted(os.listdir(class_dir)):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in _VALID_EXTS:
                continue
            rows.append((os.path.join(class_dir, fn), label))
    if not rows:
        raise ValueError(f"No images found under imagefolder bucket: {bucket_dir}")
    return rows


def _read_bucket_samples(
    bucket_dir: str, layout: Literal["imagefolder", "manifest_csv"], manifest_name: str
) -> List[Sample]:
    if layout == "manifest_csv":
        return _read_manifest_samples(bucket_dir, manifest_name)
    return _read_imagefolder_samples(bucket_dir)


def _stable_class_map(all_bucket_samples: Sequence[Sequence[Sample]]) -> Dict[str, int]:
    labels = sorted({label for rows in all_bucket_samples for _, label in rows})
    return {label: idx for idx, label in enumerate(labels)}


def _to_indexed(samples: Sequence[Sample], class_to_idx: Dict[str, int]) -> List[IndexedSample]:
    return [(img_path, class_to_idx[label]) for img_path, label in samples]


def _split_train_shadow(
    indexed_samples: Sequence[IndexedSample], shadow_ratio: float, seed: int
) -> Tuple[List[IndexedSample], List[IndexedSample]]:
    if shadow_ratio <= 0.0:
        return list(indexed_samples), []

    n = len(indexed_samples)
    if n < 2:
        return list(indexed_samples), []

    n_shadow = int(round(n * shadow_ratio))
    n_shadow = max(1, min(n - 1, n_shadow))
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    shadow_idx = set(indices[:n_shadow])

    train_rows: List[IndexedSample] = []
    shadow_rows: List[IndexedSample] = []
    for i, row in enumerate(indexed_samples):
        if i in shadow_idx:
            shadow_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, shadow_rows


def make_bucket_loaders(
    cfg: ClearDataConfig,
    shadow_holdout_ratio: float = 0.0,
    seed: int = 42,
    save_class_map_path: Optional[str] = None,
):
    if not cfg.root or not os.path.isdir(cfg.root):
        raise FileNotFoundError(f"Data root does not exist: {cfg.root}")

    bucket_dirs = _discover_bucket_dirs(cfg.root, cfg.bucket_glob)
    if len(bucket_dirs) < 2:
        raise ValueError(
            f"Need at least 2 buckets to run CLEAR streaming, found {len(bucket_dirs)} in {cfg.root}"
        )

    buckets: List[BucketInfo] = []
    per_bucket_samples: List[List[Sample]] = []
    for bucket_dir in bucket_dirs:
        layout = _detect_layout(bucket_dir, cfg.layout, cfg.manifest_name)
        bucket_name = os.path.basename(bucket_dir)
        buckets.append(BucketInfo(name=bucket_name, path=bucket_dir, layout=layout))
        per_bucket_samples.append(
            _read_bucket_samples(bucket_dir, layout, cfg.manifest_name)
        )

    class_to_idx = _stable_class_map(per_bucket_samples)
    if save_class_map_path:
        with open(save_class_map_path, "w") as f:
            json.dump(class_to_idx, f, indent=2)

    pin_memory = torch.cuda.is_available()
    train_loaders: List[DataLoader] = []
    eval_full_loaders: List[DataLoader] = []
    shadow_loaders: List[Optional[DataLoader]] = []

    for b_idx, rows in enumerate(per_bucket_samples):
        indexed_rows = _to_indexed(rows, class_to_idx)
        train_rows, shadow_rows = _split_train_shadow(
            indexed_rows, shadow_holdout_ratio, seed=seed + b_idx
        )

        train_ds = ImageListDataset(train_rows, image_size=cfg.image_size)
        eval_ds = ImageListDataset(indexed_rows, image_size=cfg.image_size)
        shadow_ds = (
            ImageListDataset(shadow_rows, image_size=cfg.image_size)
            if len(shadow_rows) > 0
            else None
        )

        train_loaders.append(
            DataLoader(
                train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=pin_memory,
            )
        )
        eval_full_loaders.append(
            DataLoader(
                eval_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=pin_memory,
            )
        )
        shadow_loaders.append(
            DataLoader(
                shadow_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=pin_memory,
            )
            if shadow_ds is not None
            else None
        )

    return buckets, class_to_idx, train_loaders, eval_full_loaders, shadow_loaders
