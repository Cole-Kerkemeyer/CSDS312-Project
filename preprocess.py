"""
Data Preprocessing Pipeline — Brain Tumor Classification & Segmentation
Author: Qianyi (Alfred) Chen
CSDS 312 Group 2

Usage:
    # Classification preprocessing (BRISC 2025):
    python preprocess.py --task classification --data-dir Classification/Data

    # Segmentation preprocessing (BraTS 2020):
    python preprocess.py --task segmentation --data-dir Segmentation/data/brats_unzipped/BraTS2020_training_data/content/data

    # Both:
    python preprocess.py --task all --clf-data Classification/Data --seg-data Segmentation/data/...
"""

import argparse
import json
import os
import re
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]


def load_manifest(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    before = len(df)
    df = df[df["is_mask"] == False].copy()
    print(f"[manifest] Loaded {before} rows, kept {len(df)} after removing mask entries.")
    return df


def summarize_split(name: str, df: pd.DataFrame) -> None:
    counts = {cls: (df["tumor_label"] == cls).sum() for cls in CLASS_NAMES if cls in df["tumor_label"].values}
    total = len(df)
    parts = "  |  ".join(f"{cls}: {n} ({n/total:.1%})" for cls, n in counts.items())
    print(f"[{name:5s}] {total:5d} samples  —  {parts}")


def compute_normalization_stats(data_dir: str, image_size: int = 224) -> tuple:
    """
    Compute per-channel mean and std from the training images.
    Uses ImageFolder so it expects train/ subfolder with class subfolders.
    Returns (mean, std) as lists of 3 floats.
    """
    train_dir = os.path.join(data_dir, "train")
    if not os.path.isdir(train_dir):
        print(f"[warn] No train/ subdirectory found at {train_dir}. Falling back to ImageFolder on data_dir.")
        train_dir = data_dir

    dataset = datasets.ImageFolder(
        root=train_dir,
        transform=transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]),
    )

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total = 0

    for images, _ in loader:
        n = images.size(0)
        mean += images.mean(dim=[0, 2, 3]) * n
        std += images.std(dim=[0, 2, 3]) * n
        total += n

    mean /= total
    std /= total
    return mean.tolist(), std.tolist()


def make_classification_split(
    df: pd.DataFrame,
    val_frac: float = 0.2,
    random_seed: int = 42,
) -> tuple:
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy() if "test" in df["split"].values else pd.DataFrame()

    val_df = train_df.sample(frac=val_frac, random_state=random_seed)
    train_df = train_df.drop(val_df.index)

    return train_df, val_df, test_df


def preprocess_classification(
    data_dir: str,
    csv_path: str,
    val_frac: float = 0.2,
    image_size: int = 224,
    output_dir: str = ".",
) -> dict:
    print("\n" + "=" * 60)
    print("CLASSIFICATION PREPROCESSING  (BRISC 2025)")
    print("=" * 60)

    df = load_manifest(csv_path)

    train_df, val_df, test_df = make_classification_split(df, val_frac=val_frac)
    summarize_split("train", train_df)
    summarize_split("val", val_df)
    if not test_df.empty:
        summarize_split("test", test_df)

    print("\nComputing normalization statistics from training images…")
    mean, std = compute_normalization_stats(data_dir, image_size=image_size)
    print(f"  mean = {[round(v, 4) for v in mean]}")
    print(f"  std  = {[round(v, 4) for v in std]}")

    # Save processed splits
    out = Path(output_dir) / "clf_splits"
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)
    if not test_df.empty:
        test_df.to_csv(out / "test.csv", index=False)

    stats = {"mean": mean, "std": std, "image_size": image_size}
    with open(out / "normalization_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved splits and stats → {out}/")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENTATION PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def scan_volumes(data_dir: str) -> list:
    pattern = re.compile(r"volume_(\d+)_slice_(\d+)")
    vol_ids = set()
    for path in glob.glob(os.path.join(data_dir, "*.h5")):
        match = pattern.search(os.path.basename(path))
        if match:
            vol_ids.add(int(match.group(1)))
    return sorted(vol_ids)


def count_slices_per_volume(data_dir: str) -> dict:
    pattern = re.compile(r"volume_(\d+)_slice_(\d+)")
    counts: dict = {}
    for path in glob.glob(os.path.join(data_dir, "*.h5")):
        match = pattern.search(os.path.basename(path))
        if match:
            vol_id = int(match.group(1))
            counts[vol_id] = counts.get(vol_id, 0) + 1
    return counts


def make_segmentation_split(
    vol_ids: list,
    val_count: int = 15,
    random_seed: int = 42,
) -> tuple:
    rng = np.random.default_rng(random_seed)
    shuffled = vol_ids.copy()
    rng.shuffle(shuffled)
    val_vols = shuffled[-val_count:]
    train_vols = shuffled[:-val_count]
    return sorted(train_vols), sorted(val_vols)


def preprocess_segmentation(
    data_dir: str,
    val_count: int = 15,
    output_dir: str = ".",
) -> dict:
    print("\n" + "=" * 60)
    print("SEGMENTATION PREPROCESSING  (BraTS 2020)")
    print("=" * 60)

    vol_ids = scan_volumes(data_dir)
    if not vol_ids:
        print(f"[error] No volume_*_slice_*.h5 files found in {data_dir}")
        return {}

    slice_counts = count_slices_per_volume(data_dir)
    total_slices = sum(slice_counts.values())

    print(f"Found {len(vol_ids)} unique patient volumes, {total_slices} total slices.")
    print(f"Slices per volume — min: {min(slice_counts.values())}, "
          f"max: {max(slice_counts.values())}, "
          f"mean: {total_slices / len(vol_ids):.1f}")

    train_vols, val_vols = make_segmentation_split(vol_ids, val_count=val_count)

    train_slices = sum(slice_counts[v] for v in train_vols)
    val_slices = sum(slice_counts[v] for v in val_vols)
    print(f"\nTrain: {len(train_vols)} volumes  ({train_slices} slices)")
    print(f"Val:   {len(val_vols)} volumes  ({val_slices} slices)")
    print("Patient-level split prevents data leakage between train and val.")

    out = Path(output_dir) / "seg_splits"
    out.mkdir(parents=True, exist_ok=True)
    split_data = {
        "train_volumes": train_vols,
        "val_volumes": val_vols,
        "total_volumes": len(vol_ids),
        "total_slices": total_slices,
        "val_count": val_count,
    }
    with open(out / "volume_split.json", "w") as f:
        json.dump(split_data, f, indent=2)

    print(f"\nSaved volume split → {out}/volume_split.json")
    return split_data


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Data preprocessing for brain tumor classification and segmentation."
    )
    parser.add_argument(
        "--task",
        choices=["classification", "segmentation", "all"],
        default="all",
        help="Which preprocessing task to run.",
    )
    parser.add_argument(
        "--clf-data",
        default="Classification/Data",
        help="Root directory of the BRISC 2025 classification dataset.",
    )
    parser.add_argument(
        "--clf-csv",
        default="Classification/Data/manifest.csv",
        help="Path to the BRISC 2025 CSV manifest.",
    )
    parser.add_argument(
        "--seg-data",
        default="Segmentation/data/brats_unzipped/BraTS2020_training_data/content/data",
        help="Directory containing BraTS 2020 .h5 slice files.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Fraction of training data to use as validation (classification).",
    )
    parser.add_argument(
        "--seg-val-count",
        type=int,
        default=15,
        help="Number of patient volumes to hold out for validation (segmentation).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image resize dimension for normalization stats computation.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save preprocessed split files and stats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.task in ("classification", "all"):
        preprocess_classification(
            data_dir=args.clf_data,
            csv_path=args.clf_csv,
            val_frac=args.val_frac,
            image_size=args.image_size,
            output_dir=args.output_dir,
        )

    if args.task in ("segmentation", "all"):
        preprocess_segmentation(
            data_dir=args.seg_data,
            val_count=args.seg_val_count,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
