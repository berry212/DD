from __future__ import annotations

import argparse
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

# When executed as a file (python src/dd_distill/prepare_nih_chest_xray14.py),
# drop the script directory from sys.path so `import datasets` resolves the
# Hugging Face package instead of local src/dd_distill/datasets.py.
if sys.path:
    try:
        script_dir = Path(__file__).resolve().parent
        if Path(sys.path[0]).resolve() == script_dir:
            sys.path.pop(0)
    except Exception:
        pass

from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image


NIH_CLASS_ORDER = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
    "No Finding",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_label_name(name: str) -> str:
    return name.strip().replace("_", " ").replace("-", " ").lower()


def build_label_index() -> tuple[dict[str, int], list[str], int]:
    label_to_index = {normalize_label_name(label): idx for idx, label in enumerate(NIH_CLASS_ORDER)}
    label_to_index["pleural thickening"] = label_to_index[normalize_label_name("Pleural_Thickening")]
    no_finding_idx = label_to_index[normalize_label_name("No Finding")]
    return label_to_index, list(NIH_CLASS_ORDER), no_finding_idx


def resolve_splits(dataset_dict: DatasetDict, seed: int) -> tuple[Dataset, Dataset, Dataset]:
    keys = {k.lower(): k for k in dataset_dict.keys()}

    train_key = keys.get("train")
    valid_key = keys.get("valid") or keys.get("validation") or keys.get("val")
    test_key = keys.get("test")

    if train_key is None:
        first_key = next(iter(dataset_dict.keys()))
        train_key = first_key

    train_split = dataset_dict[train_key]
    valid_split = dataset_dict[valid_key] if valid_key is not None else None
    test_split = dataset_dict[test_key] if test_key is not None else None

    if valid_split is None and test_split is None:
        first_split = train_split.train_test_split(test_size=0.2, seed=seed, shuffle=True)
        train_split = first_split["train"]
        tail_split = first_split["test"].train_test_split(test_size=0.5, seed=seed, shuffle=True)
        valid_split = tail_split["train"]
        test_split = tail_split["test"]
    elif valid_split is None:
        val_split = train_split.train_test_split(test_size=0.1, seed=seed, shuffle=True)
        train_split = val_split["train"]
        valid_split = val_split["test"]
    elif test_split is None:
        test_split = valid_split

    return train_split, valid_split, test_split


def to_pil_rgb(image_value: Any) -> Image.Image:
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")

    if isinstance(image_value, dict):
        if image_value.get("bytes") is not None:
            return Image.open(io.BytesIO(image_value["bytes"])).convert("RGB")
        if image_value.get("path") is not None:
            return Image.open(image_value["path"]).convert("RGB")

    arr = np.asarray(image_value)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], repeats=3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, repeats=3, axis=2)

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32, copy=False)
        if float(arr.max(initial=0.0)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    return Image.fromarray(arr).convert("RGB")


def extract_labels(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        token = raw_value.strip()
        return [token] if token else []
    if isinstance(raw_value, (list, tuple)):
        labels: list[str] = []
        for item in raw_value:
            token = str(item).strip()
            if token:
                labels.append(token)
        return labels
    token = str(raw_value).strip()
    return [token] if token else []


def pick_primary_label(
    labels: list[str],
    label_to_index: dict[str, int],
    no_finding_idx: int,
    strict_labels: bool,
) -> tuple[int, list[str], int]:
    mapped_indices: list[int] = []
    unknown_labels: list[str] = []

    for label in labels:
        key = normalize_label_name(label)
        idx = label_to_index.get(key)
        if idx is None:
            unknown_labels.append(label)
            continue
        mapped_indices.append(idx)

    if unknown_labels and strict_labels:
        raise ValueError(f"Unknown labels found: {unknown_labels}")

    disease_indices = [idx for idx in mapped_indices if idx != no_finding_idx]
    if disease_indices:
        primary = min(disease_indices)
        disease_count = len(disease_indices)
    elif no_finding_idx in mapped_indices:
        primary = no_finding_idx
        disease_count = 0
    else:
        primary = no_finding_idx
        disease_count = 0

    return primary, unknown_labels, disease_count


def convert_split(
    split_name: str,
    split: Dataset,
    image_column: str,
    label_column: str,
    image_size: int,
    max_samples: int,
    label_to_index: dict[str, int],
    no_finding_idx: int,
    strict_labels: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    total_available = len(split)
    total = total_available if max_samples <= 0 else min(max_samples, total_available)

    images = np.empty((total, image_size, image_size, 3), dtype=np.uint8)
    labels = np.empty((total,), dtype=np.int64)

    unknown_counter: Counter[str] = Counter()
    multi_label_count = 0

    for idx in range(total):
        row = split[idx]
        pil = to_pil_rgb(row[image_column]).resize((image_size, image_size), resample=Image.BILINEAR)
        images[idx] = np.asarray(pil, dtype=np.uint8)

        raw_labels = extract_labels(row.get(label_column))
        label_idx, unknown_labels, disease_count = pick_primary_label(
            labels=raw_labels,
            label_to_index=label_to_index,
            no_finding_idx=no_finding_idx,
            strict_labels=strict_labels,
        )
        labels[idx] = label_idx
        if disease_count > 1:
            multi_label_count += 1
        for unknown in unknown_labels:
            unknown_counter[unknown] += 1

        if (idx + 1) % 500 == 0 or (idx + 1) == total:
            print(f"[Prepare-NIH] {split_name}: {idx + 1}/{total}")

    stats = {
        "split": split_name,
        "total_available": int(total_available),
        "used": int(total),
        "multi_label_samples": int(multi_label_count),
        "unknown_labels": dict(sorted(unknown_counter.items())),
    }
    return images, labels, stats


def infer_columns(dataset: Dataset) -> tuple[str, str]:
    sample = dataset[0]

    image_column = "image" if "image" in sample else "img" if "img" in sample else ""
    if not image_column:
        for key, value in sample.items():
            if isinstance(value, Image.Image):
                image_column = key
                break
    if not image_column:
        raise KeyError(f"Unable to infer image column from sample keys: {list(sample.keys())}")

    label_column = "label" if "label" in sample else "labels" if "labels" in sample else ""
    if not label_column:
        raise KeyError(f"Unable to infer label column from sample keys: {list(sample.keys())}")

    return image_column, label_column


def bincount_dict(values: np.ndarray, class_names: list[str]) -> dict[str, int]:
    counts = np.bincount(values, minlength=len(class_names))
    return {class_names[idx]: int(counts[idx]) for idx in range(len(class_names))}


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    output_npz = Path(args.output_npz) if args.output_npz else data_root / f"nih_chest_xray14_{args.image_size}.npz"
    metadata_path = (
        Path(args.output_metadata)
        if args.output_metadata
        else output_npz.with_name(f"{output_npz.stem}_metadata.json")
    )

    hf_cache_dir = data_root / "hf_cache"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Prepare-NIH] Loading dataset: {args.dataset_id}")
    dataset_dict = load_dataset(args.dataset_id, cache_dir=str(hf_cache_dir))
    train_split, val_split, test_split = resolve_splits(dataset_dict, seed=args.seed)

    image_column, label_column = infer_columns(train_split)
    label_to_index, class_names, no_finding_idx = build_label_index()

    train_images, train_labels, train_stats = convert_split(
        split_name="train",
        split=train_split,
        image_column=image_column,
        label_column=label_column,
        image_size=args.image_size,
        max_samples=args.max_train,
        label_to_index=label_to_index,
        no_finding_idx=no_finding_idx,
        strict_labels=args.strict_labels,
    )
    val_images, val_labels, val_stats = convert_split(
        split_name="valid",
        split=val_split,
        image_column=image_column,
        label_column=label_column,
        image_size=args.image_size,
        max_samples=args.max_val,
        label_to_index=label_to_index,
        no_finding_idx=no_finding_idx,
        strict_labels=args.strict_labels,
    )
    test_images, test_labels, test_stats = convert_split(
        split_name="test",
        split=test_split,
        image_column=image_column,
        label_column=label_column,
        image_size=args.image_size,
        max_samples=args.max_test,
        label_to_index=label_to_index,
        no_finding_idx=no_finding_idx,
        strict_labels=args.strict_labels,
    )

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels,
        test_images=test_images,
        test_labels=test_labels,
    )

    summary = {
        "dataset_id": args.dataset_id,
        "output_npz": str(output_npz),
        "output_metadata": str(metadata_path),
        "cache_dir": str(hf_cache_dir),
        "image_size": int(args.image_size),
        "class_names": class_names,
        "label_strategy": "multi-label to single-label by fixed NIH class order; No Finding used as fallback",
        "columns": {
            "image": image_column,
            "label": label_column,
        },
        "train": {
            **train_stats,
            "class_distribution": bincount_dict(train_labels, class_names),
        },
        "valid": {
            **val_stats,
            "class_distribution": bincount_dict(val_labels, class_names),
        },
        "test": {
            **test_stats,
            "class_distribution": bincount_dict(test_labels, class_names),
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("[Prepare-NIH] Completed")
    print(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and prepare NIH Chest X-ray 14 from Hugging Face into local NPZ")
    parser.add_argument("--dataset-id", type=str, default="BahaaEldin0/NIH-Chest-Xray-14")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-npz", type=str, default="")
    parser.add_argument("--output-metadata", type=str, default="")

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--strict-labels", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
