from __future__ import annotations

import ast
import csv
import warnings
from abc import ABC
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, cast, override

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import load_dataset
from medmnist import INFO, BloodMNIST, DermaMNIST
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


NIH_HF_DATASET_ID = "BahaaEldin0/NIH-Chest-Xray-14"
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

ODIR_CLASS_ORDER = ["N", "D", "G", "C", "A", "H", "M", "O"]
ODIR_CLASS_DISPLAY = {
    "N": "normal",
    "D": "diabetic_retinopathy",
    "G": "glaucoma",
    "C": "cataract",
    "A": "age_related_macular_degeneration",
    "H": "hypertension",
    "M": "myopia",
    "O": "other_abnormalities",
}


# 用于蒸馏的数据集划分
@dataclass(frozen=True)
class DistillationSplits:
    train_set: Any
    val_set: Any
    test_set: Any
    num_classes: int
    class_names: dict[int, str]


# 用于测试的数据集划分
@dataclass(frozen=True)
class StudentEvalSplits:
    val_set: Any
    test_set: Any
    num_classes: int
    class_names: list[str]


class HFNIHChestXraySplit:
    def __init__(
        self,
        hf_split: HFDataset,
        image_column: str,
        label_column: str,
        label_to_index: dict[str, int],
        no_finding_idx: int,
    ) -> None:
        self.hf_split = hf_split
        self.image_column = image_column
        self.label_column = label_column
        self.label_to_index = label_to_index
        self.no_finding_idx = int(no_finding_idx)
        self.labels = self._build_label_array()

    def __len__(self) -> int:
        return int(len(self.hf_split))

    @staticmethod
    def _normalize_label_name(name: str) -> str:
        return name.strip().replace("_", " ").replace("-", " ").lower()

    def _extract_labels(self, raw_value: Any) -> list[str]:
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

    def _map_primary_label(self, raw_value: Any) -> int:
        labels = self._extract_labels(raw_value)
        mapped_indices: list[int] = []

        for label in labels:
            key = self._normalize_label_name(label)
            idx = self.label_to_index.get(key)
            if idx is not None:
                mapped_indices.append(int(idx))

        # Preserve source annotation order: use the first disease label if present.
        for idx in mapped_indices:
            if idx != self.no_finding_idx:
                return int(idx)

        if self.no_finding_idx in mapped_indices:
            return int(self.no_finding_idx)

        return int(self.no_finding_idx)

    def _build_label_array(self) -> np.ndarray:
        raw_labels = self.hf_split[self.label_column]
        labels = np.empty((len(raw_labels),), dtype=np.int64)
        for idx, raw in enumerate(raw_labels):
            labels[idx] = self._map_primary_label(raw)
        return labels

    @staticmethod
    def _to_rgb_uint8(image_value: Any) -> np.ndarray:
        if isinstance(image_value, Image.Image):
            return np.asarray(image_value.convert("RGB"), dtype=np.uint8)

        if isinstance(image_value, dict):
            image_bytes = image_value.get("bytes")
            image_path = image_value.get("path")
            if image_bytes is not None:
                pil = Image.open(BytesIO(image_bytes)).convert("RGB")
                return np.asarray(pil, dtype=np.uint8)
            if image_path is not None:
                pil = Image.open(image_path).convert("RGB")
                return np.asarray(pil, dtype=np.uint8)

        arr = np.asarray(image_value)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], repeats=3, axis=2)
        elif arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, repeats=3, axis=2)

        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32, copy=False)
            max_val = float(np.max(arr)) if arr.size > 0 else 0.0
            if max_val <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        return np.ascontiguousarray(arr)

    def get_image_and_label(self, index: int) -> tuple[np.ndarray, int]:
        idx = int(index)
        row = self.hf_split[idx]
        image_np = self._to_rgb_uint8(row[self.image_column])
        label = int(self.labels[idx])
        return image_np, label

    def get_all_labels(self) -> np.ndarray:
        return self.labels


class ODIR5KSplit:
    def __init__(self, image_paths: list[Path], labels: np.ndarray) -> None:
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = np.asarray(labels).reshape(-1).astype(np.int64, copy=False)
        if len(self.image_paths) != int(self.labels.shape[0]):
            raise ValueError("ODIR-5K split images/labels size mismatch.")

    def __len__(self) -> int:
        return int(len(self.image_paths))

    def get_image_and_label(self, index: int) -> tuple[np.ndarray, int]:
        idx = int(index)
        image_path = self.image_paths[idx]
        with Image.open(image_path) as pil_image:
            image_np = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
        return np.ascontiguousarray(image_np), int(self.labels[idx])

    def get_all_labels(self) -> np.ndarray:
        return self.labels


class MedMNISTImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        split_data: Any,
        transform: transforms.Compose | None = None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self._split_accessor = split_data if hasattr(split_data, "get_image_and_label") else None

        if self._split_accessor is None:
            self.images = np.ascontiguousarray(split_data.imgs)
            self.labels = np.asarray(split_data.labels).reshape(-1).astype(np.int64, copy=False)
        else:
            self.images = None
            self.labels = np.asarray(split_data.get_all_labels()).reshape(-1).astype(np.int64, copy=False)

    @override
    def __len__(self) -> int:
        if self._split_accessor is None:
            return int(self.images.shape[0])
        return int(len(self._split_accessor))

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._split_accessor is None:
            image_np = np.asarray(self.images[index])
            label_value = int(self.labels[index])
        else:
            image_np, label_value = self._split_accessor.get_image_and_label(index)

        if image_np.ndim == 2:
            image_np = np.repeat(image_np[..., None], repeats=3, axis=2)
        elif image_np.ndim == 3 and image_np.shape[0] in (1, 3) and image_np.shape[2] not in (1, 3):
            image_np = np.transpose(image_np, (1, 2, 0))
        if image_np.ndim == 3 and image_np.shape[2] == 1:
            image_np = np.repeat(image_np, repeats=3, axis=2)

        if self.transform is None:
            # torch.from_numpy expects a writable buffer; some dataset backends return read-only arrays.
            image_np_c = np.ascontiguousarray(image_np)
            if not image_np_c.flags.writeable:
                image_np_c = image_np_c.copy()
            image = torch.from_numpy(image_np_c).permute(2, 0, 1).float()
            if image.max().item() > 1.0:
                image = image / 255.0
        else:
            image = self.transform(image_np)

        label = torch.tensor(label_value, dtype=torch.long)
        return image, label


class BaseDatasetSpec(ABC):
    name: str
    prompt_prefix: str

    def class_names(self, data_root: str = "data", image_size: int = 224) -> dict[int, str]:
        raise NotImplementedError

    def load_distillation_splits(self, data_root: str, image_size: int) -> DistillationSplits:
        raise NotImplementedError

    def load_student_eval_splits(self, data_root: str, image_size: int) -> StudentEvalSplits:
        raise NotImplementedError

    def load_lora_train_split(self, data_root: str, image_size: int) -> tuple[Any, dict[int, str]]:
        split_bundle = self.load_distillation_splits(data_root=data_root, image_size=image_size)
        return split_bundle.train_set, split_bundle.class_names

    def build_class_prompts(self, class_names: dict[int, str]) -> dict[int, str]:
        prompts: dict[int, str] = {}
        for class_id, name in class_names.items():
            cleaned = str(name).replace("_", " ")
            prompts[class_id] = f"{self.prompt_prefix} {cleaned}"
        return prompts


class BaseMedMNISTSpec(BaseDatasetSpec, ABC):
    name: str
    medmnist_key: str
    split_class: type[Any]
    prompt_prefix: str

    def _class_names(self) -> dict[int, str]:
        label_meta = cast(dict[str, str], INFO[self.medmnist_key]["label"])
        return {int(k): v for k, v in label_meta.items()}

    @override
    def class_names(self, data_root: str = "data", image_size: int = 224) -> dict[int, str]:
        return self._class_names()

    @override
    def load_lora_train_split(self, data_root: str, image_size: int) -> tuple[Any, dict[int, str]]:
        train_set = self.split_class(split="train", download=True, root=data_root, size=image_size)
        return train_set, self._class_names()

    @override
    def load_distillation_splits(self, data_root: str, image_size: int) -> DistillationSplits:
        train_set = self.split_class(split="train", download=True, root=data_root, size=image_size)
        val_set = self.split_class(split="val", download=True, root=data_root, size=image_size)
        test_set = self.split_class(split="test", download=True, root=data_root, size=image_size)
        class_names = self._class_names()
        return DistillationSplits(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            num_classes=len(class_names),
            class_names=class_names,
        )

    @override
    def load_student_eval_splits(self, data_root: str, image_size: int) -> StudentEvalSplits:
        val_set = self.split_class(split="val", download=True, root=data_root, size=image_size)
        test_set = self.split_class(split="test", download=True, root=data_root, size=image_size)
        class_names = self._class_names()
        ordered_names = [class_names[i] for i in range(len(class_names))]
        return StudentEvalSplits(
            val_set=val_set,
            test_set=test_set,
            num_classes=len(class_names),
            class_names=ordered_names,
        )


class DermaMNISTSpec(BaseMedMNISTSpec):
    name = "dermamnist"
    medmnist_key = "dermamnist"
    split_class = DermaMNIST
    prompt_prefix = "dermoscopic image of"


class BloodMNISTSpec(BaseMedMNISTSpec):
    name = "bloodmnist"
    medmnist_key = "bloodmnist"
    split_class = BloodMNIST
    prompt_prefix = "microscopic image of blood cell"


class NIHChestXray14Spec(BaseDatasetSpec):
    name = "nih_chest_xray14"
    prompt_prefix = "chest x-ray showing"
    hf_dataset_id = NIH_HF_DATASET_ID

    def _class_names(self) -> dict[int, str]:
        return {idx: label for idx, label in enumerate(NIH_CLASS_ORDER)}

    @staticmethod
    def _normalize_label_name(name: str) -> str:
        return name.strip().replace("_", " ").replace("-", " ").lower()

    def _build_label_index(self) -> tuple[dict[str, int], int]:
        label_to_index = {self._normalize_label_name(label): idx for idx, label in enumerate(NIH_CLASS_ORDER)}
        label_to_index["pleural thickening"] = label_to_index[self._normalize_label_name("Pleural_Thickening")]
        no_finding_idx = label_to_index[self._normalize_label_name("No Finding")]
        return label_to_index, no_finding_idx

    def _resolve_split_key(self, dataset_dict: HFDatasetDict, candidates: tuple[str, ...]) -> str | None:
        key_map = {name.lower(): name for name in dataset_dict.keys()}
        for candidate in candidates:
            if candidate in key_map:
                return key_map[candidate]
        return None

    def _infer_columns(self, split: HFDataset) -> tuple[str, str]:
        sample = split[0]

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

    @override
    def class_names(self, data_root: str = "data", image_size: int = 224) -> dict[int, str]:
        return self._class_names()

    @override
    def load_lora_train_split(self, data_root: str, image_size: int) -> tuple[Any, dict[int, str]]:
        train_split = cast(HFDataset, load_dataset(self.hf_dataset_id, split="train"))
        image_column, label_column = self._infer_columns(train_split)
        label_to_index, no_finding_idx = self._build_label_index()
        wrapped = HFNIHChestXraySplit(
            hf_split=train_split,
            image_column=image_column,
            label_column=label_column,
            label_to_index=label_to_index,
            no_finding_idx=no_finding_idx,
        )
        return wrapped, self._class_names()

    @override
    def load_distillation_splits(self, data_root: str, image_size: int) -> DistillationSplits:
        dataset_dict = cast(HFDatasetDict, load_dataset(self.hf_dataset_id))

        train_key = self._resolve_split_key(dataset_dict, ("train",))
        if train_key is None:
            raise KeyError("NIH dataset missing train split.")

        val_key = self._resolve_split_key(dataset_dict, ("valid", "validation", "val"))
        test_key = self._resolve_split_key(dataset_dict, ("test",))

        train_split = dataset_dict[train_key]
        val_split = dataset_dict[val_key] if val_key is not None else None
        test_split = dataset_dict[test_key] if test_key is not None else None

        if val_split is None and test_split is None:
            first_split = train_split.train_test_split(test_size=0.2, seed=42, shuffle=True)
            train_split = first_split["train"]
            tail_split = first_split["test"].train_test_split(test_size=0.5, seed=42, shuffle=True)
            val_split = tail_split["train"]
            test_split = tail_split["test"]
        elif val_split is None:
            val_split = train_split.train_test_split(test_size=0.1, seed=42, shuffle=True)["test"]
        elif test_split is None:
            test_split = val_split

        image_column, label_column = self._infer_columns(train_split)
        label_to_index, no_finding_idx = self._build_label_index()

        wrapped_train = HFNIHChestXraySplit(
            hf_split=train_split,
            image_column=image_column,
            label_column=label_column,
            label_to_index=label_to_index,
            no_finding_idx=no_finding_idx,
        )
        wrapped_val = HFNIHChestXraySplit(
            hf_split=cast(HFDataset, val_split),
            image_column=image_column,
            label_column=label_column,
            label_to_index=label_to_index,
            no_finding_idx=no_finding_idx,
        )
        wrapped_test = HFNIHChestXraySplit(
            hf_split=cast(HFDataset, test_split),
            image_column=image_column,
            label_column=label_column,
            label_to_index=label_to_index,
            no_finding_idx=no_finding_idx,
        )

        class_names = self._class_names()
        return DistillationSplits(
            train_set=wrapped_train,
            val_set=wrapped_val,
            test_set=wrapped_test,
            num_classes=len(class_names),
            class_names=class_names,
        )

    @override
    def load_student_eval_splits(self, data_root: str, image_size: int) -> StudentEvalSplits:
        split_bundle = self.load_distillation_splits(data_root=data_root, image_size=image_size)
        class_names = split_bundle.class_names
        ordered_names = [class_names[i] for i in range(len(class_names))]
        return StudentEvalSplits(
            val_set=split_bundle.val_set,
            test_set=split_bundle.test_set,
            num_classes=split_bundle.num_classes,
            class_names=ordered_names,
        )


class ODIR5KSpec(BaseDatasetSpec):
    name = "odir-5k"
    prompt_prefix = "retinal fundus image of"

    def _class_names(self) -> dict[int, str]:
        return {idx: ODIR_CLASS_DISPLAY[code] for idx, code in enumerate(ODIR_CLASS_ORDER)}

    def _resolve_dataset_root(self, data_root: str) -> Path:
        root = Path(data_root) / "ODIR-5K"
        if not root.exists():
            raise FileNotFoundError(
                f"ODIR-5K root not found: {root}. Expected structure under ./data/ODIR-5K"
            )
        return root

    def _resolve_csv_path(self, dataset_root: Path) -> Path:
        candidates = [
            dataset_root / "full_df.csv",
            dataset_root / "ODIR-5K" / "full_df.csv",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            f"Cannot find ODIR-5K metadata CSV. Tried: {[str(p) for p in candidates]}"
        )

    def _resolve_image_dirs(self, dataset_root: Path) -> list[Path]:
        candidates = [
            dataset_root / "preprocessed_images",
            dataset_root / "ODIR-5K" / "Training Images",
            dataset_root / "Training Images",
            dataset_root / "ODIR-5K" / "Testing Images",
            dataset_root / "Testing Images",
        ]
        image_dirs = [path for path in candidates if path.exists()]
        if not image_dirs:
            raise FileNotFoundError(
                f"No ODIR-5K image directories found. Tried: {[str(p) for p in candidates]}"
            )
        return image_dirs

    def _resolve_image_path(self, filename: str, image_dirs: list[Path]) -> Path | None:
        cleaned = str(filename).strip()
        if not cleaned:
            return None
        base_name = Path(cleaned).name
        for image_dir in image_dirs:
            candidate = image_dir / base_name
            if candidate.exists():
                return candidate
        return None

    def _label_from_flags(self, row: dict[str, Any]) -> str | None:
        positives: list[str] = []
        for code in ODIR_CLASS_ORDER:
            raw = row.get(code)
            if raw is None:
                continue
            try:
                if float(raw) > 0.5:
                    positives.append(code)
            except (TypeError, ValueError):
                continue
        return positives[0] if positives else None

    def _label_from_target(self, row: dict[str, Any]) -> str | None:
        raw_target = row.get("target")
        if raw_target is None:
            return None
        text = str(raw_target).strip()
        if not text:
            return None
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return None
        if not isinstance(parsed, (list, tuple)) or not parsed:
            return None
        arr = np.asarray(parsed, dtype=np.float32).reshape(-1)
        if arr.size < len(ODIR_CLASS_ORDER):
            return None
        idx = int(np.argmax(arr[: len(ODIR_CLASS_ORDER)]))
        if float(arr[idx]) <= 0.0:
            return None
        return ODIR_CLASS_ORDER[idx]

    def _parse_label_code(self, row: dict[str, Any]) -> str | None:
        raw_labels = row.get("labels")
        if raw_labels is not None:
            text = str(raw_labels).strip()
            if text:
                try:
                    parsed = ast.literal_eval(text)
                except Exception:
                    parsed = text

                if isinstance(parsed, (list, tuple)):
                    for token in parsed:
                        code = str(token).strip().upper()
                        if code in ODIR_CLASS_ORDER:
                            return code
                else:
                    code = str(parsed).strip().upper()
                    if code in ODIR_CLASS_ORDER:
                        return code

        from_flags = self._label_from_flags(row)
        if from_flags is not None:
            return from_flags

        return self._label_from_target(row)

    def _load_samples(self, data_root: str) -> tuple[list[Path], np.ndarray]:
        dataset_root = self._resolve_dataset_root(data_root)
        csv_path = self._resolve_csv_path(dataset_root)
        image_dirs = self._resolve_image_dirs(dataset_root)

        image_paths: list[Path] = []
        labels: list[int] = []
        skipped_no_label = 0
        skipped_no_image = 0

        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                code = self._parse_label_code(row)
                if code is None:
                    skipped_no_label += 1
                    continue

                filename = str(row.get("filename", "")).strip()
                if not filename:
                    filename = str(row.get("Right-Fundus", "")).strip()
                if not filename:
                    filename = str(row.get("Left-Fundus", "")).strip()

                resolved = self._resolve_image_path(filename, image_dirs)
                if resolved is None:
                    raw_filepath = str(row.get("filepath", "")).strip()
                    if raw_filepath:
                        resolved = self._resolve_image_path(Path(raw_filepath).name, image_dirs)

                if resolved is None:
                    skipped_no_image += 1
                    continue

                image_paths.append(resolved)
                labels.append(ODIR_CLASS_ORDER.index(code))

        if not image_paths:
            raise RuntimeError(f"No usable ODIR-5K samples found from {csv_path}")

        if skipped_no_label > 0 or skipped_no_image > 0:
            warnings.warn(
                "ODIR-5K dropped rows during parsing: "
                f"no_label={skipped_no_label}, no_image={skipped_no_image}",
                RuntimeWarning,
            )

        return image_paths, np.asarray(labels, dtype=np.int64)

    def _build_splits(self, data_root: str) -> tuple[ODIR5KSplit, ODIR5KSplit, ODIR5KSplit]:
        image_paths, labels = self._load_samples(data_root)
        indices = np.arange(labels.shape[0], dtype=np.int64)

        try:
            train_idx, tail_idx = train_test_split(
                indices,
                test_size=0.2,
                random_state=42,
                shuffle=True,
                stratify=labels,
            )
            tail_labels = labels[tail_idx]
            val_idx, test_idx = train_test_split(
                tail_idx,
                test_size=0.5,
                random_state=42,
                shuffle=True,
                stratify=tail_labels,
            )
        except ValueError:
            warnings.warn(
                "ODIR-5K stratified split failed, fallback to random split.",
                RuntimeWarning,
            )
            train_idx, tail_idx = train_test_split(
                indices,
                test_size=0.2,
                random_state=42,
                shuffle=True,
                stratify=None,
            )
            val_idx, test_idx = train_test_split(
                tail_idx,
                test_size=0.5,
                random_state=42,
                shuffle=True,
                stratify=None,
            )

        train_split = ODIR5KSplit([image_paths[int(i)] for i in train_idx], labels[train_idx])
        val_split = ODIR5KSplit([image_paths[int(i)] for i in val_idx], labels[val_idx])
        test_split = ODIR5KSplit([image_paths[int(i)] for i in test_idx], labels[test_idx])
        return train_split, val_split, test_split

    @override
    def class_names(self, data_root: str = "data", image_size: int = 224) -> dict[int, str]:
        return self._class_names()

    @override
    def load_lora_train_split(self, data_root: str, image_size: int) -> tuple[Any, dict[int, str]]:
        train_set, _, _ = self._build_splits(data_root=data_root)
        return train_set, self._class_names()

    @override
    def load_distillation_splits(self, data_root: str, image_size: int) -> DistillationSplits:
        train_set, val_set, test_set = self._build_splits(data_root=data_root)
        class_names = self._class_names()
        return DistillationSplits(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            num_classes=len(class_names),
            class_names=class_names,
        )

    @override
    def load_student_eval_splits(self, data_root: str, image_size: int) -> StudentEvalSplits:
        _, val_set, test_set = self._build_splits(data_root=data_root)
        class_names = self._class_names()
        ordered_names = [class_names[i] for i in range(len(class_names))]
        return StudentEvalSplits(
            val_set=val_set,
            test_set=test_set,
            num_classes=len(class_names),
            class_names=ordered_names,
        )


def _iter_spec_classes() -> list[type[BaseDatasetSpec]]:
    pending = list(BaseDatasetSpec.__subclasses__())
    all_classes: list[type[BaseDatasetSpec]] = []

    while pending:
        cls = pending.pop()
        all_classes.append(cls)
        pending.extend(cls.__subclasses__())

    return all_classes


def _build_registry() -> dict[str, BaseDatasetSpec]:
    registry: dict[str, BaseDatasetSpec] = {}
    for spec_cls in _iter_spec_classes():
        if spec_cls in {BaseDatasetSpec, BaseMedMNISTSpec}:
            continue
        spec = spec_cls()
        key = getattr(spec, "name", "").lower().strip()
        if key:
            registry[key] = spec
    return registry


def supported_datasets() -> tuple[str, ...]:
    return tuple(sorted(_build_registry().keys()))


def get_dataset_spec(dataset: str) -> BaseDatasetSpec:
    key = dataset.lower().strip()
    registry = _build_registry()
    if key not in registry:
        supported = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unsupported dataset: {dataset}. Supported datasets: {supported}")
    return registry[key]
