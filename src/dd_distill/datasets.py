from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from io import BytesIO
from typing import Any, cast, override

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from datasets import load_dataset
from medmnist import INFO, BloodMNIST, DermaMNIST
from PIL import Image
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


@dataclass(frozen=True)
class DistillationSplits:
    train_set: Any
    val_set: Any
    test_set: Any
    num_classes: int
    class_names: dict[int, str]


@dataclass(frozen=True)
class StudentEvalSplits:
    val_set: Any
    test_set: Any
    num_classes: int
    class_names: list[str]


@dataclass(frozen=True)
class ArraySplitData:
    imgs: np.ndarray
    labels: np.ndarray


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

        disease_indices = [idx for idx in mapped_indices if idx != self.no_finding_idx]
        if disease_indices:
            return int(min(disease_indices))

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
            image = torch.from_numpy(np.ascontiguousarray(image_np)).permute(2, 0, 1).float()
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
