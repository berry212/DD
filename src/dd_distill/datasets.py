from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, override

import json

import numpy as np
import torch
from medmnist import INFO, BloodMNIST, DermaMNIST
from torch.utils.data import Dataset
from torchvision import transforms


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


class MedMNISTImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        split_data: Any,
        transform: transforms.Compose | None = None,
    ) -> None:
        super().__init__()
        self.images = split_data.imgs
        self.labels = split_data.labels.reshape(-1).astype(np.int64, copy=False)
        self.transform = transform

    @override
    def __len__(self) -> int:
        return int(self.images.shape[0])

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_np = np.asarray(self.images[index])
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
        label = torch.tensor(self.labels[index], dtype=torch.long)
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

    _fallback_classes = [
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

    def _resolve_npz_path(self, data_root: str, image_size: int) -> Path:
        root = Path(data_root)
        candidates = [
            root / f"{self.name}_{image_size}.npz",
            root / f"{self.name}_224.npz",
            root / f"{self.name}.npz",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            f"NIH Chest X-ray NPZ not found under {root}. "
            "Please run: uv run prepare-nih-chest-xray14 --data-root data"
        )

    def _metadata_path(self, npz_path: Path) -> Path:
        return npz_path.with_name(f"{npz_path.stem}_metadata.json")

    def _load_class_names(self, metadata_path: Path, num_classes: int) -> dict[int, str]:
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                names = payload.get("class_names")
                if isinstance(names, list) and len(names) >= num_classes:
                    return {idx: str(names[idx]) for idx in range(num_classes)}
            except Exception:
                pass

        fallback = list(self._fallback_classes)
        if len(fallback) < num_classes:
            fallback.extend([f"class_{i}" for i in range(len(fallback), num_classes)])
        return {idx: str(fallback[idx]) for idx in range(num_classes)}

    def _load_npz_splits(
        self,
        data_root: str,
        image_size: int,
    ) -> tuple[ArraySplitData, ArraySplitData, ArraySplitData, dict[int, str]]:
        npz_path = self._resolve_npz_path(data_root=data_root, image_size=image_size)
        arrays = np.load(npz_path)
        required_keys = {
            "train_images",
            "train_labels",
            "val_images",
            "val_labels",
            "test_images",
            "test_labels",
        }
        missing = sorted(required_keys.difference(arrays.files))
        if missing:
            raise KeyError(f"Missing keys in {npz_path}: {missing}")

        train_images = arrays["train_images"]
        train_labels = arrays["train_labels"].reshape(-1).astype(np.int64, copy=False)
        val_images = arrays["val_images"]
        val_labels = arrays["val_labels"].reshape(-1).astype(np.int64, copy=False)
        test_images = arrays["test_images"]
        test_labels = arrays["test_labels"].reshape(-1).astype(np.int64, copy=False)

        if train_labels.size == 0:
            raise ValueError(f"Empty train split in {npz_path}")

        max_label = int(max(train_labels.max(initial=0), val_labels.max(initial=0), test_labels.max(initial=0)))
        num_classes = max_label + 1
        class_names = self._load_class_names(self._metadata_path(npz_path), num_classes)

        train_set = ArraySplitData(imgs=train_images, labels=train_labels)
        val_set = ArraySplitData(imgs=val_images, labels=val_labels)
        test_set = ArraySplitData(imgs=test_images, labels=test_labels)
        return train_set, val_set, test_set, class_names

    @override
    def class_names(self, data_root: str = "data", image_size: int = 224) -> dict[int, str]:
        _, _, _, class_names = self._load_npz_splits(data_root=data_root, image_size=image_size)
        return class_names

    @override
    def load_distillation_splits(self, data_root: str, image_size: int) -> DistillationSplits:
        train_set, val_set, test_set, class_names = self._load_npz_splits(data_root=data_root, image_size=image_size)
        return DistillationSplits(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            num_classes=len(class_names),
            class_names=class_names,
        )

    @override
    def load_student_eval_splits(self, data_root: str, image_size: int) -> StudentEvalSplits:
        _, val_set, test_set, class_names = self._load_npz_splits(data_root=data_root, image_size=image_size)
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
