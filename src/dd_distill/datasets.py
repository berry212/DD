from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, cast, override

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


class MedMNISTImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        split_data: Any,
        transform: transforms.Compose | None = None,
    ) -> None:
        super().__init__()
        self.images = np.ascontiguousarray(split_data.imgs)
        self.labels = split_data.labels.reshape(-1).astype(np.int64, copy=False)
        self.transform = transform

    @override
    def __len__(self) -> int:
        return int(self.images.shape[0])

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_np = self.images[index]
        if self.transform is None:
            image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        else:
            image = self.transform(image_np)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label


class BaseMedMNISTSpec(ABC):
    name: str
    medmnist_key: str
    split_class: type[Any]
    prompt_prefix: str

    def _class_names(self) -> dict[int, str]:
        label_meta = cast(dict[str, str], INFO[self.medmnist_key]["label"])
        return {int(k): v for k, v in label_meta.items()}

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

    def build_class_prompts(self, class_names: dict[int, str]) -> dict[int, str]:
        return {class_id: f"{self.prompt_prefix} {name}" for class_id, name in class_names.items()}


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


def _iter_spec_classes() -> list[type[BaseMedMNISTSpec]]:
    pending = list(BaseMedMNISTSpec.__subclasses__())
    all_classes: list[type[BaseMedMNISTSpec]] = []

    while pending:
        cls = pending.pop()
        all_classes.append(cls)
        pending.extend(cls.__subclasses__())

    return all_classes


def _build_registry() -> dict[str, BaseMedMNISTSpec]:
    registry: dict[str, BaseMedMNISTSpec] = {}
    for spec_cls in _iter_spec_classes():
        spec = spec_cls()
        key = spec.name.lower().strip()
        if key:
            registry[key] = spec
    return registry


def supported_datasets() -> tuple[str, ...]:
    return tuple(sorted(_build_registry().keys()))


def get_dataset_spec(dataset: str) -> BaseMedMNISTSpec:
    key = dataset.lower().strip()
    registry = _build_registry()
    if key not in registry:
        supported = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unsupported dataset: {dataset}. Supported datasets: {supported}")
    return registry[key]
