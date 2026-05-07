from __future__ import annotations

import csv
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, override

import numpy as np
import torch
from medmnist import INFO, BloodMNIST, DermaMNIST, PathMNIST
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


DATASET_KEY_ALIASES = {
    "odir5k": "odir-5k",
    "aptos": "aptos-2019-blindness-detection",
    "aptos2019": "aptos-2019-blindness-detection",
    "aptos-2019": "aptos-2019-blindness-detection",
    "aptos-2019-blindness": "aptos-2019-blindness-detection",
}


@dataclass(frozen=True)
class DatasetSplit:
    train_set: Any
    val_set: Any
    test_set: Any
    num_classes: int
    class_names: dict[int, str]


class APTOS2019Split:
    def __init__(self, image_paths: list[Path], labels: np.ndarray) -> None:
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = np.asarray(labels).reshape(-1).astype(np.int64, copy=False)

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


class TorchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, split_data: Any, transform: Any | None = None) -> None:
        self.split_data = split_data
        self.transform = transform

        if hasattr(split_data, "get_all_labels"):
            self.labels = np.asarray(split_data.get_all_labels()).reshape(-1).astype(np.int64, copy=False)
        elif hasattr(split_data, "labels"):
            self.labels = np.asarray(split_data.labels).reshape(-1).astype(np.int64, copy=False)
        else:
            raise AttributeError("split_data must provide either get_all_labels() or labels.")

        self.images = split_data.imgs if hasattr(split_data, "imgs") else None
        self._split_accessor = split_data if hasattr(split_data, "get_image_and_label") else None
        if self._split_accessor is None and self.images is None:
            raise AttributeError("split_data must provide either get_image_and_label() or imgs.")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(index)
        if self._split_accessor is not None:
            image, label = self._split_accessor.get_image_and_label(idx)
        else:
            image = self.images[idx]
            label = int(self.labels[idx])

        image_np = np.asarray(image)
        # Some upstream datasets expose readonly numpy views; copy to avoid torch warnings.
        if not image_np.flags.writeable:
            image_np = image_np.copy()
        image_np = np.ascontiguousarray(image_np)

        if self.transform is not None:
            image_tensor = self.transform(image_np)
        else:
            image_tensor = torch.from_numpy(image_np)
            if image_tensor.ndim == 2:
                image_tensor = image_tensor.unsqueeze(0)
            elif image_tensor.ndim == 3 and image_tensor.shape[0] not in (1, 3) and image_tensor.shape[-1] in (1, 3):
                image_tensor = image_tensor.permute(2, 0, 1)
            image_tensor = image_tensor.float()
            if float(image_tensor.max().item()) > 1.0:
                image_tensor = image_tensor / 255.0

        return image_tensor, torch.tensor(int(label), dtype=torch.long)


class BaseDataset(ABC):
    name: str
    prompt_prefix: str

    def class_names(self) -> dict[int, str]:
        raise NotImplementedError

    def load_dataset_splits(self, data_root: str, image_size: int) -> DatasetSplit:
        raise NotImplementedError

    def build_class_prompts(self) -> dict[int, str]:
        prompts: dict[int, str] = {}
        for class_id, name in self.class_names().items():
            cleaned = str(name).replace("_", " ")
            prompts[class_id] = f"{self.prompt_prefix} {cleaned}"
        return prompts

class MedMNIST(BaseDataset, ABC):
    # BaseDataset.attr
    name: str
    prompt_prefix: str
    # self.attr
    split_class: type[Any]

    @classmethod
    def _clamp_image_size(cls, image_size: int) -> int:
        """Clamp to the nearest supported MedMNIST size (max 224)."""
        available = sorted(getattr(cls.split_class, "available_sizes", [28, 64, 128, 224]))
        if image_size in available:
            return image_size
        # Use the largest available size ≤ requested, or the smallest available.
        valid = [s for s in available if s <= image_size]
        clamped = max(valid) if valid else max(available)
        if clamped != image_size:
            print(
                f"[MedMNIST] image_size={image_size} not in {available}; "
                f"loading at {clamped} (transform will resize to {image_size})"
            )
        return clamped

    @override
    def class_names(self) -> dict[int, str]:
        #  INFO[dataset_name]['label'] 形如 {"0": "melanocytic nevi", "1": "melanoma", ...}
        # cast 表示类型确认，并不会真的做 cast
        label_meta = cast(dict[str, str], INFO[self.name]["label"])
        return {int(k): v for k, v in label_meta.items()}

    @override
    def load_dataset_splits(self, data_root: str, image_size: int) -> DatasetSplit:
        load_size = self._clamp_image_size(image_size)
        train_set = self.split_class(split="train", download=True, root=data_root, size=load_size)
        val_set = self.split_class(split="val", download=True, root=data_root, size=load_size)
        test_set = self.split_class(split="test", download=True, root=data_root, size=load_size)
        class_names = self.class_names()
        return DatasetSplit(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            num_classes=len(class_names),
            class_names=class_names,
        )


class DermaMNIST(MedMNIST):
    name = "dermamnist"
    prompt_prefix = "dermoscopic image of"
    split_class = DermaMNIST


class BloodMNIST(MedMNIST):
    name = "bloodmnist"
    prompt_prefix = "microscopic image of blood cell"
    split_class = BloodMNIST

class PathMNIST(MedMNIST):
    name = "pathmnist"
    prompt_prefix = "histopathology image of"
    split_class = PathMNIST


class APTOS2019BlindnessDetectionSpec(BaseDataset):
    name = "aptos-2019-blindness-detection"
    prompt_prefix = "retinal fundus image showing"

    APTOS_CLASS_DISPLAY = {
        0: "no_diabetic_retinopathy",
        1: "mild_diabetic_retinopathy",
        2: "moderate_diabetic_retinopathy",
        3: "severe_diabetic_retinopathy",
        4: "proliferative_diabetic_retinopathy",
    }

    def _resolve_dataset_root(self, data_root: str) -> Path:
        data_path = Path(data_root) / "APTOS_2019_Blindness_Detection"

        if (data_path / "train.csv").exists() and (data_path / "train_images").exists():
            return data_path

        raise FileNotFoundError(
            "APTOS-2019 dataset root not found. Expected train.csv and train_images under one of: "
            f"{data_path}"
        )

    @staticmethod
    def _resolve_image_path(train_image_root: Path, image_id: str) -> Path | None:
        stem = str(image_id).strip()
        if not stem:
            return None

        for suffix in (".png", ".jpg", ".jpeg"):
            candidate = train_image_root / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
        return None

    def _load_samples(self, data_root: str) -> tuple[list[Path], np.ndarray]:
        dataset_root = self._resolve_dataset_root(data_root)
        csv_path = dataset_root / "train.csv"
        train_image_root = dataset_root / "train_images"

        image_paths: list[Path] = []
        labels: list[int] = []

        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_id = str(row.get("id_code", "")).strip()
                raw_label = row.get("diagnosis")

                try:
                    label = int(raw_label)
                except (TypeError, ValueError):
                    continue

                if label not in self.APTOS_CLASS_DISPLAY:
                    continue

                image_path = self._resolve_image_path(train_image_root, image_id)
                if image_path is None:
                    continue

                image_paths.append(image_path)
                labels.append(label)

        if not image_paths:
            raise RuntimeError(f"No usable APTOS-2019 samples found from {csv_path}")

        return image_paths, np.asarray(labels, dtype=np.int64)

    def _build_splits(self, data_root: str) -> tuple[APTOS2019Split, APTOS2019Split, APTOS2019Split]:
        image_paths, labels = self._load_samples(data_root)
        indices = np.arange(labels.shape[0], dtype=np.int64)

        train_idx, tail_idx = train_test_split(
            indices,
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify=labels,
        )
        val_idx, test_idx = train_test_split(
            tail_idx,
            test_size=0.5,
            random_state=42,
            shuffle=True,
            stratify=labels[tail_idx],
        )

        train_split = APTOS2019Split([image_paths[int(i)] for i in train_idx], labels[train_idx])
        val_split = APTOS2019Split([image_paths[int(i)] for i in val_idx], labels[val_idx])
        test_split = APTOS2019Split([image_paths[int(i)] for i in test_idx], labels[test_idx])
        return train_split, val_split, test_split

    @override
    def load_dataset_splits(self, data_root: str, image_size: int) -> DatasetSplit:
        train_set, val_set, test_set = self._build_splits(data_root=data_root)
        class_names = dict(self.APTOS_CLASS_DISPLAY)
        return DatasetSplit(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            num_classes=len(class_names),
            class_names=class_names,
        )
    
    @override
    def class_names(self) -> dict[int, str]:
        return dict(self.APTOS_CLASS_DISPLAY)


def _normalize_dataset_key(dataset: str) -> str:
    key = str(dataset).strip().lower().replace("_", "-")
    return DATASET_KEY_ALIASES.get(key, key)


class DistilledTripletDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        images: torch.Tensor,
        weights: torch.Tensor,
        soft_labels: torch.Tensor,
        transform: transforms.Compose,
    ) -> None:
        super().__init__()
        self.images = images.float().cpu()
        self.weights = weights.float().cpu()
        self.soft_labels = soft_labels.float().cpu()
        self.hard_labels = torch.argmax(self.soft_labels, dim=1).long()
        self.transform = transform

    @override
    def __len__(self) -> int:
        return int(self.images.size(0))

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.transform(self.images[index])
        soft = self.soft_labels[index]
        hard = self.hard_labels[index]
        weight = self.weights[index]
        return image, soft, hard, weight
    

'''
递归搜索子类，返回所有支持的数据集子类列表
'''
def _iter_all_classes() -> list[type[BaseDataset]]:
    pending = list(BaseDataset.__subclasses__())
    all_classes: list[type[BaseDataset]] = []

    while pending:
        cls = pending.pop()
        all_classes.append(cls)
        pending.extend(cls.__subclasses__())

    return all_classes

'''
实例化数据集子类，并返回所有支持的数据集子类的字典
'''
def _build_registry() -> dict[str, BaseDataset]:
    registry: dict[str, BaseDataset] = {}
    for spec_cls in _iter_all_classes():
        if spec_cls in {BaseDataset, MedMNIST}:
            continue
        spec = spec_cls()
        key = spec.name.lower().strip()
        if key:
            registry[key] = spec
    return registry


def supported_datasets() -> tuple[str, ...]:
    return tuple(sorted(_build_registry().keys()))


'''
根据输入数据集名称返回特定 class
'''
def get_dataset_spec(dataset: str) -> BaseDataset:
    key = _normalize_dataset_key(dataset)
    registry = _build_registry()
    if key not in registry:
        supported = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unsupported dataset: {dataset}. Supported datasets: {supported}")
    return registry[key]
