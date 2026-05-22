'''
用法示例:

from datasets import get_dataset_spec

spec = get_dataset_spec("dermamnist")
'''

from __future__ import annotations

import csv
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, override
import os

import numpy as np
import torch
from medmnist import INFO, BloodMNIST, DermaMNIST, PathMNIST
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

## 接口
def supported_datasets() -> tuple[str, ...]:
    return tuple(sorted(_build_registry().keys()))


def get_dataset_spec(dataset: str) -> BaseDataset:
    '''
    根据输入数据集名称返回特定 class
    '''
    key = _normalize_dataset_key(dataset)
    registry = _build_registry()
    if key not in registry:
        supported = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unsupported dataset: {dataset}. Supported datasets: {supported}")
    return registry[key]


@dataclass(frozen=True)
class DatasetSplit:
    train_set: Any
    val_set: Any
    test_set: Any
    num_classes: int
    label_table: dict[int, str]

## Torch Style Dataset
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


class DistilledTripletDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
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
        self.transform = transform

    @override
    def __len__(self) -> int:
        return int(self.images.size(0))

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.transform(self.images[index])
        soft = self.soft_labels[index]
        weight = self.weights[index]
        return image, soft, weight
    

## Base and Sub-Class (auto register) and utils
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

DATASET_KEY_ALIASES = {
    "aptos": "aptos-2019-blindness-detection",
    "aptos2019": "aptos-2019-blindness-detection",
    "aptos-2019": "aptos-2019-blindness-detection",
    "aptos-2019-blindness": "aptos-2019-blindness-detection",
}

class MedMNIST(BaseDataset, ABC):
    # BaseDataset.attr
    name: str
    prompt_prefix: str
    # self.attr
    split_class: type[Any]

    @override
    def class_names(self) -> dict[int, str]:
        #  INFO[dataset_name]['label'] 形如 {"0": "melanocytic nevi", "1": "melanoma", ...}
        # cast 表示类型确认，并不会真的做 cast
        label_meta = cast(dict[str, str], INFO[self.name]["label"])
        return {int(k): v for k, v in label_meta.items()}

    @override
    def load_dataset_splits(self, data_root: str, image_size: int) -> DatasetSplit:
        train_set = self.split_class(split="train", download=True, root=data_root, size=image_size)
        val_set = self.split_class(split="val", download=True, root=data_root, size=image_size)
        test_set = self.split_class(split="test", download=True, root=data_root, size=image_size)
        class_names = self.class_names()
        return DatasetSplit(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            num_classes=len(class_names),
            label_table=class_names,
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

    class Image_Split:
        def __init__(self, image_paths: list[Path], labels: np.ndarray) -> None:
            self.image_paths = [Path(p) for p in image_paths]
            self.labels = labels

        def __len__(self) -> int:
            return len(self.image_paths)

        def get_image_and_label(self, index: int) -> tuple[np.ndarray, int]:
            image_path = self.image_paths[index]
            with Image.open(image_path) as pil_image:
                # 转换为三通道图像，Resnet 18 的输入是 224*224*3
                # image_np = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
                image_np = pil_image
            return image_np, self.labels[index]

        def get_all_labels(self) -> np.ndarray:
            return self.labels
    
    name = "aptos-2019-blindness-detection"
    prompt_prefix = "retinal fundus image showing"

    label_table = {
        0: "no_diabetic_retinopathy",
        1: "mild_diabetic_retinopathy",
        2: "moderate_diabetic_retinopathy",
        3: "severe_diabetic_retinopathy",
        4: "proliferative_diabetic_retinopathy",
    }

    def _load_data_conf(self, data_root: str, split: str = "train") -> tuple[list[Path], np.ndarray]:
        dataset_root = Path(data_root) / "APTOS_2019_Blindness_Detection"

        csv_path = dataset_root / (split + ".csv") # train.csv
        image_root = dataset_root / (split + "_images") # train_images

        image_paths: list[Path] = []
        labels: list[int] = []

        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_id = str(row['id_code']).strip()
                label = int(row['diagnosis'])

                image_path = os.path.join(image_root, image_id + '.png')

                image_paths.append(image_path)
                labels.append(label)

        return image_paths, np.asarray(labels, dtype=np.int64)


    @override
    def load_dataset_splits(self, data_root: str, image_size: int) -> DatasetSplit:
        train_set = self.Image_Split(*self._load_data_conf(data_root, split='train'))

        val_set = self.Image_Split(*self._load_data_conf(data_root, split='test'))

        test_set = self.Image_Split(*self._load_data_conf(data_root, split='val'))

        return DatasetSplit(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            num_classes=len(self.label_table),
            label_table=self.label_table,
        )
    
    @override
    def class_names(self) -> dict[int, str]:
        return self.label_table


def _normalize_dataset_key(dataset: str) -> str:
    '''
    小写, 移除空格和_和-
    '''
    key = str(dataset).strip().lower().replace("_", "-")
    return DATASET_KEY_ALIASES.get(key, key)
    

def _iter_all_classes() -> list[type[BaseDataset]]:
    '''
    递归搜索子类，返回所有支持的数据集子类列表
    '''
    pending = list(BaseDataset.__subclasses__())
    all_classes: list[type[BaseDataset]] = []

    while pending:
        cls = pending.pop()
        all_classes.append(cls)
        pending.extend(cls.__subclasses__())

    return all_classes


def _build_registry() -> dict[str, BaseDataset]:
    '''
    实例化数据集子类，并返回所有支持的数据集子类的字典
    '''
    registry: dict[str, BaseDataset] = {}
    for spec_cls in _iter_all_classes():
        if spec_cls in {BaseDataset, MedMNIST}:
            continue
        spec = spec_cls()
        key = spec.name.lower().strip()
        if key:
            registry[key] = spec
    return registry

