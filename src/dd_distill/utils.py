import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)

def normalize_batch(images: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=images.device, dtype=images.dtype)
    std = IMAGENET_STD.to(device=images.device, dtype=images.dtype)
    return (images - mean) / std


def default_data_root() -> str:
    return os.getenv("HF_DATASETS_CACHE") or os.getenv("HF_HOME", "data")


def resolve_lora_path(dataset_name: str, lora_path_arg: str) -> str:
    explicit = lora_path_arg.strip()
    if explicit:
        return explicit

    dataset_default = Path("outputs") / f"lora_{dataset_name}"
    if dataset_default.exists():
        return str(dataset_default)

    legacy_default = Path("outputs/lora_dreammnist")
    if dataset_name == "dermamnist" and legacy_default.exists():
        warnings.warn(
            "Using legacy LoRA path outputs/lora_dreammnist for dermamnist. "
            "Consider migrating to outputs/lora_dermamnist.",
            RuntimeWarning,
        )
        return str(legacy_default)

    return str(dataset_default)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")