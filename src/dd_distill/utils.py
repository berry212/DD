import json
import os
import random
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from PIL import Image
import warnings
import functools

import numpy as np
import torch
from torchvision.utils import make_grid


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


def save_preview_grid(images: torch.Tensor, output_path: Path, max_images: int = 100) -> None:
    n = min(max_images, images.size(0))
    if n <= 0:
        return
    grid = make_grid(images[:n], nrow=min(10, n), pad_value=1.0)
    grid_np = (grid.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(grid_np).save(output_path)


def _write_png(abs_path: Path, image_np: np.ndarray) -> None:
    Image.fromarray(image_np).save(abs_path)


def save_distilled_images(
    images: torch.Tensor,
    labels: torch.Tensor,
    output_dir: Path,
    start_index: int = 0,
    max_workers: int = 0,
) -> list[str]:
    image_root = output_dir / "distilled_images"
    image_root.mkdir(parents=True, exist_ok=True)

    worker_count = max(0, int(max_workers))
    executor: ThreadPoolExecutor | None = None
    futures: list[Future[None]] = []
    if worker_count > 1:
        executor = ThreadPoolExecutor(max_workers=worker_count)

    rel_paths: list[str] = []
    for idx in range(images.size(0)):
        global_idx = int(start_index + idx)
        class_id = int(labels[idx].item())
        class_dir = image_root / f"class_{class_id}"
        class_dir.mkdir(parents=True, exist_ok=True)

        rel_path = Path(f"class_{class_id}") / f"sample_{global_idx:05d}.png"
        abs_path = image_root / rel_path

        image_np = (images[idx].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        if executor is None:
            _write_png(abs_path, image_np)
        else:
            futures.append(executor.submit(_write_png, abs_path, image_np))
        rel_paths.append(str(rel_path))

    if executor is not None:
        for f in futures:
            f.result()
        executor.shutdown(wait=True)

    return rel_paths


def save_distillation_artifacts(
    output_dir: Path,
    images: torch.Tensor | None,
    weights: torch.Tensor,
    soft_labels: torch.Tensor,
    center_labels: torch.Tensor,
    counts: torch.Tensor,
    saved_paths: list[str],
    dataset_name: str,
    lora_path: str,
    image_shards: list[str] | None = None,
    store_images_in_pt: bool = True,
) -> None:
    payload: dict[str, object] = {
        "weights": weights.float().cpu(),
        "soft_labels": soft_labels.float().cpu(),
        "dataset": dataset_name,
        "lora_path": lora_path,
        "image_relative_paths": list(saved_paths),
    }
    if image_shards:
        payload["image_shards"] = list(image_shards)
    if store_images_in_pt and images is not None:
        payload["images"] = images.float().cpu()

    torch.save(payload, output_dir / "distilled_data.pt")

    num_distilled = int(soft_labels.size(0))
    if images is not None:
        num_distilled = int(images.size(0))

    metadata = {
        "dataset": dataset_name,
        "lora_path": lora_path,
        "num_distilled": num_distilled,
        "weights_sum": float(weights.sum().item()),
        "soft_labels_shape": list(soft_labels.shape),
        "center_labels": [int(v) for v in center_labels.tolist()],
        "cluster_counts": [int(v) for v in counts.tolist()],
        "image_relative_paths": saved_paths,
        "image_shards": list(image_shards or []),
    }
    with open(output_dir / "distilled_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)



def save_latent_cache(
    cache_path: Path,
    latents: torch.Tensor,
    labels: torch.Tensor,
    dataset_name: str,
    vae_model_id: str,
    image_size: int,
) -> None:
    torch.save(
        {
            "latents": latents.float().cpu(),
            "labels": labels.long().cpu(),
            "dataset": dataset_name,
            "vae_model_id": vae_model_id,
            "image_size": int(image_size),
        },
        cache_path,
    )
    print(f"[Encoding] Saved latent cache: {cache_path}")


def load_latent_cache(
    cache_path: Path,
    expected_num_samples: int,
    expected_dataset: str,
    expected_vae_model_id: str,
    expected_image_size: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not cache_path.exists():
        return None

    try:
        payload = torch.load(cache_path, map_location="cpu")
    except Exception as exc:
        warnings.warn(f"Failed to load latent cache ({cache_path}): {exc}", RuntimeWarning)
        return None

    if not isinstance(payload, dict):
        warnings.warn(f"Latent cache format invalid: {cache_path}", RuntimeWarning)
        return None

    cached_dataset = payload.get("dataset")
    if cached_dataset is not None and str(cached_dataset) != str(expected_dataset):
        warnings.warn(
            f"Latent cache dataset mismatch ({cached_dataset} != {expected_dataset}), re-encoding.",
            RuntimeWarning,
        )
        return None

    cached_model_id = payload.get("vae_model_id")
    if cached_model_id is not None and str(cached_model_id) != str(expected_vae_model_id):
        warnings.warn(
            f"Latent cache VAE mismatch ({cached_model_id} != {expected_vae_model_id}), re-encoding.",
            RuntimeWarning,
        )
        return None

    cached_image_size = payload.get("image_size")
    if cached_image_size is not None and int(cached_image_size) != int(expected_image_size):
        warnings.warn(
            f"Latent cache image size mismatch ({cached_image_size} != {expected_image_size}), re-encoding.",
            RuntimeWarning,
        )
        return None

    latents = payload.get("latents")
    labels = payload.get("labels")
    if not isinstance(latents, torch.Tensor) or not isinstance(labels, torch.Tensor):
        warnings.warn(f"Latent cache missing tensor fields: {cache_path}", RuntimeWarning)
        return None

    latents = latents.float().cpu()
    labels = labels.long().view(-1).cpu()
    if latents.size(0) != expected_num_samples or labels.size(0) != expected_num_samples:
        warnings.warn(
            f"Latent cache sample count mismatch ({latents.size(0)}/{labels.size(0)} != {expected_num_samples}), "
            "re-encoding.",
            RuntimeWarning,
        )
        return None

    print(f"[Encoding] Loaded latent cache: {cache_path}")
    return latents, labels
