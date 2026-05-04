import json
import os
import random
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)

def normalize_batch(images: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=images.device, dtype=images.dtype)
    std = IMAGENET_STD.to(device=images.device, dtype=images.dtype)
    return (images - mean) / std


def sample_random_resized_crop_params(
    image_height: int,
    image_width: int,
    scale: tuple[float, float],
    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    generator: torch.Generator | None = None,
) -> tuple[int, int, int, int]:
    height = int(max(1, image_height))
    width = int(max(1, image_width))
    area = float(height * width)

    scale_min = float(np.clip(scale[0], 1e-4, 1.0))
    scale_max = float(np.clip(scale[1], scale_min, 1.0))
    ratio_min = float(max(ratio[0], 1e-4))
    ratio_max = float(max(ratio[1], ratio_min))
    log_ratio_min = float(np.log(ratio_min))
    log_ratio_max = float(np.log(ratio_max))

    for _ in range(10):
        target_area = area * float(
            torch.empty(1).uniform_(scale_min, scale_max, generator=generator).item()
        )
        aspect_ratio = float(
            torch.empty(1).uniform_(log_ratio_min, log_ratio_max, generator=generator).exp_().item()
        )

        crop_width = int(round(np.sqrt(target_area * aspect_ratio)))
        crop_height = int(round(np.sqrt(target_area / aspect_ratio)))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            top = int(torch.randint(0, height - crop_height + 1, (1,), generator=generator).item())
            left = int(torch.randint(0, width - crop_width + 1, (1,), generator=generator).item())
            return top, left, crop_height, crop_width

    in_ratio = float(width / height)
    if in_ratio < ratio_min:
        crop_width = width
        crop_height = int(round(crop_width / ratio_min))
    elif in_ratio > ratio_max:
        crop_height = height
        crop_width = int(round(crop_height * ratio_max))
    else:
        crop_height = height
        crop_width = width

    top = max((height - crop_height) // 2, 0)
    left = max((width - crop_width) // 2, 0)
    return top, left, crop_height, crop_width


def apply_resized_crop_with_flip(
    image: torch.Tensor,
    crop_params: tuple[int, int, int, int],
    output_size: int,
    horizontal_flip: bool = False,
) -> torch.Tensor:
    top, left, crop_height, crop_width = (int(v) for v in crop_params)
    cropped = TF.resized_crop(
        image.float(),
        top=top,
        left=left,
        height=max(1, crop_height),
        width=max(1, crop_width),
        size=[int(output_size), int(output_size)],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    if bool(horizontal_flip):
        cropped = TF.hflip(cropped)
    return cropped.clamp(0.0, 1.0)


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
    teacher_temperature: float = 0.0,
    image_shards: list[str] | None = None,
    store_images_in_pt: bool = True,
    fkd_batch_path: str = "",
    fkd_batch_summary: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "weights": weights.float().cpu(),
        "soft_labels": soft_labels.float().cpu(),
        "dataset": dataset_name,
        "lora_path": lora_path,
        "teacher_temperature": float(teacher_temperature),
        "image_relative_paths": list(saved_paths),
    }
    if image_shards:
        payload["image_shards"] = list(image_shards)
    if store_images_in_pt and images is not None:
        payload["images"] = images.float().cpu()
    if fkd_batch_path:
        payload["fkd_batch_path"] = str(fkd_batch_path)
    if fkd_batch_summary:
        payload["fkd_batch_summary"] = dict(fkd_batch_summary)

    torch.save(payload, output_dir / "distilled_data.pt")

    num_distilled = int(soft_labels.size(0))
    if images is not None:
        num_distilled = int(images.size(0))

    metadata = {
        "dataset": dataset_name,
        "lora_path": lora_path,
        "teacher_temperature": float(teacher_temperature),
        "num_distilled": num_distilled,
        "weights_sum": float(weights.sum().item()),
        "soft_labels_shape": list(soft_labels.shape),
        "center_labels": [int(v) for v in center_labels.tolist()],
        "cluster_counts": [int(v) for v in counts.tolist()],
        "image_relative_paths": saved_paths,
        "image_shards": list(image_shards or []),
        "fkd_batch_path": str(fkd_batch_path),
        "fkd_batch_summary": dict(fkd_batch_summary or {}),
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
