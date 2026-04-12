from __future__ import annotations

import argparse
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
from torchvision.utils import make_grid

from .datasets import MedMNISTImageDataset, get_dataset_spec, supported_datasets


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


@dataclass
class CLVQResult:
    centers: torch.Tensor
    center_labels: torch.Tensor
    counts: torch.Tensor
    weights: torch.Tensor


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


def normalize_batch(images: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=images.device, dtype=images.dtype)
    std = IMAGENET_STD.to(device=images.device, dtype=images.dtype)
    return (images - mean) / std


def build_teacher_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def build_classifier(num_classes: int, backbone: str, imagenet_pretrained: bool) -> nn.Module:
    backbone_name = backbone.lower()
    if backbone_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
        model = resnet18(weights=weights)
    elif backbone_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if imagenet_pretrained else None
        model = resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}.")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += images.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    amp_enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true_chunks: list[np.ndarray] = []
    y_prob_chunks: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
        probs = F.softmax(logits.float(), dim=1)

        y_true_chunks.append(labels.cpu().numpy())
        y_prob_chunks.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true_chunks, axis=0)
    y_prob = np.concatenate(y_prob_chunks, axis=0)
    return y_true, y_prob


def compute_macro_auc(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> float | None:
    class_counts = np.bincount(y_true, minlength=num_classes)
    if int((class_counts > 0).sum()) < 2:
        return None

    y_true_one_hot = np.eye(num_classes, dtype=np.float32)[y_true]
    try:
        return float(roc_auc_score(y_true_one_hot, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        return None


def evaluate_teacher_baseline_metrics(
    model: nn.Module,
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    amp_enabled: bool,
    num_classes: int,
) -> dict[str, Any]:
    test_loss, test_acc = evaluate_classifier(model, test_loader, device, amp_enabled=amp_enabled)
    y_true, y_prob = predict_probabilities(model, test_loader, device, amp_enabled=amp_enabled)
    y_pred = np.argmax(y_prob, axis=1)
    test_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    test_auc = compute_macro_auc(y_true, y_prob, num_classes=num_classes)

    return {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_macro_f1": test_macro_f1,
        "test_auc": test_auc,
        "baseline_metrics": {
            "ACC": float(test_acc),
            "AUC": test_auc,
            "MacroF1": test_macro_f1,
        },
    }


def train_teacher_model(
    train_set: Any,
    val_set: Any,
    test_set: Any,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    baseline_dir: Path,
) -> tuple[nn.Module, dict[str, Any]]:
    train_transform, eval_transform = build_teacher_transforms(args.image_size)

    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        MedMNISTImageDataset(val_set, transform=eval_transform),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        MedMNISTImageDataset(test_set, transform=eval_transform),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    teacher = build_classifier(
        num_classes=num_classes,
        backbone=args.teacher_backbone,
        imagenet_pretrained=args.imagenet_pretrained,
    ).to(device)

    best_ckpt_path = baseline_dir / "teacher_best.pt"
    best_ckpt: dict[str, Any] | None = None

    if best_ckpt_path.exists():
        try:
            loaded = torch.load(best_ckpt_path, map_location=device)
            if not isinstance(loaded, dict) or "model" not in loaded:
                raise KeyError("teacher_best.pt missing required key: model")
            teacher.load_state_dict(loaded["model"])
            teacher.eval()
            best_ckpt = loaded
            print(f"[Teacher] Found existing checkpoint, skip training: {best_ckpt_path}")
        except Exception as exc:
            warnings.warn(
                f"Failed to load existing teacher checkpoint ({best_ckpt_path}), retraining. error={exc}",
                RuntimeWarning,
            )

    if best_ckpt is None:
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
            MedMNISTImageDataset(train_set, transform=train_transform),
            batch_size=args.teacher_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        optimizer = AdamW(teacher.parameters(), lr=args.teacher_lr, weight_decay=args.teacher_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.teacher_epochs, 1))
        scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)

        best_val_acc = -1.0
        history: list[dict[str, float]] = []

        for epoch in range(1, args.teacher_epochs + 1):
            teacher.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    logits = teacher(images)
                    loss = F.cross_entropy(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += float(loss.item()) * images.size(0)
                total_correct += int((logits.argmax(dim=1) == labels).sum().item())
                total_samples += images.size(0)

            train_loss = total_loss / max(total_samples, 1)
            train_acc = total_correct / max(total_samples, 1)
            val_loss, val_acc = evaluate_classifier(teacher, val_loader, device, amp_enabled=amp_enabled)
            test_loss, test_acc = evaluate_classifier(teacher, test_loader, device, amp_enabled=amp_enabled)
            scheduler.step()

            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "model": teacher.state_dict(),
                        "best_val_acc": best_val_acc,
                        "epoch": int(epoch),
                    },
                    best_ckpt_path,
                )

            print(
                f"[Teacher] epoch={epoch:03d}/{args.teacher_epochs} "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f}"
            )

        with open(baseline_dir / "teacher_history.json", "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        teacher.load_state_dict(best_ckpt["model"])
        teacher.eval()

    baseline_eval = evaluate_teacher_baseline_metrics(
        teacher,
        test_loader=test_loader,
        device=device,
        amp_enabled=amp_enabled,
        num_classes=num_classes,
    )
    best_val_acc_raw = best_ckpt.get("best_val_acc") if best_ckpt is not None else None
    best_val_acc = float(best_val_acc_raw) if best_val_acc_raw is not None else None

    teacher_summary = {
        "best_val_acc": best_val_acc,
        "test_loss": baseline_eval["test_loss"],
        "test_acc": baseline_eval["test_acc"],
        "test_auc": baseline_eval["test_auc"],
        "test_macro_f1": baseline_eval["test_macro_f1"],
        "epochs": int(args.teacher_epochs),
        "backbone": str(args.teacher_backbone),
        "checkpoint_path": str(best_ckpt_path),
        "baseline_metrics": baseline_eval["baseline_metrics"],
    }

    with open(baseline_dir / "teacher_summary.json", "w", encoding="utf-8") as handle:
        json.dump(teacher_summary, handle, indent=2)

    with open(baseline_dir / "teacher_baseline_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(baseline_eval["baseline_metrics"], handle, indent=2)

    return teacher, teacher_summary


@torch.no_grad()
def make_teacher_soft_labels(
    teacher: nn.Module,
    distilled_images: torch.Tensor,
    temperature: float,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    teacher.eval()
    soft_targets: list[torch.Tensor] = []

    for start in range(0, distilled_images.size(0), batch_size):
        end = min(distilled_images.size(0), start + batch_size)
        x = distilled_images[start:end].to(device=device)
        x = normalize_batch(x)
        logits = teacher(x)
        soft = F.softmax(logits / max(temperature, 1e-6), dim=1)
        soft_targets.append(soft.cpu())

    return torch.cat(soft_targets, dim=0)


def build_encode_loader(
    train_set: Any,
    encode_batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    return DataLoader(
        MedMNISTImageDataset(train_set, transform=None),
        batch_size=encode_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def load_vae(model_id: str, device: torch.device, dtype: torch.dtype) -> tuple[AutoencoderKL, float]:
    print(f"[Encoding] loading AutoencoderKL: {model_id}")
    vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=dtype)
    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))
    return vae, scaling_factor


@torch.no_grad()
def encode_training_images(
    vae: AutoencoderKL,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    scaling_factor: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_latents: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    total_batches = len(data_loader)
    for batch_idx, (images, labels) in enumerate(data_loader, start=1):
        images = images.to(device=device, dtype=dtype)
        images = images * 2.0 - 1.0

        posterior = vae.encode(images).latent_dist
        latents = posterior.mean * scaling_factor

        all_latents.append(latents.float().cpu())
        all_labels.append(labels.cpu())

        if batch_idx % 20 == 0 or batch_idx == total_batches:
            print(f"[Encoding] batch {batch_idx}/{total_batches}")

    return torch.cat(all_latents, dim=0), torch.cat(all_labels, dim=0)


def initialize_clvq_centers(data: np.ndarray, num_centers: int, seed: int) -> np.ndarray:
    if num_centers <= 0:
        raise ValueError("num_centers must be positive.")
    n_samples = int(data.shape[0])
    if n_samples == 0:
        raise ValueError("Cannot initialize CLVQ centers from an empty dataset.")

    rng = np.random.default_rng(seed)
    replace = num_centers > n_samples
    chosen = rng.choice(n_samples, size=num_centers, replace=replace)
    return data[chosen].astype(np.float32, copy=True)


def assign_to_centers(data: np.ndarray, centers: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    if data.size == 0 or centers.size == 0:
        raise ValueError("assign_to_centers expects non-empty data and centers.")

    center_norm = np.sum(centers * centers, axis=1)
    assignments = np.empty(data.shape[0], dtype=np.int64)

    for start in range(0, data.shape[0], batch_size):
        end = min(data.shape[0], start + batch_size)
        chunk = data[start:end]
        chunk_norm = np.sum(chunk * chunk, axis=1, keepdims=True)
        distances = chunk_norm + center_norm[None, :] - 2.0 * (chunk @ centers.T)
        assignments[start:end] = np.argmin(distances, axis=1)

    return assignments


def classwise_clvq(
    latents: torch.Tensor,
    labels: torch.Tensor,
    clusters_per_class: int,
    num_classes: int,
    seed: int,
    gamma_0: float,
    alpha: float,
    max_iter: int,
    tol: float,
    check_interval: int,
) -> CLVQResult:
    if clusters_per_class <= 0:
        raise ValueError("clusters_per_class must be positive.")

    flat_latents = latents.view(latents.size(0), -1).numpy().astype(np.float32, copy=False)
    labels_np = labels.numpy().astype(np.int64, copy=False)
    total_samples = int(flat_latents.shape[0])

    center_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    count_chunks: list[torch.Tensor] = []
    weight_chunks: list[torch.Tensor] = []

    for class_id in range(num_classes):
        class_latents = flat_latents[labels_np == class_id]
        class_samples = int(class_latents.shape[0])
        if class_samples == 0:
            warnings.warn(f"Class {class_id} has no samples; skipping.", RuntimeWarning)
            continue

        class_k = min(clusters_per_class, class_samples)
        class_seed = seed + 1009 * (class_id + 1)
        centers = initialize_clvq_centers(class_latents, class_k, class_seed)
        weights = np.full((class_k,), fill_value=1.0 / float(class_k), dtype=np.float64)
        rng = np.random.default_rng(class_seed)

        prev_centers = centers.copy()
        for step in range(max_iter):
            gamma_t = gamma_0 / ((1.0 + float(step)) ** alpha)
            sample = class_latents[int(rng.integers(0, class_samples))]

            distances = np.sum((centers - sample[None, :]) ** 2, axis=1)
            winner = int(np.argmin(distances))

            centers[winner] = (1.0 - gamma_t) * centers[winner] + gamma_t * sample

            weights *= 1.0 - gamma_t
            weights[winner] += gamma_t

            if (step + 1) % check_interval == 0 or (step + 1) == max_iter:
                denom = np.linalg.norm(prev_centers) + 1e-12
                relative_shift = float(np.linalg.norm(centers - prev_centers) / denom)
                print(
                    f"[CLVQ-Class] class={class_id} iter={step + 1}/{max_iter} "
                    f"relative_shift={relative_shift:.6e}"
                )
                if relative_shift < tol:
                    break
                prev_centers = centers.copy()

        assignments = assign_to_centers(class_latents, centers, batch_size=2048)
        counts = np.bincount(assignments, minlength=class_k).astype(np.int64)

        non_empty_mask = counts > 0
        centers = centers[non_empty_mask]
        counts = counts[non_empty_mask]

        if centers.shape[0] == 0:
            continue

        global_weights = counts.astype(np.float64) / float(total_samples)

        center_chunks.append(torch.from_numpy(centers).view(centers.shape[0], *latents.shape[1:]).float())
        label_chunks.append(torch.full((centers.shape[0],), fill_value=class_id, dtype=torch.long))
        count_chunks.append(torch.from_numpy(counts))
        weight_chunks.append(torch.from_numpy(global_weights.astype(np.float32, copy=False)).float())

        print(
            f"[CLVQ-Class] class={class_id} samples={class_samples} "
            f"kept={centers.shape[0]}"
        )

    if not center_chunks:
        raise RuntimeError("Class-wise CLVQ failed: no centers produced.")

    centers_t = torch.cat(center_chunks, dim=0)
    labels_t = torch.cat(label_chunks, dim=0)
    counts_t = torch.cat(count_chunks, dim=0)
    weights_t = torch.cat(weight_chunks, dim=0)

    weights_t = weights_t.clamp_min(0.0)
    weights_t = weights_t / weights_t.sum().clamp_min(1e-12)

    return CLVQResult(centers=centers_t, center_labels=labels_t, counts=counts_t, weights=weights_t)


class ReverseSDEDecoder:
    def __init__(
        self,
        model_id: str,
        vae: AutoencoderKL,
        device: torch.device,
        dtype: torch.dtype,
        num_inference_steps: int,
        noise_strength: float,
        lora_path: str,
        lora_scale: float,
        class_prompts: dict[int, str],
        guidance_scale: float,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.num_inference_steps = max(2, int(num_inference_steps))
        self.noise_strength = float(np.clip(noise_strength, 0.01, 1.0))
        self.class_prompts = dict(class_prompts)
        self.guidance_scale = float(max(0.0, guidance_scale))
        self._default_prompt = "dermoscopic image of skin lesion"

        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "vae": vae,
            "safety_checker": None,
            "requires_safety_checker": False,
        }
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)
        except TypeError:
            kwargs.pop("requires_safety_checker", None)
            self.pipe = StableDiffusionPipeline.from_pretrained(model_id, **kwargs)

        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self._load_lora_if_provided(lora_path=lora_path, lora_scale=lora_scale)

        self.scaling_factor = float(getattr(self.pipe.vae.config, "scaling_factor", 0.18215))
        self._prompt_cache: dict[str, torch.Tensor] = {}

    def _load_lora_if_provided(self, lora_path: str, lora_scale: float) -> None:
        if not lora_path:
            return

        adapter_dir = Path(lora_path)
        if not adapter_dir.exists():
            warnings.warn(f"LoRA path does not exist: {adapter_dir}. Continue without LoRA.", RuntimeWarning)
            return

        loaded = False
        try:
            self.pipe.load_lora_weights(str(adapter_dir))
            loaded = True
        except Exception:
            try:
                self.pipe.unet.load_attn_procs(str(adapter_dir))
                loaded = True
            except Exception:
                warnings.warn("Failed to load LoRA, continue without LoRA.", RuntimeWarning)

        if loaded:
            try:
                self.pipe.fuse_lora(lora_scale=float(lora_scale))
            except Exception:
                pass
            print(f"[LoRA] Loaded adapter from {adapter_dir} with scale={lora_scale}")

    @torch.no_grad()
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        if prompt in self._prompt_cache:
            return self._prompt_cache[prompt]

        if self.pipe.tokenizer is None or self.pipe.text_encoder is None:
            raise RuntimeError("Diffusion model must provide tokenizer and text encoder.")

        text_inputs = self.pipe.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        prompt_embeds = self.pipe.text_encoder(input_ids)[0].to(dtype=self.dtype)
        self._prompt_cache[prompt] = prompt_embeds
        return prompt_embeds

    @torch.no_grad()
    def _null_prompt_embeddings(self, batch_size: int) -> torch.Tensor:
        return self._encode_prompt("").expand(batch_size, -1, -1)

    @torch.no_grad()
    def _prompt_embeddings_for_labels(self, labels: torch.Tensor) -> torch.Tensor:
        embeds: list[torch.Tensor] = []
        for class_id in labels.tolist():
            prompt = self.class_prompts.get(int(class_id), self._default_prompt)
            embeds.append(self._encode_prompt(prompt))
        return torch.cat(embeds, dim=0)

    @torch.no_grad()
    def _decode_batch(self, centers: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        start_idx = int((1.0 - self.noise_strength) * (len(timesteps) - 1))
        start_idx = max(0, min(start_idx, len(timesteps) - 1))
        start_timestep = timesteps[start_idx]

        latents = centers.to(device=self.device, dtype=self.dtype)
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, start_timestep.expand(latents.size(0)))

        cond_prompt_embeds = self._prompt_embeddings_for_labels(labels)
        uncond_prompt_embeds = self._null_prompt_embeddings(latents.size(0)) if self.guidance_scale > 1.0 else None

        for timestep in timesteps[start_idx:]:
            model_input = self.scheduler.scale_model_input(latents, timestep)

            if uncond_prompt_embeds is not None:
                model_input = torch.cat([model_input, model_input], dim=0)
                prompt_embeds = torch.cat([uncond_prompt_embeds, cond_prompt_embeds], dim=0)
                noise_pred = self.pipe.unet(model_input, timestep, encoder_hidden_states=prompt_embeds).sample
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = self.pipe.unet(model_input, timestep, encoder_hidden_states=cond_prompt_embeds).sample

            latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample

        images = self.pipe.vae.decode(latents / self.scaling_factor).sample
        return (images / 2.0 + 0.5).clamp(0.0, 1.0)

    @torch.no_grad()
    def decode(self, centers: torch.Tensor, labels: torch.Tensor, batch_size: int) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        total = centers.size(0)
        for start in range(0, total, batch_size):
            end = min(total, start + batch_size)
            imgs = self._decode_batch(centers[start:end], labels[start:end])
            chunks.append(imgs.float().cpu())
            print(f"[Decoding] {end}/{total}")
        return torch.cat(chunks, dim=0)

    def cleanup(self) -> None:
        del self.pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def save_preview_grid(images: torch.Tensor, output_path: Path, max_images: int = 100) -> None:
    n = min(max_images, images.size(0))
    if n <= 0:
        return
    grid = make_grid(images[:n], nrow=min(10, n), pad_value=1.0)
    grid_np = (grid.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(grid_np).save(output_path)


def save_distilled_images(images: torch.Tensor, labels: torch.Tensor, output_dir: Path) -> list[str]:
    image_root = output_dir / "distilled_images"
    image_root.mkdir(parents=True, exist_ok=True)

    rel_paths: list[str] = []
    for idx in range(images.size(0)):
        class_id = int(labels[idx].item())
        class_dir = image_root / f"class_{class_id}"
        class_dir.mkdir(parents=True, exist_ok=True)

        rel_path = Path(f"class_{class_id}") / f"sample_{idx:05d}.png"
        abs_path = image_root / rel_path

        image_np = (images[idx].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(image_np).save(abs_path)
        rel_paths.append(str(rel_path))

    return rel_paths


def save_distillation_artifacts(
    output_dir: Path,
    images: torch.Tensor,
    weights: torch.Tensor,
    soft_labels: torch.Tensor,
    center_labels: torch.Tensor,
    counts: torch.Tensor,
    saved_paths: list[str],
    teacher_summary: dict[str, Any],
) -> None:
    # Keep only the triplet required by downstream training.
    torch.save(
        {
            "images": images,
            "weights": weights,
            "soft_labels": soft_labels,
        },
        output_dir / "distilled_data.pt",
    )

    metadata = {
        "num_distilled": int(images.size(0)),
        "weights_sum": float(weights.sum().item()),
        "soft_labels_shape": list(soft_labels.shape),
        "center_labels": [int(v) for v in center_labels.tolist()],
        "cluster_counts": [int(v) for v in counts.tolist()],
        "image_relative_paths": saved_paths,
        "teacher_summary": teacher_summary,
    }
    with open(output_dir / "distilled_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def run_distillation(args: argparse.Namespace) -> dict[str, Any]:
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    amp_enabled = bool(args.fp16 and device.type == "cuda")
    dataset_spec = get_dataset_spec(args.dataset)
    output_dir = Path(args.output_dir or f"outputs/{dataset_spec.name}_224_distill")
    teacher_baseline_dir = Path(args.teacher_baseline_dir or f"outputs/{dataset_spec.name}_224_distill_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_baseline_dir.mkdir(parents=True, exist_ok=True)

    run_config = vars(args).copy()
    run_config["output_dir"] = str(output_dir)
    run_config["teacher_baseline_dir"] = str(teacher_baseline_dir)

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    split_bundle = dataset_spec.load_distillation_splits(data_root=args.data_root, image_size=args.image_size)
    train_set = split_bundle.train_set
    val_set = split_bundle.val_set
    test_set = split_bundle.test_set
    num_classes = split_bundle.num_classes
    class_names = split_bundle.class_names
    print(
        f"[Setup] dataset={dataset_spec.name} device={device} "
        f"train={len(train_set)} classes={num_classes} baseline_dir={teacher_baseline_dir}"
    )

    teacher, teacher_summary = train_teacher_model(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        num_classes=num_classes,
        args=args,
        device=device,
        amp_enabled=amp_enabled,
        baseline_dir=teacher_baseline_dir,
    )

    encode_loader = build_encode_loader(
        train_set=train_set,
        encode_batch_size=args.encode_batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    vae_dtype = torch.float16 if amp_enabled else torch.float32
    vae, scaling_factor = load_vae(args.vae_model_id, device=device, dtype=vae_dtype)

    latents, latent_labels = encode_training_images(
        vae=vae,
        data_loader=encode_loader,
        scaling_factor=scaling_factor,
        device=device,
        dtype=vae_dtype,
    )

    clvq = classwise_clvq(
        latents=latents,
        labels=latent_labels,
        clusters_per_class=args.clusters_per_class,
        num_classes=num_classes,
        seed=args.seed,
        gamma_0=args.clvq_gamma0,
        alpha=args.clvq_alpha,
        max_iter=args.clvq_max_iter,
        tol=args.clvq_tol,
        check_interval=args.clvq_check_interval,
    )

    class_prompts = dataset_spec.build_class_prompts(class_names)
    decoder = ReverseSDEDecoder(
        model_id=args.diffusion_model_id,
        vae=vae,
        device=device,
        dtype=vae_dtype,
        num_inference_steps=args.sde_steps,
        noise_strength=args.sde_noise_strength,
        lora_path=args.lora_path,
        lora_scale=args.lora_scale,
        class_prompts=class_prompts,
        guidance_scale=args.guidance_scale,
    )

    try:
        distilled_images = decoder.decode(clvq.centers, clvq.center_labels, args.decode_batch_size)
    finally:
        decoder.cleanup()

    soft_labels = make_teacher_soft_labels(
        teacher=teacher,
        distilled_images=distilled_images,
        temperature=args.teacher_temperature,
        batch_size=args.eval_batch_size,
        device=device,
    )

    save_preview_grid(distilled_images, output_dir / "distilled_preview.png")
    saved_paths = save_distilled_images(distilled_images, clvq.center_labels, output_dir)

    save_distillation_artifacts(
        output_dir=output_dir,
        images=distilled_images,
        weights=clvq.weights,
        soft_labels=soft_labels,
        center_labels=clvq.center_labels,
        counts=clvq.counts,
        saved_paths=saved_paths,
        teacher_summary=teacher_summary,
    )

    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = {
        "dataset": dataset_spec.name,
        "data_root": args.data_root,
        "teacher_baseline_dir": str(teacher_baseline_dir),
        "num_classes": int(num_classes),
        "clvq_mode": "class-wise",
        "clusters_per_class": int(args.clusters_per_class),
        "num_distilled": int(distilled_images.size(0)),
        "teacher_test_acc": float(teacher_summary["test_acc"]),
        "teacher_test_auc": teacher_summary.get("test_auc"),
        "teacher_test_macro_f1": teacher_summary.get("test_macro_f1"),
        "teacher_backbone": str(args.teacher_backbone),
        "teacher_temperature": float(args.teacher_temperature),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("[Done] Distillation complete.")
    print(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MedMNIST distillation: teacher training + VAE encode + class-wise CLVQ + reverse-SDE decode"
    )
    parser.add_argument("--dataset", type=str, default="dermamnist", choices=supported_datasets())
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--teacher-baseline-dir", type=str, default="")

    parser.add_argument("--clusters-per-class", type=int, default=100)
    parser.add_argument("--clvq-gamma0", type=float, default=0.5)
    parser.add_argument("--clvq-alpha", type=float, default=0.6)
    parser.add_argument("--clvq-max-iter", type=int, default=10000)
    parser.add_argument("--clvq-tol", type=float, default=1e-5)
    parser.add_argument("--clvq-check-interval", type=int, default=500)

    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--decode-batch-size", type=int, default=32)

    parser.add_argument("--vae-model-id", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--diffusion-model-id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora-path", type=str, default="outputs/lora_dreammnist")
    parser.add_argument("--lora-scale", type=float, default=0.9)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--sde-steps", type=int, default=200)
    parser.add_argument("--sde-noise-strength", type=float, default=0.2)

    parser.add_argument("--teacher-backbone", type=str, default="resnet50", choices=["resnet18", "resnet50"])
    parser.add_argument("--teacher-epochs", type=int, default=20)
    parser.add_argument("--teacher-batch-size", type=int, default=64)
    parser.add_argument("--teacher-lr", type=float, default=3e-4)
    parser.add_argument("--teacher-weight-decay", type=float, default=1e-4)
    parser.add_argument("--teacher-temperature", type=float, default=20.0)

    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--imagenet-pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_distillation(args)


if __name__ == "__main__":
    main()
