from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from .baseline_resnet18 import load_teacher_checkpoint, train_teacher_baseline
from .datasets import TorchDataset, get_dataset_spec, supported_datasets
from .utils import *


@dataclass
class CLVQResult:
    centers: torch.Tensor
    center_labels: torch.Tensor
    counts: torch.Tensor
    influences: torch.Tensor
    weights: torch.Tensor


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
    image_size: int,
    encode_batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    encode_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    return DataLoader(
        TorchDataset(train_set, transform=encode_transform),
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


def compute_image_gradient_influence(
    teacher: nn.Module,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    log_prefix: str = "[Influence-ImageGrad]",
) -> torch.Tensor:
    teacher.eval()
    teacher.requires_grad_(False)

    score_chunks: list[torch.Tensor] = []
    total_batches = len(data_loader)
    print(f"{log_prefix} start: total_batches={total_batches}, device={device}")

    for batch_idx, (images, labels) in enumerate(data_loader, start=1):
        images = images.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device)

        if labels.ndim > 1:
            labels = labels.argmax(dim=1)
        labels = labels.long().view(-1)

        images.requires_grad_(True)
        logits = teacher(normalize_batch(images))
        per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
        input_grads = torch.autograd.grad(
            outputs=per_sample_loss.sum(),
            inputs=images,
            retain_graph=False,
            create_graph=False,
        )[0]
        grad_norm = input_grads.view(input_grads.size(0), -1).norm(p=2, dim=1).detach().cpu()
        score_chunks.append(grad_norm)

        if batch_idx % 20 == 0 or batch_idx == total_batches:
            print(f"{log_prefix} batch {batch_idx}/{total_batches}")

    if not score_chunks:
        raise RuntimeError("Failed to compute image gradient influence: no batches available.")

    return torch.cat(score_chunks, dim=0)


def stabilize_influence_scores(
    scores: torch.Tensor,
    quantile: float = 0.85,
    min_value: float = 1e-3,
    max_value: float = 1.0,
    log_prefix: str = "[Influence-Stabilize]",
) -> torch.Tensor:
    phi = scores.float().clamp_min(1e-12)
    q = float(np.clip(quantile, 0.0, 1.0))
    tau = torch.quantile(phi, q=q).clamp_min(1e-12)

    # Clip extreme influence values by quantile and map into [0, 1].
    phi_tilde = torch.minimum(phi, tau) / tau
    phi_tilde = phi_tilde.clamp(min=float(min_value), max=float(max_value))

    print(
        f"{log_prefix} q={q:.2f} tau={float(tau.item()):.6e} "
        f"phi_tilde_range=[{float(phi_tilde.min().item()):.6e}, {float(phi_tilde.max().item()):.6e}]"
    )
    return phi_tilde


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


def anchor_centers_to_medoids(
    data: np.ndarray,
    centers: np.ndarray,
    anchor: float,
    batch_size: int = 2048,
) -> np.ndarray:
    if data.size == 0 or centers.size == 0:
        raise ValueError("anchor_centers_to_medoids expects non-empty data and centers.")

    alpha = float(np.clip(anchor, 0.0, 1.0))
    if alpha <= 0.0:
        return centers

    center_norm = np.sum(centers * centers, axis=1)
    best_dist = np.full((centers.shape[0],), fill_value=np.inf, dtype=np.float64)
    best_samples = centers.copy()

    for start in range(0, data.shape[0], batch_size):
        end = min(data.shape[0], start + batch_size)
        chunk = data[start:end]
        chunk_norm = np.sum(chunk * chunk, axis=1, keepdims=True)
        distances = chunk_norm + center_norm[None, :] - 2.0 * (chunk @ centers.T)

        winner_rows = np.argmin(distances, axis=0)
        winner_dist = distances[winner_rows, np.arange(centers.shape[0])]
        improved = winner_dist < best_dist

        if np.any(improved):
            best_dist[improved] = winner_dist[improved]
            best_samples[improved] = chunk[winner_rows[improved]]

    anchored = (1.0 - alpha) * centers + alpha * best_samples
    return anchored.astype(np.float32, copy=False)


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
    medoid_anchor: float,
    sample_influence: torch.Tensor | None,
    influence_blend_beta: float,
    weight_count_power: float,
    weight_influence_power: float,
) -> CLVQResult:
    if clusters_per_class <= 0:
        raise ValueError("clusters_per_class must be positive.")

    medoid_anchor = float(np.clip(medoid_anchor, 0.0, 1.0))
    check_interval = max(1, int(check_interval))
    max_iter = max(1, int(max_iter))
    beta = float(np.clip(influence_blend_beta, 0.0, 1.0))
    count_power = float(max(weight_count_power, 1e-6))
    influence_power = float(max(weight_influence_power, 1e-6))

    latents = latents.float().cpu()
    labels = labels.long().view(-1).cpu()
    if sample_influence is None:
        sample_influence_np = np.ones((latents.size(0),), dtype=np.float32)
    else:
        sample_influence_np = sample_influence.float().view(-1).cpu().numpy().astype(np.float32, copy=False)
        if int(sample_influence_np.shape[0]) != int(latents.size(0)):
            raise ValueError(
                f"sample_influence size mismatch: expected {latents.size(0)}, got {sample_influence_np.shape[0]}"
            )
    sample_influence_np = np.clip(sample_influence_np, 1e-3, 1.0)

    flat_latents = latents.view(latents.size(0), -1).numpy().astype(np.float32, copy=False)
    labels_np = labels.numpy().astype(np.int64, copy=False)
    total_samples = int(flat_latents.shape[0])

    center_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    count_chunks: list[torch.Tensor] = []
    influence_chunks: list[torch.Tensor] = []

    for class_id in range(num_classes):
        class_selector = labels_np == class_id
        class_latents = flat_latents[class_selector]
        class_influence = sample_influence_np[class_selector]
        class_samples = int(class_latents.shape[0])
        if class_samples == 0:
            warnings.warn(f"Class {class_id} has no samples; skipping.", RuntimeWarning)
            continue

        class_k = min(clusters_per_class, class_samples)
        class_seed = seed + 1009 * (class_id + 1)
        centers = initialize_clvq_centers(class_latents, class_k, class_seed)
        running_weights = np.full((class_k,), fill_value=1.0 / float(class_k), dtype=np.float64)
        rng = np.random.default_rng(class_seed)
        prev_centers = centers.copy()

        for step in range(max_iter):
            gamma_t = gamma_0 / ((1.0 + float(step)) ** alpha)
            sample = class_latents[int(rng.integers(0, class_samples))]

            distances = np.sum((centers - sample[None, :]) ** 2, axis=1)
            winner = int(np.argmin(distances))

            centers[winner] = (1.0 - gamma_t) * centers[winner] + gamma_t * sample

            running_weights *= 1.0 - gamma_t
            running_weights[winner] += gamma_t

            if (step + 1) % check_interval == 0 or (step + 1) == max_iter:
                denom = np.linalg.norm(prev_centers) + 1e-12
                relative_shift = float(np.linalg.norm(centers - prev_centers) / denom)
                # print(
                #     f"[CLVQ-Class] class={class_id} iter={step + 1}/{max_iter} "
                #     f"relative_shift={relative_shift:.6e}"
                # )
                if relative_shift < tol:
                    break
                prev_centers = centers.copy()

        assignments = assign_to_centers(class_latents, centers, batch_size=2048)

        if medoid_anchor > 0.0:
            centers = anchor_centers_to_medoids(
                data=class_latents,
                centers=centers,
                anchor=medoid_anchor,
                batch_size=2048,
            )
            assignments = assign_to_centers(class_latents, centers, batch_size=2048)

        counts = np.bincount(assignments, minlength=class_k).astype(np.int64)
        influence_sum = np.bincount(
            assignments,
            weights=class_influence.astype(np.float64, copy=False),
            minlength=class_k,
        ).astype(np.float64)
        mean_influence = np.full((class_k,), fill_value=1e-3, dtype=np.float32)
        nonzero = counts > 0
        mean_influence[nonzero] = (influence_sum[nonzero] / np.maximum(counts[nonzero], 1)).astype(np.float32)
        mean_influence = np.clip(mean_influence, 1e-3, 1.0)

        non_empty_mask = counts > 0
        centers = centers[non_empty_mask]
        counts = counts[non_empty_mask]
        mean_influence = mean_influence[non_empty_mask]

        if centers.shape[0] == 0:
            continue

        center_chunks.append(torch.from_numpy(centers).view(centers.shape[0], *latents.shape[1:]).float())
        label_chunks.append(torch.full((centers.shape[0],), fill_value=class_id, dtype=torch.long))
        count_chunks.append(torch.from_numpy(counts))
        influence_chunks.append(torch.from_numpy(mean_influence).float())

        print(
            f"[CLVQ-Class] class={class_id} samples={class_samples} kept={centers.shape[0]} "
            f"mean_influence={float(np.mean(mean_influence)):.4f} medoid_anchor={medoid_anchor:.2f}"
        )

    if not center_chunks:
        raise RuntimeError("Class-wise CLVQ failed: no centers produced.")

    centers_t = torch.cat(center_chunks, dim=0)
    labels_t = torch.cat(label_chunks, dim=0)
    counts_t = torch.cat(count_chunks, dim=0)
    influences_t = torch.cat(influence_chunks, dim=0).float().clamp(1e-3, 1.0)

    count_signal_t = counts_t.float().clamp_min(1.0).pow(count_power)
    count_weights_t = count_signal_t / count_signal_t.sum().clamp_min(1e-12)

    influence_signal_t = influences_t.pow(influence_power)
    influence_weights_t = count_signal_t * influence_signal_t
    influence_weights_t = influence_weights_t / influence_weights_t.sum().clamp_min(1e-12)

    weights_t = (1.0 - beta) * count_weights_t + beta * influence_weights_t
    weights_t = weights_t / weights_t.sum().clamp_min(1e-12)

    return CLVQResult(
        centers=centers_t,
        center_labels=labels_t,
        counts=counts_t,
        influences=influences_t,
        weights=weights_t,
    )


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
        self._default_prompt = next(iter(self.class_prompts.values()), "medical image")

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


def run_distillation(args: argparse.Namespace) -> dict[str, Any]:
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    amp_enabled = bool(args.fp16 and device.type == "cuda")
    dataset_spec = get_dataset_spec(args.dataset)
    output_dir = Path(args.output_dir or f"outputs/{dataset_spec.name}_224_distill")
    teacher_baseline_dir = Path(args.teacher_baseline_dir or f"outputs/{dataset_spec.name}_224_distill_baseline")
    resolved_lora_path = resolve_lora_path(dataset_spec.name, args.lora_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_baseline_dir.mkdir(parents=True, exist_ok=True)

    run_config = vars(args).copy()
    run_config["output_dir"] = str(output_dir)
    run_config["teacher_baseline_dir"] = str(teacher_baseline_dir)
    run_config["lora_path"] = resolved_lora_path
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    split_bundle = dataset_spec.load_dataset_splits(data_root=args.data_root, image_size=args.image_size)
    train_set = split_bundle.train_set
    val_set = split_bundle.val_set
    test_set = split_bundle.test_set
    num_classes = split_bundle.num_classes
    print(
        f"[Setup] dataset={dataset_spec.name} device={device} "
        f"train={len(train_set)} classes={num_classes} baseline_dir={teacher_baseline_dir} "
        f"lora_path={resolved_lora_path}"
    )

    teacher_ckpt_path = teacher_baseline_dir / "teacher_best.pt"
    try:
        teacher, _ = load_teacher_checkpoint(
            checkpoint_path=teacher_ckpt_path,
            num_classes=num_classes,
            backbone=args.teacher_backbone,
            imagenet_pretrained=args.imagenet_pretrained,
            device=device,
        )
        print(f"[Teacher] Loaded existing checkpoint: {teacher_ckpt_path}")
    except Exception as exc:
        warnings.warn(
            f"Failed to load teacher checkpoint ({teacher_ckpt_path}), retraining. error={exc}",
            RuntimeWarning,
        )
        teacher, _ = train_teacher_baseline(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            num_classes=num_classes,
            args=args,
            device=device,
            amp_enabled=amp_enabled,
            baseline_dir=teacher_baseline_dir,
        )

    # Latents are independent of IPC, so cache them under dataset baseline dir.
    latent_cache_path = teacher_baseline_dir / "train_latents.pt"
    vae_dtype = torch.float16 if amp_enabled else torch.float32
    vae, scaling_factor = load_vae(args.vae_model_id, device=device, dtype=vae_dtype)

    cached_latents = load_latent_cache(
        cache_path=latent_cache_path,
        expected_num_samples=len(train_set),
        expected_dataset=dataset_spec.name,
        expected_vae_model_id=args.vae_model_id,
        expected_image_size=args.image_size,
    )

    latent_source = "cache"
    if cached_latents is None:
        encode_loader = build_encode_loader(
            train_set=train_set,
            image_size=args.image_size,
            encode_batch_size=args.encode_batch_size,
            num_workers=args.num_workers,
            device=device,
        )

        latents, latent_labels = encode_training_images(
            vae=vae,
            data_loader=encode_loader,
            scaling_factor=scaling_factor,
            device=device,
            dtype=vae_dtype,
        )
        save_latent_cache(
            cache_path=latent_cache_path,
            latents=latents,
            labels=latent_labels,
            dataset_name=dataset_spec.name,
            vae_model_id=args.vae_model_id,
            image_size=args.image_size,
        )
        latent_source = "vae_encoder"
    else:
        latents, latent_labels = cached_latents

    influence_mode = str(args.influence_mode).strip().lower()
    sample_influence: torch.Tensor | None = None
    if influence_mode == "image-grad":
        influence_loader = build_encode_loader(
            train_set=train_set,
            image_size=args.image_size,
            encode_batch_size=args.encode_batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        raw_influence = compute_image_gradient_influence(
            teacher=teacher,
            data_loader=influence_loader,
            device=device,
            log_prefix="[Influence-ImageGrad]",
        )
        sample_influence = stabilize_influence_scores(
            raw_influence,
            quantile=args.influence_quantile,
            min_value=args.influence_min_value,
            max_value=args.influence_max_value,
            log_prefix="[Influence-Sample]",
        )
    elif influence_mode != "none":
        raise ValueError(f"Unsupported influence mode: {args.influence_mode}")

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
        medoid_anchor=args.clvq_medoid_anchor,
        sample_influence=sample_influence,
        influence_blend_beta=args.influence_blend_beta,
        weight_count_power=args.weight_count_power,
        weight_influence_power=args.weight_influence_power,
    )

    class_prompts = dataset_spec.build_class_prompts()
    decoder = ReverseSDEDecoder(
        model_id=args.diffusion_model_id,
        vae=vae,
        device=device,
        dtype=vae_dtype,
        num_inference_steps=args.sde_steps,
        noise_strength=args.sde_noise_strength,
        lora_path=resolved_lora_path,
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
        influences=clvq.influences,
        soft_labels=soft_labels,
        center_labels=clvq.center_labels,
        counts=clvq.counts,
        saved_paths=saved_paths,
        dataset_name=dataset_spec.name,
        lora_path=resolved_lora_path,
    )

    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = {
        "dataset": dataset_spec.name,
        "data_root": args.data_root,
        "teacher_baseline_dir": str(teacher_baseline_dir),
        "num_classes": int(num_classes),
        "clvq_mode": "class-wise-clvq",
        "clusters_per_class": int(args.clusters_per_class),
        "clvq_medoid_anchor": float(args.clvq_medoid_anchor),
        "num_distilled": int(distilled_images.size(0)),
        "teacher_backbone": str(args.teacher_backbone),
        "teacher_temperature": float(args.teacher_temperature),
        "influence_mode": influence_mode,
        "influence_blend_beta": float(args.influence_blend_beta),
        "weight_count_power": float(args.weight_count_power),
        "weight_influence_power": float(args.weight_influence_power),
        "influence_quantile": float(args.influence_quantile),
        "influence_min_value": float(args.influence_min_value),
        "influence_max_value": float(args.influence_max_value),
        "center_influence_mean": float(clvq.influences.mean().item()),
        "center_influence_min": float(clvq.influences.min().item()),
        "center_influence_max": float(clvq.influences.max().item()),
        "lora_path": resolved_lora_path,
        "latent_cache_path": str(latent_cache_path),
        "latent_source": latent_source,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("[Done] Distillation complete.")
    print(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dataset distillation: teacher training + VAE encode + class-wise CLVQ + reverse-SDE decode"
    )
    parser.add_argument("--dataset", default="dermamnist", choices=supported_datasets())
    parser.add_argument("--data-root", type=str, default=default_data_root())
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--teacher-baseline-dir", type=str, default="")

    parser.add_argument("--clusters-per-class", type=int, default=100)
    parser.add_argument("--clvq-gamma0", type=float, default=0.5)
    parser.add_argument("--clvq-alpha", type=float, default=0.6)
    parser.add_argument("--clvq-max-iter", type=int, default=10000)
    parser.add_argument("--clvq-tol", type=float, default=1e-5)
    parser.add_argument("--clvq-check-interval", type=int, default=500)
    parser.add_argument("--clvq-medoid-anchor", type=float, default=0.0)
    parser.add_argument("--influence-mode", type=str, default="image-grad", choices=["none", "image-grad"])
    parser.add_argument("--influence-blend-beta", type=float, default=0.25)
    parser.add_argument("--weight-count-power", type=float, default=0.5)
    parser.add_argument("--weight-influence-power", type=float, default=1.5)
    parser.add_argument("--influence-quantile", type=float, default=0.85)
    parser.add_argument("--influence-min-value", type=float, default=1e-3)
    parser.add_argument("--influence-max-value", type=float, default=1.0)

    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--decode-batch-size", type=int, default=32)

    parser.add_argument("--vae-model-id", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--diffusion-model-id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora-path", type=str, default="")
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
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_distillation(args)


if __name__ == "__main__":
    main()
