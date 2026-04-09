from __future__ import annotations

import argparse
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline
from medmnist import INFO, DermaMNIST
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.models import resnet18
from torchvision.utils import make_grid
from torchvision.models import ResNet18_Weights

@dataclass
class ClusterResult:
    centers: torch.Tensor
    labels: torch.Tensor
    counts: torch.Tensor
    weights: torch.Tensor


class DermaMNISTDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data: DermaMNIST) -> None:
        super().__init__()
        self.images = np.ascontiguousarray(data.imgs)
        self.labels = data.labels.reshape(-1).astype(np.int64, copy=False)

    @override
    def __len__(self) -> int:
        return int(self.images.shape[0])

    @override
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[index]).permute(2, 0, 1).float() / 255.0
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label


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


def load_dermamnist(data_root: str) -> tuple[DermaMNIST, DermaMNIST, DermaMNIST, dict[str, Any]]:
    train_set = DermaMNIST(split="train", download=True, root=data_root, size=224)
    test_set = DermaMNIST(split="test", download=True, root=data_root, size=224)
    val_set = DermaMNIST(split="val", download=True, root=data_root, size=224)

    num_classes = len(INFO["dermamnist"]["label"])
    label_names = [INFO["dermamnist"]["label"][str(i)] for i in range(num_classes)]
    metadata = {"task_type": "multiclass", "num_classes": num_classes, "label_names": label_names}

    return train_set, test_set, val_set, metadata

def dermamnist_class_prompts() -> dict[int, str]:
    label_map = INFO["dermamnist"]["label"]
    prompts: dict[int, str] = {}
    for class_id_str, class_name in label_map.items():
        prompts[int(class_id_str)] = f"dermoscopic image of {class_name}"
    return prompts


def build_data_loaders(
    train_set: DermaMNIST,
    val_set: DermaMNIST,
    test_set: DermaMNIST,
    encode_batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
]:
    pin_memory = device.type == "cuda"

    encode_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        DermaMNISTDataset(train_set),
        batch_size=encode_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        DermaMNISTDataset(val_set),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        DermaMNISTDataset(test_set),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return encode_loader, val_loader, test_loader


def load_vae(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> tuple[AutoencoderKL, float]:
    vae_subfolder = None
    if args.vae_subfolder and args.vae_subfolder.lower() not in {"none", "null", ""}:
        vae_subfolder = args.vae_subfolder

    vae_load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if vae_subfolder is not None:
        vae_load_kwargs["subfolder"] = vae_subfolder

    vae_source = args.vae_model_id if vae_subfolder is None else f"{args.vae_model_id}/{vae_subfolder}"
    print(f"[Encoding] loading AutoencoderKL: {vae_source}")

    vae = AutoencoderKL.from_pretrained(args.vae_model_id, **vae_load_kwargs)
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

    vae.eval()
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


def cluster_latents(
    latents: torch.Tensor,
    labels: torch.Tensor,
    clusters_per_class: int,
    seed: int,
) -> ClusterResult:
    if clusters_per_class <= 0:
        raise ValueError("clusters_per_class must be positive.")

    flat_latents = latents.view(latents.size(0), -1).numpy().astype(np.float32, copy=False)
    labels_np = labels.numpy()
    unique_classes = sorted(int(v) for v in np.unique(labels_np))

    center_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    count_chunks: list[torch.Tensor] = []

    for class_id in unique_classes:
        class_indices = np.where(labels_np == class_id)[0]
        class_latents = flat_latents[class_indices]
        k = min(clusters_per_class, class_latents.shape[0])
        if k < clusters_per_class:
            warnings.warn(
                f"Class {class_id} has only {class_latents.shape[0]} samples; using {k} clusters.",
                RuntimeWarning,
            )

        kmeans = MiniBatchKMeans(
            n_clusters=k,
            n_init=10,
            random_state=seed,
            batch_size=min(4096, max(128, 8 * k)),
        )
        assignments = kmeans.fit_predict(class_latents)
        counts = np.bincount(assignments, minlength=k).astype(np.int64)

        center_chunk = torch.from_numpy(kmeans.cluster_centers_).view(k, *latents.shape[1:]).float()
        label_chunk = torch.full((k,), fill_value=class_id, dtype=torch.long)
        count_chunk = torch.from_numpy(counts)

        center_chunks.append(center_chunk)
        label_chunks.append(label_chunk)
        count_chunks.append(count_chunk)

        print(f"[Clustering] class={class_id} clusters={k} samples={int(class_latents.shape[0])}")

    centers = torch.cat(center_chunks, dim=0)
    center_labels = torch.cat(label_chunks, dim=0)
    counts = torch.cat(count_chunks, dim=0)
    weights = counts.float() / counts.float().sum()

    return ClusterResult(
        centers=centers,
        labels=center_labels,
        counts=counts,
        weights=weights,
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
        use_prompt_conditioning: bool,
        guidance_scale: float,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.num_inference_steps = max(2, int(num_inference_steps))
        self.noise_strength = float(np.clip(noise_strength, 0.01, 1.0))
        self.class_prompts = dict(class_prompts)
        self.use_prompt_conditioning = bool(use_prompt_conditioning)
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
            warnings.warn(
                f"LoRA path does not exist: {adapter_dir}. Continue without LoRA.",
                RuntimeWarning,
            )
            return

        loaded = False
        try:
            self.pipe.load_lora_weights(str(adapter_dir))
            loaded = True
        except Exception as exc_load_lora:
            warnings.warn(
                f"load_lora_weights failed ({exc_load_lora}), trying unet.load_attn_procs.",
                RuntimeWarning,
            )
            try:
                self.pipe.unet.load_attn_procs(str(adapter_dir))
                loaded = True
            except Exception as exc_load_attn:
                warnings.warn(
                    f"load_attn_procs also failed ({exc_load_attn}). Continue without LoRA.",
                    RuntimeWarning,
                )

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
        prompt_embeddings: list[torch.Tensor] = []
        for class_id in labels.tolist():
            if self.use_prompt_conditioning:
                prompt = self.class_prompts.get(int(class_id), self._default_prompt)
            else:
                prompt = ""
            prompt_embeddings.append(self._encode_prompt(prompt))
        return torch.cat(prompt_embeddings, dim=0)

    @torch.no_grad()
    def _decode_batch(self, centers: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        start_idx = int((1.0 - self.noise_strength) * (len(timesteps) - 1))
        start_idx = max(0, min(start_idx, len(timesteps) - 1))
        start_timestep = timesteps[start_idx]

        latents = centers.to(device=self.device, dtype=self.dtype)
        noise = torch.randn_like(latents)
        timestep_batch = start_timestep.expand(latents.size(0))
        latents = self.scheduler.add_noise(latents, noise, timestep_batch)

        cond_prompt_embeds = self._prompt_embeddings_for_labels(labels)
        uncond_prompt_embeds = None
        use_cfg = self.guidance_scale > 1.0
        if use_cfg:
            uncond_prompt_embeds = self._null_prompt_embeddings(latents.size(0))

        for timestep in timesteps[start_idx:]:
            model_input = self.scheduler.scale_model_input(latents, timestep)

            if use_cfg:
                assert uncond_prompt_embeds is not None
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
        decoded_batches: list[torch.Tensor] = []
        total = centers.size(0)
        for start in range(0, total, batch_size):
            end = min(total, start + batch_size)
            images = self._decode_batch(centers[start:end], labels[start:end])
            decoded_batches.append(images.float().cpu())
            print(f"[Decoding-ReverseSDE] {end}/{total}")
        return torch.cat(decoded_batches, dim=0)

    def cleanup(self) -> None:
        del self.pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_distilled_images(
    images: torch.Tensor,
    labels: torch.Tensor,
    output_dir: Path,
) -> list[str]:
    image_root = output_dir / "distilled_images"
    image_root.mkdir(parents=True, exist_ok=True)

    relative_paths: list[str] = []
    for idx in range(images.size(0)):
        class_id = int(labels[idx].item())
        class_dir = image_root / f"class_{class_id}"
        class_dir.mkdir(parents=True, exist_ok=True)

        rel_path = Path(f"class_{class_id}") / f"sample_{idx:05d}.png"
        abs_path = image_root / rel_path

        image_np = (images[idx].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(image_np).save(abs_path)
        relative_paths.append(str(rel_path))

    return relative_paths


def save_preview_grid(images: torch.Tensor, output_path: Path, max_images: int = 100) -> None:
    num_images = min(max_images, images.size(0))
    if num_images <= 0:
        return

    grid = make_grid(images[:num_images], nrow=min(10, num_images), pad_value=1.0)
    grid_np = (grid.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(grid_np).save(output_path)


def build_classifier(num_classes: int) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += images.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def train_weighted_student(
    distilled_images: torch.Tensor,
    distilled_labels: torch.Tensor,
    distilled_weights: torch.Tensor,
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    num_classes: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    train_set = TensorDataset(distilled_images.float(), distilled_labels.long(), distilled_weights.float())
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = DataLoader(
        train_set,
        batch_size=min(batch_size, len(train_set)),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = build_classifier(num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_acc = -1.0
    best_epoch = -1
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_weighted_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for images, labels, weights in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            weights = weights.to(device)

            logits = model(images)
            per_sample_ce = F.cross_entropy(logits, labels, reduction="none")
            loss = (per_sample_ce * weights).sum() / (weights.sum() + 1e-12)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_weighted_loss += float(loss.item()) * images.size(0)
            epoch_correct += int((logits.argmax(dim=1) == labels).sum().item())
            epoch_samples += images.size(0)

        train_loss = epoch_weighted_loss / max(epoch_samples, 1)
        train_acc = epoch_correct / max(epoch_samples, 1)
        val_loss, val_acc = evaluate_classifier(model, val_loader, device)
        test_loss, test_acc = evaluate_classifier(model, test_loader, device)

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
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                },
                output_dir / "student_best.pt",
            )

        print(
            f"[Weighted-Training] epoch={epoch:03d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} test_acc={test_acc:.4f}"
        )

    torch.save({"model": model.state_dict(), "history": history}, output_dir / "student_last.pt")

    with open(output_dir / "student_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    best_test_acc = 0.0
    for row in history:
        if int(row["epoch"]) == best_epoch:
            best_test_acc = row["test_acc"]
            break

    return {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc_at_best_val": best_test_acc,
        "final_test_acc": history[-1]["test_acc"],
        "final_test_loss": history[-1]["test_loss"],
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    set_global_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    print(f"[Setup] device={device}")
    print(f"[Setup] loading dataset from {args.data_root}")

    # 准备数据
    train_set, test_set, val_set, metadata = load_dermamnist(args.data_root)
    num_classes = metadata["num_classes"]
    print(f"[Setup] train_samples={len(train_set)} num_classes={num_classes}")

    encode_loader, val_loader, test_loader = build_data_loaders(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        encode_batch_size=args.encode_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    use_fp16 = bool(args.fp16 and device.type == "cuda")
    vae_dtype = torch.float16 if use_fp16 else torch.float32
    vae, scaling_factor = load_vae(args=args, device=device, dtype=vae_dtype)

    latents, latent_labels = encode_training_images(
        vae=vae,
        data_loader=encode_loader,
        scaling_factor=scaling_factor,
        device=device,
        dtype=vae_dtype,
    )
    print(f"[Encoding] latent_shape={tuple(latents.shape)}")

    clustered = cluster_latents(
        latents=latents,
        labels=latent_labels,
        clusters_per_class=args.clusters_per_class,
        seed=args.seed,
    )
    print(
        "[Clustering] "
        f"K={clustered.centers.size(0)} "
        f"weight_sum={clustered.weights.sum().item():.4f} "
        f"min_weight={clustered.weights.min().item():.6f} "
        f"max_weight={clustered.weights.max().item():.6f}"
    )

    class_prompts = dermamnist_class_prompts()
    print(
        f"[Decoding] reverse SDE with model: {args.diffusion_model_id} "
        f"prompt_conditioning={args.prompt_conditioning} guidance_scale={args.guidance_scale:.2f}"
    )
    reverse_decoder = ReverseSDEDecoder(
        model_id=args.diffusion_model_id,
        vae=vae,
        device=device,
        dtype=vae_dtype,
        num_inference_steps=args.sde_steps,
        noise_strength=args.sde_noise_strength,
        lora_path=args.lora_path,
        lora_scale=args.lora_scale,
        class_prompts=class_prompts,
        use_prompt_conditioning=args.prompt_conditioning,
        guidance_scale=args.guidance_scale,
    )

    distilled_images = reverse_decoder.decode(clustered.centers, clustered.labels, args.decode_batch_size)
    reverse_decoder.cleanup()

    distilled_images = distilled_images.float().cpu()
    save_preview_grid(distilled_images, output_dir / "distilled_preview.png")
    saved_paths = save_distilled_images(distilled_images, clustered.labels, output_dir)

    torch.save(
        {
            "centers": clustered.centers,
            "labels": clustered.labels,
            "counts": clustered.counts,
            "weights": clustered.weights,
            "images": distilled_images,
            "image_relative_paths": saved_paths,
        },
        output_dir / "distilled_data.pt",
    )

    with open(output_dir / "distilled_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "num_distilled": int(clustered.centers.size(0)),
                "class_labels": [int(v) for v in clustered.labels.tolist()],
                "cluster_counts": [int(v) for v in clustered.counts.tolist()],
                "weights": [float(v) for v in clustered.weights.tolist()],
                "weight_sum": float(clustered.weights.sum().item()),
                "prompt_conditioning": bool(args.prompt_conditioning),
                "guidance_scale": float(args.guidance_scale),
            },
            handle,
            indent=2,
        )

    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    training_summary = train_weighted_student(
        distilled_images=distilled_images,
        distilled_labels=clustered.labels,
        distilled_weights=clustered.weights,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        epochs=args.train_epochs,
        batch_size=args.train_batch_size,
        learning_rate=args.train_lr,
        weight_decay=args.weight_decay,
        device=device,
        output_dir=output_dir,
    )

    summary = {
        "data_root": args.data_root,
        "train_samples_used": int(len(train_set)),
        "num_classes": num_classes,
        "clusters_per_class": int(args.clusters_per_class),
        "num_distilled": int(clustered.centers.size(0)),
        "prompt_conditioning": bool(args.prompt_conditioning),
        "guidance_scale": float(args.guidance_scale),
        "best_val_acc": float(training_summary["best_val_acc"]),
        "test_acc_at_best_val": float(training_summary["test_acc_at_best_val"]),
        "final_test_acc": float(training_summary["final_test_acc"]),
        "final_test_loss": float(training_summary["final_test_loss"]),
        "best_epoch": int(training_summary["best_epoch"]),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("[Done] Pipeline complete.")
    print(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dermamnist 224 dataset distillation: Encoding -> Clustering -> Decoding -> Weighted Training"
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/dermamnist_224_distill")

    parser.add_argument("--clusters-per-class", type=int, default=50)
    parser.add_argument("--encode-batch-size", type=int, default=24)
    parser.add_argument("--decode-batch-size", type=int, default=8)

    parser.add_argument("--vae-model-id", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument(
        "--vae-subfolder",
        type=str,
        default="none",
        help="Subfolder for AutoencoderKL weights. Set to 'none' when loading a standalone VAE repo.",
    )
    parser.add_argument(
        "--diffusion-model-id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="outputs/lora_dreammnist",
        help="Directory of LoRA weights from train-lora-dreammnist.",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=0.9,
        help="LoRA fusion scale used during reverse-SDE decoding.",
    )
    parser.add_argument(
        "--prompt-conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use class-specific text prompts during reverse-SDE decoding.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale for reverse-SDE decoding.",
    )
    parser.add_argument("--sde-steps", type=int, default=80)
    parser.add_argument("--sde-noise-strength", type=float, default=0.2)

    parser.add_argument("--train-epochs", type=int, default=30)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--train-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
