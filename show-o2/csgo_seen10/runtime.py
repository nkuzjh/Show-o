"""Shared runtime helpers for Seen-10 training and generation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .data import CSGOSeen10Dataset, MAPS, sha256_file
from .model import CSGOSeen10Model, load_showo2_seen10


OFFICIAL_SHOWO_SHA256 = "a596cbc305c1df987c125d4f218e78f39b681621904cccfb2a3bf0ca0327f92c"
OFFICIAL_SHOWO_SIZE = 5661862314
OFFICIAL_VAE_SHA256 = "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
OFFICIAL_VAE_SIZE = 507609880
IMAGE_TOKEN_COUNT = 28 * 28 + 1


def seed_validation_rng(seed: int, sample_ids: Sequence[str] | str) -> int:
    """Seed per-sample validation noise independently of DataLoader worker startup.

    The first iteration of a persistent-worker DataLoader draws a worker base
    seed from the main process RNG. Validation targets and transport noise are
    sampled after that iterator is created, so seeding only once before the
    loop can give the first validation pass a different noise stream than later
    passes. Hashing stable manifest sample IDs and reseeding after each batch
    makes those stochastic inputs independent of iterator lifecycle and batch
    arrival timing.
    """
    ids = [sample_ids] if isinstance(sample_ids, str) else list(sample_ids)
    if not ids or any(not isinstance(sample_id, str) or not sample_id for sample_id in ids):
        raise ValueError("Validation RNG requires non-empty string sample IDs")

    digest = hashlib.sha256()
    digest.update(b"showo2-csgo-seen10-validation-v1\0")
    digest.update(str(int(seed)).encode("ascii"))
    digest.update(b"\0")
    for sample_id in ids:
        encoded = sample_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    sample_seed = int.from_bytes(digest.digest()[:8], byteorder="big") & ((1 << 63) - 1)

    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed)
    return sample_seed


@lru_cache(maxsize=32)
def _cached_sha256(path_string: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    return sha256_file(Path(path_string))


def _asset_sha256(path: Path) -> str:
    stat = path.stat()
    return _cached_sha256(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def resolve_project_path(path: str | Path, project_root: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path(project_root) / resolved
    return resolved.resolve()


def torch_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "none"}:
        return torch.float32
    raise ValueError(f"Unsupported weight dtype: {name}")


def autocast_context(device: torch.device, dtype: torch.dtype):
    if dtype == torch.float32 or device.type not in {"cuda", "cpu"}:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def load_vae(config: Any, *, device: torch.device, weight_dtype: torch.dtype, showo_dir: str | Path):
    from models import WanVAE

    vae_path = resolve_project_path(config.model.vae_model.pretrained_model_path, showo_dir)
    if not vae_path.is_file():
        raise FileNotFoundError(f"Wan2.1 VAE weights not found: {vae_path}")
    vae = WanVAE(vae_pth=str(vae_path), dtype=weight_dtype, device=device)
    vae.model.eval().requires_grad_(False)
    return vae, vae_path


def load_runtime(config: Any, *, showo_dir: str | Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch_dtype(config.training.mixed_precision)
    vae, vae_path = load_vae(config, device=device, weight_dtype=weight_dtype, showo_dir=showo_dir)
    model, tokenizer, token_ids, loading_info = load_showo2_seen10(
        config, device=device, weight_dtype=weight_dtype, showo_dir=str(showo_dir)
    )
    return model, vae, tokenizer, token_ids, device, weight_dtype, vae_path, loading_info


def create_transport_and_sampler(config: Any):
    from transport import Sampler, create_transport

    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
        snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift,
        seq_len=IMAGE_TOKEN_COUNT,
    )
    return transport, Sampler(transport)


def build_interleaved_inputs(
    samples: Mapping[str, Any],
    tokenizer: Any,
    token_ids: Mapping[str, int],
    *,
    device: torch.device,
    max_seq_length: int = 2048,
    max_prompt_tokens: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a native Show-o2 sequence with radar then target/noise modalities."""
    from models import omni_attn_mask_naive

    map_names = list(samples["map_name"])
    token_rows: List[List[int]] = []
    positions: List[List[List[int]]] = []
    target_offsets: List[int] = []
    for map_name in map_names:
        instruction = (
            "Generate a first person view game screenshot conditioned on the radar and numeric pose. "
            f"Map: {map_name}."
        )
        text = tokenizer(instruction, add_special_tokens=False).input_ids[:max_prompt_tokens]
        row = [token_ids["bos_id"], *text]
        first_offset = len(row) + 1
        row.extend([token_ids["boi_id"], *([token_ids["img_pad_id"]] * IMAGE_TOKEN_COUNT), token_ids["eoi_id"]])
        second_offset = len(row) + 1
        row.extend([token_ids["boi_id"], *([token_ids["img_pad_id"]] * IMAGE_TOKEN_COUNT), token_ids["eoi_id"]])
        row.append(token_ids["eos_id"])
        token_rows.append(row)
        positions.append([[first_offset, IMAGE_TOKEN_COUNT], [second_offset, IMAGE_TOKEN_COUNT]])
        target_offsets.append(second_offset)

    actual_length = max(len(row) for row in token_rows)
    sequence_length = int(math.ceil(actual_length / 128.0) * 128)
    if sequence_length > int(max_seq_length):
        raise ValueError(
            f"Two-image interleaved prompt needs {sequence_length} tokens, above max_seq_length={max_seq_length}"
        )
    batch_size = len(token_rows)
    required_length = 2 * (IMAGE_TOKEN_COUNT + 2) + 2
    if sequence_length < required_length:
        raise RuntimeError(
            f"Two-image sequence is too short ({sequence_length}); needs at least {required_length} tokens"
        )
    text_tokens = torch.full(
        (batch_size, sequence_length),
        int(token_ids["pad_id"]),
        dtype=torch.long,
        device=device,
    )
    image_masks = torch.zeros((batch_size, sequence_length), dtype=torch.long, device=device)
    for index, row in enumerate(token_rows):
        text_tokens[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        offset = target_offsets[index]
        image_masks[index, offset : offset + IMAGE_TOKEN_COUNT] = 1
    modality_positions = torch.tensor(positions, dtype=torch.long, device=device)
    for index, (first, second) in enumerate(positions):
        first_offset, first_length = first
        second_offset, second_length = second
        if first_length != IMAGE_TOKEN_COUNT or second_length != IMAGE_TOKEN_COUNT:
            raise RuntimeError("Each Show-o2 image modality must contain one time token and 784 patches")
        if second_offset != first_offset + first_length + 2:
            raise RuntimeError("Radar and target modalities are not interleaved in the expected order")
        if image_masks[index, first_offset : first_offset + first_length].any():
            raise RuntimeError("The clean radar modality must be excluded from flow loss")
        if not image_masks[index, second_offset : second_offset + second_length].all():
            raise RuntimeError("The target/noise modality must be included in flow loss")
    attention_mask = omni_attn_mask_naive(
        batch_size, sequence_length, modality_positions, device
    )
    return text_tokens, modality_positions, image_masks, attention_mask


@torch.no_grad()
def encode_images(vae: Any, images: torch.Tensor, *, deterministic: bool) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"Expected image batch [B,C,H,W], got {tuple(images.shape)}")
    encoded = vae.sample(images.unsqueeze(2), deterministic=deterministic)
    if encoded.ndim != 5 or encoded.shape[2] != 1:
        raise RuntimeError(f"WanVAE returned an unexpected latent shape: {tuple(encoded.shape)}")
    return encoded.squeeze(2).float()


def build_flow_pair(
    batch: Mapping[str, Any],
    *,
    vae: Any,
    transport: Any,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode the radar/target pair and create flow inputs with a clean radar endpoint."""
    radar = batch["radar"].to(device=device, dtype=weight_dtype)
    target = batch["target"].to(device=device, dtype=weight_dtype)
    radar_latents = encode_images(vae, radar, deterministic=True)
    target_latents = encode_images(vae, target, deterministic=False)
    t_target, noise, clean_target = transport.sample(target_latents)
    t_target, noisy_target, target_velocity = transport.path_sampler.plan(t_target, noise, clean_target)

    batch_size = radar_latents.shape[0]
    image_latents = torch.stack([radar_latents, noisy_target], dim=1).flatten(0, 1)
    image_labels = torch.stack([torch.zeros_like(radar_latents), target_velocity], dim=1).flatten(0, 1)
    clean_radar_t = torch.ones_like(t_target)
    timesteps = torch.stack([clean_radar_t, t_target], dim=1).reshape(-1)
    return image_latents, timesteps, image_labels


def move_model_inputs(
    batch: Mapping[str, Any],
    tokenizer: Any,
    token_ids: Mapping[str, int],
    *,
    device: torch.device,
    max_seq_length: int,
    weight_dtype: torch.dtype,
    max_prompt_tokens: int = 64,
) -> Dict[str, torch.Tensor]:
    text_tokens, modality_positions, image_masks, attention_mask = build_interleaved_inputs(
        batch,
        tokenizer,
        token_ids,
        device=device,
        max_seq_length=max_seq_length,
        max_prompt_tokens=max_prompt_tokens,
    )
    return {
        "text_tokens": text_tokens,
        "modality_positions": modality_positions,
        "image_masks": image_masks,
        "attention_mask": attention_mask.to(dtype=weight_dtype),
        "radar_pose": batch["pose"].to(device=device, dtype=torch.float32),
        "map_ids": batch["map_id"].to(device=device, dtype=torch.long),
    }


def pil_from_model_tensor(image: torch.Tensor) -> Image.Image:
    image = image.detach().float()
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected decoded RGB image [3,H,W], got {tuple(image.shape)}")
    image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0).to(torch.float32)
    pixels = (image * 255.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def save_rgb_jpeg(image: Image.Image, path: str | Path, *, skip_valid: bool = True) -> bool:
    """Write a benchmark JPEG atomically; preserve an existing valid output."""
    destination = Path(path)
    if skip_valid and is_valid_output(destination):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    rgb = image.convert("RGB")
    if rgb.size != (448, 448):
        rgb = rgb.resize((448, 448), Image.Resampling.BICUBIC)
    temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp.jpg")
    try:
        # Keep PIL's default JPEG encoding, matching the UniLIP generation output path.
        rgb.save(temporary, format="JPEG")
        if skip_valid and is_valid_output(destination):
            temporary.unlink(missing_ok=True)
            return False
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def is_valid_output(path: str | Path) -> bool:
    try:
        with Image.open(path) as image:
            return image.format == "JPEG" and image.mode == "RGB" and image.size == (448, 448)
    except (FileNotFoundError, OSError, ValueError):
        return False


@torch.no_grad()
def generate_one(
    model: CSGOSeen10Model,
    vae: Any,
    sampler: Any,
    batch: Mapping[str, Any],
    tokenizer: Any,
    token_ids: Mapping[str, int],
    *,
    device: torch.device,
    weight_dtype: torch.dtype,
    max_seq_length: int,
    max_prompt_tokens: int = 64,
    num_inference_steps: int,
    sampling_method: str,
    atol: float,
    rtol: float,
    time_shifting_factor: float,
    guidance_scale: float,
    generator: torch.Generator,
) -> Image.Image:
    if len(batch["map_name"]) != 1:
        raise ValueError("Native Show-o2 only_denoise_last_image supports inference batch size exactly 1")
    if float(guidance_scale) != 0.0:
        raise ValueError("Seen-10 builds one conditioned sequence; guidance_scale must be 0")
    radar = batch["radar"].to(device=device, dtype=weight_dtype)
    radar_latent = encode_images(vae, radar, deterministic=True)
    noise = torch.randn(
        radar_latent.shape,
        generator=generator,
        device=device,
        dtype=radar_latent.dtype,
    )
    initial_latents = torch.cat([radar_latent, noise], dim=0)
    model_inputs = move_model_inputs(
        batch,
        tokenizer,
        token_ids,
        device=device,
        max_seq_length=max_seq_length,
        weight_dtype=weight_dtype,
        max_prompt_tokens=max_prompt_tokens,
    )
    model_kwargs = {
        **model_inputs,
        "max_seq_len": model_inputs["text_tokens"].shape[1],
        "guidance_scale": float(guidance_scale),
        "only_denoise_last_image": True,
        "output_hidden_states": True,
    }
    sample_fn = sampler.sample_ode(
        sampling_method=sampling_method,
        num_steps=int(num_inference_steps),
        atol=float(atol),
        rtol=float(rtol),
        reverse=False,
        time_shifting_factor=float(time_shifting_factor),
    )
    model.eval()
    with autocast_context(device, weight_dtype):
        generated = sample_fn(initial_latents, model.t2i_generate, **model_kwargs)[-1]
    target_latent = generated[-1:].unsqueeze(2)
    decoded = vae.batch_decode(target_latent).squeeze(0).squeeze(1)
    return pil_from_model_tensor(decoded)


def single_sample_batch(sample: Mapping[str, Any]) -> Dict[str, Any]:
    batched: Dict[str, Any] = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batched[key] = value.unsqueeze(0)
        else:
            batched[key] = [value]
    return batched


def accelerator_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().to(device="cpu").contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_finetune_weights(model: CSGOSeen10Model, checkpoint_dir: str | Path) -> None:
    from safetensors.torch import save_file

    checkpoint_dir = Path(checkpoint_dir)
    backbone_state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.backbone.named_parameters()
        if parameter.requires_grad
    }
    adapter_state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.radar_adapter.state_dict().items()
    }
    save_file(backbone_state, str(checkpoint_dir / "backbone_trainable.safetensors"))
    save_file(adapter_state, str(checkpoint_dir / "radar_adapter.safetensors"))


def load_finetune_weights(model: CSGOSeen10Model, checkpoint_dir: str | Path) -> None:
    from safetensors.torch import load_file

    checkpoint_dir = Path(checkpoint_dir).resolve(strict=True)
    backbone_path = checkpoint_dir / "backbone_trainable.safetensors"
    adapter_path = checkpoint_dir / "radar_adapter.safetensors"
    if not backbone_path.is_file() or not adapter_path.is_file():
        raise FileNotFoundError(f"No inference weights found in checkpoint {checkpoint_dir}")
    state = load_file(str(backbone_path), device="cpu")
    trainable_parameters = {
        name: parameter for name, parameter in model.backbone.named_parameters() if parameter.requires_grad
    }
    if set(state) != set(trainable_parameters):
        missing = sorted(set(trainable_parameters) - set(state))
        unexpected = sorted(set(state) - set(trainable_parameters))
        raise RuntimeError(f"Trainable checkpoint key mismatch; missing={missing}, unexpected={unexpected}")
    with torch.no_grad():
        for name, parameter in trainable_parameters.items():
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))
    model.radar_adapter.load_state_dict(load_file(str(adapter_path), device="cpu"), strict=True)


def atomic_symlink(target: str, link_path: str | Path) -> None:
    link_path = Path(link_path)
    temporary = link_path.with_name(f".{link_path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target, target_is_directory=True)
    os.replace(temporary, link_path)


def sha256_paths(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(_asset_sha256(path)))
    return digest.hexdigest()


def _repository_info(project_root: str | Path) -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(project_root), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": commit, "working_tree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "working_tree_dirty": None}


def write_inference_manifest(
    output_root: str | Path,
    *,
    project_root: str | Path,
    config_path: str | Path,
    inference_settings: Mapping[str, Any],
    showo_path: str | Path,
    vae_path: str | Path,
    checkpoint_dir: str | Path,
    split: str,
    dataset: CSGOSeen10Dataset,
    train_seed: int,
    inference_seed: int,
    smoke_only: bool,
    generation_complete: bool = True,
) -> Path:
    output_root = Path(output_root)
    showo_path, vae_path = Path(showo_path), Path(vae_path)
    checkpoint_dir = Path(checkpoint_dir).resolve(strict=True)
    showo_weights = showo_path / "pytorch_model.bin"
    if not showo_weights.is_file():
        raise FileNotFoundError(f"Official Show-o2 checkpoint weight file not found: {showo_weights}")
    if showo_weights.stat().st_size != OFFICIAL_SHOWO_SIZE or _asset_sha256(showo_weights) != OFFICIAL_SHOWO_SHA256:
        raise ValueError("Local Show-o2 weights do not match the verified official 1.5B checkpoint")
    if vae_path.stat().st_size != OFFICIAL_VAE_SIZE or _asset_sha256(vae_path) != OFFICIAL_VAE_SHA256:
        raise ValueError("Local Wan2.1 VAE does not match the verified official asset")
    checkpoint_files = [
        checkpoint_dir / "backbone_trainable.safetensors",
        checkpoint_dir / "radar_adapter.safetensors",
    ]
    for path in checkpoint_files:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file is missing: {path}")
    covered_maps = list(dict.fromkeys(row["map_name"] for row in dataset.rows))
    payload = {
        "model": "Show-o2-1.5B",
        "repository": _repository_info(project_root),
        "configuration": {
            "path": str(Path(config_path).resolve()),
            "sha256": sha256_file(Path(config_path).resolve()),
        },
        "inference_settings": dict(inference_settings),
        "official_weights": {
            "showo_path": str(showo_path),
            "showo_pytorch_model_bin_sha256": OFFICIAL_SHOWO_SHA256,
            "vae_path": str(vae_path),
            "vae_sha256": OFFICIAL_VAE_SHA256,
        },
        "checkpoint": {
            "path": str(checkpoint_dir),
            "sha256": sha256_paths(checkpoint_files),
        "files": {path.name: _asset_sha256(path) for path in checkpoint_files},
        },
        "split": split,
        "maps": covered_maps,
        "expected_maps": list(MAPS),
        "sample_count": len(dataset),
        "train_seed": int(train_seed),
        "inference_seed": int(inference_seed),
        **dataset.provenance,
        "smoke_only": bool(smoke_only),
        "incomplete_limit": bool(smoke_only),
        "generation_complete": bool(generation_complete),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "inference_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        compatibility_fields = (
            "model",
            "configuration",
            "inference_settings",
            "official_weights",
            "checkpoint",
            "split",
            "maps",
            "expected_maps",
            "sample_count",
            "train_seed",
            "inference_seed",
            "benchmark_manifest_sha256",
            "asset_report_sha256",
            "selected_images_sha256",
            "selected_images_checksum_file_sha256",
            "smoke_only",
        )
        differences = [
            field for field in compatibility_fields if existing.get(field) != payload.get(field)
        ]
        if differences:
            raise ValueError(
                "Refusing to mix or re-attribute existing generation outputs; "
                f"inference manifest differs in: {differences}. Use a fresh output root."
            )
        if existing.get("generation_complete") and not generation_complete:
            return manifest_path
    else:
        generated_root = output_root / "gen_imgs"
        if generated_root.is_dir() and any(
            path.is_file() and path.suffix.lower() == ".jpg" and is_valid_output(path)
            for path in generated_root.rglob("*")
        ):
            raise ValueError(
                f"Valid generation images already exist without provenance at {output_root}; "
                "refusing to attribute them to a checkpoint. Use a fresh output root."
            )
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return manifest_path


def save_loss_curve(loss_jsonl: str | Path, output_png: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps: List[int] = []
    losses: List[float] = []
    path = Path(loss_jsonl)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                steps.append(int(record["step"]))
                losses.append(float(record["loss"]))
    if not steps:
        return
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(steps, losses, linewidth=1.0)
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Flow loss")
    axis.set_title("Show-o2 CSGO Seen-10 training loss")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=150)
    plt.close(figure)
