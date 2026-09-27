"""Aligned Seen-10 generation training with an exact 128-sample source budget."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin, broadcast_object_list, set_seed
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from .data import CSGOSeen10Dataset, sha256_file
from .finetuning import (
    EXPECTED_TRAINABLE,
    FINAL_EXPECTED_TRAINABLE,
    FINAL_POLICY,
    LEGACY_POLICY,
    assert_aligned_parameter_counts,
    policy_from_config,
    trainable_parameter_audit,
)
from .runtime import (
    OFFICIAL_SHOWO_SHA256,
    OFFICIAL_SHOWO_SIZE,
    OFFICIAL_VAE_SHA256,
    OFFICIAL_VAE_SIZE,
    atomic_symlink,
    build_flow_pair,
    create_transport_and_sampler,
    dataset_kwargs,
    load_runtime,
    move_model_inputs,
    save_finetune_weights,
    seed_validation_rng,
)
from .sampler import (
    GLOBAL_BATCH_SIZE,
    GlobalSourceStream,
    PlannedMicroBatchSampler,
    sample_order_digest,
    validate_layout,
)


PROFILE = "csgo_seen10_exp32gen_aligned"
MAX_UPDATES = 19_500
WARMUP_UPDATES = 59
CHECKPOINT_STEPS = (4_000, 8_000, 12_000, 16_000, 19_500)
VALIDATION_SAMPLES = 5_000
TRAIN_SAMPLES = 50_000


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"Aligned profile requires {label}={expected!r}; got {actual!r}")


def validate_profile(config: Any) -> None:
    """Catch changes that would silently alter the approved experiment."""
    _require_equal("experiment", str(config.get("experiment", "")), PROFILE)
    _require_equal("training.seed", int(config.training.seed), 42)
    _require_equal("training.effective_generation_batch", int(config.training.effective_generation_batch), 128)
    _require_equal("training.max_optimizer_steps", int(config.training.max_optimizer_steps), MAX_UPDATES)
    _require_equal("training.checkpoint_steps", tuple(config.training.checkpoint_steps), CHECKPOINT_STEPS)
    _require_equal("dataset.resolution", int(config.dataset.resolution), 432)
    _require_equal("dataset.resize_mode", str(config.dataset.resize_mode), "full")
    _require_equal("dataset.conditioning_mode", str(config.dataset.conditioning_mode), "text")
    _require_equal("dataset.max_seq_length", int(config.dataset.max_seq_length), 2048)
    _require_equal("dataset.max_prompt_tokens", int(config.dataset.max_prompt_tokens), 256)
    _require_equal("optimizer.learning_rate", float(config.optimizer.learning_rate), 1e-4)
    _require_equal("optimizer.beta1", float(config.optimizer.beta1), 0.9)
    _require_equal("optimizer.beta2", float(config.optimizer.beta2), 0.999)
    _require_equal("optimizer.weight_decay", float(config.optimizer.weight_decay), 0.0)
    _require_equal("optimizer.epsilon", float(config.optimizer.epsilon), 1e-8)
    _require_equal("lr_scheduler.name", str(config.lr_scheduler.name), "cosine_with_min_lr")
    _require_equal("lr_scheduler.warmup_steps", int(config.lr_scheduler.warmup_steps), WARMUP_UPDATES)
    _require_equal("lr_scheduler.min_lr", float(config.lr_scheduler.min_lr), 1e-5)
    _require_equal("training.cond_dropout_prob", float(config.training.cond_dropout_prob), 0.0)
    _require_equal("training.deterministic_algorithms", bool(config.training.deterministic_algorithms), True)
    _require_equal("finetuning.rank", int(config.finetuning.rank), 32)
    _require_equal("finetuning.alpha", int(config.finetuning.alpha), 64)
    _require_equal("finetuning.dropout", float(config.finetuning.dropout), 0.05)
    _require_equal("finetuning.bias", str(config.finetuning.bias), "none")
    policy = policy_from_config(config)
    expected_trainable = EXPECTED_TRAINABLE if policy == LEGACY_POLICY else FINAL_EXPECTED_TRAINABLE
    _require_equal("finetuning.expected_trainable_parameters", int(config.finetuning.expected_trainable_parameters), expected_trainable)
    if policy == FINAL_POLICY:
        _require_equal("finetuning.lora_bias", config.finetuning.lora_bias, False)


def configure_determinism(config: Any) -> dict[str, Any]:
    """Set strict backend controls before Accelerator or CUDA is initialized."""
    if torch.cuda.is_initialized():
        raise RuntimeError("Aligned deterministic settings must be applied before CUDA initialization")
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace is None:
        workspace = ":4096:8"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
    elif workspace not in {":4096:8", ":16:8"}:
        raise ValueError(f"Incompatible CUBLAS_WORKSPACE_CONFIG={workspace!r}; expected :4096:8 or :16:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = bool(config.training.enable_tf32)
    return {
        "cublas_workspace_config": workspace,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_algorithms_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "mixed_precision": str(config.training.mixed_precision),
    }


def resolve_layout(config: Any, args: Any, world_size: int) -> tuple[int, int]:
    micro_batch = int(args.micro_batch if args.micro_batch is not None else config.training.batch_size)
    if args.gradient_accumulation is not None:
        accumulation = int(args.gradient_accumulation)
    else:
        local_global = world_size * micro_batch
        if local_global <= 0 or GLOBAL_BATCH_SIZE % local_global:
            raise ValueError(
                f"Cannot derive accumulation: {GLOBAL_BATCH_SIZE} is not divisible by "
                f"world_size * micro_batch = {local_global}"
            )
        accumulation = GLOBAL_BATCH_SIZE // local_global
    validate_layout(world_size, micro_batch, accumulation)
    return micro_batch, accumulation


def learning_rate_for_update(update: int, *, peak: float = 1e-4, minimum: float = 1e-5) -> float:
    """One-based optimizer-update schedule; update 59 is the warmup peak."""
    if not 1 <= update <= MAX_UPDATES:
        raise ValueError(f"Optimizer update must be in [1, {MAX_UPDATES}], got {update}")
    if update <= WARMUP_UPDATES:
        return peak * update / WARMUP_UPDATES
    fraction = (update - WARMUP_UPDATES) / (MAX_UPDATES - WARMUP_UPDATES)
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * fraction))


def make_scheduler(optimizer: AdamW, config: Any) -> LambdaLR:
    peak = float(config.optimizer.learning_rate)
    minimum = float(config.lr_scheduler.min_lr)
    return LambdaLR(
        optimizer,
        lambda epoch: learning_rate_for_update(min(epoch + 1, MAX_UPDATES), peak=peak, minimum=minimum) / peak,
    )


def _semantic_config_digest(config: Any) -> str:
    settings = copy.deepcopy(OmegaConf.to_container(config, resolve=True))
    for key in ("batch_size", "gradient_accumulation_steps", "log_every", "smoke_validation_samples"):
        settings["training"].pop(key, None)
    for key in ("num_workers", "pin_memory", "persistent_workers"):
        settings["dataset"].pop(key, None)
    raw = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _resume_audit_matches(saved: Mapping[str, Any], current: Mapping[str, Any], *, policy: str) -> bool:
    """Accept omitted v1 audit schema fields only when their value is empty."""
    comparable = dict(saved)
    if policy == LEGACY_POLICY:
        comparable.setdefault("policy", LEGACY_POLICY)
        for key in ("parameter_aliases", "storage_aliases"):
            if key not in comparable and current.get(key) == []:
                comparable[key] = []
    return comparable == current


def _base_provenance(config: Any, script_dir: Path, vae_path: Path) -> dict[str, Any]:
    showo_path = Path(str(config.model.showo.pretrained_model_path)).expanduser()
    if not showo_path.is_absolute():
        showo_path = script_dir / showo_path
    showo_weights = (showo_path / "pytorch_model.bin").resolve(strict=True)
    vae_path = Path(vae_path).resolve(strict=True)
    if showo_weights.stat().st_size != OFFICIAL_SHOWO_SIZE or vae_path.stat().st_size != OFFICIAL_VAE_SIZE:
        raise ValueError("Aligned training requires the verified official Show-o2 and Wan VAE assets")
    showo_sha256 = sha256_file(showo_weights)
    vae_sha256 = sha256_file(vae_path)
    if showo_sha256 != OFFICIAL_SHOWO_SHA256 or vae_sha256 != OFFICIAL_VAE_SHA256:
        raise ValueError("Aligned training assets do not match verified official SHA-256 digests")
    return {
        "showo_path": str(showo_weights),
        "showo_sha256": showo_sha256,
        "vae_path": str(vae_path),
        "vae_sha256": vae_sha256,
    }


def _output_path(args: Any, config: Any, script_dir: Path) -> Path:
    project_root = script_dir.parent
    output_dir = Path(args.output_dir) if args.output_dir else project_root / str(config.output_root) / "seed_42"
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    return output_dir.resolve()


def _checkpoint_path(output_dir: Path, resume: str) -> Path:
    checkpoint_root = (output_dir / "checkpoints").resolve()
    requested = checkpoint_root / resume if resume in {"latest", "late", "best"} else Path(resume)
    if not requested.is_absolute():
        requested = checkpoint_root / requested
    checkpoint_dir = requested.resolve(strict=True)
    if checkpoint_dir.parent != checkpoint_root or not checkpoint_dir.name.startswith("step_"):
        raise ValueError(f"Resume checkpoint must belong to {checkpoint_root}: {checkpoint_dir}")
    return checkpoint_dir


def _validate_resume(
    metadata: Mapping[str, Any], *,
    stream: GlobalSourceStream,
    config_digest: str,
    backend_flags: Mapping[str, Any],
    base_provenance: Mapping[str, Any],
    train_dataset: CSGOSeen10Dataset,
    val_dataset: CSGOSeen10Dataset,
    smoke: bool,
    policy: str = LEGACY_POLICY,
) -> None:
    saved_policy = metadata.get("finetuning_policy", LEGACY_POLICY)
    if saved_policy != policy or (policy == FINAL_POLICY and "finetuning_policy" not in metadata):
        raise ValueError(f"Foreign or incompatible resume checkpoint: finetuning_policy differs ({saved_policy!r} != {policy!r})")
    expected = {
        "profile": PROFILE,
        "train_seed": 42,
        "effective_generation_batch": GLOBAL_BATCH_SIZE,
        "semantic_config_sha256": config_digest,
        "backend_flags": dict(backend_flags),
        "base_provenance": dict(base_provenance),
        "train_data_provenance": train_dataset.provenance,
        "validation_data_provenance": val_dataset.provenance,
        "train_sample_count": len(train_dataset),
        "validation_sample_count": len(val_dataset),
        "smoke_only": smoke,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Foreign or incompatible resume checkpoint: {key} differs")
    step = int(metadata["global_step"])
    if not 0 <= step <= MAX_UPDATES or int(metadata["global_source_count"]) != step * GLOBAL_BATCH_SIZE:
        raise ValueError("Resume checkpoint optimizer step and global source count disagree")
    if int(metadata.get("accumulation_boundary", 0)) != 0:
        raise ValueError("Resume checkpoint was not saved at an accumulation boundary")
    if int(metadata.get("scheduler_updates", step)) != step:
        raise ValueError("Resume checkpoint scheduler update count disagrees with optimizer step")
    stream.load_state_dict(dict(metadata["source_stream"]))
    if stream.consumed != step * GLOBAL_BATCH_SIZE:
        raise ValueError("Resume sampler state does not match checkpoint global step")


def _prepare_log_for_resume(path: Path, step: int) -> None:
    if not path.exists():
        if step:
            raise FileNotFoundError(f"Loss log is missing for resume at optimizer step {step}: {path}")
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept: list[str] = []
    orphaned: list[str] = []
    last = 0
    for line in lines:
        row = json.loads(line)
        row_step = int(row["step"])
        if row_step <= last:
            raise ValueError(f"Loss log has non-increasing optimizer steps at {path}")
        last = row_step
        (kept if row_step <= step else orphaned).append(line)
    if len(kept) != step:
        raise ValueError(f"Loss log has {len(kept)} steps before resume boundary {step}")
    if orphaned:
        backup = path.with_name(f"loss.orphaned_after_{step}_{time.time_ns()}.jsonl")
        backup.write_text("".join(orphaned), encoding="utf-8")
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text("".join(kept), encoding="utf-8")
        os.replace(temporary, path)


def _validation_loader(dataset: CSGOSeen10Dataset, *, rank: int, world_size: int, workers: int, seed: int) -> DataLoader:
    indices = range(rank, len(dataset), world_size)
    generator = torch.Generator(device="cpu").manual_seed(seed + 10_000 + rank)
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=workers > 0,
        generator=generator,
    )


def _validate(
    *, accelerator: Accelerator, model: torch.nn.Module, vae: Any, transport: Any,
    loader: DataLoader, tokenizer: Any, token_ids: Mapping[str, int], config: Any,
    weight_dtype: torch.dtype, seed: int,
) -> float:
    unwrapped = accelerator.unwrap_model(model)
    was_training = unwrapped.training
    unwrapped.eval()
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    total = torch.zeros(2, dtype=torch.float64, device=accelerator.device)
    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True), torch.no_grad():
            for batch in loader:
                sample_ids = batch["sample_id"]
                if len(sample_ids) != 1:
                    raise RuntimeError("Validation must use one sample per batch for sample-ID-fixed noise")
                seed_validation_rng(seed, sample_ids)
                model_inputs = move_model_inputs(
                    batch, tokenizer, token_ids, device=accelerator.device,
                    max_seq_length=int(config.dataset.max_seq_length),
                    max_prompt_tokens=int(config.dataset.max_prompt_tokens),
                    weight_dtype=weight_dtype,
                )
                image_latents, timesteps, image_labels = build_flow_pair(
                    batch, vae=vae, transport=transport,
                    device=accelerator.device, weight_dtype=weight_dtype,
                )
                with accelerator.autocast():
                    _, loss_flow = unwrapped(
                        image_latents=image_latents,
                        t=timesteps.to(dtype=weight_dtype),
                        image_labels=image_labels,
                        max_seq_len=model_inputs["text_tokens"].shape[1],
                        device=accelerator.device,
                        **model_inputs,
                    )
                total[0] += loss_flow.detach().double()
                total[1] += 1
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        unwrapped.train(was_training)
    total = accelerator.reduce(total, reduction="sum")
    if int(total[1].item()) != len(loader.dataset.dataset):
        raise RuntimeError(f"Validation covered {int(total[1].item())} samples, expected {len(loader.dataset.dataset)}")
    return float((total[0] / total[1]).item())


def _save_checkpoint(
    *, accelerator: Accelerator, model: torch.nn.Module, scheduler: LambdaLR,
    output_dir: Path, step: int, stream: GlobalSourceStream, val_loss: float,
    best_val_loss: float, metadata_base: Mapping[str, Any], layout: Mapping[str, int],
    resume_events: list[dict[str, Any]], final: bool,
) -> bool:
    if accelerator.step % accelerator.gradient_accumulation_steps:
        raise RuntimeError("Refusing to checkpoint inside a gradient-accumulation window")
    if scheduler.last_epoch != step or stream.consumed != step * GLOBAL_BATCH_SIZE:
        raise RuntimeError("Scheduler, source cursor, and optimizer step must agree at checkpoint")
    checkpoint_dir = output_dir / "checkpoints" / f"step_{step:06d}"
    stage_name = [f".{checkpoint_dir.name}.saving_{uuid.uuid4().hex}" if accelerator.is_main_process else None]
    broadcast_object_list(stage_name, from_process=0)
    staging_dir = checkpoint_dir.parent / stage_name[0]
    stage_error = [None]
    if accelerator.is_main_process:
        try:
            if checkpoint_dir.exists() or checkpoint_dir.is_symlink():
                raise FileExistsError(f"Refusing to overwrite existing checkpoint {checkpoint_dir}")
            staging_dir.mkdir(parents=False, exist_ok=False)
        except OSError as error:
            stage_error[0] = str(error)
    broadcast_object_list(stage_error, from_process=0)
    if stage_error[0] is not None:
        raise FileExistsError(stage_error[0])
    accelerator.wait_for_everyone()
    accelerator.save_state(str(staging_dir / "accelerator_state"))
    accelerator.wait_for_everyone()
    is_best = val_loss < best_val_loss
    if accelerator.is_main_process:
        save_finetune_weights(accelerator.unwrap_model(model), staging_dir)
        torch.save(scheduler.state_dict(), staging_dir / "scheduler.pt")
        metadata = {
            **metadata_base,
            "global_step": step,
            "global_source_count": stream.consumed,
            "accumulation_boundary": 0,
            "scheduler_updates": scheduler.last_epoch,
            "source_stream": stream.state_dict(),
            "validation_flow_loss": val_loss,
            "best_validation_flow_loss": min(val_loss, best_val_loss),
            "layout": dict(layout),
            "resume_events": list(resume_events),
            "numerically_equivalent_resume": not any(event["layout_changed"] for event in resume_events),
        }
        (staging_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    accelerator.wait_for_everyone()
    finalize_error = [None]
    if accelerator.is_main_process:
        try:
            if checkpoint_dir.exists() or checkpoint_dir.is_symlink():
                raise FileExistsError(f"Refusing to overwrite existing checkpoint {checkpoint_dir}")
            os.rename(staging_dir, checkpoint_dir)
        except OSError as error:
            finalize_error[0] = str(error)
    broadcast_object_list(finalize_error, from_process=0)
    if finalize_error[0] is not None:
        raise RuntimeError(f"Checkpoint staging was preserved at {staging_dir}: {finalize_error[0]}")
    if accelerator.is_main_process:
        atomic_symlink(checkpoint_dir.name, output_dir / "checkpoints" / "latest")
        if final:
            atomic_symlink(checkpoint_dir.name, output_dir / "checkpoints" / "late")
        if is_best:
            atomic_symlink(checkpoint_dir.name, output_dir / "checkpoints" / "best")
    accelerator.wait_for_everyone()
    return is_best


def main(args: Any, config: Any, script_dir: Path) -> None:
    """Execute the aligned profile after the legacy entry point has loaded YAML."""
    validate_profile(config)
    if args.seed is not None and int(args.seed) != 42:
        raise ValueError("Aligned profile fixes training seed at 42")
    if args.max_steps is not None and not args.smoke:
        raise ValueError("--max-steps is available only with --smoke for the aligned profile")
    max_steps = int(args.max_steps) if args.max_steps is not None else (1 if args.smoke else MAX_UPDATES)
    if args.smoke and not 1 <= max_steps <= 2:
        raise ValueError("Aligned smoke runs permit one or two optimizer steps")
    if not args.smoke and max_steps != MAX_UPDATES:
        raise ValueError("Aligned formal training must run exactly 19,500 optimizer steps")
    backend_flags = configure_determinism(config)

    output_dir = _output_path(args, config, script_dir)
    # A first Accelerator discovers the distributed layout before accumulation
    # is derived.  The final one owns the model and checkpoint state.
    layout_probe = Accelerator(mixed_precision=str(config.training.mixed_precision))
    output_error = [None]
    if layout_probe.is_main_process:
        if args.resume is None and output_dir.exists() and any(output_dir.iterdir()):
            output_error[0] = f"Aligned output directory is not empty: {output_dir}; pass --resume"
        elif args.resume is not None and not output_dir.is_dir():
            output_error[0] = f"Resume output directory is missing: {output_dir}"
    broadcast_object_list(output_error, from_process=0)
    if output_error[0] is not None:
        if args.resume is None:
            raise FileExistsError(output_error[0])
        raise FileNotFoundError(output_error[0])
    if layout_probe.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    layout_probe.wait_for_everyone()
    micro_batch, accumulation = resolve_layout(config, args, layout_probe.num_processes)
    del layout_probe
    accelerator = Accelerator(
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=accumulation, sync_with_dataloader=False
        ),
        mixed_precision=str(config.training.mixed_precision),
        project_dir=str(output_dir / "logs"),
    )
    layout = {
        "world_size": accelerator.num_processes,
        "micro_batch": micro_batch,
        "gradient_accumulation": accumulation,
    }
    validate_layout(**{"world_size": layout["world_size"], "micro_batch": layout["micro_batch"], "accumulation": layout["gradient_accumulation"]})
    set_seed(42, device_specific=True)

    data_root = args.data_root or str(config.benchmark.data_root)
    train_dataset = CSGOSeen10Dataset(data_root, "train", include_target=True, **dataset_kwargs(config))
    if len(train_dataset) != TRAIN_SAMPLES:
        raise ValueError(f"Aligned training requires all {TRAIN_SAMPLES} source samples; found {len(train_dataset)}")
    val_dataset = CSGOSeen10Dataset(
        data_root, "validation", include_target=True,
        limit=int(config.training.smoke_validation_samples) if args.smoke else None,
        **dataset_kwargs(config),
    )
    if not args.smoke and len(val_dataset) != VALIDATION_SAMPLES:
        raise ValueError(f"Aligned validation requires all {VALIDATION_SAMPLES} samples; found {len(val_dataset)}")
    stream = GlobalSourceStream(len(train_dataset), 42)

    model, vae, tokenizer, token_ids, device, weight_dtype, vae_path, loading_info = load_runtime(
        config, showo_dir=script_dir
    )
    if loading_info.get("missing_keys"):
        raise RuntimeError(f"Unexpected missing official weights: {loading_info['missing_keys']}")
    accelerator_index = accelerator.device.index
    if accelerator.device.type == "cuda" and accelerator_index is None:
        accelerator_index = torch.cuda.current_device()
    if device.type != accelerator.device.type or (
        device.type == "cuda" and device.index != accelerator_index
    ):
        raise RuntimeError(f"Runtime device {device} differs from Accelerator device {accelerator.device}")
    trainable_parameter_audit(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        parameters,
        lr=float(config.optimizer.learning_rate),
        betas=(float(config.optimizer.beta1), float(config.optimizer.beta2)),
        weight_decay=float(config.optimizer.weight_decay),
        eps=float(config.optimizer.epsilon),
    )
    audit = trainable_parameter_audit(model, optimizer)
    assert_aligned_parameter_counts(audit)
    if accelerator.is_main_process:
        audit_path = output_dir / "parameter_audit.json"
        if args.resume is not None and audit_path.is_file():
            prior_audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if not _resume_audit_matches(prior_audit, audit, policy=policy_from_config(config)):
                raise ValueError("Trainable parameter audit differs from the checkpointed run")
        else:
            audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        group_summary = {name: group["numel"] for name, group in audit["groups"].items() if name != "frozen"}
        print(
            f"aligned audit trainable={audit['trainable_numel']} total={audit['total_numel']} "
            f"groups={group_summary} optimizer_lr={optimizer.param_groups[0]['lr']}",
            flush=True,
        )
    accelerator.wait_for_everyone()
    model, optimizer = accelerator.prepare(model, optimizer)
    model.train()
    scheduler = make_scheduler(optimizer, config)
    transport, _ = create_transport_and_sampler(config)
    config_digest = _semantic_config_digest(config)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path
    config_path = config_path.resolve(strict=True)
    config_provenance = {"path": str(config_path), "sha256": sha256_file(config_path)}
    # Hash the multi-gigabyte base weights once, then share their provenance.
    base_payload = [_base_provenance(config, script_dir, Path(vae_path)) if accelerator.is_main_process else None]
    broadcast_object_list(base_payload, from_process=0)
    base_provenance = base_payload[0]
    metadata_base = {
        "profile": PROFILE,
        "finetuning_policy": policy_from_config(config),
        "train_seed": 42,
        "effective_generation_batch": GLOBAL_BATCH_SIZE,
        "max_optimizer_steps": MAX_UPDATES,
        "semantic_config_sha256": config_digest,
        "backend_flags": backend_flags,
        "config_provenance": config_provenance,
        "base_provenance": base_provenance,
        "train_data_provenance": train_dataset.provenance,
        "validation_data_provenance": val_dataset.provenance,
        "train_sample_count": len(train_dataset),
        "validation_sample_count": len(val_dataset),
        "smoke_only": bool(args.smoke),
    }
    checkpoint_root = output_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    loss_path = output_dir / "loss.jsonl"
    resume_events: list[dict[str, Any]] = []
    best_val_loss = math.inf
    global_step = 0

    if args.resume is not None:
        checkpoint_dir = _checkpoint_path(output_dir, args.resume)
        metadata = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))
        _validate_resume(
            metadata, stream=stream, config_digest=config_digest, backend_flags=backend_flags,
            base_provenance=base_provenance,
            train_dataset=train_dataset, val_dataset=val_dataset, smoke=bool(args.smoke),
            policy=policy_from_config(config),
        )
        global_step = int(metadata["global_step"])
        best_val_loss = float(metadata["best_validation_flow_loss"])
        newer = [
            path for path in checkpoint_root.glob("step_*")
            if path.is_dir() and int(path.name.removeprefix("step_")) > global_step
        ]
        if newer:
            raise FileExistsError(f"Output has checkpoints newer than resume boundary {global_step}: {newer}")
        if global_step >= max_steps:
            raise ValueError(f"Resume step {global_step} is already at or beyond requested step {max_steps}")
        accelerator.load_state(str(checkpoint_dir / "accelerator_state"))
        # Accelerator persists its microstep counter.  At an optimizer boundary
        # its modulo must be recalculated when accumulation changes on resume.
        accelerator.step = global_step * accumulation
        scheduler.load_state_dict(torch.load(checkpoint_dir / "scheduler.pt", map_location="cpu", weights_only=True))
        if scheduler.last_epoch != global_step:
            raise RuntimeError(f"Scheduler state is at {scheduler.last_epoch}, expected {global_step}")
        old_layout = dict(metadata["layout"])
        resume_events = list(metadata.get("resume_events", []))
        resume_events.append({
            "at_step": global_step,
            "from_layout": old_layout,
            "to_layout": layout,
            "layout_changed": old_layout != layout,
            "from_config_provenance": metadata.get("config_provenance"),
            "to_config_provenance": config_provenance,
        })
        if accelerator.is_main_process:
            _prepare_log_for_resume(loss_path, global_step)
            if old_layout != layout:
                print(
                    "Aligned resume changed the microbatch/DDP layout; source order and effective batch "
                    "remain fixed, but floating-point updates may differ numerically.",
                    flush=True,
                )
    else:
        if accelerator.is_main_process:
            OmegaConf.save(config, output_dir / "config.yaml")
    accelerator.wait_for_everyone()

    workers = int(config.dataset.num_workers)
    if workers < 0:
        raise ValueError("dataset.num_workers cannot be negative")
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=PlannedMicroBatchSampler(
            stream, world_size=accelerator.num_processes, rank=accelerator.process_index,
            micro_batch=micro_batch, accumulation=accumulation,
            remaining_updates=max_steps - global_step,
        ),
        num_workers=workers,
        pin_memory=bool(config.dataset.pin_memory) and accelerator.device.type == "cuda",
        persistent_workers=bool(config.dataset.persistent_workers) and workers > 0,
        generator=torch.Generator(device="cpu").manual_seed(42 + 1_000 + accelerator.process_index),
    )
    val_loader = _validation_loader(
        val_dataset, rank=accelerator.process_index, world_size=accelerator.num_processes,
        workers=workers, seed=42,
    )
    milestones = tuple(range(1, max_steps + 1)) if args.smoke else CHECKPOINT_STEPS
    optimizer.zero_grad(set_to_none=True)
    batch_iterator = iter(train_loader)
    for _ in range(global_step, max_steps):
        current_step = global_step + 1
        used_lr = float(optimizer.param_groups[0]["lr"])
        loss_sum = torch.zeros(2, device=accelerator.device, dtype=torch.float64)
        nonfinite_loss = torch.zeros((), device=accelerator.device, dtype=torch.int32)
        started = time.monotonic()
        for micro_index in range(accumulation):
            batch = next(batch_iterator)
            if len(batch["sample_id"]) != micro_batch:
                raise RuntimeError("Training loader produced a partial or padded microbatch")
            with accelerator.accumulate(model):
                model_inputs = move_model_inputs(
                    batch, tokenizer, token_ids, device=accelerator.device,
                    max_seq_length=int(config.dataset.max_seq_length),
                    max_prompt_tokens=int(config.dataset.max_prompt_tokens),
                    weight_dtype=weight_dtype,
                )
                image_latents, timesteps, image_labels = build_flow_pair(
                    batch, vae=vae, transport=transport,
                    device=accelerator.device, weight_dtype=weight_dtype,
                )
                with accelerator.autocast():
                    _, loss_flow = model(
                        image_latents=image_latents,
                        t=timesteps.to(dtype=weight_dtype),
                        image_labels=image_labels,
                        max_seq_len=model_inputs["text_tokens"].shape[1],
                        device=accelerator.device,
                        **model_inputs,
                    )
                nonfinite_loss = torch.maximum(
                    nonfinite_loss, (~torch.isfinite(loss_flow.detach()).all()).to(torch.int32)
                )
                accelerator.backward(loss_flow)
                expected_sync = micro_index == accumulation - 1
                if accelerator.sync_gradients != expected_sync:
                    raise RuntimeError("Accelerator synchronized gradients outside the optimizer boundary")
                if expected_sync:
                    missing_gradients: list[str] = []
                    if current_step == 1 or args.smoke:
                        missing_gradients = [
                            name for name, parameter in accelerator.unwrap_model(model).named_parameters()
                            if parameter.requires_grad and parameter.grad is None
                        ]
                    max_norm = (
                        float(config.training.max_grad_norm)
                        if config.training.max_grad_norm is not None else float("inf")
                    )
                    gradient_norm = accelerator.clip_grad_norm_(parameters, max_norm)
                    nonfinite_gradient = (~torch.isfinite(torch.as_tensor(
                        gradient_norm, device=accelerator.device
                    ))).to(torch.int32)
                    failure_counts = accelerator.reduce(torch.stack((
                        nonfinite_loss,
                        nonfinite_gradient.reshape(()),
                        torch.tensor(int(bool(missing_gradients)), device=accelerator.device, dtype=torch.int32),
                    )), reduction="sum")
                    if bool(failure_counts.any().item()):
                        raise RuntimeError(
                            f"Refusing nonfinite or incomplete optimizer update {current_step}: "
                            f"loss_ranks={int(failure_counts[0])}, gradient_ranks={int(failure_counts[1])}, "
                            f"missing_gradient_ranks={int(failure_counts[2])}; "
                            f"local_missing={missing_gradients}"
                        )
                    optimizer.step()
                    if accelerator.optimizer_step_was_skipped:
                        raise RuntimeError("Optimizer step was skipped; the exact source/update budget cannot continue")
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            loss_sum[0] += loss_flow.detach().double() * micro_batch
            loss_sum[1] += micro_batch
            if args.smoke and accelerator.is_main_process and (
                (micro_index + 1) % 4 == 0 or micro_index + 1 == accumulation
            ):
                elapsed = max(time.monotonic() - started, 1e-6)
                local_samples = (micro_index + 1) * micro_batch
                print(
                    f"smoke_progress step={current_step} micro={micro_index + 1}/{accumulation} "
                    f"local_samples={local_samples} local_samples_per_second={local_samples / elapsed:.3f}",
                    flush=True,
                )
        stream.commit_update()
        global_step = current_step
        loss_sum = accelerator.reduce(loss_sum, reduction="sum")
        if int(loss_sum[1].item()) != GLOBAL_BATCH_SIZE:
            raise RuntimeError(f"Optimizer update consumed {int(loss_sum[1].item())} samples, expected 128")
        loss_value = float((loss_sum[0] / loss_sum[1]).item())
        global_indices = stream.global_indices(start=stream.consumed - GLOBAL_BATCH_SIZE)
        digest = sample_order_digest([train_dataset.rows[index]["sample_id"] for index in global_indices])
        if accelerator.is_main_process:
            with loss_path.open("a", encoding="utf-8") as stream_file:
                stream_file.write(json.dumps({
                    "step": global_step,
                    "global_source_count": stream.consumed,
                    "loss": loss_value,
                    "lr": used_lr,
                    "sample_order_digest": digest,
                }, sort_keys=True) + "\n")
            if global_step % int(config.training.log_every) == 0:
                print(f"step={global_step} source={stream.consumed} flow_loss={loss_value:.6f} lr={used_lr:.8g}", flush=True)

        if global_step in milestones:
            val_loss = _validate(
                accelerator=accelerator, model=model, vae=vae, transport=transport,
                loader=val_loader, tokenizer=tokenizer, token_ids=token_ids, config=config,
                weight_dtype=weight_dtype, seed=42 + 1701,
            )
            if args.smoke:
                repeated = _validate(
                    accelerator=accelerator, model=model, vae=vae, transport=transport,
                    loader=val_loader, tokenizer=tokenizer, token_ids=token_ids, config=config,
                    weight_dtype=weight_dtype, seed=42 + 1701,
                )
                if not math.isclose(val_loss, repeated, rel_tol=1e-6, abs_tol=1e-6):
                    raise RuntimeError(f"Sample-ID-fixed validation changed: {val_loss} versus {repeated}")
            is_best = _save_checkpoint(
                accelerator=accelerator, model=model, scheduler=scheduler,
                output_dir=output_dir, step=global_step, stream=stream,
                val_loss=val_loss, best_val_loss=best_val_loss,
                metadata_base=metadata_base, layout=layout, resume_events=resume_events,
                final=global_step == max_steps,
            )
            if is_best:
                best_val_loss = val_loss
            if accelerator.is_main_process:
                print(f"validation step={global_step} samples={len(val_dataset)} flow_loss={val_loss:.6f}", flush=True)

    accelerator.wait_for_everyone()
    accelerator.end_training()
    if accelerator.is_main_process:
        print(f"aligned training complete: output={output_dir} step={global_step} source={stream.consumed}", flush=True)
