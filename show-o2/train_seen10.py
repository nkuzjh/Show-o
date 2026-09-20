"""Train Show-o2-1.5B on the CSGO Benchmark v2 Seen-10 generation split."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader

from csgo_seen10.data import CSGOSeen10Dataset
from csgo_seen10.runtime import (
    atomic_symlink,
    build_flow_pair,
    create_transport_and_sampler,
    generate_one,
    load_runtime,
    move_model_inputs,
    save_loss_curve,
    save_rgb_jpeg,
    single_sample_batch,
    save_finetune_weights,
    seed_validation_rng,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/showo2_1.5b_csgo_seen10.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", nargs="?", const="latest", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _fixed_torch_rng(seed: int):
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    return torch.random.fork_rng(devices=devices, enabled=True), seed


def _run_validation(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    vae: Any,
    val_loader: DataLoader,
    val_dataset: CSGOSeen10Dataset,
    tokenizer: Any,
    token_ids: Dict[str, int],
    transport: Any,
    sampler: Any,
    config: Any,
    output_dir: Path,
    step: int,
    seed: int,
    weight_dtype: torch.dtype,
    create_preview: bool = True,
) -> float:
    previous_training_mode = model.training
    model.eval()
    rng_context, fixed_seed = _fixed_torch_rng(seed)
    with rng_context:
        torch.manual_seed(fixed_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fixed_seed)
        total = torch.zeros(2, device=accelerator.device, dtype=torch.float64)
        with torch.no_grad():
            for batch in val_loader:
                sample_ids = batch.get("sample_id")
                if sample_ids is None:
                    raise KeyError("Validation batches must include manifest sample_id values")
                seed_validation_rng(seed, sample_ids)
                model_inputs = move_model_inputs(
                    batch,
                    tokenizer,
                    token_ids,
                    device=accelerator.device,
                    max_seq_length=int(config.dataset.max_seq_length),
                    weight_dtype=weight_dtype,
                    max_prompt_tokens=int(config.dataset.max_prompt_tokens),
                )
                image_latents, timesteps, image_labels = build_flow_pair(
                    batch,
                    vae=vae,
                    transport=transport,
                    device=accelerator.device,
                    weight_dtype=weight_dtype,
                )
                with accelerator.autocast():
                    model_inputs["image_masks"] = model_inputs["image_masks"].clone()
                    _, loss_flow = model(
                        image_latents=image_latents,
                        t=timesteps.to(dtype=weight_dtype),
                        image_labels=image_labels,
                        max_seq_len=model_inputs["text_tokens"].shape[1],
                        device=accelerator.device,
                        **model_inputs,
                    )
                total[0] += loss_flow.detach().double()
                total[1] += 1
        total = accelerator.reduce(total, reduction="sum")
        val_loss = float((total[0] / total[1].clamp_min(1)).item())

        accelerator.wait_for_everyone()
        if create_preview and accelerator.is_main_process:
            # Preview generation is condition-only: do not even open a held-out
            # target file while preparing its input.
            preview = val_dataset.get_condition_only(0)
            preview_batch = single_sample_batch(preview)
            device = accelerator.device
            generator = torch.Generator(device=str(device)).manual_seed(seed + 31)
            image = generate_one(
                accelerator.unwrap_model(model),
                vae,
                sampler,
                preview_batch,
                tokenizer,
                token_ids,
                device=device,
                weight_dtype=weight_dtype,
                max_seq_length=int(config.dataset.max_seq_length),
                max_prompt_tokens=int(config.dataset.max_prompt_tokens),
                num_inference_steps=int(config.transport.num_inference_steps),
                sampling_method=str(config.transport.sampling_method),
                atol=float(config.transport.atol),
                rtol=float(config.transport.rtol),
                time_shifting_factor=float(config.transport.time_shifting_factor),
                guidance_scale=float(config.transport.guidance_scale),
                generator=generator,
            )
            preview_path = (
                output_dir
                / "validation_samples"
                / f"step_{step:06d}"
                / preview["map_name"]
                / f"{preview['file_frame']}.jpg"
            )
            save_rgb_jpeg(image, preview_path, skip_valid=True)
        accelerator.wait_for_everyone()
    if previous_training_mode:
        model.train()
    return val_loss


def _save_checkpoint(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: AdamW,
    lr_scheduler: Any,
    output_dir: Path,
    step: int,
    val_loss: float,
    best_val_loss: float,
    seed: int,
) -> bool:
    del optimizer, lr_scheduler  # Saved by Accelerator as part of the resumable state.
    checkpoint_dir = output_dir / "checkpoints" / f"step_{step:06d}"
    state_dir = checkpoint_dir / "accelerator_state"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if checkpoint_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing checkpoint {checkpoint_dir}")
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(state_dir))
    accelerator.wait_for_everyone()
    is_best = val_loss < best_val_loss
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_finetune_weights(unwrapped, checkpoint_dir)
        metadata = {
            "global_step": int(step),
            "train_seed": int(seed),
            "validation_flow_loss": float(val_loss),
            "best_validation_flow_loss": float(min(val_loss, best_val_loss)),
        }
        (checkpoint_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # The benchmark contract names the final-checkpoint alias ``late``.
        # Keep ``latest`` as a compatibility alias for the conventional CLI.
        atomic_symlink(checkpoint_dir.name, output_dir / "checkpoints" / "late")
        atomic_symlink(checkpoint_dir.name, output_dir / "checkpoints" / "latest")
        if is_best:
            atomic_symlink(checkpoint_dir.name, output_dir / "checkpoints" / "best")
    accelerator.wait_for_everyone()
    return is_best


def _load_resume(
    accelerator: Accelerator,
    output_dir: Path,
    resume_arg: Optional[str],
) -> tuple[int, float]:
    if resume_arg is None:
        checkpoints = output_dir / "checkpoints"
        if checkpoints.exists() and any(checkpoints.glob("step_*")):
            raise FileExistsError(
                f"Training output already has checkpoints at {checkpoints}; pass --resume to continue"
            )
        return 0, math.inf
    checkpoint_dir = (
        output_dir / "checkpoints" / resume_arg
        if resume_arg in {"late", "latest", "best"}
        else Path(resume_arg)
    )
    checkpoint_dir = checkpoint_dir.resolve(strict=True)
    metadata_path = checkpoint_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    state_dir = checkpoint_dir / "accelerator_state"
    if not state_dir.is_dir():
        raise FileNotFoundError(f"Accelerator resume state missing: {state_dir}")
    accelerator.load_state(str(state_dir))
    return int(metadata["global_step"]), float(metadata["best_validation_flow_loss"])


def main() -> None:
    args = _args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path
    config = OmegaConf.load(config_path)
    if args.seed is not None:
        config.training.seed = args.seed
    seed = int(config.training.seed)
    max_steps = int(config.training.max_train_steps if args.max_steps is None else args.max_steps)
    if args.smoke:
        max_steps = min(max_steps, 1)
    if max_steps <= 0:
        raise ValueError("max train steps must be positive")
    data_root = args.data_root or str(config.benchmark.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else project_root / config.output_root / f"seed_{seed}"
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir = output_dir.resolve()

    accelerator = Accelerator(
        gradient_accumulation_steps=int(config.training.gradient_accumulation_steps),
        mixed_precision=str(config.training.mixed_precision),
        project_dir=str(output_dir / "logs"),
    )
    set_seed(seed, device_specific=True)
    if bool(config.training.enable_tf32) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    model, vae, tokenizer, token_ids, device, weight_dtype, vae_path, loading_info = load_runtime(
        config, showo_dir=script_dir
    )
    if loading_info.get("missing_keys"):
        raise RuntimeError(f"Unexpected missing official weights: {loading_info['missing_keys']}")
    train_dataset = CSGOSeen10Dataset(
        data_root,
        "train",
        include_target=True,
        resolution=int(config.dataset.resolution),
    )
    if args.smoke:
        val_dataset = CSGOSeen10Dataset(
            data_root,
            "validation",
            include_target=True,
            resolution=int(config.dataset.resolution),
            limit=int(config.training.smoke_validation_samples),
        )
    else:
        val_dataset = CSGOSeen10Dataset(
            data_root,
            "validation",
            include_target=True,
            resolution=int(config.dataset.resolution),
            limit_per_map=int(config.dataset.validation_samples_per_map),
        )
    num_workers = int(config.dataset.num_workers)
    loader_options: Dict[str, Any] = {
        "batch_size": int(config.training.batch_size),
        "num_workers": num_workers,
        "pin_memory": bool(config.dataset.pin_memory) and device.type == "cuda",
        "persistent_workers": bool(config.dataset.persistent_workers) and num_workers > 0,
        "drop_last": True,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **{**loader_options, "drop_last": False, "batch_size": 1},
    )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("The Show-o2 downstream freeze policy left no trainable parameters")
    optimizer = AdamW(
        trainable_parameters,
        lr=float(config.optimizer.learning_rate),
        betas=(float(config.optimizer.beta1), float(config.optimizer.beta2)),
        weight_decay=float(config.optimizer.weight_decay),
        eps=float(config.optimizer.epsilon),
    )
    from models.lr_schedulers import get_scheduler

    lr_scheduler = get_scheduler(
        str(config.lr_scheduler.name),
        optimizer=optimizer,
        num_warmup_steps=int(config.lr_scheduler.warmup_steps),
        num_training_steps=max_steps,
    )
    model, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_loader, val_loader
    )
    transport, sampler = create_transport_and_sampler(config)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        OmegaConf.save(config, output_dir / "config.yaml")
    accelerator.wait_for_everyone()
    global_step, best_val_loss = _load_resume(accelerator, output_dir, args.resume)
    if global_step > max_steps:
        raise ValueError(f"Resume checkpoint step {global_step} exceeds max_train_steps={max_steps}")

    milestones = sorted(
        {
            max(1, math.ceil(max_steps * fraction / int(config.training.validation_milestones)))
            for fraction in range(1, int(config.training.validation_milestones) + 1)
        }
    )
    if args.smoke:
        milestones = [max_steps]
    log_path = output_dir / "loss.jsonl"
    if accelerator.is_main_process:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    epoch = global_step // max(1, len(train_loader))
    skip_batches = global_step % max(1, len(train_loader))
    optimizer.zero_grad(set_to_none=True)
    adapter_gradient_checked = False
    for epoch in itertools.count(start=epoch):
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)
        epoch_loader = accelerator.skip_first_batches(train_loader, skip_batches) if skip_batches else train_loader
        skip_batches = 0
        for batch in epoch_loader:
            if global_step >= max_steps:
                break
            with accelerator.accumulate(model):
                model_inputs = move_model_inputs(
                    batch,
                    tokenizer,
                    token_ids,
                    device=accelerator.device,
                    max_seq_length=int(config.dataset.max_seq_length),
                    weight_dtype=weight_dtype,
                    max_prompt_tokens=int(config.dataset.max_prompt_tokens),
                )
                image_latents, timesteps, image_labels = build_flow_pair(
                    batch,
                    vae=vae,
                    transport=transport,
                    device=accelerator.device,
                    weight_dtype=weight_dtype,
                )
                with accelerator.autocast():
                    model_inputs["image_masks"] = model_inputs["image_masks"].clone()
                    _, loss_flow = model(
                        image_latents=image_latents,
                        t=timesteps.to(dtype=weight_dtype),
                        image_labels=image_labels,
                        max_seq_len=model_inputs["text_tokens"].shape[1],
                        device=accelerator.device,
                        **model_inputs,
                    )
                accelerator.backward(loss_flow)
                if args.smoke and not adapter_gradient_checked:
                    unwrapped = accelerator.unwrap_model(model)
                    adapter_grads = [
                        parameter.grad.detach()
                        for parameter in unwrapped.radar_adapter.parameters()
                        if parameter.grad is not None
                    ]
                    if not adapter_grads:
                        raise RuntimeError("Smoke step produced no radar adapter gradient")
                    if not all(torch.isfinite(grad).all() for grad in adapter_grads):
                        raise RuntimeError("Smoke step produced a non-finite radar adapter gradient")
                    if not any(torch.count_nonzero(grad).item() for grad in adapter_grads):
                        raise RuntimeError("Smoke step produced only zero radar adapter gradients")
                    adapter_gradient_checked = True
                if accelerator.sync_gradients and config.training.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(trainable_parameters, float(config.training.max_grad_norm))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue

            global_step += 1
            gathered_loss = accelerator.gather(loss_flow.detach().float().reshape(1)).mean().item()
            if accelerator.is_main_process:
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "step": global_step,
                                "loss": gathered_loss,
                                "lr": float(optimizer.param_groups[0]["lr"]),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                if global_step % int(config.training.log_every) == 0 or global_step == 1:
                    print(f"step={global_step} flow_loss={gathered_loss:.6f}", flush=True)

            if global_step in milestones:
                val_loss = _run_validation(
                    accelerator=accelerator,
                    model=model,
                    vae=vae,
                    val_loader=val_loader,
                    val_dataset=val_dataset,
                    tokenizer=tokenizer,
                    token_ids=token_ids,
                    transport=transport,
                    sampler=sampler,
                    config=config,
                    output_dir=output_dir,
                    step=global_step,
                    seed=seed + 1701,
                    weight_dtype=weight_dtype,
                )
                if args.smoke:
                    repeated_val_loss = _run_validation(
                        accelerator=accelerator,
                        model=model,
                        vae=vae,
                        val_loader=val_loader,
                        val_dataset=val_dataset,
                        tokenizer=tokenizer,
                        token_ids=token_ids,
                        transport=transport,
                        sampler=sampler,
                        config=config,
                        output_dir=output_dir,
                        step=global_step,
                        seed=seed + 1701,
                        weight_dtype=weight_dtype,
                        create_preview=False,
                    )
                    if not math.isclose(val_loss, repeated_val_loss, rel_tol=1e-6, abs_tol=1e-6):
                        raise RuntimeError(
                            f"Fixed-seed validation changed across runs: {val_loss} vs {repeated_val_loss}"
                        )
                is_best = _save_checkpoint(
                    accelerator=accelerator,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    output_dir=output_dir,
                    step=global_step,
                    val_loss=val_loss,
                    best_val_loss=best_val_loss,
                    seed=seed,
                )
                if is_best:
                    best_val_loss = val_loss
                if accelerator.is_main_process:
                    print(
                        f"validation step={global_step} flow_loss={val_loss:.6f} best={best_val_loss:.6f}",
                        flush=True,
                    )
            if global_step >= max_steps:
                break
        if global_step >= max_steps:
            break

    accelerator.wait_for_everyone()
    if global_step not in milestones:
        raise RuntimeError(f"Training stopped at step {global_step} without a validation/checkpoint milestone")
    if accelerator.is_main_process:
        save_loss_curve(log_path, output_dir / "loss_curve.png")
    accelerator.wait_for_everyone()
    accelerator.end_training()
    print(f"training complete: output={output_dir} step={global_step}", flush=True)


if __name__ == "__main__":
    main()
