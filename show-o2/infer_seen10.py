"""Generate discrete or continuous Seen-10 outputs from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from omegaconf import OmegaConf

from csgo_seen10.data import CSGOSeen10Dataset, declared_sample_counts
from csgo_seen10.runtime import (
    generate_one,
    create_transport_and_sampler,
    is_valid_output,
    load_finetune_weights,
    load_runtime,
    resolve_project_path,
    save_rgb_jpeg,
    single_sample_batch,
    write_inference_manifest,
)


INFERENCE_RNG_STRATEGY = (
    "per-sample torch.Generator(device).manual_seed(inference_seed + manifest_index); "
    "manifest_index is zero-based dataset row order"
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/showo2_1.5b_csgo_seen10.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Training seed, used in the output directory")
    parser.add_argument("--inference-seed", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None, help="Seed output root; contains discrete/ and continuous/")
    parser.add_argument(
        "--checkpoint", default="best", help="Checkpoint directory or best/late/latest alias"
    )
    parser.add_argument("--task", choices=("all", "discrete", "continuous"), default="all")
    parser.add_argument("--limit", type=int, default=None, help="Optional prefix limit per selected task")
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def _checkpoint_path(value: str, output_root: Path) -> Path:
    if value in {"best", "late", "latest"}:
        checkpoint = output_root / "checkpoints" / value
    else:
        checkpoint = Path(value).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = output_root / checkpoint
    return checkpoint.resolve(strict=True)


def _inference_settings(config: Any) -> Dict[str, Any]:
    return {
        "resolution": int(config.dataset.resolution),
        "max_seq_length": int(config.dataset.max_seq_length),
        "max_prompt_tokens": int(config.dataset.max_prompt_tokens),
        "num_inference_steps": int(config.transport.num_inference_steps),
        "sampling_method": str(config.transport.sampling_method),
        "atol": float(config.transport.atol),
        "rtol": float(config.transport.rtol),
        "time_shifting_factor": float(config.transport.time_shifting_factor),
        "guidance_scale": float(config.transport.guidance_scale),
        "rng_strategy": INFERENCE_RNG_STRATEGY,
    }


def _sample_seed(inference_seed: int, manifest_index: int) -> int:
    """Return the deterministic seed assigned to a split row."""
    if manifest_index < 0:
        raise ValueError("manifest_index must be non-negative")
    return int(inference_seed) + int(manifest_index)


def _run_task(
    *,
    task: str,
    config: Any,
    model: torch.nn.Module,
    vae: Any,
    sampler: Any,
    tokenizer: Any,
    token_ids: Dict[str, int],
    device: torch.device,
    weight_dtype: torch.dtype,
    data_root: str,
    output_root: Path,
    checkpoint_dir: Path,
    train_seed: int,
    inference_seed: int,
    limit: Optional[int],
    smoke_only: bool,
    config_path: Path,
    showo_path: Path,
    vae_path: Path,
    project_root: Path,
) -> None:
    split = "discrete_test" if task == "discrete" else "continuous"
    dataset = CSGOSeen10Dataset(
        data_root,
        split,
        include_target=False,
        resolution=int(config.dataset.resolution),
        limit=limit,
    )
    task_root = output_root / task
    prediction_root = task_root / "gen_imgs"
    write_inference_manifest(
        task_root,
        project_root=project_root,
        config_path=config_path,
        inference_settings=_inference_settings(config),
        showo_path=showo_path,
        vae_path=vae_path,
        checkpoint_dir=checkpoint_dir,
        split=split,
        dataset=dataset,
        train_seed=train_seed,
        inference_seed=inference_seed,
        smoke_only=smoke_only,
        generation_complete=False,
    )
    model.eval()

    generated = 0
    preserved = 0
    for index, row in enumerate(dataset.rows):
        map_name = row["map_name"]
        file_frame = row["file_frame"]
        destination = prediction_root / map_name / f"{file_frame}.jpg"
        if is_valid_output(destination):
            preserved += 1
        else:
            batch = single_sample_batch(dataset[index])
            image = generate_one(
                model,
                vae,
                sampler,
                batch,
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
                generator=torch.Generator(device=str(device)).manual_seed(
                    _sample_seed(inference_seed, index)
                ),
            )
            if save_rgb_jpeg(image, destination, skip_valid=True):
                generated += 1
            else:
                preserved += 1
        if (index + 1) % 50 == 0 or index + 1 == len(dataset):
            print(
                f"{task}: {index + 1}/{len(dataset)} processed "
                f"(generated={generated}, preserved_valid={preserved})",
                flush=True,
            )

    manifest_path = write_inference_manifest(
        task_root,
        project_root=project_root,
        config_path=config_path,
        inference_settings=_inference_settings(config),
        showo_path=showo_path,
        vae_path=vae_path,
        checkpoint_dir=checkpoint_dir,
        split=split,
        dataset=dataset,
        train_seed=train_seed,
        inference_seed=inference_seed,
        smoke_only=smoke_only,
    )
    print(
        f"{task}: output={prediction_root} generated={generated} "
        f"preserved_valid={preserved} inference_manifest={manifest_path}",
        flush=True,
    )


def main() -> None:
    args = _args()
    showo_dir = Path(__file__).resolve().parent
    project_root = showo_dir.parent
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = showo_dir / config_path
    config = OmegaConf.load(config_path.resolve())

    train_seed = int(config.training.seed if args.seed is None else args.seed)
    inference_seed = int(config.inference.seed if args.inference_seed is None else args.inference_seed)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if float(config.transport.guidance_scale) != 0.0:
        raise ValueError("Seen-10 inference uses a single conditioned pass; guidance_scale must be 0")

    output_root = (
        Path(args.output_root).expanduser()
        if args.output_root
        else project_root / config.output_root / f"seed_{train_seed}"
    )
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root = output_root.resolve()
    checkpoint_dir = _checkpoint_path(args.checkpoint, output_root)
    data_root = str(args.data_root or config.benchmark.data_root)

    model, vae, tokenizer, token_ids, device, weight_dtype, vae_path, loading_info = load_runtime(
        config, showo_dir=showo_dir
    )
    if loading_info.get("missing_keys"):
        raise RuntimeError(f"Unexpected missing official weights: {loading_info['missing_keys']}")
    load_finetune_weights(model, checkpoint_dir)
    transport, sampler = create_transport_and_sampler(config)

    showo_path = resolve_project_path(config.model.showo.pretrained_model_path, showo_dir)
    vae_path = resolve_project_path(config.model.vae_model.pretrained_model_path, showo_dir)
    limit = args.limit
    if limit is None and config.inference.max_samples is not None:
        limit = int(config.inference.max_samples)
    tasks: Iterable[str] = ("discrete", "continuous") if args.task == "all" else (args.task,)
    counts = declared_sample_counts(data_root)
    for task in tasks:
        split = "discrete_test" if task == "discrete" else "continuous"
        declared = sum(
            counts[map_name]["discrete_test" if task == "discrete" else "continuous_frames"]
            for map_name in counts
        )
        task_limit = limit
        task_is_smoke = bool(args.smoke_only or (task_limit is not None and task_limit < declared))
        _run_task(
            task=task,
            config=config,
            model=model,
            vae=vae,
            sampler=sampler,
            tokenizer=tokenizer,
            token_ids=token_ids,
            device=device,
            weight_dtype=weight_dtype,
            data_root=data_root,
            output_root=output_root,
            checkpoint_dir=checkpoint_dir,
            train_seed=train_seed,
            inference_seed=inference_seed,
            limit=task_limit,
            smoke_only=task_is_smoke,
            config_path=config_path.resolve(),
            showo_path=showo_path,
            vae_path=vae_path,
            project_root=project_root,
        )


if __name__ == "__main__":
    main()
