"""Generate discrete or continuous Seen-10 outputs from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from omegaconf import OmegaConf

from csgo_seen10.data import CSGOSeen10Dataset, declared_sample_counts
from csgo_seen10.runtime import (
    Seen10InferenceConditionCache,
    generate_batch,
    create_transport_and_sampler,
    is_valid_output,
    load_finetune_weights,
    load_runtime,
    resolve_project_path,
    save_rgb_jpeg,
    write_inference_manifest,
    dataset_kwargs,
    is_aligned,
    sample_identity_seed,
    sha256_paths,
)


INFERENCE_RNG_STRATEGY = (
    "per-sample torch.Generator(device).manual_seed(inference_seed + manifest_index); "
    "manifest_index is zero-based dataset row order"
)
INFERENCE_ALGORITHM = "showo2_seen10_native_batch_v1"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/showo2_1.5b_csgo_seen10.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Training seed, used in the output directory")
    parser.add_argument("--inference-seed", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None, help="Seed output root; contains discrete/ and continuous/")
    parser.add_argument("--prediction-output-root", default=None, help="Aligned-only isolated prediction root")
    parser.add_argument(
        "--checkpoint", default="best", help="Checkpoint directory or best/late/latest alias"
    )
    parser.add_argument("--task", choices=("all", "discrete", "continuous"), default="all")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size (default: inference.batch_size from config)",
    )
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


def _configured_batch_size(config: Any) -> int:
    inference = config.get("inference", {})
    return int(inference.get("batch_size", 16))


def _inference_settings(config: Any, batch_size: Optional[int] = None) -> Dict[str, Any]:
    if batch_size is None:
        batch_size = _configured_batch_size(config)
    settings = {
        "resolution": int(config.dataset.resolution),
        "max_seq_length": int(config.dataset.max_seq_length),
        "max_prompt_tokens": int(config.dataset.max_prompt_tokens),
        "num_inference_steps": int(config.transport.num_inference_steps),
        "sampling_method": str(config.transport.sampling_method),
        "atol": float(config.transport.atol),
        "rtol": float(config.transport.rtol),
        "time_shifting_factor": float(config.transport.time_shifting_factor),
        "guidance_scale": float(config.transport.guidance_scale),
        "batch_size": int(batch_size),
        "generation_algorithm": INFERENCE_ALGORITHM,
        "ode_implementation": (
            "explicit_euler_final_v1"
            if str(config.transport.sampling_method).lower() == "euler"
            else "torchdiffeq"
        ),
        "condition_cache": "one preprocessed radar latent and sequence template per map; pose per sample",
        "rng_strategy": INFERENCE_RNG_STRATEGY,
    }
    if is_aligned(config):
        settings.pop("batch_size")  # Execution batching does not define a different experiment.
        settings.update(generation_algorithm="showo2_aligned_native_v1", condition_cache="radar latent per map; text pose per sample",
                        rng_strategy="sha256(showo2-aligned-v1, seed, sample_id) low63; per-sample generator",
                        nfe=int(config.transport.num_inference_steps) - 1,
                        precision=str(config.training.mixed_precision), output_resolution=448,
                        vae="Wan2.1", resize_mode=str(config.dataset.resize_mode))
    return settings


def bind_prediction_checkpoint(output_root: Path, checkpoint_dir: Path) -> None:
    """Bind both task directories to one immutable checkpoint before any generation."""
    files = [checkpoint_dir / "backbone_trainable.safetensors", checkpoint_dir / "finetuning.json"]
    record = {"path": str(checkpoint_dir), "sha256": sha256_paths(files)}
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "checkpoint_binding.json"
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError("Prediction root is bound to a different checkpoint; use a fresh root")
    else:
        if any(output_root.glob("*/gen_imgs/*/*.jpg")):
            raise ValueError("Unattributed prediction files already exist; use a fresh root")
        with path.open("x") as stream:
            json.dump(record, stream, indent=2)


def _sample_seed(inference_seed: int, manifest_index: int) -> int:
    """Return the deterministic seed assigned to a split row."""
    if manifest_index < 0:
        raise ValueError("manifest_index must be non-negative")
    return int(inference_seed) + int(manifest_index)


def iter_manifest_batches(dataset: CSGOSeen10Dataset, batch_size: int) -> Iterable[list[int]]:
    """Yield stable batch blocks without crossing map boundaries."""
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows = dataset.rows
    map_start = 0
    while map_start < len(rows):
        map_name = rows[map_start]["map_name"]
        map_end = map_start + 1
        while map_end < len(rows) and rows[map_end]["map_name"] == map_name:
            map_end += 1
        for start in range(map_start, map_end, batch_size):
            yield list(range(start, min(start + batch_size, map_end)))
        map_start = map_end


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
    batch_size: int,
    condition_cache: Seen10InferenceConditionCache,
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
        **dataset_kwargs(config),
        limit=limit,
    )
    task_root = output_root / task
    prediction_root = task_root / "gen_imgs"
    write_inference_manifest(
        task_root,
        project_root=project_root,
        config_path=config_path,
        inference_settings=_inference_settings(config, batch_size),
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
    if is_aligned(config):
        identities = [{key: row.get(key) for key in ("sample_id", "map_name", "file_frame", "clip_id", "frame_index")} for row in dataset.rows]
        identity_path = task_root / "sample_manifest.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identities:
            raise ValueError("Existing sample manifest does not match requested split")
        temporary = identity_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(identities, indent=2) + "\n")
        os.replace(temporary, identity_path)
        print(f"Aligned inference: batch_size={batch_size}, target=False, rows={len(dataset)}, seed={inference_seed}", flush=True)

    generated = 0
    preserved = 0
    processed = 0
    for indices in iter_manifest_batches(dataset, batch_size):
        rows = [dataset.rows[index] for index in indices]
        destinations = [
            prediction_root / row["map_name"] / f"{row['file_frame']}.jpg"
            for row in rows
        ]
        missing = [not is_valid_output(destination) for destination in destinations]
        preserved += sum(not needs_generation for needs_generation in missing)
        if any(missing):
            radar_latents, model_inputs = condition_cache.prepare_batch(dataset, indices)
            generators = [
                torch.Generator(device=str(device)).manual_seed(
                    sample_identity_seed(inference_seed, str(dataset.rows[index]["sample_id"]))
                    if is_aligned(config) else _sample_seed(inference_seed, index))
                for index in indices
            ]
            images = generate_batch(
                model,
                vae,
                sampler,
                {"map_name": [row["map_name"] for row in rows]},
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
                generators=generators,
                radar_latents=radar_latents,
                model_inputs=model_inputs,
                **({"vae_batch_size": int(config.inference.get("vae_batch_size", 1))} if is_aligned(config) else {}),
            )
            for image, destination, needs_generation in zip(images, destinations, missing):
                if needs_generation:
                    if save_rgb_jpeg(image, destination, skip_valid=True):
                        generated += 1
                    else:
                        preserved += 1
        processed += len(indices)
        previous_processed = processed - len(indices)
        if processed // 50 > previous_processed // 50 or processed == len(dataset):
            print(
                f"{task}: {processed}/{len(dataset)} processed "
                f"(generated={generated}, preserved_valid={preserved})",
                flush=True,
            )

    manifest_path = write_inference_manifest(
        task_root,
        project_root=project_root,
        config_path=config_path,
        inference_settings=_inference_settings(config, batch_size),
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
    batch_size = _configured_batch_size(config) if args.batch_size is None else int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
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
    if is_aligned(config):
        label = args.checkpoint if args.checkpoint in {"best", "late", "latest"} else checkpoint_dir.name
        output_root = Path(args.prediction_output_root).resolve() if args.prediction_output_root else output_root / "predictions" / label
        bind_prediction_checkpoint(output_root, checkpoint_dir)
    elif args.prediction_output_root:
        raise ValueError("--prediction-output-root is only supported by the aligned profile")
    data_root = str(args.data_root or config.benchmark.data_root)

    model, vae, tokenizer, token_ids, device, weight_dtype, vae_path, loading_info = load_runtime(
        config, showo_dir=showo_dir
    )
    if loading_info.get("missing_keys"):
        raise RuntimeError(f"Unexpected missing official weights: {loading_info['missing_keys']}")
    load_finetune_weights(model, checkpoint_dir)
    transport, sampler = create_transport_and_sampler(config)
    condition_cache = Seen10InferenceConditionCache(
        vae=vae,
        tokenizer=tokenizer,
        token_ids=token_ids,
        device=device,
        weight_dtype=weight_dtype,
        max_seq_length=int(config.dataset.max_seq_length),
        max_prompt_tokens=int(config.dataset.max_prompt_tokens),
    )

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
            batch_size=batch_size,
            condition_cache=condition_cache,
            limit=task_limit,
            smoke_only=task_is_smoke,
            config_path=config_path.resolve(),
            showo_path=showo_path,
            vae_path=vae_path,
            project_root=project_root,
        )


if __name__ == "__main__":
    main()
