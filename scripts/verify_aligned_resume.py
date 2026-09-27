"""Read-only, bitwise comparison of two trusted local aligned smoke checkpoints.

Run with show-o2/.venv/bin/python. RNG files use pickle: only compare checkpoints
created by your own trusted training runs, never downloaded untrusted files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def equal(left, right) -> bool:
    if torch.is_tensor(left):
        return torch.is_tensor(right) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return isinstance(right, (tuple, list)) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def compare(left: Path, right: Path) -> dict:
    torch.set_num_threads(4)
    left, right = left.resolve(strict=True), right.resolve(strict=True)
    a = load_file(str(left / "backbone_trainable.safetensors"))
    b = load_file(str(right / "backbone_trainable.safetensors"))
    changed = sorted(name for name in a.keys() | b.keys() if name not in a or name not in b or not equal(a[name], b[name]))
    report = {"left": str(left), "right": str(right), "trainable_tensor_count": len(a),
              "trainable_parameters": sum(value.numel() for value in a.values()),
              "trainable_bitwise_equal": not changed, "changed_trainable_tensor_count": len(changed),
              "changed_trainable_tensors_preview": changed[:20]}
    del a, b
    for filename in ("model.safetensors", "optimizer.bin"):
        lp, rp = left / "accelerator_state" / filename, right / "accelerator_state" / filename
        if filename.endswith("safetensors"):
            report["full_model_file_sha256_equal"] = digest(lp) == digest(rp)
        else:
            a = torch.load(lp, map_location="cpu", weights_only=False)
            b = torch.load(rp, map_location="cpu", weights_only=False)
            report["optimizer_bitwise_equal"] = bool(equal(a, b))
            del a, b
    report["scheduler_equal"] = bool(equal(
        torch.load(left / "scheduler.pt", map_location="cpu", weights_only=True),
        torch.load(right / "scheduler.pt", map_location="cpu", weights_only=True)))
    rng_names = sorted(path.name for path in (left / "accelerator_state").glob("random_states_*.pkl"))
    if not rng_names or rng_names != sorted(path.name for path in (right / "accelerator_state").glob("random_states_*.pkl")):
        report["all_rank_rng_equal"] = False
    else:
        report["all_rank_rng_equal"] = all(equal(
            torch.load(left / "accelerator_state" / name, map_location="cpu", weights_only=False),
            torch.load(right / "accelerator_state" / name, map_location="cpu", weights_only=False)) for name in rng_names)
    a, b = json.loads((left / "metadata.json").read_text()), json.loads((right / "metadata.json").read_text())
    fields = ("profile", "finetuning_policy", "train_seed", "global_step", "global_source_count", "source_stream", "layout",
              "accumulation_boundary", "scheduler_updates", "validation_flow_loss", "semantic_config_sha256",
              "train_data_provenance", "validation_data_provenance", "backend_flags")
    report["training_state_equal"] = all(equal(a.get(key), b.get(key)) for key in fields)
    def losses(checkpoint):
        return [json.loads(line) for line in (checkpoint.parents[1] / "loss.jsonl").read_text().splitlines()
                if line and json.loads(line)["step"] <= a["global_step"]]
    left_losses, right_losses = losses(left), losses(right)
    report["loss_and_source_order_equal"] = left_losses == right_losses
    report["source_order_equal"] = [row["sample_order_digest"] for row in left_losses] == [row["sample_order_digest"] for row in right_losses]
    report["passed"] = all(report[key] for key in ("trainable_bitwise_equal", "full_model_file_sha256_equal",
        "optimizer_bitwise_equal", "scheduler_equal", "all_rank_rng_equal", "training_state_equal", "loss_and_source_order_equal"))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    result = compare(args.left, args.right)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
