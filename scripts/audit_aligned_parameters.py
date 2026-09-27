"""CPU audit of actual local Show-o2 weights; optional tiny native backward smoke.

This is not a 432px training run, benchmark inference, or a quality evaluation.
The output directory must be new. No VAE, GPU, dataset target, or external asset
download is needed. Run using show-o2/.venv/bin/python.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "show-o2"))

from csgo_seen10.finetuning import ALIGNED_LR, assert_aligned_parameter_counts, trainable_parameter_audit
from csgo_seen10.model import load_showo2_seen10
from csgo_seen10.aligned_training import make_scheduler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "show-o2/configs/csgo_seen10_exp32gen_aligned.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backward-smoke", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    config = OmegaConf.load(args.config)
    model, _, token_ids, _ = load_showo2_seen10(
        config, device=torch.device("cpu"), weight_dtype=torch.bfloat16,
        showo_dir=str(ROOT / "show-o2"),
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=ALIGNED_LR,
        betas=(float(config.optimizer.beta1), float(config.optimizer.beta2)),
        weight_decay=float(config.optimizer.weight_decay), eps=float(config.optimizer.epsilon),
    )
    audit = trainable_parameter_audit(model, optimizer)
    assert_aligned_parameter_counts(audit)
    (args.output_dir / "parameter_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"trainable": audit["trainable_numel"], "total": audit["total_numel"],
                      "groups": {k: v["numel"] for k, v in audit["groups"].items()}}, indent=2), flush=True)

    if not args.backward_smoke:
        return
    scheduler = make_scheduler(optimizer, config)
    model.train()
    # Two image slots, each with one spatial patch and one time token. This
    # deliberately tests the real graph at reduced spatial/sequence sizes.
    tokens = torch.full((1, 12), token_ids["pad_id"], dtype=torch.long)
    tokens[0, 0] = token_ids["bos_id"]
    tokens[0, 1] = token_ids["boi_id"]
    tokens[0, 2:4] = token_ids["img_pad_id"]
    tokens[0, 4] = token_ids["eoi_id"]
    tokens[0, 5] = token_ids["boi_id"]
    tokens[0, 6:8] = token_ids["img_pad_id"]
    tokens[0, 8] = token_ids["eoi_id"]
    tokens[0, 9] = token_ids["eos_id"]
    masks = torch.zeros(1, 12, dtype=torch.long)
    masks[:, 6:8] = 1
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits, loss = model(
            text_tokens=tokens, image_latents=torch.randn(2, 16, 2, 2),
            image_labels=torch.randn(2, 16, 2, 2), text_labels=None,
            image_masks=masks, t=torch.tensor([1.0, 0.5]),
            modality_positions=torch.tensor([[[2, 2], [6, 2]]]),
            attention_mask=torch.zeros(1, 1, 12, 12), max_seq_len=12,
            device=torch.device("cpu"),
        )
    assert logits is None and torch.isfinite(loss)
    loss.backward()
    named = dict(model.named_parameters())
    checks = {}
    for name, group in audit["groups"].items():
        if name == "frozen":
            assert all(named[n].grad is None for n in group["names"])
            continue
        grads = [named[n].grad for n in group["names"] if named[n].grad is not None]
        finite = bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
        nonzero = any(bool(torch.count_nonzero(g)) for g in grads)
        checks[name] = {"finite": finite, "nonzero": nonzero, "tensors_with_grad": len(grads)}
        assert finite and nonzero, (name, checks[name])
    # An optimizer step also exercises FP32 master parameters and Conv LoRA.
    optimizer.step()
    scheduler.step()
    model.eval()
    assert not any(m.training for m in model.modules())
    result = {"device": "cpu", "native_weights": True, "reduced_shape_only": True,
              "loss": float(loss.detach()), "optimizer_steps": 1, "scheduler_updates": scheduler.last_epoch,
              "groups": checks,
              "not_full_resolution_or_budget_test": True, "passed": True}
    (args.output_dir / "backward_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
