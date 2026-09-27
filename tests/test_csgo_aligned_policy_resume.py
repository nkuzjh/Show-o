"""Exact CPU checkpoint resume for the final aligned trainable policy."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn


SHOWO_DIR = Path(__file__).resolve().parents[1] / "show-o2"
if str(SHOWO_DIR) not in sys.path:
    sys.path.insert(0, str(SHOWO_DIR))

from csgo_seen10 import runtime  # noqa: E402
from csgo_seen10.aligned_training import make_scheduler  # noqa: E402
from csgo_seen10.finetuning import (  # noqa: E402
    ALIGNED_LR,
    FINAL_POLICY,
    _target_groups,
    configure_aligned_finetuning,
    trainable_parameter_audit,
)
from csgo_seen10.model import AlignedCSGOSeen10Model  # noqa: E402
from test_csgo_aligned_model import TinyAlignedBackbone  # noqa: E402


def _make_model() -> AlignedCSGOSeen10Model:
    backbone = TinyAlignedBackbone()
    # Reproduce the official loader's distinct Parameter objects backed by
    # one vocabulary tensor. The final policy must isolate the frozen head.
    backbone.showo.lm_head.weight = nn.Parameter(backbone.showo.model.embed_tokens.weight.data)
    configure_aligned_finetuning(backbone, policy=FINAL_POLICY)
    model = AlignedCSGOSeen10Model(backbone)
    model.train()
    return model


def _make_optimizer_and_scheduler(model: AlignedCSGOSeen10Model):
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=ALIGNED_LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    )
    trainable_parameter_audit(model, optimizer)
    config = SimpleNamespace(
        optimizer=SimpleNamespace(learning_rate=ALIGNED_LR),
        lr_scheduler=SimpleNamespace(min_lr=1e-5),
    )
    return optimizer, make_scheduler(optimizer, config)


def _objective(model: AlignedCSGOSeen10Model) -> torch.Tensor:
    """Run actual PEFT/affine modules, covering every final trainable group."""
    backbone = model.backbone
    terms = []
    for group, paths in _target_groups(backbone, FINAL_POLICY).items():
        for path in paths:
            module = backbone.get_submodule(path)
            if group in {"und_image_lora", "gen_image_lora"}:
                value = module(torch.randn(2, 4, 2, 2))
            else:
                value = module(torch.randn(2, module.base_layer.in_features))
            terms.append(value.square().mean())

    terms.extend((
        backbone.showo.model.embed_tokens(torch.tensor([[1, 2, 3]])).square().mean(),
        backbone.position_embedding(torch.tensor([[0, 1, 2]])).square().mean(),
        backbone.fusion_proj(torch.randn(2, 4)).square().mean(),
        backbone.time_embed.mlp(torch.randn(2, 4)).square().mean(),
        backbone.time_embed_proj(torch.randn(2, 4)).square().mean(),
        backbone.diff_proj(torch.randn(2, 4)).square().mean(),
        backbone.diffusion_head_b.norm_final(torch.randn(2, 4)).square().mean(),
        backbone.diffusion_head_b.linear(torch.randn(2, 4)).square().mean(),
        backbone.diffusion_head_b.adaLN_modulation(torch.randn(2, 4)).square().mean(),
    ))
    return torch.stack(terms).sum()


def _update(model, optimizer, scheduler) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    loss = _objective(model)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
    optimizer.step()
    scheduler.step()
    return loss.detach().clone()


def _parameter_snapshot(model, *, trainable: bool) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad == trainable
    }


def _assert_equal_tree(actual: Any, expected: Any) -> None:
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys()
        for key in expected:
            _assert_equal_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_equal_tree(left, right)
    else:
        assert actual == expected


def test_final_policy_exact_cpu_resume(tmp_path: Path, monkeypatch) -> None:
    previous_rng = torch.get_rng_state()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(20260927)
        model = _make_model()
        optimizer, scheduler = _make_optimizer_and_scheduler(model)
        audit = trainable_parameter_audit(model)
        assert {name for name in audit["groups"] if name != "frozen"} == {
            "qwen_lora", "diffusion_lora", "qwen_embedding", "und_image_lora",
            "position_embedding", "vision_lora", "gen_image_lora", "fusion_proj",
            "time", "diff_proj", "diffusion_head_b",
        }
        for path in ("image_embedder_und.proj", "image_embedder_gen.proj"):
            module = model.backbone.get_submodule(path)
            assert isinstance(module.base_layer, nn.Conv2d)
            assert module.lora_dropout["default"].training
        assert model.backbone.und_trans.layers[0].self_attn.q_proj.lora_dropout["default"].training
        frozen_initial = _parameter_snapshot(model, trainable=False)

        _update(model, optimizer, scheduler)
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        monkeypatch.setattr(runtime, "_expected_trainable_parameters", lambda _: audit["trainable_numel"])
        runtime.save_finetune_weights(model, checkpoint_dir)
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": torch.get_rng_state(),
        }, checkpoint_dir / "state.pt")
        after_step_one = _parameter_snapshot(model, trainable=True)

        reference_loss = _update(model, optimizer, scheduler)
        reference_trainable = _parameter_snapshot(model, trainable=True)
        reference_optimizer = optimizer.state_dict()
        reference_scheduler = scheduler.state_dict()
        reference_rng = torch.get_rng_state()
        assert all(torch.equal(parameter, frozen_initial[name]) for name, parameter in
                   _parameter_snapshot(model, trainable=False).items())

        torch.manual_seed(20260927)
        resumed_model = _make_model()
        resumed_optimizer, resumed_scheduler = _make_optimizer_and_scheduler(resumed_model)
        assert all(torch.equal(parameter, frozen_initial[name]) for name, parameter in
                   _parameter_snapshot(resumed_model, trainable=False).items())
        runtime.load_finetune_weights(resumed_model, checkpoint_dir)
        assert all(torch.equal(parameter, after_step_one[name]) for name, parameter in
                   _parameter_snapshot(resumed_model, trainable=True).items())
        saved = torch.load(checkpoint_dir / "state.pt", map_location="cpu", weights_only=True)
        resumed_optimizer.load_state_dict(saved["optimizer"])
        resumed_scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["rng"])
        resumed_loss = _update(resumed_model, resumed_optimizer, resumed_scheduler)

        assert torch.equal(resumed_loss, reference_loss)
        assert all(torch.equal(parameter, reference_trainable[name]) for name, parameter in
                   _parameter_snapshot(resumed_model, trainable=True).items())
        assert all(torch.equal(parameter, frozen_initial[name]) for name, parameter in
                   _parameter_snapshot(resumed_model, trainable=False).items())
        _assert_equal_tree(resumed_optimizer.state_dict(), reference_optimizer)
        _assert_equal_tree(resumed_scheduler.state_dict(), reference_scheduler)
        assert torch.equal(torch.get_rng_state(), reference_rng)
    finally:
        torch.set_rng_state(previous_rng)
        torch.set_num_threads(previous_threads)
