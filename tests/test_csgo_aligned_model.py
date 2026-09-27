"""CPU contracts for the aligned Seen-10 adapter inventory and model wrapper."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


SHOWO_DIR = Path(__file__).resolve().parents[1] / "show-o2"
if str(SHOWO_DIR) not in sys.path:
    sys.path.insert(0, str(SHOWO_DIR))

from csgo_seen10.finetuning import (  # noqa: E402
    ALIGNED_LR,
    EXPECTED_COUNTS,
    EXPECTED_TRAINABLE,
    FINAL_EXPECTED_COUNTS,
    FINAL_EXPECTED_TRAINABLE,
    FINAL_POLICY,
    LEGACY_POLICY,
    assert_aligned_parameter_counts,
    configure_aligned_finetuning,
    policy_from_config,
    trainable_parameter_audit,
)
from csgo_seen10.model import AlignedCSGOSeen10Model  # noqa: E402


class TinyQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.model = nn.Module()
        self.model.config = SimpleNamespace(use_cache=True)
        self.model.layers = nn.ModuleList([self._layer() for _ in range(28)])
        self.model.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8)
        self.checkpoint_kwargs = None

    @staticmethod
    def _layer() -> nn.Module:
        layer = nn.Module()
        layer.self_attn = nn.Module()
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(layer.self_attn, projection, nn.Linear(4, 4))
        layer.mlp = nn.Module()
        for projection in ("gate_proj", "up_proj", "down_proj"):
            setattr(layer.mlp, projection, nn.Linear(4, 4))
        return layer

    def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs: dict) -> None:
        self.checkpoint_kwargs = gradient_checkpointing_kwargs


class TinyAlignedBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.showo = TinyQwen()
        self.diffusion_head_a = nn.ModuleList([self._diffusion_layer() for _ in range(10)])
        self.image_embedder_gen = nn.Module()
        self.image_embedder_gen.proj = nn.Conv2d(4, 4, 1)
        self.image_embedder_und = nn.Module()
        self.image_embedder_und.proj = nn.Conv2d(4, 4, 1)
        self.position_embedding = nn.Embedding(4, 4)
        self.und_trans = nn.Module()
        self.und_trans.layers = nn.ModuleList([self._vision_layer() for _ in range(26)])
        self.fusion_proj = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 4))
        self.time_embed = nn.Module()
        self.time_embed.mlp = nn.Sequential(nn.Linear(4, 4), nn.SiLU(), nn.Linear(4, 4))
        self.time_embed_proj = nn.Linear(4, 4)
        self.diff_proj = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 4))
        self.diffusion_head_b = nn.Module()
        self.diffusion_head_b.norm_final = nn.LayerNorm(4)
        self.diffusion_head_b.linear = nn.Linear(4, 4)
        self.diffusion_head_b.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(4, 4))

    @staticmethod
    def _diffusion_layer() -> nn.Module:
        layer = TinyQwen._layer()
        layer.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(4, 4))
        layer.input_layernorm = nn.LayerNorm(4)
        return layer

    @staticmethod
    def _vision_layer() -> nn.Module:
        layer = nn.Module()
        layer.self_attn = nn.Module()
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(layer.self_attn, projection, nn.Linear(4, 4))
        layer.mlp = nn.Module()
        layer.mlp.fc1 = nn.Linear(4, 8)
        layer.mlp.fc2 = nn.Linear(8, 4)
        layer.layer_norm1 = nn.LayerNorm(4)
        layer.layer_norm2 = nn.LayerNorm(4)
        return layer

    def forward(self, *, image_latents: torch.Tensor, **kwargs):
        return image_latents, kwargs

    def t2i_generate(self, *, image_latents: torch.Tensor, t: torch.Tensor, **kwargs):
        return image_latents + t, kwargs


def test_in_place_lora_exact_targets_and_affine_only() -> None:
    backbone = TinyAlignedBackbone().to(dtype=torch.bfloat16)
    configure_aligned_finetuning(backbone, policy=LEGACY_POLICY)
    assert backbone.showo.checkpoint_kwargs == {"use_reentrant": False}
    assert backbone.diffusion_gradient_checkpointing
    assert backbone.aligned_flow_only
    assert backbone.showo.config.use_cache is False
    assert backbone.showo.model.config.use_cache is False

    model = AlignedCSGOSeen10Model(backbone)
    assert not hasattr(model, "radar_adapter")
    assert model.conditioning_mode == "text"
    model.train()
    assert not backbone.image_embedder_und.training
    assert not backbone.und_trans.training
    assert not backbone.position_embedding.training
    assert backbone.fusion_proj.training

    audit = trainable_parameter_audit(model)
    assert audit["embedding_head"]["same_parameter"] is False
    assert audit["embedding_head"]["shared_storage"] is False
    assert audit["registered_parameter_bytes"] == audit["unique_parameter_storage_bytes"]
    assert len([n for n, _ in model.named_modules() if n.endswith(".lora_A")]) == 276
    assert len(audit["groups"]["qwen_lora"]["names"]) == 196 * 2
    assert len(audit["groups"]["diffusion_lora"]["names"]) == 80 * 2
    assert not any(row["trainable"] for row in audit["parameters"] if row["name"].startswith("backbone.fusion_proj."))
    assert all(row["dtype"] == "torch.float32" for row in audit["parameters"] if row["trainable"])
    json.dumps(audit)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=ALIGNED_LR)
    assert trainable_parameter_audit(model, optimizer)["trainable_numel"] == audit["trainable_numel"]
    wrong = torch.optim.AdamW([next(p for p in model.parameters() if not p.requires_grad)], lr=ALIGNED_LR)
    with pytest.raises(AssertionError, match="membership mismatch"):
        trainable_parameter_audit(model, wrong)


def test_policy_selection_and_official_count_contracts() -> None:
    assert policy_from_config({}) == LEGACY_POLICY
    assert policy_from_config({"finetuning": {}}) == LEGACY_POLICY
    assert policy_from_config({"finetuning": {"policy": FINAL_POLICY}}) == FINAL_POLICY
    with pytest.raises(ValueError, match="Unknown aligned"):
        policy_from_config({"finetuning": {"policy": "typo"}})
    assert EXPECTED_TRAINABLE == 79_445_056
    assert_aligned_parameter_counts({
        "policy": LEGACY_POLICY,
        "groups": {name: {"numel": count} for name, count in EXPECTED_COUNTS.items()},
        "trainable_numel": EXPECTED_TRAINABLE,
        "total_numel": 3_119_347_936,
    })
    assert FINAL_EXPECTED_TRAINABLE == sum(FINAL_EXPECTED_COUNTS.values())
    synthetic = {
        "policy": FINAL_POLICY,
        "groups": {name: {"numel": count} for name, count in FINAL_EXPECTED_COUNTS.items()},
        "trainable_numel": FINAL_EXPECTED_TRAINABLE,
        "total_numel": 3_136_184_544,
    }
    assert_aligned_parameter_counts(synthetic)
    synthetic["groups"]["fusion_proj"]["numel"] -= 1
    with pytest.raises(AssertionError, match="group counts differ"):
        assert_aligned_parameter_counts(synthetic)


def test_final_policy_exact_targets_full_modules_and_gradients() -> None:
    backbone = TinyAlignedBackbone().to(dtype=torch.bfloat16)
    # The official low-memory loader can expose distinct Parameters backed by
    # one vocab tensor. The final policy must separate the unused frozen head.
    backbone.showo.lm_head.weight = nn.Parameter(backbone.showo.model.embed_tokens.weight.data)
    assert backbone.showo.lm_head.weight is not backbone.showo.model.embed_tokens.weight
    configure_aligned_finetuning(backbone)
    assert backbone.aligned_finetuning_policy == FINAL_POLICY
    model = AlignedCSGOSeen10Model(backbone)
    model.train()
    assert backbone.und_trans.training
    assert backbone.image_embedder_und.training
    assert backbone.position_embedding.training
    vision_proj = backbone.und_trans.layers[0].self_attn.q_proj
    assert vision_proj.lora_dropout["default"].training
    model.eval()
    assert not vision_proj.lora_dropout["default"].training
    model.train()

    audit = trainable_parameter_audit(model)
    assert audit["policy"] == FINAL_POLICY
    assert not audit["embedding_head"]["same_parameter"]
    assert not audit["embedding_head"]["shared_storage"]
    assert not audit["parameter_aliases"]
    assert len([n for n, _ in model.named_modules() if n.endswith(".lora_A")]) == 434
    assert len(audit["groups"]["vision_lora"]["names"]) == 156 * 2
    assert len(audit["groups"]["und_image_lora"]["names"]) == 2
    assert len(audit["groups"]["gen_image_lora"]["names"]) == 2
    expected_full = (
        "backbone.showo.model.embed_tokens.weight",
        "backbone.position_embedding.weight",
        "backbone.fusion_proj.0.weight", "backbone.fusion_proj.0.bias",
        "backbone.fusion_proj.1.bias", "backbone.diffusion_head_b.norm_final.weight",
        "backbone.diffusion_head_b.norm_final.bias",
    )
    by_name = {row["name"]: row for row in audit["parameters"]}
    assert all(by_name[name]["trainable"] for name in expected_full)
    frozen = (
        "backbone.showo.lm_head.weight",
        "backbone.und_trans.layers.0.self_attn.q_proj.base_layer.weight",
        "backbone.und_trans.layers.0.self_attn.q_proj.base_layer.bias",
        "backbone.und_trans.layers.0.layer_norm1.weight",
        "backbone.diffusion_head_a.0.input_layernorm.weight",
        "backbone.image_embedder_und.proj.base_layer.bias",
        "backbone.image_embedder_gen.proj.base_layer.bias",
    )
    assert all(not by_name[name]["trainable"] for name in frozen)
    assert all(row["dtype"] == "torch.float32" for row in audit["parameters"] if row["trainable"])
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=ALIGNED_LR)
    trainable_parameter_audit(model, optimizer)

    # A compact path through each newly enabled branch produces gradients.
    tokens = torch.tensor([[1, 2]])
    image = torch.randn(1, 4, 2, 2, dtype=torch.bfloat16)
    embedding = backbone.showo.model.embed_tokens(tokens).sum()
    und = backbone.image_embedder_und.proj(image).sum()
    gen = backbone.image_embedder_gen.proj(image).sum()
    vision = vision_proj(torch.randn(1, 4, dtype=torch.bfloat16)).sum()
    full = backbone.fusion_proj(torch.randn(1, 4)).sum()
    full = full + backbone.diffusion_head_b.norm_final(torch.randn(1, 4)).sum()
    (embedding + und + gen + vision + full).backward()
    assert backbone.showo.model.embed_tokens.weight.grad is not None
    assert backbone.showo.lm_head.weight.grad is None
    assert backbone.image_embedder_und.proj.lora_B["default"].weight.grad is not None
    assert vision_proj.lora_B["default"].weight.grad is not None
    assert backbone.fusion_proj[0].weight.grad is not None
    assert backbone.diffusion_head_b.norm_final.weight.grad is not None
    assert vision_proj.base_layer.weight.grad is None

    # An original bias under a selected projection must not silently join the optimizer.
    vision_proj.base_layer.bias.requires_grad_(True)
    with pytest.raises(AssertionError, match="Trainable name mismatch"):
        trainable_parameter_audit(model)
    vision_proj.base_layer.bias.requires_grad_(False)
    backbone.peft_config["default"].lora_dropout = 0.0
    with pytest.raises(AssertionError, match="PEFT LoRA configuration mismatch"):
        trainable_parameter_audit(model)


def test_wrapper_passes_text_conditioning_without_radar_film() -> None:
    backbone = TinyAlignedBackbone()
    model = AlignedCSGOSeen10Model(backbone)
    latents = torch.randn(2, 4, 2, 2)
    passed, kwargs = model(
        image_latents=latents,
        radar_pose=torch.randn(1, 5),
        map_ids=torch.zeros(1, dtype=torch.long),
        image_labels=latents,
        text_labels=None,
    )
    assert passed is latents
    assert kwargs["generation_backbone_only"] is True
    assert "radar_pose" not in kwargs and "map_ids" not in kwargs
    result, kwargs = model.t2i_generate(image_latents=latents, t=torch.ones_like(latents))
    assert torch.equal(result, latents + 1)
    assert kwargs["generation_backbone_only"] is True


def test_flow_only_training_skips_vocabulary_head_with_image_labels() -> None:
    from models.modeling_showo2_qwen2_5 import Showo2Qwen2_5

    class ImagePatch(nn.Module):
        def forward(self, latents: torch.Tensor) -> torch.Tensor:
            return latents.flatten(2).transpose(1, 2)

    class LanguageCore(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(8, 4)

        def forward(self, *, inputs_embeds, **kwargs):
            return SimpleNamespace(last_hidden_state=inputs_embeds)

    class NoVocabularyHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = LanguageCore()

        def forward(self, **kwargs):
            raise AssertionError("Vocabulary logits must not be computed in flow-only training")

    class TinyDiffusionBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        def forward(self, *, hidden_states, **kwargs):
            return (hidden_states * self.scale,)

    diffusion_block = TinyDiffusionBlock()
    fake = SimpleNamespace(
        showo=NoVocabularyHead(),
        image_embedder_und=ImagePatch(),
        image_embedder_gen=ImagePatch(),
        position_embedding=nn.Embedding(4, 4),
        und_trans=lambda hidden: {"last_hidden_state": hidden},
        fusion_proj=nn.Linear(8, 4),
        time_embed=lambda times, dtype: torch.zeros(times.numel(), 4, dtype=dtype),
        time_embed_proj=nn.Identity(),
        diff_proj=nn.Identity(),
        diffusion_head_a=[diffusion_block],
        diffusion_gradient_checkpointing=True,
        aligned_flow_only=True,
        diffusion_head_b=lambda hidden, times, positions: hidden,
        config=SimpleNamespace(patch_size=1, clip_latent_dim=4, hidden_size=4, add_time_embeds=True),
        training=True,
    )
    latents = torch.randn(2, 4, 2, 2)
    logits, flow_loss = Showo2Qwen2_5.forward(
        fake,
        text_tokens=torch.zeros(2, 7, dtype=torch.long),
        image_latents=latents,
        image_labels=torch.randn_like(latents),
        image_masks=torch.ones(2, 7),
        t=torch.zeros(2),
        modality_positions=torch.tensor([[[2, 5]], [[2, 5]]]),
        max_seq_len=7,
        device="cpu",
        generation_backbone_only=True,
    )
    assert logits is None
    assert flow_loss.ndim == 0 and torch.isfinite(flow_loss)
    flow_loss.backward()
    assert fake.fusion_proj.weight.grad is not None
    assert diffusion_block.scale.grad is not None
