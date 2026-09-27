"""Exact Show-o2 LoRA and affine fine-tuning policy for aligned Seen-10."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import torch
from torch import nn


LEGACY_POLICY = "aligned_v1"
FINAL_POLICY = "aligned_v2_final"

EXPECTED_COUNTS = {
    "qwen_lora": 36_929_536,
    "diffusion_lora": 18_677_760,
    "image_embedder_gen": 99_840,
    "time": 7_869_952,
    "diff_proj": 7_344_128,
    "diffusion_head_b": 8_523_840,
}
EXPECTED_TRAINABLE = 79_445_056
FINAL_EXPECTED_COUNTS = {
    "qwen_lora": 36_929_536,
    "diffusion_lora": 18_677_760,
    "qwen_embedding": 232_963_584,
    "und_image_lora": 38_912,
    "position_embedding": 839_808,
    "vision_lora": 16_746_496,
    "gen_image_lora": 51_200,
    "fusion_proj": 6_493_824,
    "time": 7_869_952,
    "diff_proj": 7_344_128,
    "diffusion_head_b": 8_525_888,
}
FINAL_EXPECTED_TRAINABLE = 336_481_088
# The released .bin shares the Qwen embedding/lm_head tensor storage. The
# low-memory Show-o2 loader installs these as two registered Parameters, and
# their live model count therefore includes both full vocab matrices.
EXPECTED_TOTAL = 3_119_347_936
FINAL_EXPECTED_TOTAL = 3_136_184_544
CHECKPOINT_DEDUPLICATED_TOTAL = 2_886_384_352
ALIGNED_LR = 1e-4


def _check_policy(policy: str) -> str:
    if policy not in (LEGACY_POLICY, FINAL_POLICY):
        raise ValueError(f"Unknown aligned fine-tuning policy: {policy!r}")
    return policy


def policy_from_config(config: Any) -> str:
    """Missing policy selects the original aligned_v1 experiment behavior."""
    finetuning = config.get("finetuning") if hasattr(config, "get") else getattr(config, "finetuning", None)
    if finetuning is None:
        return LEGACY_POLICY
    policy = finetuning.get("policy") if hasattr(finetuning, "get") else getattr(finetuning, "policy", None)
    return _check_policy(policy if policy is not None else LEGACY_POLICY)


def _target_groups(backbone: nn.Module, policy: str) -> dict[str, list[str]]:
    """Resolve every intended LoRA target by its complete path."""
    qwen_layers = backbone.showo.model.layers
    diffusion_layers = backbone.diffusion_head_a
    if len(qwen_layers) != 28 or len(diffusion_layers) != 10:
        raise ValueError("Aligned policy requires 28 Qwen and 10 diffusion layers")

    qwen = [
        f"showo.model.layers.{i}.{branch}.{projection}"
        for i in range(28)
        for branch, projections in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        )
        for projection in projections
    ]
    diffusion = [
        f"diffusion_head_a.{i}.{branch}.{projection}"
        for i in range(10)
        for branch, projections in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        )
        for projection in projections
    ] + [f"diffusion_head_a.{i}.adaLN_modulation.1" for i in range(10)]
    if len(qwen) != 196 or len(diffusion) != 80:
        raise AssertionError("Incorrect aligned LoRA target inventory")
    groups = {"qwen_lora": qwen, "diffusion_lora": diffusion}
    if policy == FINAL_POLICY:
        if len(backbone.und_trans.layers) != 26:
            raise ValueError("Final aligned policy requires 26 retained SigLIP layers")
        groups.update({
            "vision_lora": [
                f"und_trans.layers.{i}.{branch}.{projection}"
                for i in range(26)
                for branch, projections in (
                    ("self_attn", ("q_proj", "k_proj", "v_proj", "out_proj")),
                    ("mlp", ("fc1", "fc2")),
                )
                for projection in projections
            ],
            "und_image_lora": ["image_embedder_und.proj"],
            "gen_image_lora": ["image_embedder_gen.proj"],
        })
    for group, paths in groups.items():
        expected_type = nn.Conv2d if group in ("und_image_lora", "gen_image_lora") else nn.Linear
        for path in paths:
            module = backbone.get_submodule(path)
            base = getattr(module, "base_layer", module)
            if not isinstance(base, expected_type):
                raise TypeError(f"Aligned LoRA target is not {expected_type.__name__}: {path}")
    return groups


def _full_groups(policy: str) -> dict[str, tuple[str, ...]]:
    if policy == LEGACY_POLICY:
        return {
            "image_embedder_gen": ("image_embedder_gen.proj",),
            "time": ("time_embed.mlp.0", "time_embed.mlp.2", "time_embed_proj"),
            "diff_proj": ("diff_proj.0", "diff_proj.2"),
            "diffusion_head_b": ("diffusion_head_b.linear", "diffusion_head_b.adaLN_modulation.1"),
        }
    return {
        "qwen_embedding": ("showo.model.embed_tokens",),
        "position_embedding": ("position_embedding",),
        "fusion_proj": ("fusion_proj",),
        "time": ("time_embed.mlp", "time_embed_proj"),
        "diff_proj": ("diff_proj",),
        "diffusion_head_b": ("diffusion_head_b",),
    }


def _shares_storage(a: nn.Parameter, b: nn.Parameter) -> bool:
    return (a.device.type != "meta" and a.device == b.device
            and a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr())


def _isolate_frozen_lm_head(backbone: nn.Module) -> None:
    embedding = backbone.showo.model.embed_tokens.weight
    lm_head = backbone.showo.lm_head.weight
    if lm_head is embedding or _shares_storage(embedding, lm_head):
        backbone.showo.lm_head.weight = nn.Parameter(lm_head.detach().clone(), requires_grad=False)


def _set_affine_trainable(backbone: nn.Module) -> None:
    paths = (
        "image_embedder_gen.proj",
        "time_embed.mlp.0",
        "time_embed.mlp.2",
        "time_embed_proj",
        "diff_proj.0",
        "diff_proj.2",
        "diffusion_head_b.linear",
        "diffusion_head_b.adaLN_modulation.1",
    )
    for path in paths:
        module = backbone.get_submodule(path)
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            raise TypeError(f"Aligned affine target is not Linear/Conv2d: {path}")
        for parameter in module.parameters(recurse=False):
            parameter.requires_grad_(True)


def configure_aligned_finetuning(backbone: nn.Module, policy: str = FINAL_POLICY) -> nn.Module:
    """Freeze the loaded backbone, then apply the selected exact policy.

    PEFT injects into ``backbone`` in place; the Qwen model remains at its
    checkpoint-compatible ``showo`` path. The Wan VAE is external to this model.
    """
    from peft import LoraConfig, inject_adapter_in_model

    policy = _check_policy(policy)
    target_groups = _target_groups(backbone, policy)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    if policy == FINAL_POLICY:
        _isolate_frozen_lm_head(backbone)

    # A single anchored regex prevents PEFT's usual suffix matching from
    # accidentally selecting the identically named SigLIP vision projections.
    targets = [path for paths in target_groups.values() for path in paths]
    exact_targets = "^(?:" + "|".join(re.escape(path) for path in targets) + ")$"
    config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        lora_bias=False,
        init_lora_weights=True,
        use_rslora=False,
        use_dora=False,
        target_modules=exact_targets,
    )
    injected = inject_adapter_in_model(config, backbone)
    if injected is not backbone:
        raise RuntimeError("PEFT did not inject the adapter in place")
    actual = {
        name for name, module in backbone.named_modules()
        if hasattr(module, "lora_A") and hasattr(module, "lora_B")
    }
    if actual != set(targets):
        raise RuntimeError(f"LoRA target mismatch: missing={sorted(set(targets) - actual)}, extra={sorted(actual - set(targets))}")

    if policy == LEGACY_POLICY:
        _set_affine_trainable(backbone)
    else:
        for paths in _full_groups(policy).values():
            for path in paths:
                for parameter in backbone.get_submodule(path).parameters():
                    parameter.requires_grad_(True)
    for parameter in backbone.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()

    backbone.showo.config.use_cache = False
    backbone.showo.model.config.use_cache = False
    backbone.showo.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    backbone.aligned_flow_only = True
    backbone.diffusion_gradient_checkpointing = True
    backbone.aligned_finetuning_policy = policy
    return backbone


def _expected_groups(backbone: nn.Module, policy: str) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    for group, paths in _target_groups(backbone, policy).items():
        groups[group] = {
            f"backbone.{path}.lora_{letter}.default.weight"
            for path in paths for letter in ("A", "B")
        }
    for group, paths in _full_groups(policy).items():
        groups[group] = {
            f"backbone.{path}.{subname}"
            for path in paths
            for subname, _ in backbone.get_submodule(path).named_parameters()
        }
    if policy == FINAL_POLICY:
        required = {
            "backbone.showo.model.embed_tokens.weight", "backbone.position_embedding.weight",
            "backbone.fusion_proj.0.weight", "backbone.fusion_proj.1.weight",
            "backbone.fusion_proj.1.bias", "backbone.fusion_proj.3.weight",
            "backbone.fusion_proj.3.bias", "backbone.diffusion_head_b.norm_final.weight",
            "backbone.diffusion_head_b.linear.weight", "backbone.diffusion_head_b.linear.bias",
            "backbone.diffusion_head_b.adaLN_modulation.1.weight",
            "backbone.diffusion_head_b.adaLN_modulation.1.bias",
        }
        expected = set().union(*groups.values())
        if not required <= expected:
            raise AssertionError(f"Required final full-weight/norm/bias inventory missing: {sorted(required - expected)}")
    return groups


def _verify_lora_config(backbone: nn.Module, policy: str) -> None:
    target_paths = [path for paths in _target_groups(backbone, policy).values() for path in paths]
    targets = set(target_paths)
    expected_regex = "^(?:" + "|".join(re.escape(path) for path in target_paths) + ")$"
    config = getattr(backbone, "peft_config", {}).get("default")
    if (config is None or config.r != 32 or config.lora_alpha != 64
            or config.lora_dropout != 0.05 or config.bias != "none"
            or config.lora_bias is not False or config.init_lora_weights is not True
            or config.use_rslora is not False or config.use_dora is not False
            or config.target_modules != expected_regex):
        raise AssertionError("Aligned PEFT LoRA configuration mismatch")
    actual = {name for name, module in backbone.named_modules()
              if hasattr(module, "lora_A") and hasattr(module, "lora_B")}
    if actual != targets:
        raise AssertionError(f"LoRA target mismatch: missing={sorted(targets - actual)}, extra={sorted(actual - targets)}")
    for path in targets:
        module = backbone.get_submodule(path)
        if (module.r.get("default") != 32 or module.lora_alpha.get("default") != 64
                or module.scaling.get("default") != 2.0
                or not isinstance(module.lora_dropout["default"], nn.Dropout)
                or module.lora_dropout["default"].p != 0.05):
            raise AssertionError(f"LoRA hyperparameter mismatch: {path}")
        if getattr(module, "lora_bias", {}).get("default", False):
            raise AssertionError(f"Unexpected LoRA bias: {path}")
        if getattr(module, "use_dora", {}).get("default", False):
            raise AssertionError(f"Unexpected DoRA: {path}")
        if getattr(module, "use_rslora", {}).get("default", False):
            raise AssertionError(f"Unexpected rank-stabilized LoRA: {path}")
        if module.active_adapters != ["default"] or module.disable_adapters:
            raise AssertionError(f"Approved LoRA adapter is not active: {path}")


def trainable_parameter_audit(model: nn.Module, optimizer: Any = None) -> dict[str, Any]:
    """Validate exact trainable names, adapters, aliases, and optimizer membership.

    The report is JSON serializable. ``total_numel`` counts registered live
    parameters. It does not subtract the checkpoint's shared embedding/head
    storage, since the loader exposes them as distinct Parameter objects.
    """
    backbone = model.backbone
    policy = _check_policy(getattr(backbone, "aligned_finetuning_policy", LEGACY_POLICY))
    _verify_lora_config(backbone, policy)
    expected_groups = _expected_groups(backbone, policy)
    expected_by_name = {name: group for group, names in expected_groups.items() for name in names}
    parameters = list(model.named_parameters())
    actual_trainable = {name for name, parameter in parameters if parameter.requires_grad}
    expected_trainable = set(expected_by_name)
    if actual_trainable != expected_trainable:
        raise AssertionError(f"Trainable name mismatch: missing={sorted(expected_trainable - actual_trainable)}, "
                             f"extra={sorted(actual_trainable - expected_trainable)}")

    aliases: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    duplicated = [names for names in aliases.values() if len(names) > 1]
    if policy == FINAL_POLICY and duplicated:
        raise AssertionError(f"Final policy has aliased Parameter objects: {duplicated}")
    if any(any(name in expected_trainable for name in names) for names in duplicated):
        raise AssertionError(f"Trainable Parameter has an alias: {duplicated}")

    # Distinct Parameter objects can still alias one storage, as in the
    # released Qwen embedding and vocabulary head before isolation.
    storage_names: dict[tuple[str, int | None, int], list[str]] = {}
    for name, parameter in parameters:
        if parameter.device.type != "meta":
            key = (parameter.device.type, parameter.device.index, parameter.untyped_storage().data_ptr())
            storage_names.setdefault(key, []).append(name)
    storage_aliases = [names for names in storage_names.values() if len(names) > 1]
    if policy == FINAL_POLICY and any(
        any(name in expected_trainable for name in names) for names in storage_aliases
    ):
        raise AssertionError(f"Trainable storage aliases another Parameter: {storage_aliases}")

    named = []
    groups: dict[str, dict[str, Any]] = {}
    trainable_ids: set[int] = set()
    for name, parameter in parameters:
        group = expected_by_name[name] if parameter.requires_grad else "frozen"
        count = parameter.numel()
        named.append({
            "name": name,
            "shape": list(parameter.shape),
            "numel": count,
            "dtype": str(parameter.dtype),
            "trainable": bool(parameter.requires_grad),
            "group": group,
        })
        entry = groups.setdefault(group, {"numel": 0, "names": []})
        entry["numel"] += count
        entry["names"].append(name)
        if parameter.requires_grad:
            trainable_ids.add(id(parameter))

    optimizer_groups = []
    if optimizer is not None:
        seen: Counter[int] = Counter()
        name_by_id = {id(parameter): name for name, parameter in parameters}
        for index, group in enumerate(optimizer.param_groups):
            lr = float(group["lr"])
            if lr != ALIGNED_LR:
                raise AssertionError(f"Optimizer group {index} has lr={lr}, expected {ALIGNED_LR}")
            names = []
            for parameter in group["params"]:
                pid = id(parameter)
                seen[pid] += 1
                names.append(name_by_id.get(pid, f"<unknown:{pid}>"))
            optimizer_groups.append({"index": index, "lr": lr, "names": names})
        if set(seen) != trainable_ids or any(value != 1 for value in seen.values()):
            missing = [name_by_id[pid] for pid in trainable_ids - set(seen)]
            extra = [name_by_id.get(pid, f"<unknown:{pid}>") for pid in set(seen) - trainable_ids]
            duplicate = [name_by_id.get(pid, f"<unknown:{pid}>") for pid, count in seen.items() if count > 1]
            raise AssertionError(f"Optimizer membership mismatch: missing={missing}, extra={extra}, duplicate={duplicate}")

    embedding = model.get_parameter("backbone.showo.model.embed_tokens.weight")
    lm_head = model.get_parameter("backbone.showo.lm_head.weight")
    embedding_head = {
        "embedding_numel": embedding.numel(),
        "lm_head_numel": lm_head.numel(),
        "same_parameter": embedding is lm_head,
        "shared_storage": _shares_storage(embedding, lm_head),
    }
    if policy == FINAL_POLICY and (embedding_head["same_parameter"] or embedding_head["shared_storage"]):
        raise AssertionError("Trainable Qwen embedding still aliases frozen lm_head")
    total_numel = sum(parameter.numel() for _, parameter in parameters)
    registered_parameter_bytes = sum(parameter.numel() * parameter.element_size() for _, parameter in parameters)
    unique_storages: dict[tuple[str, int | None, int], int] = {}
    for _, parameter in parameters:
        storage = parameter.untyped_storage()
        key = (parameter.device.type, parameter.device.index, storage.data_ptr())
        unique_storages[key] = storage.nbytes()
    return {
        "policy": policy,
        "total_numel": total_numel,
        "trainable_numel": sum(parameter.numel() for _, parameter in parameters if parameter.requires_grad),
        "registered_parameter_bytes": registered_parameter_bytes,
        "unique_parameter_storage_bytes": sum(unique_storages.values()),
        "embedding_head": embedding_head,
        "parameter_aliases": duplicated,
        "storage_aliases": storage_aliases,
        "checkpoint_storage_deduplicated_reference_numel": (
            CHECKPOINT_DEDUPLICATED_TOTAL if total_numel == EXPECTED_TOTAL else None
        ),
        "groups": groups,
        "parameters": named,
        "optimizer_groups": optimizer_groups,
    }


def assert_aligned_parameter_counts(audit: dict[str, Any]) -> None:
    """Fail before training if the official checkpoint or policy differs."""
    policy = _check_policy(audit.get("policy", LEGACY_POLICY))
    expected_counts = EXPECTED_COUNTS if policy == LEGACY_POLICY else FINAL_EXPECTED_COUNTS
    expected_trainable = EXPECTED_TRAINABLE if policy == LEGACY_POLICY else FINAL_EXPECTED_TRAINABLE
    expected_total = EXPECTED_TOTAL if policy == LEGACY_POLICY else FINAL_EXPECTED_TOTAL
    actual = {name: group["numel"] for name, group in audit["groups"].items() if name != "frozen"}
    if actual != expected_counts:
        raise AssertionError(f"Aligned trainable group counts differ: expected={expected_counts}, actual={actual}")
    if audit["trainable_numel"] != expected_trainable or audit["total_numel"] != expected_total:
        raise AssertionError(
            f"Aligned parameter totals differ: trainable={audit['trainable_numel']}, total={audit['total_numel']}"
        )
