"""Numeric radar conditioning and Show-o2 model construction for Seen-10."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import nn

from .data import MAPS


class RadarPoseFiLM(nn.Module):
    """Inject numeric pose and map identity into the radar latent by FiLM."""

    def __init__(
        self,
        latent_channels: int = 16,
        map_count: int = len(MAPS),
        map_embedding_dim: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.map_embedding = nn.Embedding(map_count, map_embedding_dim)
        self.pose_mlp = nn.Sequential(
            nn.Linear(5 + map_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_channels * 2),
        )
        nn.init.zeros_(self.pose_mlp[-1].weight)
        nn.init.zeros_(self.pose_mlp[-1].bias)

    def forward(
        self,
        image_latents: torch.Tensor,
        normalized_pose: torch.Tensor,
        map_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Adapt `[radar, target]` latent pairs; leave every target latent intact."""
        if image_latents.ndim != 4:
            raise ValueError(f"Expected [2*B,C,H,W] latents, got {tuple(image_latents.shape)}")
        if normalized_pose.ndim != 2 or normalized_pose.shape[-1] != 5:
            raise ValueError(f"Expected normalized pose [B,5], got {tuple(normalized_pose.shape)}")
        batch_size = normalized_pose.shape[0]
        if map_ids.shape != (batch_size,):
            raise ValueError(f"Expected map ids [B], got {tuple(map_ids.shape)}")
        if image_latents.shape[0] != batch_size * 2:
            raise ValueError("Each sample must provide exactly one radar and one target/noise latent")
        if image_latents.shape[1] != self.latent_channels:
            raise ValueError(
                f"Expected {self.latent_channels} latent channels, got {image_latents.shape[1]}"
            )

        pairs = image_latents.reshape(batch_size, 2, *image_latents.shape[1:])
        condition_dtype = self.pose_mlp[0].weight.dtype
        conditioning = torch.cat(
            [
                normalized_pose.to(dtype=condition_dtype),
                self.map_embedding(map_ids).to(dtype=condition_dtype),
            ],
            dim=-1,
        )
        scale, shift = self.pose_mlp(conditioning).to(dtype=image_latents.dtype).chunk(2, dim=-1)
        radar = pairs[:, 0] * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        conditioned_pairs = torch.stack([radar, pairs[:, 1]], dim=1)
        return conditioned_pairs.reshape_as(image_latents)


class CSGOSeen10Model(nn.Module):
    """Wrap the official Show-o2 interleaved forward with a radar FiLM adapter."""

    def __init__(self, backbone: nn.Module, latent_channels: int = 16) -> None:
        super().__init__()
        self.backbone = backbone
        self.radar_adapter = RadarPoseFiLM(latent_channels=latent_channels)

    def condition_latents(
        self,
        image_latents: torch.Tensor,
        radar_pose: torch.Tensor,
        map_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.radar_adapter(image_latents, radar_pose, map_ids)

    def forward(
        self,
        *,
        image_latents: torch.Tensor,
        radar_pose: torch.Tensor,
        map_ids: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, ...]:
        image_latents = self.condition_latents(image_latents, radar_pose, map_ids)
        return self.backbone(image_latents=image_latents, **kwargs)

    @torch.no_grad()
    def t2i_generate(
        self,
        image_latents: torch.Tensor,
        t: torch.Tensor,
        *,
        radar_pose: torch.Tensor,
        map_ids: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        image_latents = self.condition_latents(image_latents, radar_pose, map_ids)
        return self.backbone.t2i_generate(image_latents=image_latents, t=t, **kwargs)


class AlignedCSGOSeen10Model(nn.Module):
    """Text-conditioned Seen-10 wrapper with no numeric radar adapter."""

    conditioning_mode = "text"

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self._keep_frozen_vision_eval()

    def _keep_frozen_vision_eval(self) -> None:
        from .finetuning import LEGACY_POLICY

        # Only v1 has a frozen vision path. Final-policy SigLIP/Conv adapters
        # must follow train()/eval() so their LoRA dropout is active in training.
        if getattr(self.backbone, "aligned_finetuning_policy", LEGACY_POLICY) != LEGACY_POLICY:
            return
        for name in ("image_embedder_und", "position_embedding", "und_trans"):
            getattr(self.backbone, name).eval()

    def train(self, mode: bool = True) -> "AlignedCSGOSeen10Model":
        super().train(mode)
        self._keep_frozen_vision_eval()
        return self

    def forward(self, *, image_latents: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, ...]:
        kwargs.pop("radar_pose", None)
        kwargs.pop("map_ids", None)
        if kwargs.get("text_labels") is None:
            kwargs["generation_backbone_only"] = True
        return self.backbone(image_latents=image_latents, **kwargs)

    @torch.no_grad()
    def t2i_generate(self, image_latents: torch.Tensor, t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        kwargs.pop("radar_pose", None)
        kwargs.pop("map_ids", None)
        kwargs["generation_backbone_only"] = True
        return self.backbone.t2i_generate(image_latents=image_latents, t=t, **kwargs)


def _resolve_local(path: str, project_showo_dir: str) -> str:
    from pathlib import Path

    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path(project_showo_dir) / resolved
    return str(resolved.resolve())


def load_showo2_seen10(
    config: Any,
    *,
    device: torch.device,
    weight_dtype: torch.dtype,
    showo_dir: str,
) -> Tuple[nn.Module, Any, Dict[str, int], Dict[str, Any]]:
    """Load the official 1.5B weights with the selected Seen-10 policy."""
    from models import Showo2Qwen2_5
    from models.misc import get_text_tokenizer

    showo_config = config.model.showo
    aligned = str(config.get("experiment", "")) == "csgo_seen10_exp32gen_aligned"
    showo_path = _resolve_local(showo_config.pretrained_model_path, showo_dir)
    qwen_path = _resolve_local(showo_config.llm_model_path, showo_dir)
    siglip_path = _resolve_local(config.model.clip.config_path, showo_dir)
    showo_config.llm_model_path = qwen_path

    text_tokenizer, showo_token_ids = get_text_tokenizer(
        qwen_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name="qwen2_5",
    )
    showo_token_ids = dict(showo_token_ids)
    if text_tokenizer.pad_token_id is None:
        raise ValueError("Show-o2 tokenizer did not define the required [PAD] token")
    showo_token_ids["pad_id"] = int(text_tokenizer.pad_token_id)

    # Patch-grid dimensions follow the active profile's config. A 432px image
    # uses the released checkpoint's native 27x27 position table.
    backbone, loading_info = Showo2Qwen2_5.from_pretrained(
        showo_path,
        llm_model_path=qwen_path,
        llm_vocab_size=len(text_tokenizer),
        load_from_showo=True,
        image_latent_height=int(showo_config.get("image_latent_height", 27 if aligned else 28)),
        image_latent_width=int(showo_config.get("image_latent_width", 27 if aligned else 28)),
        clip_pretrained_model_path=siglip_path,
        torch_dtype=weight_dtype,
        use_safetensors=False,
        local_files_only=True,
        output_loading_info=True,
    )
    missing_keys = loading_info.get("missing_keys", [])
    if missing_keys:
        raise RuntimeError(f"Official Show-o2 checkpoint did not initialize all model parameters: {missing_keys}")
    backbone = backbone.to(device)

    if aligned:
        from .finetuning import (
            assert_aligned_parameter_counts,
            configure_aligned_finetuning,
            policy_from_config,
            trainable_parameter_audit,
        )

        configure_aligned_finetuning(backbone, policy=policy_from_config(config))
        model = AlignedCSGOSeen10Model(backbone).to(device)
        assert_aligned_parameter_counts(trainable_parameter_audit(model))
        return model, text_tokenizer, showo_token_ids, loading_info

    # Follow the project's downstream freeze policy: preserve the language and
    # semantic understanding towers while adapting the generation path.
    frozen_substrings = ("showo", "image_embedder_und", "und_trans", "position_embedding")
    for name, parameter in backbone.named_parameters():
        parameter.requires_grad_(not any(fragment in name for fragment in frozen_substrings))
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()

    backbone.showo.config.use_cache = False
    if hasattr(backbone.showo, "gradient_checkpointing_enable"):
        try:
            backbone.showo.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            backbone.showo.gradient_checkpointing_enable()
    if hasattr(backbone.showo, "model"):
        backbone.showo.model.config.use_cache = False

    model = CSGOSeen10Model(backbone, latent_channels=int(showo_config.image_latent_dim)).to(device)
    model.radar_adapter.to(device=device, dtype=torch.float32)
    return model, text_tokenizer, showo_token_ids, loading_info
