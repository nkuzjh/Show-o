"""Small tensor helpers shared by accelerated Seen-10 inference paths."""

from __future__ import annotations

import torch


def interleave_pairs(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Return ``[first_0, second_0, first_1, second_1, ...]`` along dim 0."""
    if first.shape != second.shape:
        raise ValueError(
            f"Paired tensors must have identical shapes, got {tuple(first.shape)} and {tuple(second.shape)}"
        )
    if first.ndim < 1:
        raise ValueError("Paired tensors must have a batch dimension")
    return torch.stack((first, second), dim=1).flatten(0, 1)


def pair_timesteps_with_clean_condition(timesteps: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Set each radar time to 1 while retaining its corresponding target time."""
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if timesteps.numel() == batch_size:
        target_times = timesteps.reshape(batch_size)
    elif timesteps.numel() == 2 * batch_size:
        target_times = timesteps.reshape(batch_size, 2)[:, 1]
    else:
        raise ValueError(
            f"Expected {batch_size} or {2 * batch_size} timesteps, got {timesteps.numel()}"
        )
    clean_times = torch.ones_like(target_times)
    return interleave_pairs(clean_times, target_times)


def keep_pair_targets_only(values: torch.Tensor) -> torch.Tensor:
    """Zero each clean-condition value and preserve every paired target value."""
    if values.ndim < 1 or values.shape[0] == 0 or values.shape[0] % 2:
        raise ValueError(f"Expected an even, non-empty paired batch, got {tuple(values.shape)}")
    pairs = values.reshape(values.shape[0] // 2, 2, *values.shape[1:])
    return interleave_pairs(torch.zeros_like(pairs[:, 0]), pairs[:, 1])

