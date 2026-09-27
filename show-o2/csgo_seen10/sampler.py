"""Deterministic, unpadded source stream for aligned Seen-10 training.

The DataLoader may request batches ahead of the model.  Its iterator therefore
plans positions without changing the checkpointed *consumed* cursor.  The
trainer commits a complete global optimizer batch only after using it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch.utils.data import Sampler


GLOBAL_BATCH_SIZE = 128


class GlobalSourceStream:
    """Successive seed+epoch permutations, without dropped or padded samples."""

    def __init__(self, sample_count: int, seed: int, consumed: int = 0) -> None:
        if sample_count <= 0:
            raise ValueError("Source stream requires at least one training sample")
        if consumed < 0 or consumed % GLOBAL_BATCH_SIZE:
            raise ValueError("Consumed source count must be a nonnegative optimizer boundary")
        self.sample_count = int(sample_count)
        self.seed = int(seed)
        self.consumed = int(consumed)
        self._cached_epoch = -1
        self._cached_order: list[int] = []

    def index_at(self, position: int) -> int:
        if position < 0:
            raise ValueError("Source position cannot be negative")
        epoch, offset = divmod(int(position), self.sample_count)
        if epoch != self._cached_epoch:
            generator = torch.Generator(device="cpu").manual_seed(self.seed + epoch)
            self._cached_order = torch.randperm(self.sample_count, generator=generator).tolist()
            self._cached_epoch = epoch
        return self._cached_order[offset]

    def global_indices(self, *, start: int | None = None) -> list[int]:
        position = self.consumed if start is None else int(start)
        if position < 0 or position % GLOBAL_BATCH_SIZE:
            raise ValueError("Global batches must begin on a 128-sample boundary")
        return [self.index_at(position + offset) for offset in range(GLOBAL_BATCH_SIZE)]

    def local_micro_indices(
        self,
        *,
        world_size: int,
        rank: int,
        micro_batch: int,
        accumulation: int,
        start: int | None = None,
    ) -> list[list[int]]:
        validate_layout(world_size, micro_batch, accumulation)
        if not 0 <= rank < world_size:
            raise ValueError(f"Invalid rank {rank} for world size {world_size}")
        indices = self.global_indices(start=start)
        local_size = micro_batch * accumulation
        local = indices[rank * local_size : (rank + 1) * local_size]
        return [local[offset : offset + micro_batch] for offset in range(0, local_size, micro_batch)]

    def commit_update(self) -> None:
        self.consumed += GLOBAL_BATCH_SIZE

    def state_dict(self) -> dict[str, int]:
        epoch, epoch_offset = divmod(self.consumed, self.sample_count)
        return {
            "seed": self.seed,
            "sample_count": self.sample_count,
            "consumed": self.consumed,
            "epoch": epoch,
            "epoch_offset": epoch_offset,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {"seed": self.seed, "sample_count": self.sample_count}
        for key, value in expected.items():
            if int(state[key]) != value:
                raise ValueError(f"Source stream {key} differs: checkpoint={state[key]}, current={value}")
        consumed = int(state["consumed"])
        if consumed < 0 or consumed % GLOBAL_BATCH_SIZE:
            raise ValueError("Checkpoint source cursor is not on an optimizer boundary")
        epoch, offset = divmod(consumed, self.sample_count)
        if (int(state["epoch"]), int(state["epoch_offset"])) != (epoch, offset):
            raise ValueError("Checkpoint source epoch and offset disagree with consumed cursor")
        self.consumed = consumed


def validate_layout(world_size: int, micro_batch: int, accumulation: int) -> None:
    if min(world_size, micro_batch, accumulation) <= 0:
        raise ValueError("World size, microbatch, and accumulation must be positive")
    if world_size * micro_batch * accumulation != GLOBAL_BATCH_SIZE:
        raise ValueError(
            f"Effective generation batch must be {GLOBAL_BATCH_SIZE}; got "
            f"{world_size} * {micro_batch} * {accumulation} = "
            f"{world_size * micro_batch * accumulation}"
        )


def sample_order_digest(sample_ids: Sequence[str]) -> str:
    if len(sample_ids) != GLOBAL_BATCH_SIZE:
        raise ValueError(f"Expected {GLOBAL_BATCH_SIZE} sample IDs, got {len(sample_ids)}")
    digest = hashlib.sha256(b"csgo-seen10-global-source-order-v1\0")
    for sample_id in sample_ids:
        encoded = sample_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


class PlannedMicroBatchSampler(Sampler[list[int]]):
    """Plan batches from a fixed start; DataLoader prefetch never commits them."""

    def __init__(
        self,
        stream: GlobalSourceStream,
        *,
        world_size: int,
        rank: int,
        micro_batch: int,
        accumulation: int,
        remaining_updates: int,
    ) -> None:
        validate_layout(world_size, micro_batch, accumulation)
        if remaining_updates < 0:
            raise ValueError("Remaining update count cannot be negative")
        self.stream = stream
        self.world_size = world_size
        self.rank = rank
        self.micro_batch = micro_batch
        self.accumulation = accumulation
        self.remaining_updates = remaining_updates
        self.start = stream.consumed

    def __len__(self) -> int:
        return self.remaining_updates * self.accumulation

    def __iter__(self) -> Iterator[list[int]]:
        for update in range(self.remaining_updates):
            micros = self.stream.local_micro_indices(
                world_size=self.world_size,
                rank=self.rank,
                micro_batch=self.micro_batch,
                accumulation=self.accumulation,
                start=self.start + update * GLOBAL_BATCH_SIZE,
            )
            yield from micros
