"""CPU checks for the aligned generation source and update contract."""

from __future__ import annotations

import math
import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset


SHOWO_DIR = Path(__file__).resolve().parents[1] / "show-o2"
if str(SHOWO_DIR) not in sys.path:
    sys.path.insert(0, str(SHOWO_DIR))

from csgo_seen10.sampler import (  # noqa: E402
    GLOBAL_BATCH_SIZE,
    GlobalSourceStream,
    PlannedMicroBatchSampler,
    sample_order_digest,
)
from csgo_seen10.aligned_training import (  # noqa: E402
    CHECKPOINT_STEPS,
    _save_checkpoint,
    _resume_audit_matches,
    _semantic_config_digest,
    _validate_resume,
    configure_determinism,
    learning_rate_for_update,
    make_scheduler,
    resolve_layout,
    validate_profile,
)
from csgo_seen10 import aligned_training as training_module  # noqa: E402


CONFIG_PATH = SHOWO_DIR / "configs" / "csgo_seen10_exp32gen_aligned.yaml"
V1_CONFIG_PATH = SHOWO_DIR / "configs" / "csgo_seen10_exp32gen_aligned_v1.yaml"


def test_profile_budget_and_optimizer_schedule() -> None:
    config = OmegaConf.load(CONFIG_PATH)
    validate_profile(config)
    assert tuple(config.training.checkpoint_steps) == CHECKPOINT_STEPS
    assert math.isclose(learning_rate_for_update(1), 1e-4 / 59)
    assert math.isclose(learning_rate_for_update(59), 1e-4)
    assert math.isclose(learning_rate_for_update(19500), 1e-5)
    assert learning_rate_for_update(60) < learning_rate_for_update(59)

    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    scheduler = make_scheduler(optimizer, config)
    assert math.isclose(optimizer.param_groups[0]["lr"], learning_rate_for_update(1))
    optimizer.step()
    scheduler.step()
    assert scheduler.last_epoch == 1
    assert math.isclose(optimizer.param_groups[0]["lr"], learning_rate_for_update(2))

    parameter2 = torch.nn.Parameter(torch.tensor(0.0))
    optimizer2 = torch.optim.AdamW([parameter2], lr=1e-4)
    scheduler2 = make_scheduler(optimizer2, config)
    optimizer2.load_state_dict(optimizer.state_dict())
    scheduler2.load_state_dict(scheduler.state_dict())
    assert scheduler2.last_epoch == 1
    assert optimizer2.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


def test_policy_configs_have_separate_semantic_digests() -> None:
    final = OmegaConf.load(CONFIG_PATH)
    previous = OmegaConf.load(V1_CONFIG_PATH)
    validate_profile(final)
    validate_profile(previous)
    assert final.finetuning.policy == "aligned_v2_final"
    assert "policy" not in previous.finetuning
    assert final.finetuning.expected_trainable_parameters == 336_481_088
    assert previous.finetuning.expected_trainable_parameters == 79_445_056
    assert _semantic_config_digest(final) != _semantic_config_digest(previous)


def test_v1_resume_accepts_only_missing_empty_audit_schema_fields() -> None:
    # The original v1 audit had no policy or alias-inventory keys. Its
    # established counts, parameter list, and storage facts remain binding.
    old = {
        "trainable_numel": 79_445_056,
        "total_numel": 3_119_347_936,
        "embedding_head": {"shared_storage": False, "same_parameter": False},
        "groups": {"qwen_lora": {"numel": 36_929_536}},
        "parameters": [{"name": "backbone.showo.model.embed_tokens.weight", "trainable": False}],
        "unique_parameter_storage_bytes": 6_397_585_984,
    }
    current = {**old, "policy": "aligned_v1", "parameter_aliases": [], "storage_aliases": []}
    assert _resume_audit_matches(old, current, policy="aligned_v1")
    assert "policy" not in old and "parameter_aliases" not in old
    assert not _resume_audit_matches(old, current, policy="aligned_v2_final")
    assert not _resume_audit_matches(old, {**current, "storage_aliases": [["embedding", "head"]]}, policy="aligned_v1")
    assert not _resume_audit_matches(old, {**current, "trainable_numel": 79_445_057}, policy="aligned_v1")
    assert not _resume_audit_matches(old, {**current, "embedding_head": {"shared_storage": True, "same_parameter": False}}, policy="aligned_v1")
    assert not _resume_audit_matches({**old, "policy": "aligned_v2_final"}, current, policy="aligned_v1")


def test_strict_backend_flags_and_workspace_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    config = OmegaConf.load(CONFIG_PATH)
    old = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cuda.matmul.allow_tf32,
    )
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    try:
        flags = configure_determinism(config)
        assert flags["cublas_workspace_config"] == ":4096:8"
        assert flags["deterministic_algorithms"] is True
        assert flags["deterministic_algorithms_warn_only"] is False
        assert flags["cudnn_benchmark"] is False
        assert flags["cudnn_deterministic"] is True
        assert flags["cuda_matmul_allow_tf32"] is True
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        assert configure_determinism(config)["cublas_workspace_config"] == ":16:8"
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":invalid")
        with pytest.raises(ValueError, match="Incompatible CUBLAS_WORKSPACE_CONFIG"):
            configure_determinism(config)
    finally:
        torch.use_deterministic_algorithms(old[0], warn_only=old[1])
        torch.backends.cudnn.benchmark = old[2]
        torch.backends.cudnn.deterministic = old[3]
        torch.backends.cuda.matmul.allow_tf32 = old[4]


def test_layout_override_keeps_global_budget_and_semantics() -> None:
    config = OmegaConf.load(CONFIG_PATH)
    args = SimpleNamespace(micro_batch=None, gradient_accumulation=None)
    assert resolve_layout(config, args, 1) == (8, 16)
    assert resolve_layout(config, args, 2) == (8, 8)
    args.micro_batch = 4
    assert resolve_layout(config, args, 4) == (4, 8)
    args.gradient_accumulation = 4
    with pytest.raises(ValueError, match="Effective generation batch"):
        resolve_layout(config, args, 4)
    args.gradient_accumulation = None
    config2 = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    config2.training.batch_size = 4
    config2.training.gradient_accumulation_steps = 32
    config2.dataset.num_workers = 0
    assert _semantic_config_digest(config) == _semantic_config_digest(config2)
    config2.dataset.resolution = 448
    assert _semantic_config_digest(config) != _semantic_config_digest(config2)


def test_unpadded_source_stream_and_prefetch_safe_resume() -> None:
    # 131 creates an epoch boundary inside the second optimizer batch.
    stream = GlobalSourceStream(sample_count=131, seed=42)
    initial = stream.global_indices()
    assert len(initial) == GLOBAL_BATCH_SIZE
    assert len(set(initial)) == GLOBAL_BATCH_SIZE
    planned = PlannedMicroBatchSampler(
        stream, world_size=2, rank=0, micro_batch=8,
        accumulation=8, remaining_updates=3,
    )
    prefetched = list(iter(planned))
    assert len(prefetched) == 24
    assert stream.consumed == 0  # DataLoader prefetch is not checkpoint consumption.
    assert [item for micro in prefetched[:8] for item in micro] == initial[:64]

    stream.commit_update()
    state = stream.state_dict()
    assert state["consumed"] == 128
    assert state["epoch"] == 0 and state["epoch_offset"] == 128
    second = stream.global_indices()
    assert len(second) == 128
    assert len(set(second[:3])) == 3
    # Every source index appears exactly once in a completed epoch.
    first_epoch = [stream.index_at(position) for position in range(131)]
    assert sorted(first_epoch) == list(range(131))

    resumed = GlobalSourceStream(sample_count=131, seed=42)
    resumed.load_state_dict(state)
    assert resumed.global_indices() == second
    rank0 = resumed.local_micro_indices(world_size=4, rank=0, micro_batch=4, accumulation=8)
    rank1 = resumed.local_micro_indices(world_size=4, rank=1, micro_batch=4, accumulation=8)
    assert [item for micro in rank0 + rank1 for item in micro] == second[:64]
    ids = [f"sample/{index}" for index in second]
    assert sample_order_digest(ids) == sample_order_digest(ids)
    with pytest.raises(ValueError, match="consumed cursor"):
        resumed.load_state_dict({**state, "epoch_offset": 1})


def test_worker_prefetch_and_resume_keep_two_step_source_order() -> None:
    class IndexedDataset(Dataset):
        def __len__(self) -> int:
            return 131

        def __getitem__(self, index: int) -> int:
            return index

    dataset = IndexedDataset()

    def consume(stream: GlobalSourceStream, updates: int) -> list[list[int]]:
        loader = DataLoader(
            dataset,
            batch_sampler=PlannedMicroBatchSampler(
                stream, world_size=1, rank=0, micro_batch=8,
                accumulation=16, remaining_updates=updates,
            ),
            num_workers=2,
            generator=torch.Generator().manual_seed(1042),
        )
        iterator = iter(loader)
        result = []
        for _ in range(updates):
            batch = [index for _ in range(16) for index in next(iterator).tolist()]
            result.append(batch)
            stream.commit_update()
        return result

    uninterrupted = GlobalSourceStream(131, 42)
    expected = consume(uninterrupted, 2)
    first_run = GlobalSourceStream(131, 42)
    first = consume(first_run, 1)
    resumed = GlobalSourceStream(131, 42)
    resumed.load_state_dict(first_run.state_dict())
    second = consume(resumed, 1)
    assert first + second == expected
    assert resumed.consumed == uninterrupted.consumed == 256


def test_profile_rejects_changed_training_budget() -> None:
    config = OmegaConf.load(CONFIG_PATH)
    config.training.max_optimizer_steps = 19_499
    with pytest.raises(ValueError, match="max_optimizer_steps"):
        validate_profile(config)
    config = OmegaConf.load(CONFIG_PATH)
    config.finetuning.rank = 16
    with pytest.raises(ValueError, match="finetuning.rank"):
        validate_profile(config)


def test_resume_rejects_foreign_profile_and_cursor() -> None:
    class FakeDataset:
        provenance = {"manifest": "verified"}

        def __len__(self) -> int:
            return 131

    dataset = FakeDataset()
    saved = GlobalSourceStream(131, 42)
    saved.commit_update()
    metadata = {
        "profile": "csgo_seen10_exp32gen_aligned",
        "train_seed": 42,
        "effective_generation_batch": 128,
        "semantic_config_sha256": "same-config",
        "backend_flags": {"deterministic_algorithms": True},
        "base_provenance": {"showo": "same-base"},
        "train_data_provenance": dataset.provenance,
        "validation_data_provenance": dataset.provenance,
        "train_sample_count": 131,
        "validation_sample_count": 131,
        "smoke_only": True,
        "global_step": 1,
        "global_source_count": 128,
        "source_stream": saved.state_dict(),
    }
    kwargs = {
        "stream": GlobalSourceStream(131, 42),
        "config_digest": "same-config",
        "backend_flags": {"deterministic_algorithms": True},
        "base_provenance": {"showo": "same-base"},
        "train_dataset": dataset,
        "val_dataset": dataset,
        "smoke": True,
    }
    _validate_resume(metadata, **kwargs)
    assert kwargs["stream"].consumed == 128
    with pytest.raises(ValueError, match="profile differs"):
        _validate_resume({**metadata, "profile": "legacy"}, **kwargs)
    with pytest.raises(ValueError, match="backend_flags differs"):
        _validate_resume({**metadata, "backend_flags": {"deterministic_algorithms": False}}, **kwargs)
    with pytest.raises(ValueError, match="source count disagree"):
        _validate_resume({**metadata, "global_source_count": 127}, **kwargs)
    with pytest.raises(ValueError, match="finetuning_policy differs"):
        _validate_resume(metadata, **kwargs, policy="aligned_v2_final")
    final_metadata = {**metadata, "finetuning_policy": "aligned_v2_final"}
    _validate_resume(final_metadata, **kwargs, policy="aligned_v2_final")
    with pytest.raises(ValueError, match="finetuning_policy differs"):
        _validate_resume(final_metadata, **kwargs, policy="aligned_v1")


def _two_rank_update_worker(rank: int, result_dir: str, master_port: int) -> None:
    os.environ.update({
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(master_port),
        "ACCELERATE_USE_CPU": "true",
    })
    torch.set_num_threads(1)
    accelerator = Accelerator(
        cpu=True,
        mixed_precision="no",
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=8, sync_with_dataloader=False,
        ),
    )
    stream = GlobalSourceStream(131, 42)
    micros = stream.local_micro_indices(world_size=2, rank=rank, micro_batch=8, accumulation=8)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    model, optimizer = accelerator.prepare(model, optimizer)
    config = OmegaConf.load(CONFIG_PATH)
    scheduler = make_scheduler(optimizer, config)
    used_lr = optimizer.param_groups[0]["lr"]
    for micro_index, indices in enumerate(micros):
        x = torch.tensor([float(index % 17) / 17 for index in indices]).reshape(-1, 1)
        with accelerator.accumulate(model):
            loss = model(x).square().mean()
            accelerator.backward(loss)
            assert accelerator.sync_gradients == (micro_index == 7)
            if accelerator.sync_gradients:
                parameter = accelerator.unwrap_model(model).weight
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
    stream.commit_update()
    Path(result_dir, f"rank_{rank}.json").write_text(json.dumps({
        "indices": [index for micro in micros for index in micro],
        "weight": accelerator.unwrap_model(model).weight.item(),
        "used_lr": used_lr,
        "next_lr": optimizer.param_groups[0]["lr"],
        "scheduler_epoch": scheduler.last_epoch,
        "source_count": stream.consumed,
    }), encoding="utf-8")
    accelerator.end_training()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def test_two_cpu_ranks_match_one_global_update(tmp_path: Path) -> None:
    with socket.socket() as endpoint:
        endpoint.bind(("127.0.0.1", 0))
        port = endpoint.getsockname()[1]
    mp.spawn(_two_rank_update_worker, args=(str(tmp_path), port), nprocs=2, join=True)
    rank0 = json.loads((tmp_path / "rank_0.json").read_text(encoding="utf-8"))
    rank1 = json.loads((tmp_path / "rank_1.json").read_text(encoding="utf-8"))
    global_indices = GlobalSourceStream(131, 42).global_indices()
    assert rank0["indices"] + rank1["indices"] == global_indices
    assert set(rank0["indices"]).isdisjoint(rank1["indices"])
    for result in (rank0, rank1):
        assert result["scheduler_epoch"] == 1
        assert result["source_count"] == 128
        assert math.isclose(result["used_lr"], learning_rate_for_update(1))
        assert math.isclose(result["next_lr"], learning_rate_for_update(2))

    reference = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        reference.weight.fill_(0.4)
    optimizer = torch.optim.SGD(reference.parameters(), lr=learning_rate_for_update(1))
    x = torch.tensor([float(index % 17) / 17 for index in global_indices]).reshape(-1, 1)
    reference(x).square().mean().backward()
    optimizer.step()
    expected = reference.weight.item()
    assert rank0["weight"] == pytest.approx(expected, abs=1e-7)
    assert rank1["weight"] == pytest.approx(expected, abs=1e-7)


def test_checkpoint_publishes_only_complete_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "checkpoints").mkdir()

    class FakeAccelerator:
        is_main_process = True
        gradient_accumulation_steps = 16
        step = 16
        fail_save = False

        def wait_for_everyone(self) -> None:
            pass

        def save_state(self, state_dir: str) -> None:
            path = Path(state_dir)
            assert path.parent.name.startswith(".step_00000")
            assert not (tmp_path / "checkpoints" / f"step_{self.step // 16:06d}").exists()
            path.mkdir(parents=True)
            (path / "state.bin").write_bytes(b"resumable-state")
            if self.fail_save:
                raise RuntimeError("simulated interrupted save")

        def unwrap_model(self, model: object) -> object:
            return model

    monkeypatch.setattr(
        training_module, "save_finetune_weights",
        lambda _model, directory: (Path(directory) / "backbone_trainable.safetensors").write_bytes(b"weights"),
    )
    accelerator = FakeAccelerator()
    source = GlobalSourceStream(131, 42)
    source.commit_update()
    kwargs = {
        "accelerator": accelerator,
        "model": object(),
        "scheduler": SimpleNamespace(last_epoch=1, state_dict=lambda: {"last_epoch": 1}),
        "output_dir": tmp_path,
        "step": 1,
        "stream": source,
        "val_loss": 0.5,
        "best_val_loss": math.inf,
        "metadata_base": {"profile": "aligned"},
        "layout": {"world_size": 1, "micro_batch": 8, "gradient_accumulation": 16},
        "resume_events": [],
        "final": True,
    }
    assert _save_checkpoint(**kwargs)
    complete = tmp_path / "checkpoints" / "step_000001"
    assert (complete / "accelerator_state" / "state.bin").is_file()
    assert (complete / "metadata.json").is_file()
    assert (tmp_path / "checkpoints" / "latest").resolve() == complete
    assert not list((tmp_path / "checkpoints").glob(".step_*.saving_*"))
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _save_checkpoint(**kwargs)
    assert (complete / "metadata.json").is_file()

    accelerator.step = 32
    accelerator.fail_save = True
    source.commit_update()
    kwargs.update(step=2, scheduler=SimpleNamespace(last_epoch=2, state_dict=lambda: {"last_epoch": 2}))
    with pytest.raises(RuntimeError, match="simulated interrupted save"):
        _save_checkpoint(**kwargs)
    assert not (tmp_path / "checkpoints" / "step_000002").exists()
    assert len(list((tmp_path / "checkpoints").glob(".step_000002.saving_*"))) == 1
    assert (tmp_path / "checkpoints" / "latest").resolve() == complete
