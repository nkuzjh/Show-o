"""Small contract tests for the Show-o2 CSGO Seen-10 integration.

These tests deliberately use the checked-in benchmark bundle for dataset
identity and tiny stubs for model-side mechanics. They never load Show-o2
weights or initialize a GPU model.
"""

from __future__ import annotations

import os
import sys
import types
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SHOWO_DIR = REPO_ROOT / "show-o2"
if str(SHOWO_DIR) not in sys.path:
    sys.path.insert(0, str(SHOWO_DIR))

from csgo_seen10 import data as dataset_module  # noqa: E402
from csgo_seen10.data import CSGOSeen10Dataset, MAPS  # noqa: E402
from csgo_seen10.acceleration import (  # noqa: E402
    interleave_pairs,
    keep_pair_targets_only,
    pair_timesteps_with_clean_condition,
)
from csgo_seen10.model import RadarPoseFiLM  # noqa: E402
from csgo_seen10.runtime import (  # noqa: E402
    IMAGE_TOKEN_COUNT,
    Seen10InferenceConditionCache,
    build_flow_pair,
    build_interleaved_inputs,
    explicit_euler_final,
    generate_batch,
    is_valid_output,
    noise_for_generators,
    save_rgb_jpeg,
    seed_validation_rng,
)
from infer_seen10 import (  # noqa: E402
    INFERENCE_RNG_STRATEGY,
    _sample_seed,
    iter_manifest_batches,
)


def test_released_27_by_27_position_table_interpolates_to_28_by_28() -> None:
    helper_path = SHOWO_DIR / "models" / "position_utils.py"
    spec = importlib.util.spec_from_file_location("showo_position_utils", helper_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    embedding = SimpleNamespace(weight=torch.zeros((27 * 27, 8)))

    assert module.position_table_matches_grid(embedding, 27, 27)
    assert not module.position_table_matches_grid(embedding, 28, 28)


def _benchmark_root() -> Path:
    return Path(
        os.environ.get(
            "CSGO_BENCHMARK_V2_DATA_ROOT",
            "/home/jiahao/task/UniLIP/data/csgo_benchmark_v2",
        )
    )


def test_real_seen10_manifest_dataset_order_and_inference_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = _benchmark_root()
    assert (data_root / "minimal_dataset_report.json").is_file()

    inference = CSGOSeen10Dataset(
        data_root,
        "discrete_test",
        include_target=False,
        limit_per_map=1,
    )
    assert len(inference) == len(MAPS)
    assert [row["map_name"] for row in inference.rows] == list(MAPS)
    for map_id, row in enumerate(inference.rows):
        assert row["map_id"] == map_id
        assert row["sample_id"] == f'{row["map_name"]}/{row["file_frame"]}'
        assert row["target_path"] is None

    opened: list[Path] = []
    original_open = Image.open

    def tracked_open(fp, *args, **kwargs):
        opened.append(Path(fp).resolve())
        return original_open(fp, *args, **kwargs)

    monkeypatch.setattr(dataset_module.Image, "open", tracked_open)
    sample = inference[0]
    assert "target" not in sample
    assert sample["sample_id"] == inference.rows[0]["sample_id"]
    assert sample["map_name"] == MAPS[0]
    assert sample["pose"].shape == (5,)
    assert sample["radar"].shape == (3, 448, 448)
    assert opened == [inference.rows[0]["radar_path"]]

    train = CSGOSeen10Dataset(
        data_root,
        "train",
        include_target=True,
        limit_per_map=1,
    )
    train_sample = train[0]
    assert train_sample["radar"].shape == (3, 448, 448)
    assert train_sample["target"].shape == (3, 448, 448)


def test_real_continuous_split_keeps_clip_frame_order() -> None:
    continuous = CSGOSeen10Dataset(
        _benchmark_root(),
        "continuous",
        include_target=False,
        limit_per_map=64,
    )
    assert len(continuous) == len(MAPS) * 64

    first_clip = continuous.rows[:64]
    assert [row["map_name"] for row in first_clip] == [MAPS[0]] * 64
    assert [row["frame_index"] for row in first_clip] == list(range(64))
    assert len({row["clip_id"] for row in first_clip}) == 1
    assert [row["sample_id"] for row in first_clip] == [
        f'{MAPS[0]}/{row["file_frame"]}' for row in first_clip
    ]
    assert continuous.rows[64]["map_name"] == MAPS[1]
    assert continuous.rows[64]["frame_index"] == 0


def test_inference_rng_schedule_uses_zero_based_manifest_index() -> None:
    assert [_sample_seed(42, index) for index in (0, 1, 9)] == [42, 43, 51]
    assert "inference_seed + manifest_index" in INFERENCE_RNG_STRATEGY
    assert "zero-based dataset row order" in INFERENCE_RNG_STRATEGY
    with pytest.raises(ValueError, match="non-negative"):
        _sample_seed(42, -1)


def test_inference_pair_helpers_preserve_each_sample_target() -> None:
    radar = torch.tensor([[1.0], [2.0]])
    target = torch.tensor([[10.0], [20.0]])
    assert torch.equal(interleave_pairs(radar, target).flatten(), torch.tensor([1.0, 10.0, 2.0, 20.0]))

    times = pair_timesteps_with_clean_condition(torch.tensor([0.25, 0.75]), batch_size=2)
    assert torch.equal(times, torch.tensor([1.0, 0.25, 1.0, 0.75]))
    already_paired = pair_timesteps_with_clean_condition(torch.tensor([0.5, 0.2, 0.6, 0.4]), batch_size=2)
    assert torch.equal(already_paired, torch.tensor([1.0, 0.2, 1.0, 0.4]))

    velocities = torch.tensor([[100.0], [1.0], [200.0], [2.0]])
    assert torch.equal(keep_pair_targets_only(velocities).flatten(), torch.tensor([0.0, 1.0, 0.0, 2.0]))


def test_batched_noise_matches_independent_manifest_seed_draws() -> None:
    latent_batch = torch.zeros((3, 2, 2, 2))
    seeds = [42, 43, 99]
    generators = [torch.Generator(device="cpu").manual_seed(seed) for seed in seeds]
    observed = noise_for_generators(latent_batch, generators)
    expected = torch.cat(
        [
            torch.randn(
                (1, *latent_batch.shape[1:]),
                generator=torch.Generator(device="cpu").manual_seed(seed),
            )
            for seed in seeds
        ],
        dim=0,
    )
    assert torch.equal(observed, expected)


def test_manifest_batches_are_stable_and_never_cross_map_boundaries() -> None:
    dataset = SimpleNamespace(
        rows=[
            {"map_name": "a"},
            {"map_name": "a"},
            {"map_name": "a"},
            {"map_name": "a"},
            {"map_name": "a"},
            {"map_name": "b"},
            {"map_name": "b"},
            {"map_name": "b"},
        ]
    )
    assert list(iter_manifest_batches(dataset, 2)) == [[0, 1], [2, 3], [4], [5, 6], [7]]
    with pytest.raises(ValueError, match="positive"):
        list(iter_manifest_batches(dataset, 0))


def test_condition_cache_reuses_radar_and_prompt_but_keeps_pose_per_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"radar": 0, "tokenizer": 0, "vae": 0, "mask": 0}
    fake_models = types.ModuleType("models")

    def fake_attn_mask(batch_size, sequence_length, modality_positions, device):
        calls["mask"] += 1
        return torch.zeros((batch_size, 1, sequence_length, sequence_length), device=device)

    fake_models.omni_attn_mask_naive = fake_attn_mask
    monkeypatch.setitem(sys.modules, "models", fake_models)

    class TinyTokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool = False):
            calls["tokenizer"] += 1
            assert add_special_tokens is False
            return SimpleNamespace(input_ids=[30, 31])

    class CountingVAE:
        def sample(self, images: torch.Tensor, *, deterministic: bool) -> torch.Tensor:
            del deterministic
            calls["vae"] += 1
            values = images[:, :1, :1, :1, :1]
            return values.expand(images.shape[0], 16, 1, 2, 2).contiguous()

    class TinyDataset:
        rows = [
            {"map_name": "cs_agency", "map_id": 0, "pose": torch.tensor([0., 0., 0., 0., 0.])},
            {"map_name": "cs_agency", "map_id": 0, "pose": torch.tensor([1., 2., 3., 4., 5.])},
        ]

        def get_condition_only(self, index: int):
            calls["radar"] += 1
            return {"radar": torch.full((3, 4, 4), 10.0 + index)}

    cache = Seen10InferenceConditionCache(
        vae=CountingVAE(),
        tokenizer=TinyTokenizer(),
        token_ids={"bos_id": 1, "boi_id": 2, "img_pad_id": 3, "eoi_id": 4, "eos_id": 5, "pad_id": 0},
        device=torch.device("cpu"),
        weight_dtype=torch.float32,
        max_seq_length=1664,
        max_prompt_tokens=64,
    )
    dataset = TinyDataset()
    first_latents, first_inputs = cache.prepare_batch(dataset, [0, 1])
    repeated_latents, repeated_inputs = cache.prepare_batch(dataset, [1])

    assert first_latents.shape == (2, 16, 2, 2)
    assert torch.equal(first_latents[0], first_latents[1])
    assert torch.equal(repeated_latents[0], first_latents[0])
    assert torch.equal(first_inputs["radar_pose"], torch.stack([row["pose"] for row in dataset.rows]))
    assert not torch.equal(first_inputs["radar_pose"][0], first_inputs["radar_pose"][1])
    assert first_inputs["text_tokens"].shape[0] == 2
    assert repeated_inputs["text_tokens"].shape[0] == 1
    assert calls == {"radar": 1, "tokenizer": 1, "vae": 1, "mask": 1}


def test_explicit_euler_final_matches_torchdiffeq_fixed_euler() -> None:
    from transport import Sampler, create_transport

    transport = create_transport(path_type="Linear", prediction="velocity", do_shift=False, seq_len=785)
    sampler = Sampler(transport)

    def denoiser(x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        time = t.reshape(t.shape[0], *([1] * (x.ndim - 1)))
        return 0.25 * torch.tanh(x) + 0.1 * time

    initial = torch.linspace(-1.0, 1.0, 2 * 3 * 2 * 2).reshape(2, 3, 2, 2)
    settings = {"num_steps": 7, "time_shifting_factor": 3.0}
    reference_sampler = sampler.sample_ode(
        sampling_method="euler",
        num_steps=settings["num_steps"],
        reverse=False,
        time_shifting_factor=settings["time_shifting_factor"],
    )
    reference = reference_sampler(initial, denoiser)[-1]
    actual = explicit_euler_final(
        initial,
        denoiser,
        sampler,
        num_steps=settings["num_steps"],
        time_shifting_factor=settings["time_shifting_factor"],
        model_kwargs={},
    )
    assert torch.equal(actual, reference)


def test_generate_batch_decodes_all_targets_in_one_vae_call() -> None:
    from transport import Sampler, create_transport

    class TinyGenerationModel:
        def eval(self):
            return self

        def t2i_generate(self, image_latents, t, **kwargs):
            del t, kwargs
            return torch.zeros_like(image_latents)

    class CountingDecodeVAE:
        def __init__(self):
            self.batch_sizes: list[int] = []

        def batch_decode(self, latents: torch.Tensor) -> torch.Tensor:
            self.batch_sizes.append(latents.shape[0])
            return latents[:, :3]

    transport = create_transport(path_type="Linear", prediction="velocity", do_shift=False, seq_len=785)
    sampler = Sampler(transport)
    vae = CountingDecodeVAE()
    images = generate_batch(
        TinyGenerationModel(),
        vae,
        sampler,
        {"map_name": ["cs_agency", "cs_agency"]},
        tokenizer=None,
        token_ids={},
        device=torch.device("cpu"),
        weight_dtype=torch.float32,
        max_seq_length=8,
        max_prompt_tokens=64,
        num_inference_steps=4,
        sampling_method="euler",
        atol=1e-6,
        rtol=1e-3,
        time_shifting_factor=3.0,
        guidance_scale=0.0,
        generators=[
            torch.Generator(device="cpu").manual_seed(10),
            torch.Generator(device="cpu").manual_seed(11),
        ],
        radar_latents=torch.zeros((2, 16, 2, 2)),
        model_inputs={"text_tokens": torch.zeros((2, 8), dtype=torch.long)},
    )

    assert len(images) == 2
    assert all(image.size == (2, 2) for image in images)
    assert vae.batch_sizes == [2]


def test_validation_rng_is_stable_after_dataloader_startup_consumes_cpu_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def draw_after_loader_startup(random_draws: int) -> tuple[int, torch.Tensor]:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234)
            torch.rand(random_draws)  # Simulate iterator/worker base-seed consumption.
            sample_seed = seed_validation_rng(1701, ["de_train/frame_0001"])
            return sample_seed, torch.randn(16)

    first_seed, first_noise = draw_after_loader_startup(1)
    repeated_seed, repeated_noise = draw_after_loader_startup(37)
    assert first_seed == repeated_seed
    assert torch.equal(first_noise, repeated_noise)

    with torch.random.fork_rng(devices=[]):
        assert seed_validation_rng(1701, ["de_train/frame_0001"]) != seed_validation_rng(
            1701, ["de_train/frame_0002"]
        )


def test_two_image_sequence_has_785_token_offsets_and_target_only_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    fake_models = types.ModuleType("models")

    def fake_attn_mask(batch_size, sequence_length, modality_positions, device):
        observed["positions"] = modality_positions.detach().cpu().clone()
        observed["sequence_length"] = sequence_length
        return torch.ones((batch_size, sequence_length, sequence_length), device=device)

    fake_models.omni_attn_mask_naive = fake_attn_mask
    monkeypatch.setitem(sys.modules, "models", fake_models)

    class TinyTokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool = False):
            assert add_special_tokens is False
            token_count = 2 if "cs_agency" in text else 3
            return SimpleNamespace(input_ids=list(range(30, 30 + token_count)))

    token_ids = {
        "bos_id": 1,
        "boi_id": 2,
        "img_pad_id": 3,
        "eoi_id": 4,
        "eos_id": 5,
        "pad_id": 0,
    }
    tokens, positions, image_masks, attention_mask = build_interleaved_inputs(
        {"map_name": ["cs_agency", "de_train"]},
        TinyTokenizer(),
        token_ids,
        device=torch.device("cpu"),
    )

    assert IMAGE_TOKEN_COUNT == 785
    assert positions.tolist() == [
        [[4, 785], [791, 785]],
        [[5, 785], [792, 785]],
    ]
    assert tokens.shape == (2, 1664)
    assert image_masks.shape == (2, 1664)
    assert attention_mask.shape == (2, 1664, 1664)
    assert torch.count_nonzero(image_masks[0, 4 : 4 + 785]) == 0
    assert torch.all(image_masks[0, 791 : 791 + 785] == 1)
    assert torch.count_nonzero(image_masks[1, 5 : 5 + 785]) == 0
    assert torch.all(image_masks[1, 792 : 792 + 785] == 1)
    assert torch.equal(observed["positions"], positions.cpu())
    assert observed["sequence_length"] == 1664


class _StubVAE:
    def sample(self, images: torch.Tensor, *, deterministic: bool) -> torch.Tensor:
        del deterministic
        # Carry one pixel value through as a tiny, easily inspected 16-channel latent.
        values = images[:, :1, :1, :1, :1]
        return values.expand(images.shape[0], 16, 1, 2, 2).contiguous()


class _StubPathSampler:
    def plan(self, t: torch.Tensor, noise: torch.Tensor, clean: torch.Tensor):
        return t, clean + noise, noise - clean


class _StubTransport:
    path_sampler = _StubPathSampler()

    def sample(self, clean: torch.Tensor):
        t = torch.tensor([0.25, 0.75], dtype=clean.dtype, device=clean.device)
        noise = torch.full_like(clean, 1000.0)
        return t, noise, clean


def test_flow_pair_interleaves_radar_target_latents_timesteps_and_labels() -> None:
    batch = {
        "radar": torch.stack(
            [torch.full((3, 4, 4), 10.0), torch.full((3, 4, 4), 20.0)]
        ),
        "target": torch.stack(
            [torch.full((3, 4, 4), 100.0), torch.full((3, 4, 4), 200.0)]
        ),
    }
    latents, timesteps, labels = build_flow_pair(
        batch,
        vae=_StubVAE(),
        transport=_StubTransport(),
        device=torch.device("cpu"),
        weight_dtype=torch.float32,
    )

    assert latents.shape == (4, 16, 2, 2)
    assert labels.shape == latents.shape
    assert torch.allclose(latents[:, 0, 0, 0], torch.tensor([10.0, 1100.0, 20.0, 1200.0]))
    assert torch.allclose(timesteps, torch.tensor([1.0, 0.25, 1.0, 0.75]))
    assert torch.allclose(labels[:, 0, 0, 0], torch.tensor([0.0, 900.0, 0.0, 800.0]))


def test_pose_film_preserves_target_latents_and_has_adapter_gradient() -> None:
    torch.manual_seed(0)
    adapter = RadarPoseFiLM(latent_channels=2, map_embedding_dim=4, hidden_dim=8)
    latents = torch.arange(4 * 2 * 3 * 3, dtype=torch.float32).reshape(4, 2, 3, 3) / 10.0
    original_targets = latents[[1, 3]].clone()
    pose = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]])
    map_ids = torch.tensor([0, 9], dtype=torch.long)

    conditioned = adapter(latents, pose, map_ids)
    assert torch.equal(conditioned[[1, 3]], original_targets)
    (conditioned[[0, 2]].square().mean()).backward()
    final_layer = adapter.pose_mlp[-1]
    assert final_layer.weight.grad is not None
    assert torch.count_nonzero(final_layer.weight.grad) > 0


def test_default_jpeg_writer_emits_valid_448_rgb_and_rejects_invalid_files(tmp_path: Path) -> None:
    destination = tmp_path / "sample.jpg"
    source = Image.new("RGBA", (512, 320), color=(40, 120, 220, 80))

    assert save_rgb_jpeg(source, destination)
    assert is_valid_output(destination)
    with Image.open(destination) as saved:
        assert saved.format == "JPEG"
        assert saved.mode == "RGB"
        assert saved.size == (448, 448)

    # The benchmark writer should retain the PIL default JPEG settings used by
    # UniLIP: RGB conversion and resize are the only image transformations.
    expected = tmp_path / "expected.jpg"
    expected_rgb = source.convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
    expected_rgb.save(expected, format="JPEG")
    assert destination.read_bytes() == expected.read_bytes()

    before = destination.read_bytes()
    assert not save_rgb_jpeg(source, destination)
    assert destination.read_bytes() == before

    invalid = tmp_path / "invalid.jpg"
    source.save(invalid, format="PNG")
    assert not is_valid_output(invalid)
