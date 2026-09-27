"""Aligned conditioning, target isolation and output identity regression tests."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "show-o2"))
from csgo_seen10.data import CSGOSeen10Dataset, MAPS
from csgo_seen10 import runtime as runtime_module
from csgo_seen10.runtime import (build_interleaved_inputs, dataset_kwargs,
                                generate_batch, is_valid_output, sample_identity_seed)
from infer_seen10 import _inference_settings, bind_prediction_checkpoint

CONFIG = ROOT / "show-o2/configs/csgo_seen10_exp32gen_aligned.yaml"
DATA = "/home/jiahao/task/UniLIP/data/csgo_benchmark_v2"


def test_aligned_splits_and_no_test_target(monkeypatch):
    config = OmegaConf.load(CONFIG)
    for split, size in (("train", 50000), ("validation", 5000), ("discrete_test", 20000), ("continuous", 12800)):
        ds = CSGOSeen10Dataset(DATA, split, include_target=split in {"train", "validation"}, **dataset_kwargs(config))
        assert len(ds) == size
        assert list(dict.fromkeys(row["map_name"] for row in ds.rows)) == list(MAPS)
        assert "z_calibration_sha256" in ds.provenance
        assert "split_files_sha256" in ds.provenance
        if split in {"discrete_test", "continuous"}:
            assert all(row["target_path"] is None for row in ds.rows)
            original = Image.open
            allowed = {row["radar_path"] for row in ds.rows}
            def guarded(path, *args, **kwargs):
                assert Path(path) in allowed, f"Unexpected image read: {path}"
                return original(path, *args, **kwargs)
            with monkeypatch.context() as patch:
                patch.setattr(Image, "open", guarded)
                item = ds[0]
                assert "target" not in item
                assert item["radar"].shape == (3, 432, 432)
                assert "pitch=" in item["instruction"] and "yaw=" in item["instruction"]
        if split == "continuous":
            assert len({(r["map_name"], r["clip_id"]) for r in ds.rows}) == 200


def test_native_two_image_blocks_and_prompt_not_truncated():
    tokenizer = lambda text, **kwargs: SimpleNamespace(input_ids=list(range(10, 10 + len(text.split()))))
    token_ids = dict(bos_id=1, boi_id=2, img_pad_id=3, eoi_id=4, eos_id=5, pad_id=0)
    samples = {"map_name": ["cs_agency"], "instruction": ["map x y z pitch yaw"], "image_token_count": [730]}
    tokens, positions, masks, _ = build_interleaved_inputs(samples, tokenizer, token_ids, device=torch.device("cpu"), max_prompt_tokens=256)
    assert positions[0, :, 1].tolist() == [730, 730]
    assert int(masks.sum()) == 730
    with pytest.raises(ValueError, match="refusing truncation"):
        build_interleaved_inputs(samples, tokenizer, token_ids, device=torch.device("cpu"), max_prompt_tokens=2)


def test_seed_and_settings_batch_independent():
    assert sample_identity_seed(42, "map/a") == sample_identity_seed(42, "map/a")
    assert sample_identity_seed(42, "map/a") != sample_identity_seed(42, "map/b")
    config = OmegaConf.load(CONFIG)
    assert _inference_settings(config, 1) == _inference_settings(config, 16)


def test_binding_rejects_checkpoint_mix(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "backbone_trainable.safetensors").write_bytes(b"weights")
    (checkpoint / "finetuning.json").write_text("{}")
    output = tmp_path / "prediction"
    bind_prediction_checkpoint(output, checkpoint)
    bind_prediction_checkpoint(output, checkpoint)
    (checkpoint / "backbone_trainable.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="different checkpoint"):
        bind_prediction_checkpoint(output, checkpoint)


def test_corrupt_jpeg_not_preserved(tmp_path):
    path = tmp_path / "frame.jpg"
    Image.new("RGB", (448, 448)).save(path)
    assert is_valid_output(path)
    path.write_bytes(path.read_bytes()[:700])
    assert not is_valid_output(path)


def test_target_only_euler_keeps_radar_and_matches_paired_integration():
    from transport import Sampler, create_transport
    class Model:
        def eval(self):
            return self
        def t2i_generate(self, latents, times, **kwargs):
            assert torch.equal(latents[::2], torch.full_like(latents[::2], 0.25))
            velocity = torch.zeros_like(latents)
            velocity[1::2] = latents[1::2] * 0.1
            return velocity
    class VAE:
        def __init__(self):
            self.chunks = []
        def batch_decode(self, latents):
            self.chunks.append(len(latents))
            return latents[:, :3]
    sampler = Sampler(create_transport(path_type="Linear", prediction="velocity", do_shift=False, seq_len=730))
    model, vae = Model(), VAE()
    def generate():
        return generate_batch(model, vae, sampler, {}, None, {}, device=torch.device("cpu"),
            weight_dtype=torch.float32, max_seq_length=8, num_inference_steps=4,
            sampling_method="euler", atol=1e-6, rtol=1e-3, time_shifting_factor=3.0,
            guidance_scale=0, generators=[torch.Generator().manual_seed(i) for i in (42, 43)],
            radar_latents=torch.full((2, 16, 2, 2), .25),
            model_inputs={"text_tokens": torch.zeros((2, 8), dtype=torch.long)}, vae_batch_size=1)
    paired = generate()
    model.conditioning_mode = "text"
    targets = generate()
    assert [i.tobytes() for i in paired] == [i.tobytes() for i in targets]
    assert vae.chunks == [1, 1, 1, 1]


class _TinyAlignedModel(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.conditioning_mode = "text"
        self.backbone = torch.nn.Module()
        self.backbone.aligned_finetuning_policy = policy
        self.backbone.embedding = torch.nn.Embedding(3, 2)
        self.backbone.head = torch.nn.Linear(2, 3, bias=True)
        self.backbone.head.weight = torch.nn.Parameter(self.backbone.embedding.weight.data)
        self.backbone.norm = torch.nn.LayerNorm(2)
        self.backbone.adapter = torch.nn.Linear(2, 2, bias=False)
        self.backbone.adapter.weight.requires_grad_(False)


@pytest.mark.parametrize("policy", ["aligned_v1", "aligned_v2_final"])
def test_finetuning_checkpoint_roundtrip_and_policy_rejection(tmp_path, monkeypatch, policy):
    model = _TinyAlignedModel(policy)
    expected = {
        name: parameter.detach().clone()
        for name, parameter in model.backbone.named_parameters() if parameter.requires_grad
    }
    monkeypatch.setattr(runtime_module, "_expected_trainable_parameters", lambda _: sum(p.numel() for p in expected.values()))
    runtime_module.save_finetune_weights(model, tmp_path)
    metadata_path = tmp_path / "finetuning.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["policy"] == policy
    assert metadata["bias"] == "none" and metadata["lora_bias"] is False
    with torch.no_grad():
        for parameter in model.backbone.parameters():
            if parameter.requires_grad:
                parameter.zero_()
    wrong = "aligned_v1" if policy == "aligned_v2_final" else "aligned_v2_final"
    model.backbone.aligned_finetuning_policy = wrong
    with pytest.raises(ValueError, match="finetuning policy"):
        runtime_module.load_finetune_weights(model, tmp_path)
    assert all(torch.count_nonzero(p) == 0 for p in model.backbone.parameters() if p.requires_grad)
    model.backbone.aligned_finetuning_policy = policy
    runtime_module.load_finetune_weights(model, tmp_path)
    assert all(torch.equal(dict(model.backbone.named_parameters())[name], value) for name, value in expected.items())

    if policy == "aligned_v1":
        for key in ("policy", "bias", "lora_bias"):
            metadata.pop(key)
        metadata_path.write_text(json.dumps(metadata))
        runtime_module.load_finetune_weights(model, tmp_path)
        model.backbone.aligned_finetuning_policy = "aligned_v2_final"
        with pytest.raises(ValueError, match="finetuning policy"):
            runtime_module.load_finetune_weights(model, tmp_path)


def test_inference_manifest_identifies_only_final_policy(tmp_path, monkeypatch):
    showo = tmp_path / "showo"
    showo.mkdir()
    (showo / "pytorch_model.bin").write_bytes(b"showo")
    vae = tmp_path / "vae.bin"
    vae.write_bytes(b"vae")
    config = tmp_path / "config.yaml"
    config.write_text("experiment: aligned\n")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "backbone_trainable.safetensors").write_bytes(b"weights")
    metadata_path = checkpoint / "finetuning.json"
    monkeypatch.setattr(runtime_module, "OFFICIAL_SHOWO_SIZE", 5)
    monkeypatch.setattr(runtime_module, "OFFICIAL_VAE_SIZE", 3)
    monkeypatch.setattr(runtime_module, "OFFICIAL_SHOWO_SHA256", "test-hash")
    monkeypatch.setattr(runtime_module, "OFFICIAL_VAE_SHA256", "test-hash")
    monkeypatch.setattr(runtime_module, "_asset_sha256", lambda _: "test-hash")
    class FakeDataset:
        conditioning_mode = "text"
        rows = [{"map_name": MAPS[0]}]
        provenance = {"benchmark_manifest_sha256": "manifest"}

        def __len__(self):
            return len(self.rows)

    dataset = FakeDataset()
    options = dict(project_root=tmp_path, config_path=config, inference_settings={},
                   showo_path=showo, vae_path=vae, checkpoint_dir=checkpoint,
                   split="continuous", dataset=dataset, train_seed=42,
                   inference_seed=42, smoke_only=False)
    metadata_path.write_text(json.dumps({"experiment": "csgo_seen10_exp32gen_aligned"}))
    v1_path = runtime_module.write_inference_manifest(tmp_path / "v1", **options)
    assert "finetuning_policy" not in json.loads(v1_path.read_text())
    runtime_module.write_inference_manifest(tmp_path / "v1", **options)
    metadata_path.write_text(json.dumps({"policy": "aligned_v2_final"}))
    final_path = runtime_module.write_inference_manifest(tmp_path / "final", **options)
    assert json.loads(final_path.read_text())["finetuning_policy"] == "aligned_v2_final"
