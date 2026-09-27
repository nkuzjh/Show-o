"""No-process/no-output regression tests for public launch commands."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_csgo_seen10.sh"
PROFILE = "csgo_seen10_exp32gen_aligned"


def command(*args):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    return subprocess.run(["bash", str(SCRIPT), *args, "--dry-run"], cwd=ROOT,
                          env=env, capture_output=True, text=True)


def test_legacy_default_unchanged():
    result = command("infer")
    assert result.returncode == 0
    assert "--seed 0" in result.stdout and "--checkpoint best" in result.stdout
    assert "showo2_1.5b_csgo_seen10.yaml" in result.stdout
    assert "--prediction-output-root" not in result.stdout


def test_aligned_default_and_multicard_batch():
    result = command("train", "--experiment", PROFILE)
    assert result.returncode == 0
    assert "configs/csgo_seen10_exp32gen_aligned.yaml" in result.stdout
    assert f"/outputs/csgo_benchmark_v2_aligned/{PROFILE}_v2_final/seed_42" in result.stdout
    assert "--seed 42" in result.stdout
    assert "--micro-batch 8 --gradient-accumulation 16" in result.stdout
    result = command("train", "--experiment", PROFILE, "--num-processes", "2")
    assert result.returncode == 0
    assert "torch.distributed.run --standalone --nproc_per_node 2" in result.stdout
    assert "--micro-batch 8 --gradient-accumulation 8" in result.stdout
    assert command("train", "--experiment", PROFILE, "--micro-batch", "4", "--gradient-accumulation", "32").returncode == 0
    assert command("train", "--experiment", PROFILE, "--micro-batch", "4", "--gradient-accumulation", "16").returncode != 0


def test_aligned_prediction_and_eval_isolation():
    result = command("infer", "--experiment", PROFILE)
    assert result.returncode == 0
    assert "--checkpoint late" in result.stdout
    assert "/seed_42/predictions/late" in result.stdout
    result = command("eval", "--experiment", PROFILE, "--checkpoint", "best")
    assert result.returncode == 0
    assert "/predictions/best/continuous/gen_imgs" in result.stdout
    assert "/evaluations/best/continuous" in result.stdout


def test_aligned_smoke_does_not_pollute_formal_root():
    result = command("smoke", "--experiment", PROFILE)
    assert result.returncode == 0
    assert "seed_42_smoke_runs/" in result.stdout
    assert "seed_42/smoke_runs/" not in result.stdout
    assert "--smoke --max-steps 1" in result.stdout
    assert "smoke continuous" in result.stdout and "--frame-only" in result.stdout


def test_explicit_v1_policy_uses_original_root_and_config():
    result = command("train", "--experiment", PROFILE, "--finetuning-policy", "aligned_v1")
    assert result.returncode == 0
    assert "configs/csgo_seen10_exp32gen_aligned_v1.yaml" in result.stdout
    assert f"/outputs/csgo_benchmark_v2_aligned/{PROFILE}/seed_42" in result.stdout
    assert f"/{PROFILE}_v2_final/" not in result.stdout


def test_policy_option_rejects_legacy_and_unknown_values():
    assert command("infer", "--finetuning-policy", "aligned_v1").returncode == 2
    result = command("infer", "--experiment", PROFILE, "--finetuning-policy", "unknown")
    assert result.returncode == 2
    assert "Invalid --finetuning-policy" in result.stderr
