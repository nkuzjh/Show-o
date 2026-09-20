# Show-o2-1.5B on CSGO Benchmark v2 Seen-10

This integration covers the two GENERATION tasks only: discrete generation and
manifest-ordered continuous generation. The dataset adapter consumes the
published report, benchmark manifest, split files, radar mappings, and Z
calibration. Training uses the official Show-o2 interleaved path and Wan2.1 VAE;
the numeric normalized 5DoF pose and map identity enter through a learned
radar-latent FiLM adapter. Inference uses `include_target=False` and reads no
target or neighboring frames.

## Environment and commands

Create the project-local environment and materialize the pinned official
assets:

```bash
bash scripts/setup_csgo_seen10.sh
```

The verified project environment uses Python 3.13.11, host PyTorch
2.12.0+cu130, torchvision 0.27.0, transformers 4.47.0, diffusers 0.31.0,
timm 1.0.12, and torchdiffeq 0.2.5. Model training and inference run only in
`show-o2/.venv`; metrics run with
`/home/jiahao/miniconda3/envs/UniLIP/bin/python`.

Run the one-step forward/backward, validation, resume/checkpoint, one-image
generation, and shared-evaluator smoke checks:

```bash
bash scripts/run_csgo_seen10.sh smoke --seed 0
```

The formal training/inference/evaluation workflow is:

```bash
bash scripts/run_csgo_seen10.sh train --seed 0
bash scripts/run_csgo_seen10.sh infer --seed 0 --checkpoint best --task all
bash scripts/run_csgo_seen10.sh eval --seed 0 --task all
```

The unified runner also accepts `--task discrete` or `--task continuous` and
`--seed N`. `UNILIP_PYTHON` (or `EVAL_PYTHON`), `SHARED_EVAL_DIR`, and
`DATA_ROOT` may be overridden in the environment. The inference RNG defaults
to seed 42; the training seed selects the output directory.

### Direct entry points

The commands wrapped by `run_csgo_seen10.sh` are, from the project root:

```bash
cd /home/jiahao/task/Show-o/show-o2

.venv/bin/python train_seen10.py \
  --config configs/showo2_1.5b_csgo_seen10.yaml \
  --seed 0 \
  --data-root /home/jiahao/task/UniLIP/data/csgo_benchmark_v2 \
  --output-dir /home/jiahao/task/Show-o/outputs/csgo_benchmark_v2_seen10/Show-o2-1.5B/seed_0

.venv/bin/python infer_seen10.py \
  --config configs/showo2_1.5b_csgo_seen10.yaml \
  --seed 0 \
  --data-root /home/jiahao/task/UniLIP/data/csgo_benchmark_v2 \
  --output-root /home/jiahao/task/Show-o/outputs/csgo_benchmark_v2_seen10/Show-o2-1.5B/seed_0 \
  --checkpoint best \
  --task all
```

The corresponding direct shared-evaluator commands are:

```bash
/home/jiahao/miniconda3/envs/UniLIP/bin/python \
  /home/jiahao/task/csgo_benchmark_v2_eval_general/run_eval.py discrete \
  --config /home/jiahao/task/csgo_benchmark_v2_eval_general/benchmark_v2.yaml \
  --pred-root /home/jiahao/task/Show-o/outputs/csgo_benchmark_v2_seen10/Show-o2-1.5B/seed_0/discrete/gen_imgs \
  --data-root /home/jiahao/task/UniLIP/data/csgo_benchmark_v2 \
  --output /home/jiahao/task/Show-o/outputs/csgo_benchmark_v2_seen10/Show-o2-1.5B/seed_0/evaluation_shared/discrete

/home/jiahao/miniconda3/envs/UniLIP/bin/python \
  /home/jiahao/task/csgo_benchmark_v2_eval_general/run_eval.py continuous \
  --config /home/jiahao/task/csgo_benchmark_v2_eval_general/benchmark_v2.yaml \
  --pred-root /home/jiahao/task/Show-o/outputs/csgo_benchmark_v2_seen10/Show-o2-1.5B/seed_0/continuous/gen_imgs \
  --data-root /home/jiahao/task/UniLIP/data/csgo_benchmark_v2 \
  --output /home/jiahao/task/Show-o/outputs/csgo_benchmark_v2_seen10/Show-o2-1.5B/seed_0/evaluation_shared/continuous
```

## Outputs

Training artifacts are written under
`outputs/csgo_benchmark_v2_seen10/Show-o2-1.5B/seed_<seed>/`:

- `checkpoints/step_<step>/` contains resumable Accelerator state and the
  trainable Show-o2/adaptor weights; `checkpoints/late` and `checkpoints/best`
  point to the last and best saved milestones. `checkpoints/latest` is a
  compatibility alias of `late`.
- `loss.jsonl` and `loss_curve.png` record the flow loss.
- `discrete/gen_imgs/<map>/<file_frame>.jpg` and
  `continuous/gen_imgs/<map>/<file_frame>.jpg` contain generated images.
- Each task root contains an `inference_manifest.json` recording the official
  asset/checkpoint hashes, split, sample count, seeds, and benchmark provenance.
- Formal evaluator results are written to `evaluation_shared/<task>/`.

Inference preserves already-valid 448x448 RGB JPEGs, writes the remaining
identities in manifest order, and uses PIL's default JPEG settings to match the
benchmark output encoding. A per-sample RNG seed (`inference_seed + manifest
index`) makes interrupted runs reproducible when valid outputs are skipped. The
smoke run uses a timestamped separate output root and is explicitly marked
`smoke_only`; it is not a formal benchmark run.

The configured formal schedule is 50,000 optimizer steps with validation and
checkpoint milestones at 10,000-step intervals. Full generation coverage is
20,000 discrete and 12,800 continuous images.

## Verified smoke

The documented smoke command completed with exit status 0 on 2026-09-20
(Asia/Hong_Kong). Its isolated output root is:

```text
/home/jiahao/task/Show-o/outputs/csgo_benchmark_v2_seen10/Show-o2-1.5B/seed_0/smoke_runs/20260919T204819Z
```

The real GPU run produced training flow loss `1.2272881269454956` and repeated
fixed-seed validation flow loss `1.2011971473693848`. It saved resumable
Accelerator state plus the trainable backbone and Radar adapter; `late`,
`latest`, and `best` all resolve to `step_000001`. Inference then reloaded that
checkpoint and wrote one discrete and one continuous JPEG. Both files were
independently verified as RGB, 448x448 JPEGs, and both task manifests record
`generation_complete=true`, `smoke_only=true`, and the same resolved checkpoint.

The shared evaluator consumed both outputs successfully with one-of-one
coverage. It reported `formal=false` and wrote no official metric artifact, as
required for smoke mode. A second inference pass reported `generated=0` and
`preserved_valid=1` for both tasks; image modification times, sizes, and SHA-256
digests remained unchanged. Because `RUN_FULL=0`, no 50,000-step training,
complete 20,000/12,800-image generation, or Table 1 metric result was run or
claimed.
