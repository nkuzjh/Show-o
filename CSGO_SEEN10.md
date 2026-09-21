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

Full inference uses a fixed batch size of 16 by default. The existing command
above remains valid and picks up this default. Set `--batch-size N` to choose a
different inference batch size, for example:

```bash
bash scripts/run_csgo_seen10.sh infer --seed 0 --checkpoint best --task all --batch-size 8
```

This setting affects generation throughput only; training batch size and the
shared evaluator are unchanged. The one-sample smoke path also remains valid.

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
  --task all \
  --batch-size 16
```

`--batch-size` is optional; omitting it uses 16. The same option is available
on the unified runner, while existing inference commands that do not specify
it continue to work.

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

## Inference acceleration (without compilation)

The inference implementation uses the following four optimizations. They
preserve the benchmark inputs, the configured 28-point Euler time grid, and the
training and evaluation protocols. No `torch.compile` path or reduced-step
sampling is used.

1. **Fixed batches (default 16).** The old invocation remains supported; its
   omitted `--batch-size` now selects 16, and `--batch-size N` overrides it.
   Each example is packed as an interleaved `(radar, target)` latent pair. Its
   timesteps are `(1, t_i)`, and only that example's target velocity is kept.
   Generate the initial noise with the existing per-row seed
   `inference_seed + manifest_index`, then stack samples in manifest order.
   Resume uses stable, map-local manifest blocks. If a block contains any
   missing output, the block is recomputed with its original manifest indices,
   but only missing JPEGs are written; valid JPEGs are never overwritten.
   Formal per-map counts are divisible by 16, while a partial/smoke tail uses
   its actual size. Decode generated target latents as a batch. This keeps each
   sample's random input independent of resume state and preserves interrupted
   run behavior.

2. **Skip unused language-model outputs.** The image-only generation path
   consumes the final transformer hidden state for the diffusion head, not
   next-token logits or the full tuple of intermediate hidden states. Avoid
   calculating the full vocabulary logits and retaining intermediate states on
   each denoising evaluation while supplying the same final state to the
   diffusion head.

3. **Cache map-specific inputs.** There are ten fixed radar maps. Encode each
   map's radar once with the deterministic VAE path and reuse its latent for
   every corresponding row. Cache the map prompt tokens and the matching
   modality positions, image mask, and attention-mask template as well. Pose
   conditioning remains per sample.

4. **Use an explicit fixed Euler loop.** Reproduce the configured shifted
   28-point time grid and Euler updates, retaining only the current latent and
   final result instead of the sampler's full trajectory. This reduces
   trajectory storage without changing the number or placement of time points.

The per-row noise seed and manifest ordering are independent of batch size.
Existing valid RGB 448x448 JPEGs are still skipped on resume. As with ordinary
batched inference, changing batch size can produce small floating-point
differences from a separate batch-size-1 run; outputs remain deterministic for
the same run settings and sample seeds. The inference manifest records the
batch size and generation algorithm and rejects mixing outputs produced with
different settings in one output root.

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
