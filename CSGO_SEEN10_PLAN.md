# Show-o2-1.5B 接入 CSGO Benchmark v2 Seen-10

本次只接入 Table 1 中 GENERATION 模型负责的 discrete generation 与
continuous generation。项目参数为 seed 0、`BUILD_SHARED_EVALUATOR=0`、
`RUN_FULL=0`；因此先落地完整正式流程并执行真实数据/GPU smoke，不启动全量训练。

## 最小变更

- `show-o2/csgo_seen10/`：新增 manifest/split 驱动的数据适配、归一化数值
  5DoF + map embedding 的 Radar latent FiLM adapter，以及 Show-o2 公共加载、
  checkpoint 和采样工具。
- `show-o2/configs/showo2_1.5b_csgo_seen10.yaml`：448 分辨率、训练超参数、
  5 个等距 validation/checkpoint 里程碑及正式输出配置。
- `show-o2/train_seen10.py`：复用官方 Wan2.1 VAE、Show-o2 interleaved
  flow-matching forward、AdamW 和 transport；支持恢复、validation 选优、
  `late`/`best` 链接（同时保留 `latest` 兼容别名）、JSONL loss 日志和曲线。
- `show-o2/infer_seen10.py`：从同一冻结 best checkpoint 分别读取
  `seen_discrete_test` 与按 `clip_id`/帧顺序展开的 `seen_continuous`；推理阶段
  只读取 Radar 与 pose，不打开目标/相邻帧，按 manifest identity 保存 448×448
  RGB JPEG。
- `scripts/run_csgo_seen10.sh`：统一提供 `smoke|train|infer|eval` 与 `--seed`，
  共享评测固定调用 `/home/jiahao/task/csgo_benchmark_v2_eval_general/run_eval.py`。
- `scripts/setup_csgo_seen10.sh`、`show-o2/requirements-csgo-seen10.txt`：创建
  Show-o2 独立环境并下载/校验官方 Show-o2-1.5B 与 Wan2.1 VAE 资产。
- `CSGO_SEEN10.md`：记录实际环境、直接命令、checkpoint/结果路径和 smoke 证据。

## 数据与模型合同

数据适配器只读取 `minimal_dataset_report.json`、`benchmark_manifest.json`、
`splits/seen/...`、manifest/report 映射的 `images/`、`radars/` 与发布的
`calibration/z_calibration.json`，不扫描图片树或重划分。固定 map 顺序来自
manifest 并与 Seen-10 合同核对；pose 为
`[x/1024, y/1024, (z-z_min)/(z_max-z_min), pitch/(2*pi), yaw/(2*pi)]`。

每个样本构造成两个 Show-o2 image modality：第一个是 VAE 编码后的 Radar，
由数值 pose MLP 与 map embedding 产生的 FiLM 参数调制并固定在 clean endpoint；
第二个是训练时的目标 FPV noisy latent 或推理时的随机 latent。loss 只覆盖第二个
modality。模型输出经官方 Wan2.1 VAE 解码并保存为标准图片。

## 运行与验收

统一入口为：

```bash
bash scripts/setup_csgo_seen10.sh
bash scripts/run_csgo_seen10.sh smoke --seed 0
bash scripts/run_csgo_seen10.sh train --seed 0
bash scripts/run_csgo_seen10.sh infer --seed 0 --task all
bash scripts/run_csgo_seen10.sh eval --seed 0 --task all
```

smoke 必须实际验证：一个 manifest batch、一次 Show-o2 forward/backward、训练状态
保存和重载、同一 checkpoint 各生成一张 discrete/continuous 448 RGB 图片，以及
共享评测器对两份输出的读取。smoke 输出与正式输出隔离，不能作为 Table 1 结果。
正式训练默认 50,000 optimizer steps，并在 10,000/20,000/30,000/40,000/50,000
step 各 validation/save 一次；完整推理要求 20,000 张 discrete 与 12,800 张
continuous 全覆盖后方可正式评测。
