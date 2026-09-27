# Show-o2 CSGO aligned：实施验收记录

本文保存已发生的验收证据，不作为持续训练日志，也不把旧版本结果当作最终版结果。当前运行状态和手动命令见 [CSGO_SEEN10.md](CSGO_SEEN10.md)；最终方案与验收标准见 [CSGO_SEEN10_PLAN.md](CSGO_SEEN10_PLAN.md)。2026-09-27 文档整理只归并既有记录，未重跑下列模型测试。

## 1. 最终版 aligned_v2_final：2026-09-27

- 全仓CPU回归 **47项通过**：精确LoRA target/参数量、Transformer原始参数冻结、完整MLP/Norm/bias解冻、embedding/head共享隔离、视觉LoRA dropout模式、optimizer成员、split/calibration/测试target隔离、source stream/有效batch/CPU双进程DDP、保存恢复、版本拒绝及启动器兼容。JUnit记录：[pytest.xml](outputs/aligned_v2_final_verification_20260927/pytest.xml)。依赖弃用警告不影响结果。
- 加载本地真实官方Show-o2权重进行CPU审计，确认434个targets、890个可训练张量、336,481,088可训练参数、3,136,184,544总参数（10.728995%，不含VAE）。完整逐参数和optimizer清单：[parameter_audit.json](outputs/aligned_v2_final_verification_20260927/parameter_audit.json)。
- 同一真实模型执行缩小空间/序列尺寸的原生flow forward/backward及一次AdamW和scheduler更新；11个可训练组均有有限非零梯度，冻结参数无梯度，验证eval模式关闭训练态：[backward_smoke.json](outputs/aligned_v2_final_verification_20260927/backward_smoke.json)。这是随机latent/label的计算图检查，不是432分辨率训练、有效batch128运行或质量评估。
- 最终策略的tiny模型覆盖全部434个LoRA注入点以及所有全训模块，保存step1的权重、optimizer、scheduler、RNG后，重建模型恢复到step2；loss、所有trainable、frozen参数、optimizer、scheduler和RNG逐位一致。此为CPU精确恢复测试，不等于真实模型GPU的完整恢复验证。
- 调整前v1真实权重CPU审计仍为79,445,056可训练参数：[v1审计](outputs/aligned_v1_compat_cpu_audit_20260927/parameter_audit.json)。按正常加载流程解析LLM绝对路径后，其semantic config digest与历史v1 checkpoint一致。历史audit缺少新增policy/空alias字段可兼容，但真实共享关系、存储量或参数变化仍严格拒绝；CPU与GPU的存储关系可能不同，不能据此承诺跨设备bitwise恢复。
- `bash -n scripts/run_csgo_seen10.sh`、启动器两版本dry-run和 `git diff --check` 通过。

复核真实权重审计（使用一个不存在的新目录，脚本仅使用CPU，不下载权重）：

```bash
show-o2/.venv/bin/python scripts/audit_aligned_parameters.py \
  --output-dir outputs/aligned_v2_cpu_audit_manual --backward-smoke
```

当次实现验收时GPU正被其他项目训练占用，未启动GPU smoke，也未启动正式训练、离散/连续全量推理或评测；未停止或修改其他项目任务。最终版432分辨率GPU显存/吞吐、真实模型GPU恢复，以及新的离散/连续生成与共享evaluator端到端smoke仍需用户手动执行主文档中的隔离smoke，并另外完成同布局2步恢复对照。下面的v1历史GPU结果不作为v2已完成这些检查的证据。

## 2. 历史 aligned_v1：2026-09-27（不能作为 v2 的验收）

该版本的最终验收使用两个独立目录：

- `outputs/aligned_smoke_20260927_deterministic_control`：真实GPU不间断2个optimizer updates。
- `outputs/aligned_smoke_20260927_deterministic_resume`：复制上述完整step1保存点后，从新进程恢复到step2。复制起点先验证逐位相同；没有续写或改动control目录。

每步128条源样本，累计256；scheduler更新2次，保存时accumulation边界为0。最终同布局恢复比较全部通过：568个可训练张量（79,445,056参数）、完整模型文件SHA256、optimizer、scheduler、全部rank RNG、训练状态、loss和源样本顺序均相同。step2 validation flow loss为0.3252403885126114（仅2条smoke验证样本）。结果和完整参数审计分别保存在resume目录的 `resume_verification.json`、`parameter_audit.json`。

可只读复核（RNG/optimizer使用pickle，仅对本地可信checkpoint运行）：

```bash
show-o2/.venv/bin/python scripts/verify_aligned_resume.py \
  outputs/aligned_smoke_20260927_deterministic_control/checkpoints/step_000002 \
  outputs/aligned_smoke_20260927_deterministic_resume/checkpoints/step_000002
```

回归测试37项通过，包括真实双进程CPU/DDP全局batch128梯度等价检查；`bash -n`及`git diff --check`通过。只有一张物理GPU，没有声称完成真实多GPU训练验收。

旧方案文档记录过共享GPU下约244秒/211秒每update、约6.85GiB/checkpoint。仅为v1短测历史，不是独占GPU基准，不能用来预测最终v2的训练时间、显存或磁盘需求。

最终resume checkpoint已用batch2生成离散/连续各2张图片，全部通过完整JPEG解码、448×448和RGB检查；两个split绑定的checkpoint内容hash均为 `8a2e7a9cc7989fe4f946cd8ad194817585f0d1b25377e387238bf384424ba0d6`。预测位于resume目录的 `predictions/late/{discrete,continuous}/gen_imgs`；连续样本保留 `cs_agency_continuous_0000` 的frame0/1身份。共享evaluator离散smoke读取2张成功，连续 `--frame-only` smoke读取1张成功。

随后将batch改为1续跑同一smoke目录：两个split均generated=0、preserved_valid=2，4张JPEG的SHA256全部不变。所有本轮训练、推理、评测smoke进程已正常退出。

上述是链路smoke而非正式质量评估：未运行FID/FVD/TWE/TDE，未验证batch16全量吞吐，不使用smoke测试指标选checkpoint或调整超参。

首轮smoke暴露并修复三类问题：运行时独立embedding/head导致总参数量不同于磁盘共享存储计数；加载器保持eval导致checkpointing不生效；相同RNG下GPU backward非确定性导致Adam结果出现微小偏差。v1当时修正为按实际Parameter审计、显式进入train并保持vision eval（v2视觉LoRA需跟随train/eval，不能沿用这一冻结视觉设置）、启用严格确定性backend并记录恢复约束。此前的 `aligned_smoke_20260927_approval_resume`、`aligned_smoke_20260927_verified_resume`、`aligned_smoke_20260927_verified_uninterrupted` 目录均保留为诊断证据，不视为最终精确恢复验收或正式实验。

本轮保留的诊断与验收checkpoint共约55GiB（四个双checkpoint目录）；未自动删除任何证据。验收时文件系统剩余约407GB。旧实验best/late链接仍分别指向step_040000/step_050000，新正式seed_42目录尚未创建。

未启动正式19500-step训练或全量推理，也未修改/停止其他项目任务。

## 3. 首次接入 legacy：2026-09-20

以下为首次接入当天的历史记录；当时 `RUN_FULL=0`，不表示此后一直未执行正式训练。当前完成度见主文档的日期化快照。

首次接入smoke于2026-09-20（Asia/Hong_Kong）正常退出，隔离输出根为：

```text
/home/jiahao/task/Show-o/outputs/csgo_benchmark_v2_seen10/Show-o2-1.5B/seed_0/smoke_runs/20260919T204819Z
```

真实GPU训练flow loss为`1.2272881269454956`，固定seed重复验证loss为`1.2011971473693848`。保存可恢复Accelerator状态、可训练backbone和Radar adapter，`late`/`latest`/`best`均指向`step_000001`。重新加载后生成离散/连续各一张，独立核验RGB、448×448 JPEG；两个manifest均为`generation_complete=true`、`smoke_only=true`并绑定同一checkpoint。

共享evaluator成功读取两类输出，各为1/1覆盖，报告`formal=false`，不写正式指标产物。再次推理两任务均`generated=0`、`preserved_valid=1`；图片mtime、大小、SHA256不变。当次`RUN_FULL=0`，未执行或声称完成50,000步正式训练、20,000/12,800全量生成或Table 1结果；此后legacy实际完成的训练不倒写为当次smoke成果。
