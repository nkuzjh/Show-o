# Show-o2 aligned 文档已合并

本文件保留旧链接入口，不再单独维护方案、命令或运行状态。

- [CSGO_SEEN10.md](CSGO_SEEN10.md)：legacy / aligned_v1 / aligned_v2_final 的环境、运行命令、输出隔离和日期化状态。
- [CSGO_SEEN10_PLAN.md](CSGO_SEEN10_PLAN.md)：最终 aligned 配方来源、训练/推理两张数据流图、精确 LoRA/full/frozen 审计、五种配置对照与验收标准。
- [CSGO_ALIGNED_VALIDATION.md](CSGO_ALIGNED_VALIDATION.md)：最终版 CPU 验收与历史 v1 GPU / legacy 验收证据，分版本列出未测范围。

`--experiment csgo_seen10_exp32gen_aligned` 仍默认选择 `aligned_v2_final`；附加 `--finetuning-policy aligned_v1` 使用保留的调整前方案；不传 experiment 仍是首次 numeric-FiLM 接入。文档整理不改变代码、配置、checkpoint、预测或已批准的实验语义。
