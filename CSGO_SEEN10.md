# Show-o2 接入 CSGO Benchmark v2 Seen-10

本文统一维护已经实现的运行行为、环境权重、手动命令、输出和带日期的状态快照。命令从各服务器的Show-o仓库根执行（本地 `/home/jiahao/task/Show-o`，用户远程 `~/task/Show-o`）；实际模型工作目录是嵌套的 `show-o2/`。仅涉及 **radar/map + 当前5DoF pose → FPV**，不增加定位或时序任务，不包含 CrossMap-4。

文档分工参考 ControlAR、OmniGen2，但模型、配置与验收结论只依据 Show-o2：

- 本文：日常操作与当前状态的统一入口。
- [CSGO_SEEN10_PLAN.md](CSGO_SEEN10_PLAN.md)：设计依据、官方/exp32_gen 对比、训练/推理数据流图、模块策略、实现文件分工与验收标准。
- [CSGO_ALIGNED_VALIDATION.md](CSGO_ALIGNED_VALIDATION.md)：分版本、分日期保存的实施验收证据与未测范围，不是实时训练日志。
- [CSGO_ALIGNED.md](CSGO_ALIGNED.md)：原 aligned 文档的兼容导航，不再重复维护内容。
- [CSGO_BENCHMARK_V2_PLAN.md](CSGO_BENCHMARK_V2_PLAN.md) 与 [csgo_benchmark_v2_start.md](csgo_benchmark_v2_start.md)：首次接入方案/需求历史，不覆盖后续批准的最终方案。

## 1. 实验范围与比较口径

主参考为 UniLIP generation-only `exp32_gen`；joint `exp32` 仅作次要对照。最终 aligned 对齐发布数据、可用条件信息、generation 曝光预算和共享评测规则；432原生尺寸、视觉/文本可训练范围、原生 flow 与采样机制是明确披露的模型差异，不声称逐项复刻 exp32_gen 或 FLOPs 相等。

| 版本 | 入口与配置（配置位于 `show-o2/configs/`） | run root（项目根下） |
| --- | --- | --- |
| legacy 首次接入 | 不加 `--experiment`；`showo2_1.5b_csgo_seen10.yaml` | `outputs/csgo_benchmark_v2_seen10/Show-o2-1.5B/seed_0` |
| 调整前 aligned_v1 | `--experiment csgo_seen10_exp32gen_aligned --finetuning-policy aligned_v1`；`csgo_seen10_exp32gen_aligned_v1.yaml` | `outputs/csgo_benchmark_v2_aligned/csgo_seen10_exp32gen_aligned/seed_42` |
| 最终 aligned_v2_final | `--experiment csgo_seen10_exp32gen_aligned`；`csgo_seen10_exp32gen_aligned.yaml` | `outputs/csgo_benchmark_v2_aligned/csgo_seen10_exp32gen_aligned_v2_final/seed_42` |

| 设置 | legacy 默认单卡 | aligned_v1 | 最终 aligned_v2_final |
| --- | --- | --- | --- |
| pose 注入 | 归一化数值 RadarPoseFiLM + map embedding | 物理数值文本，无额外 adapter | 同 v1 |
| radar / 原生 FPV / 保存尺寸 | 448 / 448 / 448 | 432 / 432 / 448 | 同 v1 |
| 图像处理 | 官方确定性 resize + center crop | 全图 resize，关闭 crop | 同 v1 |
| 主干训练 | 官方 warm-up 冻结名单，生成侧 full，无 LoRA | Qwen/DiT LoRA + 部分生成层 full | Qwen/SigLIP/DiT/两侧 Conv LoRA + 指定 embedding/MLP/输出头整体 full |
| LoRA targets / trainable | 无 LoRA | 276 / 79,445,056 | 434 / 336,481,088 |
| 有效 generation batch | 1 | 128 | 128 |
| optimizer updates / 曝光 | 配置50,000 / 50,000 | 19,500 / 2,496,000 | 同 v1 |
| AdamW / peak LR | β=(0.9,0.999)，wd0，1e-4 | 同左 | 同左 |
| scheduler | warmup500，后 constant | warmup59，cosine 至1e-5 | 同 v1 |
| validation / 保存 | 每10,000步；默认每地图10条验证 | 4000/8000/12000/16000/19500；完整5000条验证 | 同 v1 |
| Euler 时间点 / NFE | 28 / 27 | 50 / 49 | 同 v1 |
| checkpoint 报告 | 旧命令默认 best | late/final 主结果，best 补充 | 同 v1 |

以上区分配置与实际完成状态（第8节）。三个版本不能交叉恢复；最终版必须从官方原始权重开始，不从历史 CSGO checkpoint 初始化。没有 policy 字段的历史 aligned 配置按 v1 解释，不静默升级。

## 2. 数据、条件与输出边界

默认数据根：`/home/jiahao/task/UniLIP/data/csgo_benchmark_v2`。读取发布的 report、manifest、selection、split、radar 映射和 frozen exact calibration，不扫描图片目录重构划分。

固定地图：`cs_agency`、`cs_italy`、`de_ancient`、`de_anubis`、`de_dust2`、`de_inferno`、`de_mirage`、`de_nuke`、`de_overpass`、`de_train`。

| 发布 split | 数量 | target FPV |
| --- | ---: | --- |
| seen_train | 50,000 | 训练可读取 |
| seen_validation | 5,000 | 验证可读取 |
| seen_discrete_test | 20,000 | 推理禁止读取 |
| seen_continuous | 12,800 = 200 clips × 64 frames | 推理禁止读取 |

aligned 将 radar 输入原生 Wan VAE/双支路视觉路径，任务文本包含 map、x/y一位小数、z三位小数、pitch/yaw转度后一位小数与发布的地图z范围。完整模板见 [data.py](show-o2/csgo_seen10/data.py)。不实例化数值 pose 模块；256 token上限超出即报错，不截断条件。历史全四split的87,800条指令检查为113–119 tokens。

legacy 数值条件保持原义：`[x/1024, y/1024, (z-z_min)/(z_max-z_min), pitch/(2π), yaw/(2π)]`，z范围来自发布的逐地图 frozen calibration。

aligned 采用RGB、确定性全图 bicubic resize432和原生归一化；无随机 crop、flip、颜色扰动或擦除。官方中心 crop 因会改变 FPV 视野而关闭。原生图像latent为16×54×54，每图729个空间patch tokens与1个time token；最终解码432再整图resize448。native mixed-modal 条件保留，CFG条件dropout=0、guidance=0。

两类推理均为 `CSGOSeen10Dataset(include_target=False)`（等价于 `load_target=False`），不打开测试target、历史/邻近/未来真实FPV，也不使用前一生成帧。连续集逐帧独立，保留原始clip/frame身份与顺序。每个condition仅一张448×448 RGB JPEG，`gen_imgs/<map>/<file_frame>.jpg`，Pillow默认JPEG编码；无best-of-N或基于GT挑样本。

## 3. 最终 aligned 训练行为与 checkpoint

### 3.1 模块与优化

最终版为434个精确LoRA注入点：Qwen196、SigLIP156、生成DiT80、Conv2d2。统一r32、alpha64、dropout0.05、bias=none、lora_bias=false；不使用DoRA/rsLoRA。Transformer内部MLP/AdaLN仍属Transformer，不能单独全训。

全量训练文本/位置Embedding、fusion整个模块、时间MLP/投影、diff_proj与整个diffusion_head_b，含这些模块自己的Norm/bias。VAE、Transformer原始weight/bias/Norm、Qwen末端Norm和未使用lm_head冻结。总trainable336,481,088，占注入后模型3,136,184,544的10.73%（不含外置VAE）。精确前缀、参数分组和五种配置对照统一见 [方案文档](CSGO_SEEN10_PLAN.md)。

所有可训练组AdamW peak LR1e-4、β=(0.9,0.999)、wd0、eps1e-8，clip norm1；59个optimizer updates warmup，cosine最低1e-5。训练用FP32可训练参数/BF16计算、TF32；严格确定性backend写入恢复合同，不开启torch.compile。完整 `parameter_audit.json` 核对每个参数及optimizer收录，冻结参数不得进入optimizer。

### 3.2 预算与多卡

```text
world_size × micro_batch_per_device × gradient_accumulation = 128
默认单卡：1 × 8 × 16；默认2卡：2 × 8 × 8；默认4卡：4 × 8 × 4
其他正整数拆分合法，例如 1 × 4 × 32、2 × 16 × 4
19,500 updates × 128 = 2,496,000 次generation曝光
2,496,000 / 50,000 = 49.92 个完整数据epoch
```

默认world取可见GPU数，micro8，累计自动按128/(world×micro)计算；只校验正整数及乘积128，不分别固定micro/累计。不按batch调整LR；CFG、两个图像槽和tokens不重复计样本。

全局确定性source stream跨epoch续接，不在epoch尾执行不足128的update，也不丢尾或padding样本。DataLoader预取与实际消费cursor分离。Accelerator负责loss累计缩放，optimizer/scheduler/global step均只在完整更新边界前进。`max_optimizer_steps=19500`是权威终止条件。

### 3.3 保存、选点与恢复

正式只在 **4000、8000、12000、16000、19500** 保存，每次完整5,000条validation计算本模型原生velocity flow loss。按rank无padding分片、每样本固定噪声，并还原训练RNG。

- `best`：这些验证点中loss最小者；相同loss保留较早者。
- `late`：正式step19500结束checkpoint，主比较结果。
- `latest`：最近完整保存点，用于恢复，不代表最新日志step。

checkpoint在梯度累计边界0原子落盘，包含模型LoRA/full、optimizer、scheduler、AMP状态（BF16无scaler）、各rank RNG、global optimizer step、累计源样本及sampler epoch/cursor。保存完成才更新别名；本项目不使用参考项目的 `COMPLETE` 标记约定。

新训练拒绝非空run root；恢复核对policy、官方基础权重、数据/config、参数审计和backend。相同world/micro/累计是同布局精确恢复前提；改变拆分但保持128可以边界恢复source stream，显式标记非bitwise等价，不承诺跨设备/依赖环境逐位一致。

legacy别名语义不同：`late`/`latest`每次保存更新，`best`按旧验证子集选取；其checkpoint不与aligned互通。外部模型best对UniLIP final不能声称严格同选点规则。

## 4. 本地 / 远程初始化、官方权重与路径

按所在服务器选择第4.2或第4.3的分步流程（先环境、后权重），也可以直接选择第4.4的一键合并命令；两种流程任选其一，不需要重复执行。必要的版本、导入和权重完整性校验由脚本内部完成，不需要另行执行检查或验证命令。初始化不会启动训练、推理或评测。

### 4.1 为什么采用两套环境

2026-09-28核实的本地环境是 **RTX PRO 6000 Blackwell Server Edition（计算能力12.0）/ 驱动580.173.02 / Python3.13.11**；用户远程计算节点是 **4×A100 80GB / 驱动550.54.14 / Python3.13.5**。远程登录节点 `cluster-master` 没有 `nvidia-smi`，不能据此判定计算节点无GPU。

| 配置 | 本地 local | 远程 remote |
| --- | --- | --- |
| Python实测版本 | 3.13.11 | 3.13.5 |
| PyTorch | 2.12.0+cu130 | 2.6.0+cu124 |
| torchvision | 0.27.0+cu130 | 0.21.0+cu124 |
| GPU适用对象 | 本地Blackwell | 远程A100 |
| 环境方式 | 兼容现有合格宿主继承venv；新建时优先复用匹配宿主，否则隔离安装 | 项目内隔离venv，不依赖或修改Conda base |
| 初始化入口 | `scripts/setup_csgo_seen10_local.sh` | `scripts/setup_csgo_seen10_remote.sh` |

本地现有cu130不能直接给550驱动使用；远程cu124也不作为本地Blackwell的受支持运行配置。PyTorch从2.7/cu128开始提供Blackwell支持，CUDA13.x通常要求580及以上驱动。这里采用两套配置保留已验证的本地环境，不升级任何驱动，也不尝试未经两机验证的折中组合；并非声称理论上不存在其他共同版本。[PyTorch Blackwell说明](https://pytorch.org/blog/pytorch-2-7/)、[NVIDIA兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

两套脚本共享其余模型依赖和同一套权重下载逻辑。本次初始化profile均要求Python3.13，补丁版本不固定，适用于上述3.13.11与3.13.5；其他Python版本不是本次部署目标。远程2.6.0/0.21.0有官方Python3.13 Linux x86_64安装包，无需仅为torch降级现有Python。[官方版本组合](https://pytorch.org/get-started/previous-versions/#v260)、[cu124安装包](https://download.pytorch.org/whl/cu124/torch/)。不同torch/CUDA环境不承诺跨机器逐位恢复；远程完整Show-o2链路仍需独立验收。

核心公共依赖为transformers4.47.0、diffusers0.31.0、timm1.0.12、torchdiffeq0.2.5、peft0.18.1，完整清单见 [requirements-csgo-seen10.txt](show-o2/requirements-csgo-seen10.txt)。[local约束](show-o2/constraints-csgo-seen10-local.txt)和[remote约束](show-o2/constraints-csgo-seen10-remote.txt)固定各自torch/torchvision，防止安装其他依赖时切换CUDA组合。

### 4.2 本地服务器：分步准备

本地Blackwell服务器，先只配置环境：

```bash
cd /home/jiahao/task/Show-o
bash scripts/setup_csgo_seen10_local.sh --env-only
```

环境准备成功后，单独下载/校验权重：

```bash
bash scripts/setup_csgo_seen10_local.sh --assets-only
```

复用已符合版本的本地环境；新建时优先继承匹配的宿主torch，否则在项目内隔离安装。脚本不修改系统Python、Conda base或驱动。

### 4.3 远程服务器：分步准备

先将此次脚本变更同步到远程仓库，再只配置环境；不要照抄原机器的 `/home/jiahao/...`：

```bash
cd ~/task/Show-o
bash scripts/setup_csgo_seen10_remote.sh --env-only
```

环境准备成功后，单独下载/校验权重：

```bash
bash scripts/setup_csgo_seen10_remote.sh --assets-only
```

远程创建 `show-o2/.venv`，在其中安装固定的cu124 torch/torchvision及共用依赖；不会向Conda base安装torch，也不要求登录节点具有GPU。请不要将本地 `.venv` 复制到远程或反向复制。

之前报错留下的空继承环境也使用上面的环境配置命令（或第4.4合并命令），无需修复参数：脚本只对经内部检查确认仅含pip/setuptools/wheel等初始化组件、没有torch的未完成环境，自动先备份再重建。备份保留为 `show-o2/.venv.incomplete-backup-<时间>-<PID>-<序号>`，实际路径会打印，不自动删除。存在用户包、已安装的错误torch版本、异常目录或符号链接时仍拒绝自动替换。已有checkpoint、下载缓存和实验输出不移动、不删除。

两个初始化入口都使用本机仓库下的 `show-o2/.venv`，不是同一个目录内并存两套环境。不得在同一checkout里交替执行local/remote来切换已安装环境，更不能让正在运行的任务使用的环境被覆盖。

### 4.4 可选：一键配置环境并下载权重

不想分步时，使用以下对应服务器的合并命令，替代第4.2/4.3的两步，不需要再单独下载。

本地服务器：

```bash
cd /home/jiahao/task/Show-o
bash scripts/setup_csgo_seen10_local.sh
```

远程服务器：

```bash
cd ~/task/Show-o
bash scripts/setup_csgo_seen10_remote.sh
```

两套入口的无参模式均为“环境＋权重”；`--env-only`只准备环境，`--assets-only`只下载/校验资产且要求环境已准备好，不安装依赖或修复环境。权重相同，下载逻辑共用，但命令仍需选择本机的local/remote入口以匹配环境。

### 4.5 官方权重与运行路径

分步下载和一键合并准备的资产完全相同。已满足的依赖和校验通过的大权重复用；下载中断后可重跑对应下载命令或合并命令。

| 本地资产（均位于 `show-o2/checkpoints/`） | 来源/固定方式 |
| --- | --- |
| `show-o2-1.5B/pytorch_model.bin` 与config | 官方revision `07ec16589d4fc5422a74dddbbc4b2cd11e551039`；权重5,661,862,314 bytes，固定SHA256 |
| `Wan2.1_VAE.pth` | 官方Wan2.1 VAE，507,609,880 bytes，固定SHA256（下载URL使用main） |
| `Qwen2.5-1.5B-Instruct/` | 仅config/tokenizer，不额外下载完整Qwen权重；获取使用main |
| `siglip-so400m-patch14-384/` | 仅config/preprocessor；视觉权重来自Show-o2 checkpoint；获取使用main |

权重SHA与校验逻辑见 [setup脚本](scripts/setup_csgo_seen10.sh)。两大权重合计约6.17GB，另有配置/tokenizer与下载缓存；不能把所有小文件都称作已固定不可变revision。已存在但不匹配的文件拒绝覆盖。配置相对路径由模型工作目录解析，不从CSGO训练产物初始化。

统一runner当前路径规则：

| 用途 | 实际选择 |
| --- | --- |
| 训练/推理Python | 固定 `show-o2/.venv/bin/python` |
| 数据 | 环境变量 `DATA_ROOT`，否则 `/home/jiahao/task/UniLIP/data/csgo_benchmark_v2` |
| 共享评测器 | `SHARED_EVAL_DIR`，否则 `/home/jiahao/task/csgo_benchmark_v2_eval_general` |
| 评测Python | `EVAL_PYTHON` > `UNILIP_PYTHON` > `/home/jiahao/miniconda3/envs/UniLIP/bin/python` |
| 旧配置覆盖 | `CONFIG`用于legacy；显式aligned选择其canonical配置 |
| run目录覆盖 | `--output-root PATH`是完整seed run root，不再自动追加seed |

远程数据/共享评测器另行部署；环境与权重准备成功不表示这些路径已准备好。下例按远程常见 `~/task` 布局设置，若实际不同请替换；评测环境按共享评测器自己的README准备：

```bash
export DATA_ROOT="$HOME/task/UniLIP/data/csgo_benchmark_v2"
export SHARED_EVAL_DIR="$HOME/task/csgo_benchmark_v2_eval_general"
export EVAL_PYTHON="$SHARED_EVAL_DIR/.venv/bin/python"
```

本次只修改初始化，不改runner的路径优先级、不修改aligned YAML、预算/LoRA/采样策略或checkpoint合同。脚本不会把新机器的环境差异伪装成原机器的精确复现；旧run恢复若被身份校验拒绝，不可改写metadata绕过。

## 5. 手动训练、恢复、推理与评测

以下命令由用户按需手动执行，不要求先运行额外的检查或smoke命令。训练与恢复二选一；训练结束后再推理和评测。推理、评测默认同时处理离散和连续两个任务。现有任务运行时，不再启动写入同一目录的进程。

### 5.1 最终 aligned_v2_final

```bash
cd ~/task/Show-o  # 两台服务器都使用各自实际仓库路径

# 首次训练：默认全部可见GPU，micro8，累计自动使有效batch=128。
bash scripts/run_csgo_seen10.sh train --experiment csgo_seen10_exp32gen_aligned

# 同一run的最近完整checkpoint恢复；保持原GPU布局与batch拆分。
bash scripts/run_csgo_seen10.sh train --experiment csgo_seen10_exp32gen_aligned --resume latest
```

正式结束后，同一冻结late生成两个测试集。以下省略值均来自当前runner/config：checkpoint=late、task=all、推理seed42、batch16；训练seed默认并限定42：

```bash
bash scripts/run_csgo_seen10.sh infer --experiment csgo_seen10_exp32gen_aligned
bash scripts/run_csgo_seen10.sh eval --experiment csgo_seen10_exp32gen_aligned

# best仅作补充，独立预测/评测目录。
bash scripts/run_csgo_seen10.sh infer --experiment csgo_seen10_exp32gen_aligned --checkpoint best
bash scripts/run_csgo_seen10.sh eval --experiment csgo_seen10_exp32gen_aligned --checkpoint best
```

runner对aligned默认选择late，legacy仍默认best。无需OmniGen2那样的convert步骤，直接加载当前policy的LoRA/full权重。若仅运行一个测试集，可指定 `--task discrete` 或 `--task continuous`；设备和batch拆分仍可覆盖，默认行为不变。

### 5.2 保留 aligned_v1

所有对应命令增加 `--finetuning-policy aligned_v1`；train、resume、infer、eval必须始终选择同一个版本。例如：

```bash
bash scripts/run_csgo_seen10.sh train --experiment csgo_seen10_exp32gen_aligned --finetuning-policy aligned_v1
bash scripts/run_csgo_seen10.sh train --experiment csgo_seen10_exp32gen_aligned --finetuning-policy aligned_v1 --resume latest
bash scripts/run_csgo_seen10.sh infer --experiment csgo_seen10_exp32gen_aligned --finetuning-policy aligned_v1
bash scripts/run_csgo_seen10.sh eval --experiment csgo_seen10_exp32gen_aligned --finetuning-policy aligned_v1
```

这些是保留旧实验的操作入口，不是最终方案的推荐新训练命令。

### 5.3 legacy：原命令继续有效

历史seed0已训练结束且存在未完成预测，下面是复现/恢复入口，不要直接对已完成run重启训练，也不要把它作为aligned结果：

```bash
bash scripts/run_csgo_seen10.sh train
bash scripts/run_csgo_seen10.sh train --resume latest
bash scripts/run_csgo_seen10.sh infer
bash scripts/run_csgo_seen10.sh eval
```

不传experiment继续使用numeric-FiLM、seed0和旧目录；推理默认batch16，`--batch-size 8`等覆盖仍兼容。legacy的best/late不分预测根，不可混用两个checkpoint续写同一预测目录。

直接Python入口仍为 `show-o2/train_seen10.py` 与 `show-o2/infer_seen10.py`。日常只需使用以上runner命令，不需要手动激活venv或维护第二份直接Python命令。旧初始化入口默认local以及原有检查参数继续兼容，但不作为日常操作步骤。

## 6. 推理加速、输出隔离与共享评测

### 6.1 无编译的四级加速

| 优先级 | 实现 | 保留的语义 |
| --- | --- | --- |
| 1 | 原生批量，默认16；`--batch-size`覆盖 | 每个样本成对radar/target，只去噪各自target |
| 2 | 地图静态radar预处理与VAE latent缓存 | aligned文本pose逐样本构造；legacy可缓存地图文本/序列模板 |
| 3 | 跳过无用词表logits/中间状态，显式Euler只留最终状态 | 保留原时间网格；aligned50点/49NFE、shift3、CFG0 |
| 4 | aligned分块VAE decode（配置chunk1）、JPEG完整性检查、原子保存与可续跑跳过 | 不覆盖有效图片，不混入其他checkpoint |

不启用compile，不减少采样步数，不提供未经批准的20NFE主配置。legacy使用28点/27NFE，VAE批量解码；不能用ControlAR compiled batch16约9小时的历史数据推算Show-o2速度。

aligned随机流由seed+sample_id的SHA256派生；legacy为seed+split manifest index。批量/恢复不改变每样本初始噪声，但不同batch kernel的BF16误差不保证图片字节相同。aligned允许同一实验改变执行batch，legacy会把batch写入推理身份合同并拒绝在同一目录混用。

续跑按稳定的map-local manifest块处理：块中存在缺图就重算该块，但只写缺失/损坏输出，保留有效JPEG。只缓存静态radar，不缓存包含样本pose的aligned完整条件。

### 6.2 输出结构与身份

最终版run下：

```text
seed_42/
  config.yaml, parameter_audit.json, loss.jsonl
  checkpoints/
    step_004000/ ... step_019500/
    best -> validation loss最小者
    late -> step_019500
    latest -> 最近完整保存
  predictions/{late,best}/
    checkpoint_binding.json
    {discrete,continuous}/
      inference_manifest.json
      sample_manifest.json
      gen_imgs/<map>/<file_frame>.jpg
  evaluations/{late,best}/{discrete,continuous}/
```

调整前v1目录结构相同、run root不同。legacy使用run根下的 `discrete/`、`continuous/` 和 `evaluation_shared/<task>/`，其旧checkpoint另含Radar adapter。

aligned两split绑定同一checkpoint内容hash；policy/config、数据manifest/selection/calibration/split、seed或checkpoint不一致即拒绝续写。已有JPEG完整解码并核对448×448 RGB，损坏图只能由同一实验重建。`--output-root`不能用于绕过身份检查；不改写旧manifest来混入新模型图片。

### 6.3 唯一共享评测器

仅调用 `/home/jiahao/task/csgo_benchmark_v2_eval_general/run_eval.py` 和该目录 `benchmark_v2.yaml`，不在Show-o2复制/修改指标实现。

- 离散：PSNR、SSIM、LPIPS、Boundary_F1、FID。
- 连续：PSNR、SSIM、LPIPS、TWE、TDE、FVD。
- 共享配置：equal-map macro，clip length16、stride16、FVD size224，frame-difference threshold2、min_track_len4等tracking设置由共享评测器维护。

先完成20,000/12,800图片及identity/coverage检查，再运行正式eval。少量smoke及连续frame-only不等于FID/FVD或时序指标验收，也不构成正式结果。

## 7. 验收范围说明（不是额外操作步骤）

本文只保留日常初始化、训练、恢复、推理和评测命令，不要求用户另行执行检查或smoke。版本/导入/权重校验已内置初始化；数据、checkpoint身份、输出完整性等保护仍由各执行流程内部实施，不因精简文档而关闭。

完整模型验收标准仍见 [方案文档](CSGO_SEEN10_PLAN.md)，已测与未测证据见 [验收记录](CSGO_ALIGNED_VALIDATION.md)。最终版全尺寸GPU显存/吞吐、多卡训练及同布局精确恢复仍未全部实测；简化操作不等于这些检查已经通过，也不把初始化成功当作远程A100端到端模型验收。

原有隔离smoke能力继续保留，但不加入日常执行链，也不由初始化自动触发。`RUN_FORMAL=0`是此前实施阶段不自动启动正式实验的边界，不是runner选项；正式训练/全量生成/评测仍由用户手动执行。

## 8. 状态快照与未完成项

只读快照：**2026-09-27 18:55 HKT**。不是常驻监控；未来状态以实际日志、checkpoint与manifest为准。

| 版本 | 已有证据 | 不能据此声称的结果 |
| --- | --- | --- |
| legacy seed0 | loss日志到step50000；best→step040000，late/latest→step050000；离散目录有96个JPEG文件，manifest `generation_complete=false` | 非20,000张完整离散结果；无正式continuous目录/共享正式评测结果 |
| aligned_v1 | 2026-09-27真实GPU两步及同布局逐位恢复；离散/连续各2张、共享evaluator小测 | 非正式19500步训练；不是v2验证 |
| aligned_v2_final | 47项CPU回归、真实权重参数审计、缩小图backward、tiny模型精确恢复 | 未做432全尺寸GPU训练/恢复、新版图像生成和evaluator端到端smoke |

legacy证据：[loss.jsonl](outputs/csgo_benchmark_v2_seen10/Show-o2-1.5B/seed_0/loss.jsonl)、[离散manifest](outputs/csgo_benchmark_v2_seen10/Show-o2-1.5B/seed_0/discrete/inference_manifest.json)。96仅为目录文件计数，本次未重新解码全部文件；manifest绑定旧step040000，不是aligned checkpoint。

未发现指向Show-o项目的训练/推理/评测进程；其他项目任务仍在运行，未停止或修改。两个aligned正式默认run根尚未创建。最终版验收摘要来自既有产物，不是本次重新运行测试。

最终版micro8显存、batch16吞吐、全模型GPU精确恢复和完整指标尚待验证，不复用旧v1资源数字给出ETA。后续由用户手动验收与启动；本次仅整理文档，未下载权重、修改实现或触碰已有结果。

本次整理核对了文档本地链接、Bash代码块语法及三版本runner的代表性dry-run；149个代码/脚本/配置/依赖文件整理前后的SHA256完全一致。既有47项模型相关回归记录仍属于前次实施验收，没有在文档整理中重跑。

### 8.1 双服务器初始化增补：2026-09-28

第4节现已提供local/remote两个初始化入口，旧入口无参兼容local。本地现有环境的CPU依赖/版本检查通过；16项初始化mock/CPU测试与6项旧runner回归共22项通过，另检查了Shell语法、文档链接和Bash示例。记录见 [本轮pytest.xml](outputs/setup_env_verification_20260928.AaoZBU/pytest.xml)，范围说明见 [验收记录](CSGO_ALIGNED_VALIDATION.md)。

本轮模型/数据/训练/推理/config与runner的45个受保护文件SHA256未变，公共依赖有效条目、权重下载实现和官方资产revision/URL/大小/SHA也未变。没有安装或重装真实环境、下载大依赖/权重、执行GPU检查或启动正式实验。尚未在远程A100实际运行安装与模型smoke；模拟安装测试不等于远程实机验收。

### 8.2 初始化命令精简：2026-09-28

按后续要求，第4.2/4.3分别列出本地/远程的环境、权重分步命令，第4.4单列可选的一键合并命令；不要求额外的手动检查、修复或smoke步骤。环境配置/合并模式自动识别并备份重建严格为空的失败继承venv；仅下载和只读模式不修复环境。原有高级参数兼容，模型与实验配置不变。

主代理运行34项初始化测试及6项旧runner回归，共40项通过；11个Bash代码块语法、14条运行命令的只解析测试与文档本地链接通过。未真实安装/替换环境、下载权重、执行GPU操作或启动正式实验；远程实机验收仍未完成。详细范围见 [验收记录](CSGO_ALIGNED_VALIDATION.md)。
