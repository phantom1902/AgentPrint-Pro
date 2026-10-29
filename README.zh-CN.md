# ANT：流量采集与 risk / scenario 分类实验

[English](README.md) | [简体中文](README.zh-CN.md)

本仓库连接 ANT 的两个阶段：**智能体流量采集**，以及**基于公开固定划分的 risk / scenario 分类实验**。主实验套件以 `Benchmark/data/ANT/{train,val,test}` 为唯一任务数据输入，完成 **risk 二分类、scenario 五分类 × 13 个 baseline** 的预处理、训练、预测、episode 聚合与指标汇总。模型代码取自本项目实际使用的本地适配版本；来源见 [SOURCES.md](SOURCES.md)。

实验部分仍是可独立使用的最小套件，保留原模型模块，不包含 primitive、消融、超参数搜索、旧实验结果、特征缓存、数据和权重。新增的[采集源码](data_collection/README.zh-CN.md)说明上游采集环节，但**收到的采集压缩包不完整，目前无法端到端运行采集**。使用公开数据训练和评估不依赖采集代码。主实验命令均在仓库根目录执行。

项目自有代码采用 **GPL-3.0-only**；单独发布的 ANT 数据及原创标注中作者持有的权利采用 **CC-BY-4.0**。第三方实现保留原许可，未解决的授权问题已列入[许可范围说明](LICENSING.md)和[第三方许可清单](THIRD_PARTY_LICENSES.md)，根许可证不替代这些授权。

## 需要一起公开什么

| 内容 | 套件中的位置 | 用途 |
| --- | --- | --- |
| 采集源码及导入审计 | `data_collection/` | 批量调度、会话导出、任务索引与诊断工具；运行依赖不完整 |
| 公开数据及固定划分 | 从 [Hugging Face](https://huggingface.co/datasets/benzenenene/ANT) 获取并放入 `Benchmark/data/ANT/{train,val,test}` | 只读取 episode 的 `network.pcap` 和目录标签 |
| 标签定义 | `configs/scenario.json`、`suite.py::task_label` | 固定 risk / scenario 标签 |
| 模型及 PCAP 适配代码 | `Benchmark/`、`engines.py` | 从公开 PCAP 生成输入并训练 |
| 主实验配置 | `configs/{risk,scenario}/`、`Benchmark/all_default_configs/` | 模型参数和训练预算 |
| 评估与聚合 | `suite.py`、`Benchmark/evaluation.py` | validation 选聚合规则，test 报告结果 |
| 运行入口 | `suite.py`、`run_all.py` | 单模型或全部 26 个任务 |
| 依赖、来源与验证记录 | `requirements-*.txt`、`SOURCES.md`、`VALIDATION.md` | 安装、审计及已验证范围 |
| 三个起始 checkpoint 的获取方式或文件 | `WEIGHTS.json`、`verify_weights.py` | 需另外提供，代码包不包含权重 |

**本套件支持从公开数据重新训练，不承诺逐位恢复论文表格。** 历史 checkpoint、恢复训练状态和部分参数快照并不完整，已知差异见下方“与历史结果的关系”。

## 数据集发布

ANT 数据集存放于 Hugging Face：[benzenenene/ANT](https://huggingface.co/datasets/benzenenene/ANT)。

该发布账户为本次提交新注册的专用账户，未绑定任何作者身份信息，也没有与本次提交无关的历史提交。

从数据仓库获取已发布的 `train`、`val`、`test` 划分后，放入 `Benchmark/data/ANT/`，保留下文的目录层级。也可以将数据存放在其他位置，通过 `suite.py` 的 `--data` 参数指定实际 ANT 目录。使用发布的固定划分，不重新分割。数据许可及适用范围见 [LICENSING.md](LICENSING.md)。

## 数据约定

保留以下层级，不能只把 PCAP 平铺在三个 split 下：

```text
Benchmark/data/ANT/
├── train/<engine>/<family>/Axxxx/network.pcap
├── val/<engine>/<family>/Axxxx/network.pcap
└── test/<engine>/<family>/Axxxx/network.pcap
```

- risk：`attack_code_generation`、`attack_execution` 为 `malicious`，其他已收录 family 为 `benign`。
- scenario：按 `Axxxx` 的前两位任务编号分为 `artifact_processing`、`general_knowledge`、`scientific_evidence`、`business_services`、`security_testing`。精确任务编号在 `configs/scenario.json`。
- 不读取 session 日志、prompt、私有 annotation 或 flow-layer 标注。
- 使用发布的固定划分，不重新随机分割。不允许重复的 `engine/episode` 身份或同一真实 PCAP 路径跨 split；每个 split 必须覆盖全部类别。此检查不等同于逐包内容去重。
- 本次对现有公开目录检查得到 train **2703**、val **202**、test **209** 个 episode；重新发布数据时以生成的 `dataset_counts.json` 为准。

## 从采集到公开实验数据

根据目的选择入口：

1. **运行主实验：**取得公开的固定划分，按下文配置模型环境并提供所需起始权重，运行 `suite.py`。不需要采集脚本或智能体账号。
2. **检查或扩展采集：**阅读 [data_collection/README.zh-CN.md](data_collection/README.zh-CN.md)，查看已提供的源码、缺失组件、环境边界与诊断方法。采集依赖单独管理，不与模型环境合并。

导入包包含批量调度和会话工具，但原始 521 项文件清单中有 **480 项未找到、13 项对应空脚本**，runner、网络环境工具、8 个 mock 服务和任务辅助代码尚未形成可运行的依赖集合。此次排除了空脚本、私有开发笔记、macOS 元数据及损坏的内部 ZIP。[IMPORT_REPORT.json](data_collection/IMPORT_REPORT.json)记录缺口；新的采集 manifest 只记录本次实际交付的文件。

采集调度器将原始 episode 写入类似 `data_collection/episodes/Agent/<engine>/Axxxx/` 的目录，不能直接当作公开的 train / val / test。原始输出可能包含会话、prompt、日志和工作目录，而分类实验仅读取公开 PCAP 和目录标签。数据发布还需要原始 episode 筛选、family 标签和固定划分信息，才能生成上文约定的层级。本次导入没有提供导出或生成划分的工具；重新随机划分会改变实验。采集任务总索引也不能替代 `configs/scenario.json` 的分类映射。

## 环境

在 Linux 使用两个 Python 环境。下面的版本来自已通过小样本检查的本机环境；未在全新机器上重建环境，GPU 驱动和 CUDA 编译工具需与 PyTorch 匹配。

主环境：Python 3.12，适用于除 NetMamba 外的 12 个模型。

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-main.txt
```

TrafficFormer 的 PCAP 转换还调用系统 `tshark`；运行前确认它在 PATH 中。CPU 可用于 FlowLens 和小样本检查，正式深度模型实验使用 CUDA。

NetMamba：单独使用 Python 3.9、PyTorch 2.0.1 / CUDA 11.7、torchvision 0.15.2。需要 C++ 编译工具、CUDA toolkit（包括 `nvcc`）及两个 CUDA 扩展。以下命令中的 `python` 指此环境：

```bash
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements-netmamba.txt
python -m pip install --no-build-isolation --no-deps causal-conv1d==1.1.0
MAMBA_FORCE_BUILD=TRUE python -m pip install --no-build-isolation --no-deps ./Benchmark/2024-ICNP-NetMamba/mamba-1p1p1
```

必须保留套件中的 NetMamba / mamba 源码。代码包不携带与某台机器绑定的 `.so`；本次检查使用了本机已有的匹配扩展，未重新验证上述编译步骤。缺少 `selective_scan_cuda` 时需要先完成扩展安装，不能以 CPU 随机森林替代 NetMamba。

## 起始权重

默认在套件根目录建立 `weights/` 并放入三个文件：

```text
weights/yatc-pretrained.pth
weights/netmamba-pretrained.pth
weights/trafficformer-pretrained.bin
```

实际实验起始文件的来源路径、SHA-256 和已知上游入口在 [WEIGHTS.json](WEIGHTS.json)。文件名只是套件统一命名，不表示三个文件都是未经微调的官方预训练权重。

```bash
python verify_weights.py --weights ./weights
```

特别是 TrafficFormer：实际文件是本地 `finetuned_model.bin`，其 SHA-256 与本地官方预训练文件 `nomoe_bertflow_pre-trained_model.bin-120000` 不同。目前没有证据证明它们等价。需要另外提供实际文件及其生成说明，才能准确衔接历史实验；直接换用官方文件属于新的起始条件。其余 10 个模型无需外部 checkpoint，TFusion encoder 从 train 的 benign flows 学习。

## 运行

先检查并生成两类任务的 manifest。数据可以在套件之外，使用 `--data` 指向实际 ANT 目录即可。

```bash
python suite.py check --data ./Benchmark/data/ANT
python suite.py prepare --data ./Benchmark/data/ANT --output ./outputs
```

单模型示例：

```bash
python suite.py run --task risk --model FlowLens --device cpu
python suite.py run --task scenario --model DF --device cuda:0
python suite.py run --task scenario --model YaTC --device cuda:0 --weights ./weights
```

运行全部 26 个主实验，指定两个环境的 Python。顺序运行以避免相互争抢显存；已完成项会跳过，遇到失败停止：

```bash
python run_all.py \
  --python-main /path/to/main-env/bin/python \
  --python-netmamba /path/to/netmamba-env/bin/python \
  --data ./Benchmark/data/ANT --weights ./weights \
  --output ./outputs --device cuda:0
```

所有模型名：`FlowLens YaTC NetMamba TFusion TrafficFormer DF AWF TikTok VarCNN NetCLR TMWF ARES BAPM`。可通过 `run_all.py --models DF AWF --tasks risk` 运行子集。NetMamba 单独运行时，应使用它自己的 Python 调用 `suite.py`。

仅构建输入：加 `--prepare-only`，不要求预训练权重。之后使用相同参数去掉该选项即可训练。小样本验证可显式加 `--epochs 1 --batch-size 8`，这种覆盖不是正式主实验配置。`--epochs` 对 TFusion 控制 encoder 训练，对 FlowLens 无作用。

同一输出目录中的已完成或失败实验不会被单模型入口覆盖。重新实验或修正失败后使用新的 `--output`；不要把不同数据、参数或模型版本的结果混在一起。`run_all.py` 的已完成项跳过只适用于继续同一批实验。

```bash
python suite.py summarize --output ./outputs
```

汇总生成 `outputs/results.csv`，包括 26 个任务的状态及已有指标；未运行项为空，不填入历史结果。

## 评估约定与输出

FlowLens、YaTC、NetMamba、TFusion、TrafficFormer 先输出 flow 预测。仅根据 **val 的 episode accuracy** 从 `mean_soft`、`majority_vote`、`confidence_weighted_mean`、`max_confidence` 中选择规则；并列时按此前顺序选取，再将选定规则用于 test。entropy confidence 为 `1 − H(p)/log(C)`，全部置信度为零时使用均值。

八个 WF 模型直接使用 episode 输入，不做 flow 聚合。输入使用全部可解析协议，与本项目 all-protocol 主实验入口一致。NetCLR 使用本地公共 WF runner 的监督分类流程，套件不另外追加原方法的对比预训练阶段。

每次运行的主要输出：

```text
outputs/<risk|scenario>/<model>/
├── suite_status.json          # 参数、解释器、完成/失败状态
├── resolved_config.json       # 四个集成模型的展开配置
├── native/ 或 algorithm_output/ # 原模型 checkpoint 与原生输出
├── val/、test/                # flow 模型的预测及 episode 对应关系
├── aggregation.json           # flow 模型的 validation 选规则记录
├── episode/test/              # 最终 episode 预测和 evaluation_metrics.json
└── coverage.json              # 实际评估数及缺失 episode ID
```

预处理可能过滤过短或无法解析的流，指标基于实际产生预测的 episode。报告数值时必须同时检查 `coverage.json`，不能把子集覆盖写成完整测试集。全局统计在 `dataset_counts.json`；各模型预处理细节留在各自运行目录。

随机性沿用原实现：控制器、WF、YaTC 的默认 seed 为 42；NetMamba 原生训练 seed 为 0；TrafficFormer 为 7、转换为 10；TFusion contrast batch 为 777，RF 为 42；FlowLens 配置中 random_state 为 42。`suite.py --seed` 只覆盖控制器、WF、YaTC 和 TFusion RF，不统一覆盖这些原生配置。CUDA 算子不保证逐位确定性。

## 与历史结果的关系

1. `configs/origins.json` 记录冻结配置的来源说明及校验值，匿名版本不公开内部运行路径。risk 的 TFusion、TrafficFormer 从 best-result 参数和本地默认值重建，无法证明每个未记录参数都与当时完全一致。
2. scenario 的 TFusion 历史运行复用已有 benign encoder；本套件根据原 encoder 参数从公开 train 重训，再训练 20 棵树、最大深度 50 的 RF。风险任务使用其自己的调优参数。没有把另一任务的 encoder checkpoint 冒充原文件。
3. scenario 的 NetMamba 历史配置中的本地 `--resume` 已移除，以所提供的起始权重重新微调。保留历史显式学习率和其他已记录参数。
4. NetMamba / TrafficFormer 保留原生 validation 阈值 0.99 的停止机制，因此 epoch 配置为上限。部分原生日志包含停止决定之后的 test 指标；test 不参与本套件选择聚合规则。
5. YaTC 沿用实际 adapter 的 `zero_ip_addresses=false`，NetMamba 使用 `remove_ip=true`。本套件没有把论文中的统一描述自动改成另一种预处理。
6. 执行的是当前公开目录的固定划分，实际计数为 3114。套件不填补缺失 episode，不使用论文表格的旧分母替代当前 manifest。

检查范围、未覆盖范围和移植修改分别见 [VALIDATION.md](VALIDATION.md) 与 [SOURCES.md](SOURCES.md)。

匿名版本的检查范围及提交前复查步骤见 [ANONYMITY.md](ANONYMITY.md)。
