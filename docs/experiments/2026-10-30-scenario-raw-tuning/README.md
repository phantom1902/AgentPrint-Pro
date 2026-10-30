# 五类场景 raw 参数搜索结果（2026-10-30）

本轮选择学习率衰减系数为 **0.5** 的双宽度模型：三个种子的最佳验证 Macro-F1 均值为 **0.785323 ± 0.016825**（样本标准差，seed=42/43/44）。同一配置 seed=42 为 **0.802530**；原配置同种子为 **0.649131**，提高 **15.34 个百分点**。这是预算内已测配置的结果，不代表全局最优或测试集表现。原配置仅运行一个种子，不能将其与三种子均值解释为配对的统计显著性比较。

## 最佳参数

| 参数 | 原配置 | 本轮选择 |
|---|---:|---:|
| learning_rate | .0005 | .001 |
| decay_per_30 | .2 | .5 |
| weight_decay | .001 | .001 |
| batch_size | 200 | 200 |
| dropout_2d | .1 | .05 |
| dropout_1d | .3 | .1 |
| width | 1 | 2 |
| 参数量 | 891,855 | 3,540,879 |

Adam，普通交叉熵，FP32，workers=0，CPU 数值库线程数 4。第 e 轮使用 `max(1e-6, .001 * .5 ** (e / 30))`，e 从 0 开始。只缩放隐藏卷积通道，输入和五分类输出维持原定义。

## 预算、数据和停止规则

实际 **12 次**：1 次基线、7 个新配置、4 次复验，study wall time **3532.15 秒（58 分 52 秒）**，失败 0，pruned 0。批准上限为 17 次或 3600 秒，先到为准；为了在时间上限内完成前两名复验，未用尽新配置名额。审计、备份和文档整理在训练结束后完成，没有追加训练。

固定 `scenario/raw`，训练 2697、验证 202 个 episode，类别数为 5；train 支持数 `[444,271,871,734,377]`，val `[36,21,71,59,15]`。缓存版本 `agentprint-mtam-v2`，使用原有官方划分，没有加载 test。训练、验证清单哈希记录于 audit.json 的 study_protocol.manifests。

全部从头训练。seed=42 顺序搜索；验证 Macro-F1 前两名各追加 seed=43、44。最终按三种子最佳验证 Macro-F1 的均值选择，Accuracy 只报告。每次至少 60、最多 200 轮；在首 30 轮锚点后，Macro-F1 连续 20 轮无至少 .001 的提升，且相邻两个 10 轮窗口平均验证损失相对改善不足 1%，才按平台条件停止。所有 12 次均因联合平台条件结束。

## 全部试验

轮数和最佳轮次均按从 1 开始计数；Macro-F1 和 Accuracy 来自最佳 F1 checkpoint。每次完整七维参数见 trials.csv，提案理由及原始命令见备份的 proposal.json。

| 试验 | seed | 配置变化 | 停止轮数 | 最佳轮次 | 验证 Macro-F1 | 验证 Accuracy |
|---|---:|---|---:|---:|---:|---:|
| trial-00 | 42 | 原始参数基线 | 90 | 70 | 0.649131 | 0.623762 |
| trial-01 | 42 | 基线，仅宽度 0.5 | 121 | 101 | 0.552762 | 0.574257 |
| trial-02 | 42 | 基线，仅宽度 2 | 74 | 54 | 0.714789 | 0.698020 |
| trial-03 | 42 | trial-02，提高学习率至 .001 | 91 | 70 | 0.759739 | 0.747525 |
| trial-04 | 42 | trial-03，weight decay 降至 .0001 | 94 | 74 | 0.748214 | 0.727723 |
| trial-05 | 42 | trial-03，Dropout 降至 .05/.1 | 90 | 70 | 0.772301 | 0.767327 |
| trial-06 | 42 | trial-05，batch 降至 128 | 85 | 61 | 0.725440 | 0.693069 |
| trial-07 | 42 | trial-05，衰减系数改为 .5 | 154 | 132 | 0.802530 | 0.777228 |
| trial-08 | 43 | trial-05 复验 | 111 | 91 | 0.770476 | 0.767327 |
| trial-09 | 43 | trial-07 复验 | 102 | 71 | 0.768908 | 0.772277 |
| trial-10 | 44 | trial-07 复验 | 86 | 65 | 0.784529 | 0.767327 |
| trial-11 | 44 | trial-05 复验 | 78 | 52 | 0.733943 | 0.732673 |

| 衰减系数 | Macro-F1（seed 42 / 43 / 44） | 均值 ± 样本标准差 | 平均 Accuracy |
|---:|---|---|---:|
| 0.5 | 0.802530 / 0.768908 / 0.784529 | 0.785323 ± 0.016825 | 0.772277 |
| 0.2 | 0.772301 / 0.770476 / 0.733943 | 0.758907 ± 0.021638 | 0.755776 |

在已测点中，扩大网络、提高学习率、降低 Dropout、减缓衰减后分数上升；降低 weight decay 或 batch 未带来提升。搜索以逐步改动为主，未遍历整个参数空间，也没有穷尽参数交互。最佳配置三次停止在 154、102、86 轮，对应最佳轮次 132、71、65；不应直接将最佳轮次固定为通用训练轮数。

## 验证证据与限制

- 最新相关测试：14 项通过，6.741 秒；包括新调参入口 7 项、既有模型 1 项、收敛规则 6 项。没有重跑全套项目测试。
- 同环境原配置复跑 90 轮，与保留的原收敛实验每轮 9 个确定性字段逐项一致。
- 独立 audit.py 核验逐轮学习率、首次停止时点、最佳轮次/F1、Adam 步数、ask/tell 对应关系、三种子选择，并重载前两名共 6 个 checkpoint 在完整验证集上重算 F1/Accuracy，全部通过。
- 验证集仅 202 个样本（最小类别 15 个），且多次用于选参；结果存在验证集过拟合和种子波动。三次复验不能替代独立测试集评估。
- GPU 为 RTX 3080 Ti，PyTorch 2.12.1+cu130、Python 3.12。每组详细环境与数据身份保留在原始产物中。

## 产物、备份与 Git

源码分支 `codex/scenario-raw-optim`，实现提交 `167118c`。独立入口 `agentprint/tuning.py`；原正式训练配置与历史结果保留。轻量最佳参数见 best-params.json；完整审计见 audit.json。

远程实验目录：

```text
/root/autodl-tmp/ant-workspace/src/AgentPrint-Pro/.optim-agent-runs/scenario-raw-20261030
```

最佳 seed=42 模型位于该目录 `trial-07/best.pt`；`trial-09` 和 `trial-10` 是同参数的另两种子。所有完整训练状态及历史最佳快照保留在数据盘。

本地备份：

```text
D:/AllenLeeSpace/studyziliao/paper_code/.agentprint-worktrees/reports/scenario-raw-tuning-20261030/scenario-raw-results.tar.gz
```

备份包含 125 个文件：study/提案/身份/环境/逐轮指标/验证逐类及混淆矩阵、全部 12 个 best.pt，以及最佳参数 seed=42 的 last.pt。其余 last.pt 与历史最佳快照保留远程。归档 SHA256 及每个 best.pt 已完成远程/本地一致性核验；详细哈希见 backup-manifest.json 和 audit.json。

## 精确复跑命令（仅供之后执行）

在服务器上述源码目录下运行，输出目录必须新建；已结束的 study 有固定 60 分钟截止时间，不应重置旧 study 来追加试验。下面直接调用单试验入口从头复跑，不会改动既有结果：

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/root/autodl-tmp/ant-workspace/envs/optim-agent/bin/python - <<'PY'
import json
from agentprint.tuning import evaluate_trial
params = json.load(open("docs/experiments/2026-10-30-scenario-raw-tuning/best-params.json"))
evaluate_trial(
    params,
    "/root/autodl-tmp/ant-workspace/runs/scenario-raw-selected-reproduction-seed42",
    "/root/autodl-tmp/ant-workspace/data/processed/agentprint-full-v2",
    seed=42, device="cuda:0",
)
PY
```

此命令会重新训练并消耗 GPU，未在本轮结束后执行。原 production train 配置不支持此次新增宽度/Dropout 字段，复跑请使用上述调参入口。独立审计可在原服务器用同样的线程/CUBLAS 设置运行本目录 audit.py；脚本中的固定路径对应本轮保留产物。
