# 五类场景 raw 第二轮参数探索（2026-10-30）

**结论：保留上一轮最佳参数。** 本轮新增候选未提高三种子平均最佳验证 Macro-F1。旧最佳为 **0.785323 ± 0.016825**；新候选最好为 **0.770611 ± 0.006130**，均值低 1.47 个百分点。后者这三次种子的波动更小，但预先约定的选优目标是均值，因此不替换旧最佳。

## 范围与预算

沿用用户已经确认的最多 17 次、3600 秒硬上限，5 类 scenario/raw、原官方 train/val、从头训练、seed42 搜索，前两名新配置补 seed43/44；不读 test。实际 **12 次**（1 基线＋7 新配置＋4 复验），**3324.08 秒（55 分 24 秒）**；失败 0、pruned 0，全部以联合验证平台规则停止。复验结束即停止，本轮未因剩余时间再增加未经复验的新点。

训练 2697、验证 202 个 episode，缓存 `agentprint-mtam-v2`。普通 CE、Adam、确定性 FP32、workers=0、数值库线程4。每次至少60、最多200轮；F1连续20轮无至少.001提升，且相邻10轮平均验证损失相对改善不足1%，才按平台规则停止。LR公式 `max(1e-6, initial_lr * decay_per_30 ** (epoch / 30))`，epoch从0开始。

## 推断如何转为试验

上一轮扩大通道、提高学习率、降低Dropout及放慢衰减后表现提高。本轮先保持双宽度，分别探索学习率上下两侧（.0015、.0007）、更慢衰减（.75），再检查低学习率与慢衰减的组合。组合得分略高于两个单独改动，但未超过旧最佳，因此继续检查更慢的.9衰减和更靠近原点的.0011学习率，最后检查进一步降低Dropout。

这些改动都未在seed42超过旧最佳。随后仍对本轮前两名做完整复验，避免仅凭一个种子下结论。weight decay=.001、batch=200、width=2保持固定；本轮不涉及损失函数、数据划分或输入变换修改。原默认基线的Dropout为.1/.3、width=1，其余试验均width=2。

## 全部试验

轮数与最佳轮次均从1开始，F1为最佳验证F1 checkpoint的结果。完整七维参数及Accuracy见 trials.csv，逐轮指标、命令、提案理由在原始备份中。

| trial | seed | 学习率 | 每30轮衰减 | Dropout 2D/1D | 停止轮数 | 最佳轮次 | 验证 Macro-F1 |
|---|---:|---:|---:|---|---:|---:|---:|
| trial-00 | 42 | 0.0005 | 0.2 | 0.1/0.3 | 90 | 70 | 0.649131 |
| trial-01 | 42 | 0.0015 | 0.5 | 0.05/0.1 | 97 | 74 | 0.755256 |
| trial-02 | 42 | 0.0007 | 0.5 | 0.05/0.1 | 100 | 79 | 0.777139 |
| trial-03 | 42 | 0.001 | 0.75 | 0.05/0.1 | 94 | 88 | 0.767506 |
| trial-04 | 42 | 0.0007 | 0.75 | 0.05/0.1 | 100 | 74 | 0.779149 |
| trial-05 | 42 | 0.0007 | 0.9 | 0.05/0.1 | 85 | 83 | 0.748250 |
| trial-06 | 42 | 0.0011 | 0.5 | 0.05/0.1 | 101 | 81 | 0.754444 |
| trial-07 | 42 | 0.001 | 0.5 | 0.025/0.05 | 93 | 73 | 0.767666 |
| trial-08 | 43 | 0.0007 | 0.75 | 0.05/0.1 | 60 | 39 | 0.717992 |
| trial-09 | 44 | 0.0007 | 0.75 | 0.05/0.1 | 109 | 83 | 0.771369 |
| trial-10 | 43 | 0.0007 | 0.5 | 0.05/0.1 | 77 | 56 | 0.764977 |
| trial-11 | 44 | 0.0007 | 0.5 | 0.05/0.1 | 101 | 68 | 0.769717 |

## 三种子最终比较

± 为三个种子的样本标准差，不是置信区间。旧最佳分数沿用上一轮；其源代码、依赖环境、缓存身份已核对，且本轮默认基线90轮×9个确定性字段逐项一致。

| 配置来源 | 学习率 | 每30轮衰减 | seed42 / 43 / 44 | 均值 ± 标准差 |
|---|---:|---:|---|---|
| 上一轮最佳 | 0.001 | 0.5 | 0.802530 / 0.768908 / 0.784529 | 0.785323 ± 0.016825 |
| 本轮候选 | 0.0007 | 0.5 | 0.777139 / 0.764977 / 0.769717 | 0.770611 ± 0.006130 |
| 本轮候选 | 0.0007 | 0.75 | 0.779149 / 0.717992 / 0.771369 | 0.756170 ± 0.033291 |

较低学习率.0007与原衰减.5的三次结果更集中，但均值降低；衰减.75的候选在seed43明显下降；进一步降Dropout仅单种子检查，没有改善。这些结果支持继续保留原参数，不能证明该点是全局最优，也不能据此断言各参数的普遍因果作用。验证集仅202条，多轮选择会使验证表现乐观；本轮没有测试集结论。

## 保留的最佳参数

```json
{
  "batch_size": 200,
  "decay_per_30": 0.5,
  "dropout_1d": 0.1,
  "dropout_2d": 0.05,
  "learning_rate": 0.001,
  "weight_decay": 0.001,
  "width": 2.0
}
```

参数量3,540,879，最佳seed42验证Macro-F1=.802530，三种子均值=.785323。

## 验证、Git与产物

- 独立审计通过：逐轮LR、首次停止、最佳checkpoint、Adam步数、ask/tell对应、三种子排序、源代码/依赖/缓存身份一致性。
- 重新加载本轮两候选6个模型和旧最佳3个模型，在完整验证集复算F1/Accuracy；共9个checkpoint全部匹配。
- 14项相关测试再次通过（6.421秒）；不声称重跑全套项目测试。
- 分支 `codex/scenario-raw-optim-round2`，计划提交 `eb68b0e`；生产实现自`167118c`未修改。
- 所有远程训练产物位于数据盘 `.optim-agent-runs/scenario-raw-20261030-round2`，其源码根目录为 `/root/autodl-tmp/ant-workspace/src/AgentPrint-Pro`。
- 最佳模型仍为上一轮目录 `scenario-raw-20261030/trial-07/best.pt`，另两种子在trial-09、trial-10；不会用本轮同名trial替换它们。
- 本地归档位于 `D:/AllenLeeSpace/studyziliao/paper_code/.agentprint-worktrees/reports/scenario-raw-tuning-20261030-round2/scenario-raw-round2-results.tar.gz`，共145个文件，包含本轮全部最佳模型和逐轮指标、旧最佳三种子模型与指标，以及选中seed42的last状态。归档和各最佳模型均经SHA256一致性校验，详细哈希见backup-manifest.json/audit.json。历史最佳快照及其他last状态保留服务器。

## 复现（未额外执行训练）

在服务器源码目录执行以下命令会从头训练最佳参数；更改seed为43/44可复验另两个种子。已结束study的截止时间不重置，单次重现另写新输出目录。固定环境与详细入口说明见 `docs/agentprint-tuning.md`。

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/root/autodl-tmp/ant-workspace/envs/optim-agent/bin/python - <<'PY'
import json
from agentprint.tuning import evaluate_trial
params = json.load(open('docs/experiments/2026-10-30-scenario-raw-tuning-round2/best-params.json'))
evaluate_trial(params,
    '/root/autodl-tmp/ant-workspace/runs/scenario-raw-round2-selected-reproduce-seed42',
    '/root/autodl-tmp/ant-workspace/data/processed/agentprint-full-v2',
    seed=42, device='cuda:0')
PY
```

本目录 audit.py 可在原服务器用上述线程/CUBLAS环境独立重跑审计；repeat_finalists.py 保存本轮复验调度逻辑，结束后不要重新启动它。所有GPU训练现已结束。
