# DF 五类场景分类：仓库默认配置（2026-10-30）

已使用真实 ANT 数据完成 DF 默认配置的30轮训练和评估，无调参。最佳模型按验证集 Accuracy 选取，第 **27** 轮；测试集 Accuracy **0.5789**、Macro-F1 **0.6049**。这是单种子默认基线，不宣称收敛或最优。

## 配置与输入

| 项目 | 设置 |
|---|---|
| 模型 | WFlib DF，5分类，3,668,613参数 |
| 轮数 | 30，仓库预设 |
| 优化器／初始学习率 | Adamax／0.002，固定学习率 |
| batch size／随机种子 | 128／42 |
| 损失／精度 | 普通交叉熵／FP32 |
| 选模型规则 | 验证Accuracy最大，并列取较晚轮次 |
| 输入 | 每episode前5000个可解析包的方向序列，DIR |
| 方向定义 | PCAP首包源IP为正，其他源IP为负；继承套件规则 |
| 包协议 | all，与套件主实验一致 |
| 数据加载 | workers=0；CPU数值库线程4 |

原始序列为方向×相对时间，DF实际取sign，只保留方向，不使用包长。长度不足补零，超出截断。首包相对时间为0，同时间戳包也可能为0；这是原适配实现行为，本次保留。它与AgentPrint的完整episode双统计MTAM、运行端推断、训练样本过滤规则不同。

## 样本与划分

保留官方train/val/test，不重新分割；逐条核对目录场景与任务编号映射一致，检查engine/episode身份及真实路径不跨split重复。默认总入口suite.py会同时检查risk标签，但当前数据已按五类场景目录组织，不含attack_execution等旧risk family，导致无关risk检查失败。本次直接调用仓库已有WF runner，未改模型、训练器或数据标签。

| split | 原始数量 | 实际使用 | 未产生输入 |
|---|---:|---:|---:|
| train | 2703 | 2703 | 0 |
| val | 202 | 202 | 0 |
| test | 209 | 209 | 0 |

训练每轮使用 2688 条，继承drop_last；每轮shuffle后末尾不足128的batch丢弃，所丢样本随轮次变化。val/test不丢尾批。逐类支持数、缺失episode列表和非零方向槽位统计见audit.json。

## 结果

均为验证Accuracy所选checkpoint的分数；测试集在完成选模后进行一次正式评估，之后独立重载核验同一模型；未按测试指标调参或选轮次。

| split | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|
| val | 0.564356 | 0.585582 | 0.608376 |
| test | 0.578947 | 0.604851 | 0.611400 |

30轮中最高验证Macro-F1为 **0.585582**（第27轮），此记录仅描述曲线，不替换原先按Accuracy选出的checkpoint。完整逐轮数据见epochs.csv。

测试集逐类结果：

| 类别 | 支持数 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| artifact_processing | 35 | 0.6842 | 0.7429 | 0.7123 |
| business_services | 22 | 0.5600 | 0.6364 | 0.5957 |
| general_knowledge | 72 | 0.5278 | 0.5278 | 0.5278 |
| scientific_evidence | 60 | 0.5455 | 0.5000 | 0.5217 |
| security_testing | 20 | 0.6842 | 0.6500 | 0.6667 |

测试混淆矩阵如下，行是真实类别、列是预测类别，类别顺序见label_mapping.json：

```json
[
  [
    26,
    1,
    2,
    4,
    2
  ],
  [
    0,
    14,
    5,
    2,
    1
  ],
  [
    8,
    5,
    38,
    18,
    3
  ],
  [
    1,
    3,
    26,
    30,
    0
  ],
  [
    3,
    2,
    1,
    1,
    13
  ]
]
```

单种子和30轮预算不足以证明收敛。类别数量不均，Accuracy与Macro-F1应同时报告。此前AgentPrint调参结果是验证集多种子选参分数，且样本过滤与选择指标不同；不能直接把本次DF测试分数与之当作公平的最终测试对比。

## 执行和验证

- 使用现有数据盘环境，无新增依赖安装、无生产代码改动。
- 合成PCAP小样本GPU端到端检查通过，然后运行真实完整数据。
- 30轮loss与验证指标均有限；独立核验最佳轮次、并列处理、输入标签对齐、覆盖数量与跨split身份。
- 重载checkpoint，完整验证集和测试集重新推理；F1/Accuracy匹配，测试概率与原产物在数值容差内一致。
- 总耗时（含生成DF输入）288.1秒；训练与原生最终评估15.9秒。两者不含合成检查和后续审计/备份。
- 环境：NVIDIA GeForce RTX 3080 Ti，PyTorch 2.12.1+cu130；完整环境及来源SHA256见audit.json。

## 复现与产物

在服务器源码目录 `/root/autodl-tmp/ant-workspace/src/AgentPrint-Pro`，下列命令会重新训练，请使用新的out-dir，避免覆盖既有产物：

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/root/autodl-tmp/ant-workspace/envs/optim-agent/bin/python \
  Benchmark/scripts/run_ant_wf_scenario_smoke.py \
  --data-root /root/autodl-tmp/ant-workspace/data/raw/ANT \
  --out-dir /root/autodl-tmp/ant-workspace/runs/df-scenario-default-reproduce \
  --per-engine-scenario 1000000 --label-field Scenario --protocols all \
  --models DF --device cuda:0 --seed 42 --feature-workers 4 \
  --epoch-metrics-path /root/autodl-tmp/ant-workspace/runs/df-scenario-default-reproduce/progress.json \
  --use-default-config-training
```

`--per-engine-scenario 1000000`让现有选择器保留各engine/类别的所有样本；没有均衡采样。`--use-default-config-training`明确使用DF预设30轮，而非脚本的小样本默认1轮。首次命令与来源commit见launch.json。

远程完整实验：`/root/autodl-tmp/ant-workspace/runs/df-scenario-default-20261030`。模型为其中 `WFlib-DF/checkpoint-best.pt`；输入缓存为 `wflib_raw/`。

本地完整归档：`D:/AllenLeeSpace/studyziliao/paper_code/.agentprint-worktrees/reports/df-scenario-20261030/df-scenario-default-results.tar.gz`，包括模型、输入缓存、指标、预测、日志和审计。归档及模型SHA256已校验，见backup-manifest.json/audit.json。轻量报告归入Git分支`codex/df-scenario-baseline`。
