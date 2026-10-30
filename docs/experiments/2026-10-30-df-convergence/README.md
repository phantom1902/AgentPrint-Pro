# DF训练至验证平台（2026-10-30）

**已满足联合验证平台停止条件。** 本次独立确定性训练共 **60轮**，按验证Accuracy选中第 **36轮**。测试Accuracy **0.6603**，Macro-F1 **0.6892**。

## 实验身份与原30轮的关系

原WF runner只保存验证Accuracy最佳模型（第27轮），没有第30轮末模型、Adamax或RNG状态。因此先尝试从seed42重建训练状态，并逐轮核验。GPU上第一轮train loss与原记录不符（1.475378 vs 1.474040），入口当即停止，未进入续训。另用未修改的原入口和同一缓存运行1轮也未复现原记录，得到1.481573；没有用这次诊断的测试结果决定配置。

用户已被告知无法完整接续原训练。本次执行的是**独立从头训练**，启用确定性执行、关闭TF32/cuDNN benchmark，以首30轮作为新轨迹的锚点；`baseline_parity_passed=false`。CPU集成测试证明重建逻辑在可复现轨迹上匹配原入口，但不能据此声称旧GPU训练已被复现。保留失败日志和旧30轮结果，不将它们拼成一条训练曲线。

## 固定配置与停止判据

- WFlib DF五分类，3,668,613参数；输入为原缓存DIR-5000，沿用此前固定的train/val/test=2703/202/209，没有重新预处理或重分割。
- Adamax，固定学习率0.002，batch128，普通交叉熵，seed42，FP32，workers0，数值库线程4；继承训练drop_last，不做调参，不使用AgentPrint的LR衰减。
- 确定性算法开启，TF32关闭，cuDNN benchmark关闭，CUBLAS工作空间`:4096:8`。这与原默认30轮的执行设置不同。
- 前30轮作为锚点，至少追加30轮；F1连续20轮没有至少0.001提升，并且相邻两段各10轮的平均验证CE相对改善不足1%，两者同时满足才按平台停止。相对改善定义为`(previous-recent)/previous`。
- 累计200轮保护上限；达到上限仍未满足上述条件时标为未收敛。
- 全程按验证Accuracy选模型，并列取较晚轮次；F1只用于停止。测试在最终模型选定后评估，再由独立审计重载核验，未用测试集选轮次或改参数。

停止时的证据：总轮数60，追加30，无显著F1提升24轮；前10轮平均验证CE=1.483086，后10轮=1.713415，相对改善=-15.5303%。停止原因`validation_plateau`，converged=`True`。后10轮验证损失比前10轮升高15.53%，而训练损失继续降低，表明后期出现过拟合；因此保存第36轮最佳模型。这里“收敛”指已达到约定的验证停止标准。

## 结果

| 实验 | 验证Accuracy | 验证Macro-F1 | 测试Accuracy | 测试Macro-F1 |
|---|---:|---:|---:|---:|
| 原默认30轮 | 0.564356 | 0.585582 | 0.578947 | 0.604851 |
| 本次独立训练 | 0.574257 | 0.589651 | 0.660287 | 0.689170 |

本次所有轮次的最高验证Macro-F1为 **0.589651**，恰好也由按Accuracy选中的第36轮模型取得。新轨迹首30轮最高验证Macro-F1为0.573704、最高Accuracy为0.559406。两次实验执行设置不同且仅一个种子，表格是描述性对照，不能把全部差异归因为训练轮数。

测试集逐类指标：

| 类别 | 支持数 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| artifact_processing | 35 | 0.7647 | 0.7429 | 0.7536 |
| business_services | 22 | 0.8235 | 0.6364 | 0.7179 |
| general_knowledge | 72 | 0.6792 | 0.5000 | 0.5760 |
| scientific_evidence | 60 | 0.5529 | 0.7833 | 0.6483 |
| security_testing | 20 | 0.7500 | 0.7500 | 0.7500 |

混淆矩阵与验证集逐类指标见audit.json，逐轮train loss、val CE、Accuracy、F1与时间见epochs.csv。

## 实现与验证

独立入口`tools/df_convergence.py`复用原DF、输入转换和优化器，以及已验证的ConvergencePolicy；未修改原baseline入口。先写CPU集成测试并观察缺少入口时失败，再实现。最新9项相关测试通过（13.485秒），包括3项真实CPU集成测试与6项既有停止规则测试。检查了原结果保留、逐轮及checkpoint重建、重建差异拒绝、输入身份拒绝、保护上限不误报收敛、幂等、中断后模型/Adamax/RNG完全一致，以及选模结束前不需要test缓存。

独立GPU审计通过：首次满足停止条件的时点、Accuracy选模及并列处理、原数据/原结果SHA256、完整last中Adamax步数1260，以及重载best后完整val/test指标和测试概率一致性。实际轮内训练/验证累计耗时262.2秒，不包含文件写入、测试、审计和备份。

源码提交`9a101d6`，分支`codex/df-convergence`；环境见audit.json。本次最后状态保存模型、优化器、RNG和历史，因此本实验自身的中断恢复不再依赖原30轮缺失的状态。

## 复现与产物

以下命令会启动训练；若要新复现，应更改output目录。对本实验相同output重复调用会核验身份并恢复完整epoch，已有result时返回原结果。

```bash
cd /root/autodl-tmp/ant-workspace/src/AgentPrint-Pro
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/root/autodl-tmp/ant-workspace/envs/optim-agent/bin/python tools/df_convergence.py \
  --raw-root /root/autodl-tmp/ant-workspace/runs/df-scenario-default-20261030/wflib_raw \
  --parent-root /root/autodl-tmp/ant-workspace/runs/df-scenario-default-20261030/WFlib-DF \
  --output /root/autodl-tmp/ant-workspace/runs/df-scenario-convergence-fresh-20261030 \
  --device cuda:0 --fresh
```

远程数据盘产物：`/root/autodl-tmp/ant-workspace/runs/df-scenario-convergence-fresh-20261030`。`best.pt`为最终选中模型，`last.pt`为最终完整训练状态，历史最佳快照保留在checkpoints/。原30轮目录不变，未替换旧checkpoint。

本地完整状态备份：`D:/AllenLeeSpace/studyziliao/paper_code/.agentprint-worktrees/reports/df-convergence-20261030/df-convergence-results.tar.gz`。包含best/last、曲线、预测、身份、审计、测试日志及失败重建记录；归档及best/last均完成SHA256校验，见backup-manifest.json。历史最佳快照和原输入缓存留在服务器；原DF缓存已有上一轮本地备份。
