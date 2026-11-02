# 全量IAT缓存与六组训练

2026-11-02，用户在六组GPU小样本验收通过后明确要求“开始缓存 和 训练”，授权本次全量train/val IAT拟合、缓存及六组正式训练。沿用已确认协议，不追加参数搜索或真实test评估。历史实现规格中的仅smoke预算在此次执行请求中被扩大；特征与训练规则保持不变。

复用2697个train和202个val的有效基础MTAM，不重建原缓存。仅train拟合共享边界，train/val生成独立IAT补充缓存；数组预计667,929,600字节，另有清单及拟合进度。启动前服务器实际内存90GiB、CPU配额12核，GPU为RTX 3080 Ti 12GiB，数据盘约21GiB空闲。

六组按single/raw、single/log1p、replace/raw、replace/log1p、prepend/raw、prepend/log1p顺序从头串行运行。seed42，width2，Adam，初始学习率0.001、weight_decay0.001、batch200；dropout_2d=0.05、dropout_1d=0.1。学习率为max(1e-6,0.001×0.5^(epoch/30))。至少60轮、最多200轮；连续20轮无至少0.001的验证Macro-F1提升且最近两段各10轮平均验证损失改善不足1%时停止。最佳模型按实际验证Macro-F1选择，同分取较早轮；达到上限不自动等同收敛。

FP32、关闭AMP/TF32/benchmark、workers0、数值线程4，CUDA确定性环境变量为CUBLAS_WORKSPACE_CONFIG=:4096:8。拟合和prepare使用已验收公共命令的顺序实现，不引入未验证的并行预处理。失败时停止队列并记录，不静默跳过或降低样本、模型、batch。

正式配置位于[configs](../../../configs/agentprint/iat-full-20261102/single-raw.yaml)。[执行入口](run.py)从已提交版本调用公共fit-bins→prepare→六组train/evaluate，所有新增数据和日志均在`/root/autodl-tmp/ant-workspace`：

- 拟合：`artifacts/agentprint-iat-full-20261102/fit.json`及其`.progress`。
- 补充缓存：`data/processed/agentprint-iat-full-20261102`。
- 模型与指标：`runs/agentprint-iat-full-20261102`。
- 队列状态、命令及日志：`reports/iat-full-20261102/state.json`与同目录日志。

缓存完成后自动启动训练，无需再次发送指令。`state.json`记录实际源码提交、命令、退出码及完成配置；状态以服务器文件为准，本说明不代表执行完成。执行入口拒绝覆盖已有状态；需要恢复时先检查失败原因，拟合按Episode、缓存按完整分片、训练按完整epoch恢复。
