# IAT多视角与多核：执行记录

2026-11-01，用户明确授权按已拆分票据多agent执行、允许并行，提供当前CPU无卡服务器；授权覆盖实现、测试和有限真实小样本，不覆盖全量预处理或正式训练。Tracker发布待配置，不阻塞此次直接执行授权。

集成分支：codex/agentprint-iat-multikernel-dev；实现基线a23cf45。主agent负责逐票集成、最终全套测试和Standards/Spec双轴审查；各实现agent使用隔离Git工作区和独立提交。

| 票据 | 当前状态 | 证据 |
|---|---|---|
| T01 | 已集成 | 5d0cf48；新v2公共测试6项通过，4项旧模型/场景/Primitive/恢复回归通过；mypy和compileall通过 |
| T02 | 已集成 | 8e704c7；8项新拟合测试及3项回归通过 |
| T03 | 已集成 | 3ddc344；替换模型及旧模型接口回归通过 |
| T04 | 已集成 | 049e08e；11项针对性检查通过 |
| T05 | 已集成 | 93569f1；6项公共缓存/训练/身份/恢复检查通过 |
| T06 | 已集成 | 86e82a9；10项新检查及6项原收敛检查通过 |
| T07 | 已完成 | 746e021；组合IAT多核完整epoch恢复、六配置合成闭环和真正旧v1产物兼容 |
| T08 | 已完成 | 28776ff；六组真实GPU小样本闭环于2026-11-02全部通过，完整报告见下 |

初始远端为旧功能提交faddd44、无GPU。宿主机显示48逻辑CPU和数百GiB内存，后核验容器当时仅2GiB，不能以宿主机free输出作为预算。首次真实验收退出137，证据保留；2026-11-02用户更新资源后，实际cgroup为90GiB、12核CPU配额及RTX 3080 Ti 12GiB。实现28776ff通过Git bundle部署至独立数据盘工作区，不重置原目录。

原生工作树工具因目标仓库未列入本聊天workspace roots拒绝创建；在既有外部工作区目录使用Git工作树隔离，不改动用户.idea目录。测试优先公共命令全流程及产物，必要公开模型接口；实施时进行red→green、单文件检查，主agent在末尾运行全套一次。

已在新功能合入之前，以真正旧代码57883c6创建（相对a23cf45仅文档变化）v1完整2轮参考运行及第0轮后中断的checkpoint，保留来源版本和SHA256，供T07用新代码验证评估与恢复。夹具为合成流量，不是研究结果。

旧v1产物已在合入新实现后完成评估和续训验证：model、Adam、Python/NumPy/Torch/shuffle RNG、epoch和最佳指标与旧代码连续运行完全一致，原完整checkpoint SHA256未改变。证据已交付至[旧v1兼容报告](../../experiments/2026-11-01-iat-multikernel-smoke/legacy-v1-compatibility.json)。

T05合入冲突仅涉及功能守卫；保留T04前置模型并移除已实现IAT的临时拒绝条件。集成后的6个新模块mypy检查通过。最终七个模块mypy和compileall通过；Standards/Spec双轴审查均无剩余实质问题。Windows全套122项：116通过、5跳过、1项WinError1314；Linux针对性9项补测全部通过，覆盖Windows未完成的6项。不同环境合计覆盖全部测试，不声称某次完整运行全绿。六组真实GPU验收各2轮全部通过，选样20train/10val，不读真实test、无全量IAT预处理或正式训练。

最终[验收报告](../../experiments/2026-11-01-iat-multikernel-smoke/README.md)包含环境、选样、fit/cache身份、六组命令与指标、checkpoint和完成后恢复SHA256。服务器新增文件均位于 `/root/autodl-tmp/ant-workspace`。GitHub同步与tracker发布未完成；不阻塞本次已授权实现和验收。
