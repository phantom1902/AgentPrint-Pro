# AgentPrint ANT 执行记录

用户于2026-10-29明确授权：按DAG派发subagent完成所有任务，主agent集成测试，使用Git。原先的整体启动与拆分确认由这条直接指令满足；tracker配置与issue发布待办不阻塞编码，不假称issue已发布。

集成分支：codex/agentprint-ant-baselines。实现基线：bc08dc8。主agent负责合入、服务器资源串行、集成验证与最终复核。后续代码审查比较基线固定为bc08dc8。

## 当前状态

| Ticket | 状态 | 执行者 | 提交/证据 |
|---|---|---|---|
| T01 | 已完成并集成 | t01_implement / t01_review | 实现e265a2b，集成13264f0；7定向测试、compileall、mypy通过；审查无重要问题 |
| T02 | 已完成并集成 | t02_implement / t02_review | 实现7e18e7b，集成d991900；片段6项及T01 7项通过，审查通过 |
| T03 | 已完成并集成，审查通过 | t03_implement / t03_review | 实现afc04da，集成e8c3167；8项质量测试与T01 7项通过 |
| T04 | 已完成并集成，复核通过 | t04_cursor_fix / t04_fix_review | 5c2fceb及8e66b21，集成d8ae303；新增3项集成测试通过 |
| T05 | 已完成并集成，审查通过 | t05_implement / t05_review | 实现94069dd，集成8009052；主agent合并后四配置命令测试通过123.135秒 |
| T06 | 已完成并集成，审查通过 | t06_implement / t06_review | 实现e8f7993，集成5593670；4项恢复测试通过，合并后mypy11文件通过 |
| T07 | 执行中 | t07_implement / root | 子agent实现可复现验收工具，root独占服务器执行 |
| T08 | 等待T07 | 未派发 | |
| T09 | 等待T08 | 未派发 | |

## 工作区与测试

- 当前应用项目目录是研究工作区，代码仓库是另一个Git仓库；内置worktree工具无法定位代码仓库的ref。已在代码仓库外用Git worktree创建隔离任务分支，避免修改原始研究仓库。
- 各票先执行公开命令/产物边界的red→green测试，模型只补充已确认的公开输入输出、CE和反传边界；主agent在合入后核对，不测试私有函数结构。
- 本地创建独立Python3.12测试环境，供subagent快速单文件测试和类型检查；服务器仍按规格执行真实CPU验收与全量预处理。服务器当前Python3.12.3、torch2.12.1+cu130、无GPU，数据盘可用约23.3GB。
- root独占共享服务器变更与重负载任务，不并行运行多个真实训练。GPU正式实验不在本次执行范围。
- 每票实施后记录报告与提交、进行任务审查；全票合入后执行完整测试与独立Standards/Spec代码审查，修复重要问题后才宣布完成。

## 已发现事项

- T01审查通过，仅有一项非阻塞问题：嵌套列表形式的非法splits配置可能抛TypeError而非配置诊断；后续配置票中补公开命令回归并修复。
- Linux旧协议3项测试通过；Windows旧测试受默认GBK和创建symlink权限限制，最终完整回归在Linux进行。既有sklearn单类警告与新模块分开记录。
- 服务器GitHub传输多次失败；使用本地Git bundle经SSH传输，再由服务器git fetch/fast-forward同一提交，记录SHA。不是未提交工作区拷贝，也不假称GitHub fetch成功。

## 恢复提示

任务报告位于代码仓库外的.agentprint-worktrees/reports。以本表、dag.json、Git历史和报告为恢复依据；不要重复派发已完成并集成的任务。

- T04任务审查发现完整片段源恰逢分片边界时游标仍停在当前PCAP，需一行前瞻后推进至下一源；修复期间不标记完成。元数据清单随样本数增长的内存开销记录为规模限制，数组缓存仍按分片有界。

- T03审查通过：普通IPv6长度40+payload length按规格执行。主agent核对T02/T04共享生成器只捕获预期EpisodeExcluded，程序异常继续向上传播；整源聚合后才发布片段；支持边界文档在T09更新。T04修复8e66b21已集成d8ae303，主agent新增3项集成测试全部通过（12.682秒）。

- T05/T06交叉合并核对：保留shuffle generator与完整恢复循环；训练loader后立即缺类门控；保存protocol.json；getRF按任务K；固定47逐epoch指标与验证排除信息；版本v2、正式全split默认及LR/早轮次平分规则保留。主agent合并后片段47类prepare/train/evaluate通过21.510秒，四配置公开闭环通过123.135秒，mypy11文件通过。

- T07 helper0a9e114已合并8feec33，目标Linux6项公开工具测试通过（12.898秒）。真实尝试选择15/6个train/val Episode（64/31条完整标注），有效场景12/6、片段35/22；场景raw训练/验证成功，后续log1p训练SIGKILL(-9)且stderr为空，因此验收未通过、不启动T08。容器内存限额2GB、历史峰值触及限额；正在将父进程依赖探测改为短生命周期进程，保留原batch4/FP32及失败工作区。helper审查记录PyYAML缺失时环境报告捕获较晚的次要问题，最终复核时处理。
