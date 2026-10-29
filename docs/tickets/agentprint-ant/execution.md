# AgentPrint ANT 执行记录

用户于2026-10-29明确授权：按DAG派发subagent完成所有任务，主agent集成测试，使用Git。原先的整体启动与拆分确认由这条直接指令满足；tracker配置与issue发布待办不阻塞编码，不假称issue已发布。

集成分支：codex/agentprint-ant-baselines。实现基线：bc08dc8。主agent负责合入、服务器资源串行、集成验证与最终复核。后续代码审查比较基线固定为bc08dc8。

## 当前状态

| Ticket | 状态 | 执行者 | 提交/证据 |
|---|---|---|---|
| T01 | 已完成并集成 | t01_implement / t01_review | 实现e265a2b，集成13264f0；7定向测试、compileall、mypy通过；审查无重要问题 |
| T02 | 执行中 | t02_implement | 隔离分支codex/ant-t02，基线13264f0 |
| T03 | 执行中 | t03_implement | 隔离分支codex/ant-t03，基线13264f0 |
| T04 | 执行中 | t04_implement | 隔离分支codex/ant-t04，基线13264f0 |
| T05 | 等待T02 | 未派发 | |
| T06 | 已就绪，等待可用agent槽位 | 未派发 | 隔离分支codex/ant-t06已创建 |
| T07 | 等待T03/T04/T05/T06 | 未派发 | |
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
