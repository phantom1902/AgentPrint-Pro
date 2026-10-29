# ANT 采集源码

[English](README.md) | [简体中文](README.zh-CN.md) | [主实验入口](../README.zh-CN.md)

本目录提供构建 ANT 公开划分之前使用的智能体 episode 流量采集调度和辅助代码。分类套件不调用这些采集脚本，使用已发布数据时无需运行它们。

## 已提供的内容

从 `data_collection_code.zip` 导入了 25 个非空源码或支持文件；少量修改记录在 [IMPORT_REPORT.json](IMPORT_REPORT.json)。

| 位置 | 实际内容 |
| --- | --- |
| `scripts/batch_run_*.py` | CLI / Web 批量调度、并行启动器及不同服务商变体 |
| `scripts/copy_*_sessions.py`、`scripts/codex_session_utils.py` | 会话查找与复制工具 |
| `tools/import_tasks.py` | 从另行提供的任务目录或 ZIP 导入输入，不覆盖已有文件 |
| `tools/check_runtime.py` | 检查 manifest 哈希、源码引用、模块、系统工具和 mock 服务 |
| `tools/smoke_collection.py`、`tools/fixture_agent.py`、`examples/mock_smoke.yaml` | 基础设施检查入口和测试输入；所需运行组件缺失 |
| `tasks/index.yaml` | 任务目录索引，不代表任务实现或分类标签映射 |
| `requirements*.txt` | 采集直接依赖和较宽泛的任务环境快照 |

**当前是部分源码导入，不是可运行的完整采集版本。** 原始 manifest 声明 521 个文件，其中 480 个声明路径未找到、13 个文件为空、3 个非空文件的哈希与记录不符。一份私有笔记另有文件名编码不一致问题。因此“480 个缺失”是按声明路径统计，并非缺失 480 个可执行程序。

缺失或为空的组件包括 `scripts/runner/`、`scripts/env/`、8 个 `scripts/mock/mock_*_server.py`、`setup_server.sh`、`label_pcap.py`、`visualize_episode.py`、生产任务包装脚本和任务辅助文件。损坏或截断的 `tasks/.zip`、空文件、私有开发笔记和 macOS 元数据未导入。原 README 声称包含全部运行组件，与实际收到的文件不符，已由当前说明和仓库根 README 替代。

[IMPORT_REPORT.json](IMPORT_REPORT.json)记录压缩包原始哈希和文件缺口；[MANIFEST.json](MANIFEST.json)仅记录实际交付文件及当前哈希，不代表运行环境完整。会话查找已改为读取 session 中的 workspace 元数据，不再硬编码开发机器路径。基础设施检查会在创建测试文件或探测端口前拒绝缺失的运行组件。

## 检查当前导入

在仓库根目录使用 Python 3.11 或更高版本：

```bash
python data_collection/tools/check_runtime.py --source-only
```

当前版本**预期返回非零退出码**，原因是 runner 引用和 mock 服务缺失。`--source-only` 只跳过已安装依赖检查，不忽略缺失源码。此次整合另行检查了已交付源码的语法和文件完整性；安装依赖不能补回缺失源码。

恢复采集需要从原项目补齐非空的 runner、环境工具、mock、后处理和任务辅助代码，再复查身份信息、凭据、许可及文件哈希。还需完整任务输入和智能体配置。完成后才能执行完整运行检查和基础设施测试。当前文档不将任何安装或采集命令标为已经可用。

## 环境与任务输入

采集环境面向 Linux / Python 3.11+，与主实验和 NetMamba 环境分别管理。`requirements.txt` 是直接依赖；`requirements-task-environment.txt` 是原任务环境快照，不是经过全新机器验证的锁文件。预期还需要网络命名空间、iproute2、iptables、tcpdump、bubblewrap、智能体 CLI，以及具体任务所需的 Docker 等工具；Web 采集需要 Playwright、浏览器及账号配置。导入目录不会安装这些组件。

完整任务实例和外部任务数据需另行提供。保留的导入工具可以把可信任务目录或 ZIP 复制到 `tasks/`，不覆盖已有文件；仅导入任务输入不能恢复缺失的采集运行代码。外部任务材料保留原许可。

服务商变体从环境变量读取凭据，如 `MIMO_API_KEY`、`XIAOAI_API_KEY`，并保留 `ANTHROPIC_AUTH_TOKEN` 回退。端点保持源文件配置，使用时应按服务商调整。真实密钥、浏览器配置、会话历史和生成结果不得加入源码发布包。部分批量入口会执行启动清理，使用共享机器前需检查选项。本次整合未执行智能体调用、采集任务、抓包或清理操作。

## 与分类实验的衔接

原始输出可能位于 `episodes/Agent/<engine>/Axxxx/`，按 runner 产生 PCAP、会话、事件及工作目录。公开分类数据要求 `Benchmark/data/ANT/<train|val|test>/<engine>/<family>/Axxxx/network.pcap`。本次导入没有提供已验证的数据导出或固定划分生成流程，需要保留发布时的 episode 筛选和划分记录，不能自行随机重分。分类标签以仓库根[数据约定](../README.zh-CN.md#数据约定)为准。

项目原创贡献遵循 [GPL-3.0-only](../LICENSE)；单独的数据授权为 [CC-BY-4.0](../LICENSE-DATA)，具体边界见 [LICENSING.md](../LICENSING.md)。外部任务输入、软件包、服务和模型权重不因本次导入改变许可。
