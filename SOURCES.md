# 实际实现来源与套件修改

本清单取自原项目 `Benchmark/SOURCES.md` 的实现路径审计。上游仓库说明方法实现的来源；本套件实际执行的是收录的本地适配代码，不在运行时下载上游最新版本。

| Baseline | 实际实现上游 | 套件执行代码 |
| --- | --- | --- |
| FlowLens | [dmbb/FlowLens](https://github.com/dmbb/FlowLens) | `Benchmark/2021-NDSS-FlowLens/run_unified_classifier.py`，本地离线特征和 RF 适配，并非完整 P4 系统 |
| YaTC | [NSSL-SJTU/YaTC](https://github.com/NSSL-SJTU/YaTC) | `Benchmark/2023-AAAI-YaTC/fine-tune.py`、`models_YaTC.py`，本地代码副本及 ANT 适配 |
| NetMamba | [wangtz19/NetMamba](https://github.com/wangtz19/NetMamba) | `Benchmark/2024-ICNP-NetMamba/src/`、其自带 `mamba-1p1p1/` |
| TFusion | [fuchuanpu/TFusion](https://github.com/fuchuanpu/TFusion) | `Benchmark/2025-CCS-TFusion/model/` 原生 encoder / detector；PCAP 特征由 ANT 本地适配构建 |
| TrafficFormer | [IDP-code/TrafficFormer](https://github.com/IDP-code/TrafficFormer) | `Benchmark/2025-SP-TrafficFormer/fine-tuning/run_classifier.py`、`uer/` |
| DF | [WFlib](https://github.com/Xinhao-Deng/Website-Fingerprinting-Library) | `Benchmark/2024_CCS_WFlib_Early_Stage_WF/WFlib/models/DF.py` |
| AWF | [WFlib](https://github.com/Xinhao-Deng/Website-Fingerprinting-Library) | 同目录 `AWF.py` |
| TikTok | [WFlib](https://github.com/Xinhao-Deng/Website-Fingerprinting-Library) | 同目录 `TikTok.py` |
| VarCNN | [WFlib](https://github.com/Xinhao-Deng/Website-Fingerprinting-Library) | 同目录 `VarCNN.py` |
| NetCLR | [WFlib](https://github.com/Xinhao-Deng/Website-Fingerprinting-Library) | 同目录 `NetCLR.py`，本地公共 runner 的监督训练 |
| TMWF | [WFlib](https://github.com/Xinhao-Deng/Website-Fingerprinting-Library) | 同目录 `TMWF.py` |
| ARES | [WFlib](https://github.com/Xinhao-Deng/Website-Fingerprinting-Library) | 同目录 `ARES.py` |
| BAPM | [WFlib](https://github.com/Xinhao-Deng/Website-Fingerprinting-Library) | 同目录 `BAPM.py` |

八个 WF 模型统一由 `Benchmark/scripts/run_ant_wf_scenario_smoke.py` 的构建、特征处理及训练函数调用。文件名含 `smoke`，但它是本项目主实验复用的公共 runner。未以各方法同名独立仓库替换实际 WFlib 实现。

WFlib 的本地 checkout HEAD 为 `537ec8e03c805540b04b95b144fa158ce50aa255`；八个模型文件无已跟踪修改，`tools/data_processor.py` 存在原项目本地修改。历史审计记录其 remote 重定向到 FIND-Lab 的同一仓库。此 HEAD 是检查时的本地版本，不是每次历史运行的版本证明。其余五个本地模型目录缺少独立 Git 元数据，无法可靠补写上游 commit。

本文件及第三方目录中的姓名、邮箱、机构、仓库和版权声明属于被引用的 baseline 或依赖项目，不是本 benchmark 的作者声明。

## 从原项目抽取后的改动

- 增加独立入口 `suite.py` / `run_all.py`、模型调度 `engines.py` 和固定标签映射；路径均基于套件根目录或命令参数。
- 删除对历史缓存、外部训练目录、调度器以及 NetMamba 恢复训练 checkpoint 的运行依赖；冻结实际记录中的配置参数。重建配置的证据在 `configs/origins.json`。
- NetMamba / TrafficFormer 的输入转换器额外保存 train / validation 的 flow 与 episode 对应关系，供 validation 聚合选择使用；validation 转换结果为空时报错，不以 train 替代。
- NetMamba validation 使用其原生 `eval.py` 与最佳 checkpoint。排除训练专用 `--lr` 参数；为该脚本加入 `--no_amp`，使其与训练导出的全精度 test 推理一致；checkpoint 加载失败时报错，不静默评估随机初始化模型。
- 公共命令执行函数只向当前 run 日志写入命令，不再向代码根目录追加 `command_history.md`。
- WFlib 的 `models/__init__.py` 为空，避免导入未收录模型；模型由现有 runner 按文件加载。默认配置只保留指定的八个 WF 模型。
- 汇总从实际新生成的预测计算，额外保存 test 覆盖数和缺失 episode ID；不导入旧指标。TFusion 的运行配置显式记为实际使用的 RF。

`VENDORED_FILES.json` 对每个抽取文件同时记录原项目副本的 `source_sha256` 和套件副本的 `suite_sha256`，可以区分抽取后补丁。两者均不代表上游原始发布版本的校验值。`FILES.sha256` 覆盖交付代码包的文件内容，不包含数据、权重或运行输出。

## 匿名版本中的处理

- 内部实验运行目录与时间标识替换为来源说明；保留配置证据的校验值。
- TrafficFormer 辅助代码的临时文件改为相对当前工作目录或指定输出路径，移除开发者机器绝对路径与调试注释。
- 第三方文档增加来源提示，原有版权、许可证和方法归属予以保留。

## 许可证文件

保留原目录现有的 TFusion、TrafficFormer、WFlib 及 mamba 许可证和源码声明。FlowLens、YaTC、NetMamba 顶层本地副本未找到独立 LICENSE 文件；没有为这些第三方代码补写或统一套用新的许可证。项目自有代码已采用 GPL-3.0-only，数据授权采用 CC-BY-4.0；具体范围和第三方例外见 `LICENSING.md` 与 `THIRD_PARTY_LICENSES.md`。预训练权重未装入代码包。

## 采集源码导入

`data_collection/` 来自单独提供的 `data_collection_code.zip`，不替换上表任何 baseline 实现。实际导入文件、源文件哈希、缺失组件和局部修改见该目录的 `IMPORT_REPORT.json`；发布文件哈希见 `MANIFEST.json`。压缩包并不完整，未据原 README 声明其可以端到端运行。许可边界见根目录 `LICENSING.md` 与 `THIRD_PARTY_LICENSES.md`。
