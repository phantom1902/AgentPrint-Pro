# IAT多视角与多核：实现与验收记录

实现分支：`codex/agentprint-iat-multikernel-dev`。远端实际执行源码为 `28776ff0949fdd3778d56d485d97791f4ef21779`；后续提交仅补充文档和证据，最终源码范围见Git记录。

状态：T01–T08实现及验收完成。2026-11-02更换实例资源后，六组真实小样本GPU验收全部通过；本地Windows权限受限的检查也在Linux补测通过。历史失败证据保留。未执行全量IAT预处理、正式训练或真实test评估。

## 已实现并有针对性证据

- 显式fit-bins→prepare→train→evaluate；训练Episode等权精确拟合共享IAT桶，独立256样本IAT分片、sample_id关联与十通道raw/log1p。
- 单核、替换型3/5/7多核、无残差前置多核；12配置公开模型接口及IAT开启六配置的合成公共命令闭环。
- 完整epoch恢复模型、Adam、Python/NumPy/Torch/shuffle RNG、最佳checkpoint和停止历史；组合IAT＋前置多核＋log1p实际中断恢复与连续运行等价。
- 真正旧v1源码生成的checkpoint评估/续训通过，完整旧参考checkpoint未改写，见[兼容证据](legacy-v1-compatibility.json)。
- 七个新/扩展模块mypy（check-untyped-defs）及compileall通过；[双轴审查](code-review.md)Standards和Spec各0项剩余问题。早期v2关闭IAT的缓存完整性缺口已修复并有公共回归。

## 全套回归

本地Windows完整CPU回归已结束：`python -m unittest discover -s tests -q`共122项，116通过、5跳过、1错误，用时1761.429秒（约29分21秒），退出码1，不能记为全绿。唯一错误为`test_protocol.ProtocolTests.test_public_splits_and_duplicate_rejection`在创建夹具符号链接时触发`WinError 1314`，尚未进入重复样本拒绝断言；5项跳过来自旧CPU验收辅助测试的符号链接权限守卫。应在支持符号链接的Linux环境补测这些检查；本次未为消除错误而修改测试或生产代码。命令、环境、日志SHA256和错误摘要见[本地全套证据](local-full-suite.json)。

旧远端完整回归启动后SSH入口无法连接，未取得最终exit/log，不能算通过。上述本地自动化回归使用合成夹具，不代表六组真实数据GPU验收或正式训练已完成。

## 真实数据运行

远端输出根目录：`/root/autodl-tmp/ant-workspace/acceptance/iat-multikernel-20261101`。源码独立工作区为`/root/autodl-tmp/ant-workspace/src/AgentPrint-Pro-IAT`，原`AgentPrint-Pro`工作区未重置。新增源码、bundle、临时文件、缓存、运行和报告均位于指定数据盘工作区。

[选样清单](selection.json)严格为20个train、10个val，分别每类4/2，按PCAP大小和sample_id选择，不以消息标注存在为条件。已有MTAM仅链接复用，没有重建或复制数组。

[训练集拟合](fit.json)完成：20个Episode、9902个有效包、9607个正间隔、275个零间隔，报告4次时间逆序。六个共享边界为2000、7000、13000、35000、1953000、26023000纳秒；这是smoke拟合，不能替代全量拟合。指纹为`59ed4f4d9da958487e51cb43fe09ef28583cf609ade9ef282e33a821875116f1`。

[train补充缓存](iat-train-metadata.json)与[val补充缓存](iat-val-metadata.json)均完成，使用共同拟合；第一组single/raw开始训练后、首个epoch checkpoint提交前遭SIGKILL，退出137，完成配置数0/6。未自动重试、跳过、减少样本或修改batch/模型。其余五组未执行；不报告任何真实验证分数或收敛结果。

[外部失败证据](remote-failure.json)：容器cgroup内存上限和历史峰值均为2147483648字节（2GiB）。`free -h`显示的408GiB为宿主机内存，不能作为容器预算。本次全套测试与真实验收并行运行；启动前未核对容器配额是执行检查的遗漏。该峰值为整个容器总值，不能据此推断单模型独占内存需求大于2GiB；未另做单独重试。退出137后读取到上述限制和历史峰值，支持资源限制的怀疑；未采集失败前峰值基线，内核OOM计数也未提供额外确认，因此不把触顶时刻或具体杀进程机制写成已证实事实。原[进程终止时summary](remote-summary-at-kill.json)仍为running，因为SIGKILL不能由Python异常处理捕获，应结合外部失败证据阅读。

## 参数量和复现

width2下，IAT十通道单核3,550,095、替换多核3,552,403、前置多核3,560,497个参数；raw/log1p不改变参数量。公开模型实际计数见[参数证据](model-parameter-counts.json)。

公共辅助入口和预算说明见[使用文档](../../agentprint-iat-smoke.md)。全量IAT和正式训练继续等待独立指令。

## 2026-11-02 GPU补验收通过

新实例实际cgroup内存上限96,636,764,160字节（90GiB）、CPU配额12核；RTX 3080 Ti显存12,288MiB，驱动580.76.05，PyTorch 2.12.1+cu130确认CUDA可用。记录见[环境和补测](gpu-20261102/reports/iat-multikernel-20261102/environment-and-tests.json)。CUDA确定性设置使用 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，保持FP32、关闭AMP/TF32、workers0、数值线程4。

以原实现提交28776ff运行，新增输出目录 `/root/autodl-tmp/ant-workspace/acceptance/iat-multikernel-20261102-gpu`，保留之前失败目录。先完成CPU补测，再单独串行运行六组。选样仍20 train/10 val、每类4/2，基础MTAM链接复用；重新拟合的指纹与之前完全相同。没有扩样或读取真实test。

[通过摘要](gpu-20261102/acceptance/iat-multikernel-20261102-gpu/summary.json)及[公共命令日志](gpu-20261102/acceptance/iat-multikernel-20261102-gpu/commands.log)记录六组各2轮、最佳checkpoint验证集重载和完成后resume。single/replace/prepend × raw/log1p全部成功，退出码0；六组last.pt、best.pt、metrics.jsonl和result.json恢复前后SHA256均相同，下载前也重新核验。不是GPU中途断电续训测试；完整epoch中断恢复等价由CPU集成测试另行覆盖。

入口整段执行为北京时间2026-11-02 14:47:13至14:47:35（约22秒，包含拟合、缓存、六组训练/评估/完成后恢复）；仅30个小样本，不能据此估计全量耗时。六组smoke验证Accuracy均0.2；只有2轮，不能排名模型、推断正式效果或宣称收敛。

初次启动器因实例缺少可选的 `/usr/bin/time` 而退出127，发生在Python入口启动前、未创建验收目录；移除该计时包装后成功运行，模型、样本和预算没有调整。[启动器失败日志](gpu-20261102/reports/iat-multikernel-20261102/launcher-missing-time.log)保留。训练checkpoint与大缓存仍留在服务器数据盘，本地只保存轻量证据。

实现已本地Git提交并通过Git bundle部署远端Git分支；本轮没有声称GitHub已同步或tracker已发布。Tracker配置仍是独立待办。
