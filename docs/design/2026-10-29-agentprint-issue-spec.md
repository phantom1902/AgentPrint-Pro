## Problem Statement

研究者已经在服务器上准备了 ANT 原始流量和消息标注，希望依据《Exposing LLM User Privacy via Traffic Fingerprint Analysis — A Study of Privacy Risks in LLM Agent Interactions》的方法，分别评估完整 Episode 的五类场景分类和给定片段的 47 类 Primitive 分类。

现有套件提供既有风险/场景基线，尚未提供本次需要的标注片段构造、运行端方向判定、双统计 MTAM、RF CNN 适配及独立训练评估流程。直接使用旧风险标签、逐连接切分或既有解析器会改变本次任务定义。当前无卡模式要求先在 CPU 完成开发验收，再准备完整数据，正式 GPU 实验留到后续阶段。

## Solution

新增独立的 agentprint 模块，提供 prepare、train、evaluate 三个命令。沿用 ANT 官方划分，将完整 Episode 和由现有标注构造的给定片段分别转换成 MTAM；用适配的 Robust Fingerprinting CNN 分别训练五类场景和 47 类 Primitive 分类器。

两个任务分别拥有样本清单、损失、权重和结果，共用适用的数据解析、MTAM 和模型定义。每个任务均比较 raw 与 log1p，以验证集主 Macro-F1 选择 checkpoint 和输入配置。交付可恢复的数据处理、可追溯的实验配置、质量/覆盖报告、CPU 小样本验收及全量预处理产物。

## User Stories

1. As a researcher, I want five-class Scenario classification on complete Episodes, so that I can measure task-category recognition from traffic.
2. As a researcher, I want 47-class Primitive classification on given annotated segments, so that I can evaluate behavior-category recognition independently.
3. As a researcher, I want independent samples, losses, checkpoints, and metrics for the two tasks, so that their results have clear meanings.
4. As a researcher, I want the official train/validation/test assignments preserved, so that my baseline uses the released benchmark protocol.
5. As a researcher, I want fixed label identities and output sizes, so that missing classes never silently redefine the task.
6. As a researcher, I want the current N001–N047 ontology used, so that historical G labels do not enter the experiment.
7. As a researcher, I want adjacent annotation starts to define segments, so that every supervised interval is traceable to existing annotations.
8. As a researcher, I want the final annotation to end at capture end, so that its sample uses the available observation interval.
9. As a researcher, I want only duplicate-start groups and their uncertain intervals excluded, so that other usable segments from the Episode remain available.
10. As a researcher, I want unalignable annotation intervals excluded with reasons, so that the system never guesses clock offsets.
11. As a researcher, I want Scenario eligibility assessed separately from annotation availability, so that missing logs do not unnecessarily discard complete-Episode samples.
12. As a researcher, I want empty or zero-duration samples excluded without extra minimum-size thresholds, so that short valid observations remain represented.
13. As a researcher, I want the Agent runtime endpoint identified once per Episode, so that all its segments share the same direction reference.
14. As a researcher, I want a fixed, auditable IPv4 endpoint rule, so that direction is reproducible without manual per-Episode labeling.
15. As a researcher, I want ambiguous endpoints skipped rather than guessed, so that uncertainty remains visible in the dataset report.
16. As a researcher, I want eligible IPv6 addresses associated through the runtime endpoint's unique source MAC, so that determinable IPv6 traffic can be included.
17. As a researcher, I want all determinable runtime connections included, so that the observation covers mixed Agent and tool traffic.
18. As a researcher, I want byte totals based on IP total length, so that link-layer framing does not change the chosen size definition.
19. As a researcher, I want ACKs, connection-control packets, and observed retransmissions retained, so that their observed traffic patterns remain in the input.
20. As a researcher, I want usable IPv4/IPv6 packets included regardless of transport protocol, so that UDP and ICMP are not silently lost.
21. As a researcher, I want readable IP headers counted even when payload capture is truncated, so that known packet lengths remain usable.
22. As a researcher, I want damaged capture record streams excluded, so that partial-prefix processing cannot appear as complete data.
23. As a researcher, I want every sample's full interval divided into 1800 equal bins, so that all inputs have the agreed representation.
24. As a researcher, I want exact packet ownership at internal boundaries and capture end, so that packets are neither duplicated nor shifted across excluded intervals.
25. As a researcher, I want raw and log1p inputs derived from the same statistics, so that their comparison uses identical samples.
26. As an operator, I want direct PCAP-to-MTAM preparation without saved segment PCAPs, so that I avoid an unnecessary traffic copy.
27. As an operator, I want integer MTAM counts stored in bounded shards and read with memory mapping, so that the dataset fits a modest-memory server.
28. As an operator, I want compatible completed preprocessing shards reused after interruption, so that I do not repeatedly process the full dataset.
29. As an operator, I want incompatible cache reuse rejected with a reason, so that stale artifacts cannot silently change the experiment.
30. As a researcher, I want separate valid and excluded sample manifests, so that every inclusion and exclusion can be inspected.
31. As a researcher, I want per-sample provenance, timestamps, endpoint evidence, and traffic totals, so that unexpected results can be traced to their inputs.
32. As a researcher, I want an RF CNN adapted for two statistics and the task's class count, so that the architecture has an explicit reference implementation.
33. As a researcher, I want ordinary unweighted cross-entropy for each task, so that the first baseline does not add unapproved loss variants.
34. As a researcher, I want raw/log1p to share all other training conditions, so that the validation comparison isolates the input transform.
35. As a researcher, I want checkpoint and transform selection based on validation Macro-F1, so that test results cannot influence selection.
36. As a researcher, I want the main macro average to use classes supported by the evaluated split, so that its definition remains explicit when classes are absent.
37. As a researcher, I want fixed-47 Macro-F1, class support, accuracy, and per-class results reported, so that coverage limitations remain visible.
38. As a researcher, I want deterministic tie rules for checkpoint and transform selection, so that ties do not depend on incidental execution order.
39. As a researcher, I want exactly four predeclared single-seed formal configurations, so that the experiment budget remains bounded.
40. As a researcher, I want all four configurations evaluated at their validation-best checkpoints, so that the raw/log1p comparison is fully reported.
41. As a researcher, I want formal training paused for a task missing an expected training class, so that an incomplete training ontology is not hidden in results.
42. As an operator, I want common YAML settings with four experiment overlays and validated runtime overrides, so that experiments remain easy to reproduce.
43. As an operator, I want each new run to use a unique output location, so that previous artifacts are preserved.
44. As an operator, I want the best checkpoint and the latest complete training state saved, so that I can both evaluate and resume an experiment.
45. As an operator, I want resume compatibility checks and recorded runtime changes, so that continuation preserves the intended experiment.
46. As an operator, I want metrics beyond the committed recovery epoch archived, so that resumed logs do not double-count effective epochs.
47. As a researcher, I want epoch metrics and periodic training loss in files and TensorBoard, so that I can inspect learning and retain reviewable evidence.
48. As an operator, I want reproducible FP32 execution with explicit failures, so that the system never silently changes batch size or precision.
49. As an operator, I want the server's existing Python/PyTorch checked and only missing dependencies added, so that setup stays focused on this module.
50. As a developer, I want hand-checkable capture fixtures exercised through public commands, so that tests catch incorrect statistics rather than merely copying implementation logic.
51. As a developer, I want both classifier outputs and cross-entropy backpropagation checked, so that each task's model contract is verified.
52. As a researcher, I want a small real-data CPU run for all four configurations, so that end-to-end functionality is established before full processing.
53. As a researcher, I want CPU acceptance data taken only from official training and validation splits, so that test data is not used for debugging model behavior.
54. As an operator, I want full preprocessing after CPU acceptance, so that all official splits are ready before GPU experiments.
55. As a researcher, I want full-data inclusion, exclusion, and class-coverage reports, so that the effective benchmark population is known.
56. As an operator, I want server execution tied to a Git commit, so that every result identifies its exact code version.
57. As a researcher, I want selected lightweight summaries committed with the code, so that the experimental evidence can be reviewed without large binary artifacts.
58. As an operator, I want original data, historical runs, and existing checkout changes preserved, so that development does not overwrite my work.
59. As a researcher, I want implementation progress reported at milestones and new decisions raised explicitly, so that accepted requirements remain under my control.
60. As a researcher, I want the first delivery to finish with CPU acceptance and full preprocessing, so that I can defer GPU cost until hardware is available.

## Implementation Decisions

- **任务与领域边界**：遵循已有四项架构决定：两个独立分类任务、使用 RF 参考模型、mixed-flow 观察、以相邻标注起点构造给定片段。共享通用定义不等于共享训练权重。现有风险标签和逐连接观察协议不用于定义本次任务。
- **模块与接口**：新增独立 agentprint 模块，提供 prepare、train、evaluate 命令。流式包解析、区间/样本构造、MTAM 缓存、RF 模型和训练评估按这些公共入口组合；仅复用已核验适用的公共工具，不直接继承旧解析器的整包列表、截断返回前缀和运输层头不全即丢包行为。
- **划分与标签**：保留官方 train/val/test。场景按字典序编号：artifact_processing=0、business_services=1、general_knowledge=2、scientific_evidence=3、security_testing=4；当前 N001–N047 对应 0–46。保存固定标签映射，任何 split 缺类均不临时缩减或重编号输出头。
- **给定片段**：当前标注起点到下一起点，最后一段到完整捕获结束。它们是操作性的日志区间，不宣称精确物理行为边界。同起点组的全部标注及从该起点到下一个不同起点的区间排除；末组延伸到捕获结束。不跨排除区间拼接、不任取重复组一条。无法可靠对齐的区间排除，不猜测偏移；保留其他有效区间。
- **样本资格**：片段无可用包或零时长时排除；不另设最少包数或最短时长。场景样本为空捕获、无可用方向包或零捕获时长时排除，不构造伪全零样本。缺日志或标注无法对齐只影响片段资格；方向缺失或记录流损坏等共同条件可影响两个任务。所有排除只作用于派生样本，不删除原始数据。
- **IPv4 运行端**：已有明确身份映射可直接使用。当前交付材料没有找到这类映射，按已确认规则，在完整 Episode 中找出唯一发出初始 TCP SYN（SYN=1、ACK=0）的 RFC1918 地址，并要求该地址同时是参与 IP 包数的唯一第一名。无候选、多候选或条件冲突则排除 Episode。没有覆盖率阈值、地址字典序消歧或 UDP/53 回退。
- **IPv6 与方向**：已识别 IPv4 的发包源 MAC 唯一时，将相同源 MAC 发出的 IPv6 单播源地址关联到运行端；排除未指定地址和组播地址，多 MAC 或地址冲突不关联。整个 Episode 的全部片段共用相同端点依据。仅源属于运行端为上行，仅目的属于运行端为下行；两端都属于或都不属于时跳过并计数。不把多播目的 MAC 直接当作运行端下行身份。保存规则版本和地址证据，明确这是推断身份。
- **包范围与长度**：观察样本区间内全部可确定方向的运行端连接。纳入 IP 总长度可可靠读取的 IPv4/IPv6，不限于 TCP；保留 ACK、SYN/FIN 及捕获到的每一次重传。字节量为 IPv4 total length 或普通 IPv6 的 40 字节固定头加 payload length，包含 IP/运输层头，不含链路层头。不用帧长、捕获长度或载荷长替代；ARP 等非 IP 包不纳入。
- **解析质量与失败语义**：IP 头完整且声明长度可读的载荷截断包仍计入，并记录截断；IP 头不可读跳过该包。PCAP 记录流损坏到不能可靠继续时排除整个 Episode，不使用已读前缀冒充完整捕获。预期质量排除继续处理，未预料程序异常停止命令、报告错误并保留已完成产物。按实际 linktype 解析，不将未知格式静默当作 Ethernet。
- **MTAM 定义**：一个样本为两项统计乘两个方向乘 1800 个时间槽，批输入为 [B,2,2,1800]。统计顺序是包数、IP 字节量；方向顺序是上行、下行。每个 Episode 按完整捕获时间范围、片段按完整标注区间各自等分 1800 槽；不固定秒级截断，不缩成过滤后首末包范围。内部区间/槽左闭右开，边界包进入新区间/槽；完整捕获终点进入最后槽，片段终点仅按其所属有效末片段处理，不把排除区间的包移到前一有效片段。
- **缓存和读取**：直接从原始 PCAP 生成 MTAM，不保存片段 PCAP。原始统计缓存为 uint64，按任务和 split 分开，每片最多 1024 个完整样本，末片可不足；使用分片 NPY 和内存映射。训练读取后转 float32，按配置保持 raw 或逐元素 log1p。同一缓存和有效清单支撑两个输入变换，1024 个样本的文件容量与训练 batch 无关。
- **缓存恢复**：按完整分片提交和复用，未完成分片重建；不增加另一套 Episode 级临时 MTAM。兼容性检查包括预处理版本、配置、文件清单，以及 PCAP 相对位置、大小、mtime；小标注和端点元数据用内容哈希。拒绝不兼容复用并解释原因，不每次全量哈希 PCAP；不能识别大小和 mtime 都未变的 PCAP 内容替换。
- **样本契约**：有效与排除记录分别保存为 JSONL。有效记录包含 sample_id、task、split、engine、scenario、episode_id、label_name、label_id、pcap_relpath、annotation_relpath、annotation_row、start_ns、end_ns、duration_ns、runtime_endpoints、endpoint_source、packet_count、ip_bytes、shard_relpath、row_index、quality。时间为整数纳秒，来源与缓存位置为相对各自根目录的引用；场景样本的片段专属字段为空。排除记录保留可用来源/标签/时间及 reason、detail，不伪造缓存索引。另存格式版本、配置指纹和标签映射标识，不保存消息正文或包载荷。
- **RF 结构适配**：采用 Robust Fingerprinting 原 CNN，仅将实际首个 Conv2d 输入通道改为 2，任务输出改为 5 或 47。保留参考卷积、池化、BN、ReLU、Dropout、reshape 和初始化。Conv2d 两级通道为 2→32→32、32→64→64，核为 (3,6)、padding 为 (1,1)，池化依次为 (1,3)、(2,2)，两处 Dropout 为 0.1；之后将每样本 64×1×296 重排为 32×592。Conv1d 为 32→128→128、128→256→256、256→512→K，核3、padding1，前两组各接池化3与 Dropout 0.3，最终自适应平均池化输出 K 个分数。各卷积保留参考 BN/ReLU，包括最后类别卷积；不新增投影。自定义初始化仍只覆盖参考中的 Conv2d、BN2d、Linear，其余沿用框架默认。
- **损失与输入比较**：两任务各用普通无类别权重 CrossEntropyLoss，模型分数直接送入损失；模型内部不加入 softmax，仅展示概率时使用。raw/log1p 的其他训练条件、有效清单和评估类别集合一致，不增加联合损失或首轮超参数搜索。
- **正式训练预算**：seed=42，两个任务乘两种输入变换，共四次正式训练。Adam，初始学习率 5e-4，weight_decay=1e-3，batch=200，总计30 epochs；第 e 轮开始的学习率为 5e-4 乘 0.2 的 e/30 次方，e 从0到29。每轮验证，不把 CPU 验收算作正式实验，不报告未运行的多种子均值/标准差。
- **指标与选择**：主 Macro-F1 的类别集合来自当前 split 固定有效清单中有真实样本的类别，不能随预测变化。同步报告 Accuracy、逐类支持数、覆盖率和排除分布；片段另报固定47类、缺类计零的辅助 Macro-F1。验证主 Macro-F1 选择 checkpoint 和每任务输入变换；同分取较早 epoch，输入同分取 raw。四个预声明配置均报告各自验证最优 checkpoint 的测试结果，标记验证选出的主配置，测试不参与选择。全量训练缺预定类别时，先报告并停止受影响任务的正式训练，不影响另一任务或有意缩小的 CPU 验收。
- **配置契约**：公共 YAML 加四个实验覆盖配置，未知字段或非法值报错。实验参数通过 YAML 修改；命令行运行项限于配置选择、数据根、缓存根、输出根、设备、恢复和 checkpoint，以及 evaluate 的 val/test split 选择。保存最终合并配置；正式实验和 smoke 使用独立配置，不静默变更参数。
- **checkpoint 与恢复**：保存 best 权重和 last 完整状态，后者包含模型、优化器、已完成 epoch、最佳指标/位置和随机状态。从最近完整 epoch 继续至原目标总轮数，不额外再跑30轮。校验任务、变换、标签映射、模型/优化器参数、seed、train/val 清单标识、预处理和模型版本。路径或设备可改变并记录，不宣称跨设备逐位一致；超过恢复点的未提交指标归档后续写，避免重复有效 TensorBoard epoch。
- **日志与实验目录**：每次新建唯一 run_id，包含任务、变换、seed、UTC 时间和短随机串；显式指定旧运行才可恢复，不静默覆盖。每 epoch 保存训练/验证 Loss、验证 Accuracy、主 Macro-F1、片段辅助 Macro-F1、学习率、耗时、样本数；每50个训练 step 增加训练 Loss。提供最终合并 YAML、JSONL 指标、JSON 汇总、逐类和混淆矩阵 CSV，以及同一 run_id 的 TensorBoard 曲线。
- **执行条件**：FP32，关闭 AMP、TF32 和 cuDNN benchmark，固定随机种子并启用可用的确定性执行要求。预处理单进程、DataLoader workers=0，正式配置串行执行。OOM、非有限损失或不支持要求模式时停止报告，不自动调整 batch、精度或训练参数。核验沿用镜像现有 Python/PyTorch，仅补齐独立最小依赖并记录实际版本；运行依赖为 torch、numpy、PyYAML、dpkt、scikit-learn、tensorboard，测试沿用 unittest、scapy。
- **开发与留档**：本地编辑、提交和推送当前开发分支，服务器检出相同提交执行并记录 SHA。代码、原始数据、派生缓存和运行结果使用各自配置根；遇到已有 checkout 修改或位置冲突保留并反馈，不自动 reset/覆盖。Git 保存源码、配置、测试、文档和选定轻量配置/汇总/逐类/混淆矩阵/清单标识/环境证据，忽略原始数据、MTAM、权重和 TensorBoard 事件。服务器历史产物保留，清理由用户另行指定。
- **首次交付范围**：依次完成数据适配与 MTAM、模型与训练评估、自动化与真实 CPU 验收，再全量预处理。CPU 验收最多20个训练和10个验证 Episode，尽量覆盖五场景；片段从中最多取64个训练和32个验证片段。四配置各1 epoch、batch=4、seed42，只使用官方 train/val；数量不足时报告实际值，不放宽质量规则，无法运行则报告验收未完成。CPU通过后处理全部原始捕获及已有完整标注，生成两个任务所有官方 split 的缓存、清单和覆盖报告；构造 test 数据不等于使用测试成绩调试。记录实际耗时和存储，不预先承诺GPU耗时或成本。
- **协作节奏**：完整规格与启动确认后连续开发、按上述里程碑报告，新增决策或与规格冲突的事实再磋商，不要求每个里程碑例行暂停。本次规格整理或发布本身不表示实现、验收或正式实验已经完成。

## Testing Decisions

- **好测试的标准**：核验对使用者可见的输入/输出、统计结果和失败语义；预期值来自可手算的包、时间区间或已知标签预测，不用待测代码重复计算期望值，不锁定私有函数调用或内部模块划分。
- **主要边界**：沿用已确认 Q37 的验收范围，以 prepare→train→evaluate 的公共命令和输出产物为主边界，复用相同构造数据覆盖解析、区间、MTAM、清单、缓存和指标。以公开模型输入/输出补充五类与47类输出形状、有限交叉熵和反向梯度检查，避免为每个内部 helper 新建测试接口。用户已回复“符合，沿用上述边界”，此测试边界已确认。
- **包与方向行为**：构造具有已知 SYN 发起方、竞争候选、IPv6/MAC 关联和不可判定包的捕获，检查端点依据、上下行和跳过原因。覆盖无TCP不回退、MAC/地址冲突、控制包、重传、UDP/ICMP、非IP、IP长度与链路长度不同、载荷截断、IP头损坏和记录流损坏。验证损坏 Episode 不产生可用前缀缓存。
- **区间与MTAM行为**：用已知时间和长度的包检查1800槽完整区间、统计/方向顺序、内部边界、捕获终点、重复起点组、末组、无包/零时长和不可对齐区间。检查被排除区间不被前后片段吸收，片段异常不自动取消独立场景资格。
- **缓存与配置行为**：检查1024样本分片边界、末片、uint64精确计数、raw/log1p同源性、有效/排除清单定位和版本标识；中断后仅复用完成且兼容分片，不兼容解释原因。检查未知配置字段、非法值和运行项覆盖限制，不以大规模真实PCAP重复扫描充当单元测试。
- **训练评估行为**：用已知真实/预测标签核对主宏平均类别集合、固定47辅助指标、缺类支持数、同分选择和验证/测试隔离。验证best/last产物、兼容性拒绝、目标总轮数、恢复点之后指标归档、有效epoch不重复，以及正式缺训练类暂停不误伤smoke模式。
- **先例与工具**：现有仓库使用 unittest、临时目录和 scapy 构造 PCAP，并有官方划分、验证集选择及有限结果检查的先例。沿用这些方式；现有 fixture 的旧场景名称与仅UDP流量不满足新标签/端点契约，不能直接作为新任务的成功样本。按新契约构造 fixture，不把生产方向规则改成迎合旧测试。
- **真实数据验收**：在服务器无卡模式跑通两个任务、两种变换的预处理→训练→评估，并记录命令、提交、配置、实际样本数、输出和环境。它验证功能而非预测质量，不使用test调试、不要求覆盖47类，不作为正式30轮实验结果。
- **全量产物检查**：CPU验收通过后报告两任务各split有效/排除数、原因、逐类支持数和覆盖率，核验清单引用与缓存行可读取、标签映射一致。没有预设分类分数门槛，也不把抽样方向覆盖率当作全量验收结论。

## Out of Scope

- 风险二分类、复用旧风险目录作为本次监督目标、联合多任务模型、共享训练权重、层级分类和自动定位行为边界。
- 新的数据采集系统、重做官方划分、恢复全部缺失消息、猜测时钟偏移、人工补录端点或 UDP/53 方向回退。
- 保存切分PCAP、另存Episode级MTAM恢复缓存、每次复用都重哈希全部原始捕获、自动清理历史数据或实验。
- 自行设计额外投影/CNN、类别加权损失、超参数搜索、多种子追加、AMP/TF32/多进程优化或静默OOM降配。
- 在本次开发交付中开卡或启动四次正式GPU实验；后续训练预算与指标协议已明确，但执行需等待GPU可用及相应阶段启动。
- 声称逐参数复现AgentPrint、日志区间等同精确动作边界、规则端点等同身份真值、当前方向抽样等同全量验证，或将单种子结果包装成多种子统计。
- 由本规格整理任务代替此前尚待完成的开发启动整体确认。

## Further Notes

- 本规格从已确认的 Q1–Q20、Q22–Q62 合成；Q21 已撤回，原 Q20 的“仅明确身份、不推断”已由 Q54/Q58/Q59 的用户选择修订。其余未由用户采纳的历史提议不进入实施要求。
- 现有交付清点为3114份PCAP、2634份完整消息标注、56368条完整标注记录，原始捕获约105.36GB。它们是核验时的输入数量，不是已生成的有效样本数量；全量筛选后的支持数尚待实施统计。
- 全量文件头核验为classic PCAP/Ethernet，不代表全部报文记录完整。26个训练engine×scenario组合的小文件抽样中，当前IPv4规则可选23/26；该抽样偏向小文件，不提供全量覆盖率或方向准确率。
- 按所有输入均生成候选估算，uint64 MTAM约3.43GB，不含清单、文件头、权重、日志和临时产物；实际值在全量预处理后报告，分片不复制约105GB原始捕获。
- RF参考来源为 https://github.com/robust-fingerprinting/RF ，模型源码按UTF-8/LF规范化的SHA256为4cf9ff1a71a3601aeab61756327467add251475fa3bfd9c19eeb43e658effa19。逐层尺寸已核对，实际前向/反向仍是实现验收事项。
- 文档交付状态：这是待发布的issue正文草稿。尚未找到项目的issue-tracker与triage标签配置；按to-spec要求，先运行/setup-matt-pocock-skills，再发布到配置的tracker并应用ready-for-agent标签。草稿保存或Git推送不等同于issue已经发布，不创建额外triage步骤。
