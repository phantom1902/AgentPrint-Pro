## Problem Statement

研究者已在ANT官方划分上完成AgentPrint五类场景分类及相关基线实验，希望进一步评估包到达间隔和不同时间观察范围是否有助于识别完整Episode的场景。

现有MTAM保存各方向的包数和IP字节量，没有保留逐包时间，因此无法直接从已有缓存还原IAT分布；现有CNN也未提供本次确定的3/5/7槽并行多核前端。仅增加输入通道或卷积层不足以保证拟合无验证集泄漏、缓存关联正确、模型恢复可靠，以及旧实验继续可读。

本需求是增加可复现、可审计的时间视角与多核网络能力，并先以自动化测试和有限真实样本确认完整流程可执行。本轮不以小样本分数证明效果提升，也不启动全量预处理或正式训练。

## Solution

扩展既有AgentPrint公共命令流程，提供显式fit-bins，再执行prepare、train、evaluate。以训练Episode等权分布拟合共享IAT桶，新增8个按方向区分的IAT计数通道，与原2个统计通道共同输入模型。

支持单核CNN、替换首单元多核、前置多核三种结构，以及IAT开/关和raw/log1p，共12种配置。多核表示网络分别观察3/5/7个时间槽后融合，保持原1800槽输入，不构造多分辨率分槽特征。

本轮只对IAT开启的6种配置进行真实小样本训练，每组2轮；复用原MTAM，保持旧协议兼容，并验证拟合、补充缓存、训练和评估的身份及恢复结果。正式训练接口实现已确认的停止规则，但本轮不执行正式训练。

## User Stories

1. As a researcher, I want five-class Scenario classification on complete Episodes, so that the extension answers the current task-category question.
2. As a researcher, I want the official split assignments preserved, so that observations remain comparable with existing experiments.
3. As a researcher, I want the original effective-packet and runtime-endpoint rules reused, so that IAT and MTAM describe the same traffic.
4. As a researcher, I want effective packets stably ordered by integer timestamps, so that IAT is nonnegative and reproducible.
5. As a researcher, I want equal-timestamp packets to retain capture-record order, so that their directional attribution is deterministic.
6. As a researcher, I want timestamp inversions reported, so that capture ordering problems remain visible.
7. As a researcher, I want adjacent effective packets across all directions and connections to define IAT, so that the feature describes the full Episode arrival sequence.
8. As a researcher, I want each interval assigned to the current packet's direction and time slot, so that counts have an explicit interpretation.
9. As a researcher, I want the first effective packet to contribute no fabricated interval, so that zero IAT means simultaneous observed arrivals.
10. As a researcher, I want the original full capture interval divided into 1800 slots, so that adding IAT does not redefine the observation horizon.
11. As a researcher, I want a dedicated zero-IAT bucket and seven buckets for intervals greater than zero, so that simultaneous arrivals remain distinguishable.
12. As a researcher, I want incoming and outgoing traffic to share bucket boundaries, so that a bucket has the same time meaning in both directions.
13. As a researcher, I want boundaries fitted using train only, so that validation and test observations do not determine the representation.
14. As a researcher, I want each eligible Episode to have equal total fitting weight, so that high-packet-count Episodes do not dominate the boundaries.
15. As an operator, I want at most 4096 fitting intervals sampled per Episode, so that fitting uses a bounded per-Episode sample.
16. As a researcher, I want sampling derived from seed 42 and stable sample identity, so that traversal order and training seed do not alter the fit.
17. As a researcher, I want weighted empirical quantiles to use observed integer values, so that boundary definitions are exact and inspectable.
18. As a researcher, I want duplicate boundaries retained and empty buckets reported, so that the fixed eight-channel representation never relies on invented timing noise.
19. As a researcher, I want intervals exactly on a boundary assigned to the right bucket, so that every edge case follows a documented rule.
20. As a researcher, I want Episodes without intervals greater than zero excluded only from fitting, so that their classification eligibility remains unchanged.
21. As an operator, I want fitting to fail when no usable positive interval exists anywhere in the fitting pool, so that invalid boundaries are never fabricated.
22. As a researcher, I want all effective intervals counted during feature generation, so that the fitting sample does not become a feature-count subsample.
23. As a researcher, I want all model variants and transforms to share one fitted artifact, so that comparisons use the same timing categories.
24. As an operator, I want independent IAT shards joined to existing MTAM by sample identity, so that original caches can be reused without matching shard boundaries.
25. As an operator, I want uint64 counts and 256-sample IAT shards, so that storage precision and shard size are predictable.
26. As an operator, I want missing or incompatible base caches rejected explicitly, so that preparation never silently changes the base statistics.
27. As a researcher, I want raw and log1p applied consistently to every input statistic, so that IAT counts follow the same transform choice as MTAM.
28. As a researcher, I want parallel time kernels of lengths 3, 5, and 7, so that the model can combine different local observation ranges.
29. As a researcher, I want both replacement and prepend variants available independently, so that their different integration choices can be evaluated later.
30. As a researcher, I want a fixed shared branch-channel budget and actual parameter counts reported, so that capacity changes remain visible.
31. As a researcher, I want every model initialized from scratch under the existing RF initialization rules, so that the extension does not depend on transferring a previous best checkpoint.
32. As a researcher, I want all twelve feature/model/transform configurations supported, so that later controlled comparisons do not require another interface redesign.
33. As a researcher, I want only the six IAT-enabled configurations trained during current acceptance, so that original MTAM experiments are not repeated.
34. As a researcher, I want one approved optimization anchor shared across the current variants, so that this implementation does not introduce another parameter search.
35. As a researcher, I want ordinary cross-entropy and validation Macro-F1 selection, so that the training objective and model-selection rule stay explicit.
36. As a researcher, I want early checkpoint ties resolved in favor of the earlier epoch, so that selection is deterministic.
37. As a researcher, I want formal training to stop under the approved joint F1/loss plateau rule with a 200-epoch cap, so that budget exhaustion is not misreported as convergence.
38. As an operator, I want deterministic FP32 execution with explicit worker and thread settings, so that interrupted and uninterrupted runs can be compared reliably.
39. As an operator, I want nonfinite losses, resource exhaustion, and OOM reported without silently reducing the run, so that failures do not redefine the experiment.
40. As an operator, I want fitting, preparation, and training to resume at their committed boundaries, so that interruptions do not force all work to restart.
41. As a researcher, I want feature, fit, cache, model, transform, and training identities checked together, so that artifacts from different experiments cannot be mixed.
42. As a researcher, I want evaluation to reconstruct the saved model definition, so that current command-line switches cannot reinterpret a checkpoint.
43. As an existing user, I want genuine old configurations, caches, and checkpoints to remain readable and resumable, so that this extension preserves completed research.
44. As an existing user, I want disabled extensions to preserve original initialization and output under the original settings, so that compatibility is measured against the real old protocol.
45. As a researcher, I want a deterministic small selection of train and validation Episodes without annotation-presence filtering, so that Scenario acceptance uses its own eligibility rules.
46. As a researcher, I want smoke fitting artifacts distinguished from full-data artifacts, so that acceptance outputs cannot accidentally become the formal representation.
47. As a researcher, I want each of the six acceptance runs limited to seed 42 and two epochs, so that the agreed small-sample budget is respected.
48. As a researcher, I want real test data untouched during this work, so that implementation validation cannot influence later test evaluation.
49. As a maintainer, I want tests to assert public commands, outputs, and restoration behavior, so that internal refactoring does not invalidate the test suite unnecessarily.
50. As an operator, I want all remote artifacts kept on the designated data disk, so that replacing the system image does not erase the experiment state.
51. As a reviewer, I want Git revision, sample selection, artifact identities, hardware, parameter counts, and failures recorded, so that acceptance claims can be audited.

## Implementation Decisions

1. **任务与资格**：新增功能用于五类完整Episode场景分类；沿用官方划分、现有标签、运行端方向、有效包、样本资格和完整捕获区间。保留既有Primitive流程兼容，不在本轮新增47类功能。
2. **IAT语义**：单Episode有效包按整数纳秒稳定排序，同时间戳保留原记录顺序，并报告时间逆序情况。前驱跨方向、跨连接；间隔等于当前包时间减去前一有效包时间，按当前包方向计数。首包无IAT，但仍贡献原MTAM。
3. **时间归属**：每个间隔归当前包所在的原1800槽，内部边界归新槽，捕获结束点归最后槽。过滤后首末包不重定义捕获范围。退化时间区间沿用已有资格规则。
4. **分桶**：固定8桶，零间隔独占桶0，桶1–7用于大于零的间隔。入站/出站共享六个边界，分别统计；末桶无上限。大于零的间隔恰落边界时归右桶，重复边界保留，不加抖动，允许并报告空桶。
5. **拟合权重**：仅train参与。每个Episode最多4096个大于零间隔，均匀无放回抽样，不足全取；固定seed 42结合稳定sample_id派生种子。参与Episode总数为E，第e个抽样数为m_e，每值权重1/(E×m_e)，不是全局逐包等权或平均各Episode分位点。
6. **经验分位点**：按加权经验逆CDF取1/7至6/7六个分位点，取累计权重达到目标比例的实际整数纳秒值，不插值。无大于零间隔的Episode只退出拟合并报告，分类资格不变；全拟合池无此类间隔则失败。
7. **拟合与统计分离**：抽样只用于边界拟合，prepare统计全部有效间隔。所有模型、输入变换和训练种子共用拟合产物；产物记录train清单、来源身份、规则、边界、统计及指纹。smoke与全量拟合身份明确区分。
8. **排序资源**：逐Episode保存紧凑数值记录并在内存稳定排序，不保存原始包内容或完整逐包序列产物。资源不足停止报告，不自动跳过样本，也不引入外部排序。
9. **表示**：基础MTAM为uint64、2统计×2方向×1800槽；IAT补充为uint64、8桶×2方向×1800槽。合并输入为批次×10通道×2方向×1800槽，关闭IAT时保留2通道。字节统计继续使用IP包总长度。
10. **变换**：raw保留计数；log1p作用于全部输入统计，包含IAT计数；不添加标准化、额外归一化或特征重分槽。
11. **缓存关联**：IAT独立缓存，256样本一片，末片可不足；通过sample_id与原1024样本分片关联，不要求分片边界一致。每个IAT样本225 KiB，整片数组56.25 MiB，不含元数据；新增数组空间为原MTAM的4倍。
12. **缓存前置条件**：补充缓存绑定sample_id、来源身份、基础缓存身份和拟合指纹。基础缓存缺失、身份不匹配或缺少所选sample_id时停止报告，不自动补建。IAT仍须读取原PCAP，不能从原聚合缓存反推。
13. **模型矩阵**：支持IAT开/关×单核/替换型/前置型×raw/log1p共12配置。网络保留RF二维前端、原reshape及一维分类器总体结构，输出五类logits；多核置于reshape之前，作用于原始时间轴。
14. **多核结构**：三分支方向核高均为3，时间核长3/5/7，stride 1，方向补零1，时间补零1/2/3，分支输出时间长度1800。输出通道22/21/21，每支Conv→BN→ReLU，拼接64通道后以1×1 Conv→BN→ReLU融合。核高3不意味着存在第三种方向。
15. **两种接法**：替换型整体替换原第一个Conv/BN/ReLU单元，融合投影64→64，再接原后续前端，不重复旧首个BN/ReLU，不裁剪回旧首层长度。前置型投影64→输入2或10通道，再进入完整CNN；无输入残差。两种接法独立，默认不串联。
16. **容量与初始化**：共同width=2，原CNN首层输出64。分支分摊固定64通道中间预算，不保证严格等参数，报告实际参数量。各变体从头训练；新增Conv2d沿用RF初始化规则，Conv1d保留既有默认行为，不迁移旧最优权重。
17. **训练锚点**：Adam，初始学习率0.001，weight_decay 0.001，batch 200，dropout_2d 0.05，dropout_1d 0.1。学习率为初始值乘0.5的epoch/30次方，下限0.000001，epoch从0计。两种输入变换共享参数，不调用旧硬编码0.0005/0.2学习率公式。
18. **损失和选模**：不加权交叉熵；按实际最佳验证Macro-F1选checkpoint，同分取较早epoch。停止判断的0.001提升阈值不限制最佳checkpoint更新；不使用test挑参数或选模型。
19. **正式停止接口**：至少60、最多200轮，沿用既有显著提升追踪语义：连续20轮无至少0.001的验证Macro-F1提升，且最近两段各10轮平均验证交叉熵相对改善不足1%，两条件同时满足才判定收敛。前段损失均值为0时相对改善按0处理；达到上限但不满足条件须明确报告未收敛。本轮实际为2轮smoke预算，不宣称收敛。
20. **执行规则**：固定种子、FP32，关闭AMP、TF32及cudnn benchmark；DataLoader workers为0，数值线程为4。OOM、资源不足或非有限损失停止报告，不自动更换样本规模、模型规模或数值协议。
21. **公共接口**：扩展既有AgentPrint公共CLI；新增显式fit-bins，prepare读取已完成拟合产物，不隐式拟合，再进入train、evaluate。新特征及模型使用独立扩展版本，新配置schema_version为2。
22. **职责边界**：拟合模块负责train分布和边界；补充缓存模块负责IAT计数和身份；数据加载负责sample_id关联和输入变换；模型构造负责三种前端；训练/评估负责指标、checkpoint和恢复。优先复用现有解析、指标与恢复能力，不进行无关重构。
23. **旧协议兼容**：v1继续严格按旧规则解析，不自动注入新字段或改写旧预处理指纹；旧缓存与真正旧checkpoint仍可读取、评估、恢复。关闭新增模块的初始化/输出回归采用原协议设置，不把width=2锚点误当旧默认模型。
24. **实验身份**：特征定义、模型结构、拟合指纹、manifest、基础及补充缓存、输入变换、训练参数与种子共同进入身份。候选选模范围包含特征/模型身份，不只按raw/log1p分组。评估和恢复按checkpoint保存定义重建模型并核验身份，不只信任当前CLI开关。
25. **恢复边界**：fit-bins按Episode保存完成进度、最多4096抽样值及身份；prepare按完整256样本IAT分片恢复；train按完整epoch恢复模型、Adam、随机数、shuffle、最佳模型及停止历史，沿用超前日志处理。身份变化必须拒绝复用。
26. **真实验收选择**：从现有有效清单按每类最多4个train、2个val选取，总计最多20/10；优先PCAP小者，并列按sample_id。不额外要求消息标注存在。仅所选train用于smoke拟合，所选train/val用于补充缓存和真实验收。
27. **真实验收预算**：只运行IAT开启的六配置，各seed 42、2轮，batch 200自然形成小批次，其余锚点不变，串行执行；CPU做单元/集成测试，真实六配置优先可用GPU，无GPU则CPU。原MTAM不重复训练实验。
28. **产物和版本**：本地编辑及Git提交，服务器全部新增产物位于已指定的ant-workspace数据盘根目录内；不改写原始数据。远程验收前先提交实现，报告记录Git版本、样本清单、拟合和缓存身份、参数量、硬件、执行轮数、指标、恢复证据和失败项。

## Testing Decisions

1. **已确认测试边界**：用户在本次to-spec核对中确认，以公共fit-bins→prepare→train→evaluate命令及拟合产物、缓存、指标、身份核验和恢复结果作为主要边界；仅补充公开模型接口的形状、交叉熵、反向传播及旧模型一致性测试。
2. **好测试的标准**：断言可观察行为、可手算计数、明确失败原因及恢复后的等价结果，不断言内部辅助函数调用次序、私有结构或纯实现镜像。以少量确定性合成PCAP和已知清单驱动公共入口，覆盖正常路径和关键反例。
3. **拟合和特征**：通过拟合产物与prepare输出验证跨方向/连接前驱、稳定时间排序、首包、零间隔、当前包方向/槽号、六个加权分位点、边界归右、重复边界、末桶、无大于零间隔、train-only、等权抽样及全量计数。
4. **确定性和身份**：改变输入遍历顺序不改变拟合结果；不同训练种子共享同一拟合。smoke与全量身份分离。对错误拟合、变更来源、不匹配缓存、缺失sample_id和不同模型候选混用，公共流程必须拒绝并报告。
5. **缓存和恢复**：验证uint64计数与维度、256样本末片、按sample_id跨原分片关联；在拟合Episode提交和IAT完整分片提交附近制造中断，对比恢复与一次完成的最终产物及身份。
6. **模型公开接口**：支持矩阵具有五类logits、有限交叉熵和有效反向梯度；覆盖两种独立多核接法与全部输入变换。使用原参数和固定种子验证关闭模块时旧初始化和输出一致。此类模型接口检查不作为新增真实训练实验。
7. **训练和评估**：通过公共命令的日志、checkpoint及评估输出验证验证集选模、同分规则、特征/模型候选隔离、模型重建、学习率、严格停止阈值及上限不误报收敛。完整epoch恢复后，模型、Adam、随机数、shuffle及选模结果与不中断执行一致。
8. **旧协议**：使用真正旧v1配置、缓存和checkpoint验证读取、评估与恢复；不能只让新代码生成一套带旧标签的新产物冒充兼容证明。
9. **test隔离**：真实验收只读取train/val，不读取真实test；显式test评估路径只用合成夹具验证。选模阶段不依赖test缓存存在。
10. **已有测试先例**：复用既有合成PCAP手算与公共CLI端到端测试、缓存来源与恢复测试、完整epoch模型/优化器/RNG等价测试、调参模型初始化/输出一致性测试、CPU小样本验收方法。现有双任务验收器的标注筛选和任务矩阵不能直接套用于本次六配置场景验收。
11. **真实有界验收**：六组IAT开启配置分别完成所选样本拟合、补充缓存、2轮训练、最佳checkpoint验证集评估和恢复路径核验。报告逐项区分通过、失败与未执行；smoke指标不排名方法、不证明论文效果或收敛。

## Out of Scope

- 本轮全量IAT预处理、正式长程训练、参数优化和多种子效果比较；实现正式停止接口不意味着获得正式训练执行预算。
- IAT关闭配置的真实重复训练、原MTAM重新训练、已有基线实验重跑。
- 47类Primitive新功能、自动片段/Primitive边界定位、Episode分层决策流程和TCN。
- 1800/900/450等多分辨率分槽、外部排序、残差前置、多核两种接法默认串联、预训练权重迁移、严格等参数容量匹配。
- 验证集或测试集参与桶边界拟合、真实test读取、test挑参数或选模。
- 隐式拟合、基础缓存自动补建、静默忽略资源失败、自动降规模、额外标准化及无关工程重构。

## Further Notes

- 本规格由Q1–Q43已确认决定及本次明确确认的测试边界整理而成，不重新启动需求访谈。实验效果是否提升仍需后续正式实验验证。
- 上一轮功能已在本地合入main；本轮规格位于独立的codex/agentprint-iat-multikernel-dev分支。当前交付只包含规格整理，尚未开始IAT实现或远程验收。
- 公共命令名、维度、数学边界和预算为本规格合同；实际内部拆分可以在不改变合同和测试边界的前提下形成实施计划。
- 发布状态：这是待发布的Issue正文。项目尚未配置Issue tracker及triage标签映射，应按to-spec技能要求运行/setup-matt-pocock-skills。配置完成后发布至所选tracker，并应用ready-for-agent标签，无需额外triage；保存文档或提交Git不等同于发布Issue或已应用标签。
- GitHub连接在最近一次同步尝试中失败；本地分支整理已完成，不据此声称远端main已更新。
