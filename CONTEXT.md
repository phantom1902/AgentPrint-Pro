# AgentPrint 在 ANT 上的分类

本术语表定义两个独立流量分类任务使用的语言。

## Language

**Episode**：
一次 Agent 任务执行对应的完整流量轨迹，是场景分类的样本单位。
_Avoid_：将 episode 与其中一个行为片段混称为“样本段”

**场景（Scenario）**：
Episode 的五类任务类别：artifact_processing、general_knowledge、scientific_evidence、business_services、security_testing。
_Avoid_：使用“任务类别”同时指场景和 primitive 类别

**Primitive 类别**：
消息标注中的行为类别，本次研究采用当前 N001–N047 标签本体。
_Avoid_：把 primitive 类别称为五类场景

**标注起点（Annotation start）**：
消息标注记录的 start_time，是构造给定片段的时间依据。
_Avoid_：未经对齐核验就称其为工具执行的精确物理起点

**给定片段（Annotated segment）**：
根据现有标注及明确区间规则构造、带有一个 primitive 类别的流量区间，是片段分类的样本单位。
_Avoid_：将给定片段与模型自动定位的片段混用

**MTAM**：
按时间槽分别汇总上下行包数和字节量的多视角流量表示。

**RF 参考模型**：
AgentPrint 引用的 Robust Fingerprinting 卷积分类器，用于补充网络实现的可追溯参考。
_Avoid_：将此处 RF 解释为随机森林
