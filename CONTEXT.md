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

**重复起点组（Coincident-start group）**：
同一 episode 中 start_time 相同的两条或以上标注记录，其内部的时间先后关系无法仅由该字段确定。

**给定片段（Annotated segment）**：
根据现有标注及明确区间规则构造、带有一个 primitive 类别的流量区间，是片段分类的样本单位。
_Avoid_：将给定片段与模型自动定位的片段混用

**MTAM**：
按时间槽分别汇总上下行包数和字节量的多视角流量表示。

**Agent 运行端（Runtime endpoint）**：
执行该 episode 中 Agent 的网络通信端。方向以此端为参照，而不是以每条连接的客户端/服务端身份为参照。

**上行／下行（Outgoing / Incoming）**：
Agent 运行端向外发送的包为上行，外部发送至运行端的包为下行。
_Avoid_：将“谁发起某条连接”直接等同于整个 episode 的上下行参照

**包长（Packet length）**：
本项目字节量统计采用的 IP 包总长度，包含 IP 头、运输层头和数据，不包含链路层头。
_Avoid_：将链路帧长、抓取长度或运输层载荷长与本项目包长混用

**RF 参考模型**：
AgentPrint 引用的 Robust Fingerprinting 卷积分类器，用于补充网络实现的可追溯参考。
_Avoid_：将此处 RF 解释为随机森林
