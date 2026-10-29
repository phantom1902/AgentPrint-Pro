# ANT 运行端信息与抓包文件头：只读核验

本轮为设计磋商补充现场事实。核验通过服务器 SSH 只读完成，未安装依赖、修改原始数据或更改 GPU 状态；未全量扫描 PCAP 报文正文。

## 检查范围

数据根目录：`/root/autodl-tmp/ant-workspace/data/raw/ANT`。

全量文件清单共 5751 个文件：3114 个 network.pcap、2634 个完整 messages.jsonl、1 个 messages.jsonl.incomplete，以及根 README.md、.gitattributes。未发现逐 episode 的采集配置、主机/接口记录或端点映射文件。

全量 PCAP 检查仅读取每文件前 32 字节：3114/3114 均为 classic PCAP，小端 magic d4c3b2a1、微秒时间精度、版本 2.4、linktype=1（Ethernet）。未见 pcapng、短或异常文件头。该检查不能证明全部包记录完整，也不能证明消息与流量的时间对应关系。

完整标注共 2634 个文件、61953713 字节（约 59.084 MiB）；先按现存 engine×scenario 组合抽查 26 份首记录，随后流式遍历全部完整文件。共 56368 条记录，JSON 解析错误、非对象记录和空行均为 0；未读取 .incomplete。

全部完整标注的结构化字段中，未发现明确的 host/client/endpoint/interface/IP 身份字段；嵌套字典（包括数组中的嵌套字典）为 0。字段内容为消息、任务、类别、primitive、时间等，tool_evidence 字段仅出现 7 次。未输出消息正文，也未将文本中可能出现的 IP 当作运行端元数据。

同时检查数据根和工作区 README.md，以及 docs/environment.json、docs/ant_repository_info.json、docs/ant_repository_metadata.json、docs/ANT_DOWNLOAD_PROGRESS.md；没有找到原始采集运行端地址或接口配置。environment.json 的 hostname 属于当前工作区，不能证明原始采集端身份；README 对 user-side traffic 的概述也不提供具体地址映射。

## 可确定与不可确定

- 可确定：当前实际交付文件、全部完整标注记录和上述全局资料中，没有找到可直接读取的 Agent 运行端 IP/接口身份映射。
- 可确定：经典 PCAP 全局头只描述捕获格式等信息，不提供“哪个地址运行 Agent”的角色映射。
- 不能据此确定：作者原始采集系统是否曾保存这类配置；本轮未检查作者未交付的材料。
- 普通报文源/目的 IP、主要私网端点、首包源地址和 SYN 发起方可用于规则推断，但不能冒充明确的运行端元数据。

## 对当前设计的影响

Q20 曾按“必须有已有明确运行端信息，缺失跳过”记录。现场事实说明这不是已知仅影响少量异常样本的分支；按该严格口径，目前没有足够的直接身份依据来构造所需方向输入。

Q54 将请用户澄清是否允许将 PCAP 中的地址/连接特征纳入“已有信息”，使用固定、可审计的规则识别运行端，无法明确时再跳过。推荐不代表采纳；Q54 回答前不静默改用任何启发式，也不把本报告当成已验证方向准确率。

其余已确认任务、标签、MTAM、损失、训练预算与工程选择保留。本报告是设计事实核验，不是预处理或训练完成报告。
