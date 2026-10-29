# 验证记录

此记录区分接口验证与完整主实验；没有用小样本分数代替论文结果。

## 已完成

- 在当前 `Benchmark/data/ANT` 上检查目录身份、split、标签覆盖及 manifest 生成，共 3114 个 episode：train 2703 / val 202 / test 209。risk 中 malicious 分别为 381 / 15 / 20。
- 三项协议测试通过：固定标签与跨 split 重复拒绝；聚合规则只根据 validation 选择（即使 test 偏好另一规则）；均匀概率下的 entropy confidence 退化处理。
- Python 源码语法及 JSON 配置解析检查通过。
- 三个测试时起始 checkpoint 与 `WEIGHTS.json` 的 SHA-256 均匹配。
- 使用合成 PCAP 完成下表的预处理、训练、预测及 episode 指标输出。合成集共 30 个 episode，三个 split 各 10 个、每个 scenario 类各 2 个；每 episode 4 条 UDP 双向流，每流 12 包。

| 模型 | scenario 端到端 | risk 端到端 | 计算环境 |
| --- | --- | --- | --- |
| FlowLens | 通过 | 通过 | CPU |
| YaTC | 通过 | 通过 | CUDA，1 epoch，batch 8 |
| NetMamba | 通过 | 通过 | 独立 Python 3.9 / CUDA 环境，1 epoch，batch 8 |
| TFusion | 通过 | 通过 | CPU，encoder 1 epoch |
| TrafficFormer | 通过 | 未单独运行 | CUDA，1 epoch，batch 8 |
| DF | 通过 | 未单独运行 | CUDA，1 epoch，batch 8 |
| AWF | 通过 | 通过 | CUDA / CPU，1 epoch，batch 8 |
| TikTok | 通过 | 未单独运行 | CUDA，1 epoch，batch 8 |
| VarCNN | 通过 | 未单独运行 | CUDA，1 epoch，batch 8 |
| NetCLR | 通过 | 未单独运行 | CUDA，1 epoch，batch 8 |
| TMWF | 通过 | 未单独运行 | CUDA，1 epoch，batch 8 |
| ARES | 通过 | 未单独运行 | CUDA，1 epoch，batch 8 |
| BAPM | 通过 | 未单独运行 | CUDA，1 epoch，batch 8 |

各模型成功输出的 test coverage 均为合成集的 10 / 10。risk / scenario 的模型构建与训练接口共用，未单独运行的组合不标为已经验证。

另将代码目录复制到原工程之外，清除 PYTHONPATH 后，使用独立副本完成公开数据检查，以及 FlowLens / AWF / TFusion 两个任务的顺序调度与汇总，确认这些路径不依赖原工程的代码目录和缓存。

## 重做小样本检查

```bash
python -m unittest discover -s tests -p 'test_*.py'
python tests/make_fixture.py /tmp/ant-fixture
python run_all.py --models FlowLens AWF TFusion --tasks risk scenario \
  --data /tmp/ant-fixture --output /tmp/ant-smoke \
  --device cpu --epochs 1 --batch-size 8
```

有匹配的 GPU 环境和起始权重后，可用相同数据检查其他模型。NetMamba 需使用独立解释器及已安装的 CUDA 扩展。

## 未覆盖

- 未重新训练全部公开数据上的 26 个正式实验，未检验论文表格数值是否一致。
- 未在空白系统上重新安装全部依赖或重编译 NetMamba 扩展；本次使用既有环境与匹配的本地扩展。包中只提供源码，不含扩展二进制。
- 小样本仅覆盖合成 UDP 流，不能替代真实数据规模、其他协议和异常 PCAP 的全面测试。
- 不证明历史各次运行与当前代码逐文件一致；现有记录不足以恢复所有历史上游 commit、checkpoint 与未保存参数。

安装版本、实际权重差异、数据数量和历史配置重建范围见 `README.md`、`WEIGHTS.json` 与 `configs/origins.json`。

## 采集源码整合检查

- 对新增的 20 个 Python 源文件进行静态语法检查，未调用真实智能体或网络采集。
- 通过工作目录元数据匹配会话的检查：目录命名可变化，忽略损坏行和不匹配的 episode。
- 验证缺少运行组件时，基础设施检查会在探测端口、创建文件或执行子进程前退出。
- 原有三项分类协议测试和上述两项离线检查均通过。
- 采集目录 `MANIFEST.json` 的已交付文件哈希一致。`check_runtime.py --source-only` 按预期失败：缺少 runner 等脚本引用及全部 8 个 mock 服务；不将此诊断失败表述为完整采集验证通过。
- 未验证完整采集、真实任务输入、第三方服务、全新环境安装或原始固定划分的生成。此次整合未重新训练 baseline。
