## What to build

T07：在服务器完成四配置CPU真实数据验收。

研究者可以在当前无卡服务器上，用真实ANT训练/验证小样本跑通两个任务的raw/log1p完整流程，得到提交SHA、配置、环境、测试与产物证据，判断全量预处理是否可以启动。

## Acceptance criteria

- [ ] 将同一开发提交同步至已约定的服务器代码位置，记录SHA；已有checkout修改或位置冲突保留并报告，不reset覆盖，不在并发代理间相互修改同一checkout。
- [ ] 重新核验当前Python/PyTorch与最小依赖，仅补缺失项，记录实际版本；运行torch、numpy、PyYAML、dpkt、scikit-learn、tensorboard及unittest/scapy所需检查，不安装整套无关baseline依赖。
- [ ] 集成T02–T06后执行已确认自动化测试，以公共命令产物为主边界，补充两输出头的CE/反向检查；验证Primitive使用完整方向与可恢复缓存，而不只各分支局部成功。
- [ ] 场景最多20个训练、10个验证Episode，尽量覆盖五场景；Primitive从中最多取64个训练、32个验证片段。固定seed42，独立smoke配置和选择清单，仅用官方train/val，不用test调试。
- [ ] 四配置各1epoch、batch4在CPU实际完成prepare→train→evaluate；支持数不足报告实际值，不要求小样本覆盖47类，不降低质量规则，无法运行则明确报告验收未完成。
- [ ] 核验日志、TensorBoard、best/last、逐类/混淆矩阵、缓存与清单引用可读取；记录可复现命令、实际样本数、耗时和限制，不以smoke分数声明模型性能。
- [ ] 服务器环境变更和重负载验证串行运行，不开卡、不启动正式GPU训练；所有阻塞票完成且集成后本票才可标记验收通过。

## Blocked by

- T03：按完整方向与流量质量规则生成样本。
- T04：中断后恢复兼容的分片预处理。
- T05：完成raw/log1p比较与正式实验协议。
- T06：恢复训练并保留一致的日志与结果。
