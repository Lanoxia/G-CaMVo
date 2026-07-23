# G-CaMVo 正式实验协议

本文档区分“代码/管线验收”和“可写进论文的经验结论”。任何真实改进声明都应在冻结测试集和真实模型输出上产生。

## 1. 研究问题

- **RQ1（成本—质量）**：在相同模型池和相同模型输出下，CaMVo 是否比固定策略减少成本，同时维持 Macro-F1？
- **RQ2（图增益）**：G-CaMVo 相对 CaMVo 是否改善 Macro-F1、malicious recall 或达到相同质量时的费用？
- **RQ3（适用条件）**：增益是否随事件难度、攻击阶段、图同质性、节点度数和时间间隔变化？
- **RQ4（稳健性）**：结果是否跨随机种子、attack day、host、模型池和价格扰动稳定？
- **RQ5（风险控制）**：TRACE-GCaMVo 能否在预先声明的质量差容忍度内，以更低成本停止，并把未解决案例升级给分析员？

## 2. 三层证据

### Layer A：工程验收

使用 CASIE 真实文本/金标签或 OpTC 真实正例拓扑，模型响应和价格模拟。验证 parser、graph、router、cache、budget、metric、sweep 均可运行。不得表述为真实 LLM 性能。

### Layer B：小额真实模型 PoC

每个数据集先选 30—100 个样本，使用至少 3 个模型、固定解码和固定 prompt。检查输出合法率、token 估计误差、缓存命中、预算账本和图边质量。若 Dify/Adams 不允许修改 temperature，必须记录平台固定值，并依靠共享缓存保证所有路由策略比较同一组响应；不能把固定 `0.7` 误写成确定性采样。该层用于排错和估算正式实验成本。

### Layer C：冻结的正式评估

在验证集完成模型 prior、`delta`、`lambda`、边权和 prompt 选择后冻结配置。测试集只运行一次主结果；任何后续选择都必须视为新的探索实验。

## 3. 数据和标签

### CASIE

- 单位：事件 mention；
- 金标签：五类 event subtype；
- 分组：`document_id`；
- 图：同 hopper 强边，同文档弱边；
- 限制：文档关系图不是主机 provenance graph，只适合辅助验证方法通用性。

### OpTC

- 正例：社区 best-effort event-ID 标签与 attack-period eCAR 的 1:1 join；
- 负例：官方 benign collection period 中，与正例 object/action 分布匹配的 eCAR；
- 禁止：将 attack-period 中“未标注”的行当作 benign；
- 图：共享 actor/object/host/principal 并带时间衰减的 correlation graph；
- 限制：benign-period negatives 存在 temporal/domain shift，应增加时间特征审计和外部 benign 数据复核。

## 4. 切分

推荐比例为 calibration 20%、validation 10%、test 70%，实际比例可按场景数调整，但必须先固定。

- CASIE：整篇文档进入同一 partition，使用 `grouped_split(..., group_key=document_id)`；
- OpTC：优先按 attack day/scenario 切分；场景不足时按 host-time block 分组，并明确这不是跨场景泛化；
- 图边：切分后移除跨 partition 边；在线测试时只允许当前节点读取测试流中已经出现的邻居；
- 抽样：大 eCAR 使用确定性 bottom-k hash 全流抽样，禁止只读文件前 N 行。

## 5. 模型和 prompt 控制

- 至少包含 cheap、mid、strong 三档；
- 固定 provider model version、temperature、输出上限、system prompt 和 JSON schema；
- 保存当前官方价格来源和查询日期；
- `prior_quality` 仅用 calibration/validation 估计，不能用 test；
- 同一模型 × 样本响应通过共享 cache 供所有策略复用；
- 更换 endpoint/model/token 参数会改变缓存指纹。
- 模型 confidence 必须只用 calibration split 校准；如果接口不返回 confidence，统一使用可审计的缺省值，不能在 test 上调参。

## 6. 必须比较的策略

1. Cheapest single；
2. Strongest single；
3. Fixed cheap ensemble；
4. Full ensemble；
5. CaMVo；
6. G-CaMVo；
7. TRACE-GCaMVo；
8. G-CaMVo `lambda=0` 消融（应与 CaMVo 相同）；
9. TRACE `no graph`、`no redundancy correction`、`no calibration` 消融；
10. 若预算允许：随机成本匹配策略和简单 degree/uncertainty escalation heuristic。

## 7. 参数选择

最小网格：

- `delta ∈ {0.85, 0.92, 0.97}`；
- `lambda ∈ {0, 0.3, 1, 3}`；
- 另做 graph edge-weight cap、warm-up 和 correlation window 消融。
- TRACE 风险阈值至少扫描 `{0.20, 0.10, 0.05, 0.03, 0.02, 0.01}`，先在 validation 上选定，再冻结到 test；
- 图关系转移的最小观测数、信息性门槛和最大消息权重也只能在 validation 上选择。

选择标准应提前定义，例如：在 validation Macro-F1 不低于 strongest single 1 个百分点的条件下最小化成本；若无策略满足，则如实报告不可行。

## 8. 报告指标

质量：Accuracy、Macro Precision/Recall/F1、每类 F1、malicious precision/recall、按难度/攻击阶段分层结果。

效率：总成本、每项成本、平均模型数、升级率、全池率、模型选择率、并行/串行平均延迟、P95 延迟。

稳健性：至少 5 个随机种子，报告 mean ± std；正式主结论对逐样本配对差异做 cluster bootstrap confidence interval。CASIE 以 document 为重采样单位；OpTC 以 incident/scenario 或 host-time block 为单位，不能把高度相关的事件当独立样本。

风险报告：TRACE 额外报告决策风险、风险阈值、abstention/analyst-escalation rate、可靠性图（reliability diagram）和 expected calibration error。内部后验风险在完成独立校准前不是安全保证。

## 9. 失败与敏感性分析

必须记录：

- 图正则造成的新增 false positives/false negatives；
- 高度节点、跨用户/跨主机边是否导致错误扩散；
- 模型相关错误导致 full ensemble 弱于 strong single 的情况；
- `delta` 增大但质量不单调的情况；
- API 解析失败、超时、预算阻断和缓存异常；
- 价格 ±25%、延迟 ±25% 时策略选择是否稳定。
- 图采样前后节点数、边数、孤立率、连通分量、最大分量和平均度；若抽样后图退化，停止做图增益结论。

## 10. 可接受的结论表述

在无 Key 模拟层，只能说“实现可运行”“观察到候选信号”“发现某些参数下负增益”。只有 Layer C 在真实模型、冻结测试和多场景验证后，才可说“G-CaMVo/TRACE-GCaMVo 相对基线改善 X，并节省 Y”。若质量置信区间跨 0，应使用“在给定区间内未观察到明确质量差异，同时成本下降 Y”的等价性措辞，不能写“显著提升”。
