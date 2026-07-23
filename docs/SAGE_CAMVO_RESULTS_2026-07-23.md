# SAGE-CaMVo：CASIE + Mordor 冻结真实响应结果

## 实验协议

- 模型响应：Adams 四模型真实输出，离线冻结重放；
- 新增模型调用：0；
- CASIE：24,470 个响应快照中的 2,820 条四模型完整、公平事件；
- Mordor：1,000 条任务中的 913 条四模型完整事件；
- 超参数：一个通用配置，由 CASIE 与 Mordor 的平均 validation Macro-F1 增量共同选择；
- 测试标签：不参与配置选择；
- 在线图反馈：预测后一步到达，只影响未来节点；
- 统计：按 document / incident-window 做 2,000 次 paired cluster bootstrap。

## 主结果

| 数据集 | 方法 | Accuracy | Macro-F1 | 平均模型数 | 成本（USD） | 图介入率 |
|---|---|---:|---:|---:|---:|---:|
| CASIE | CaMVo | 0.8886 | 0.8874 | 2.028 | 2.7082 | 0% |
| CASIE | **SAGE-CaMVo** | **0.8916** | **0.8905** | **2.028** | **2.7082** | 2.89% |
| Mordor | CaMVo | 0.6006 | 0.4495 | 2.297 | 2.8201 | 0% |
| Mordor | **SAGE-CaMVo** | **0.6054** | **0.4598** | **2.297** | **2.8201** | 6.23% |

配对差异：

| 数据集 | Macro-F1 增量 | 95% cluster-bootstrap CI | 候选更优概率 |
|---|---:|---:|---:|
| CASIE | **+0.00309** | **[+0.00099, +0.00576]** | 99.7% |
| Mordor | **+0.01033** | **[+0.00248, +0.03025]** | 97.55% |

这是目前仓库中第一次在同一通用配置下，让图增强相对 CaMVo 在 CASIE 和 Mordor 两个真实响应缓存上同时取得正增量，并保持调用模型数和费用完全不变。

## 图真正学到了什么

CASIE 测试结束时：

| 关系 | Gate mean | Active utility | 转移有效观测 |
|---|---:|---:|---:|
| same_hopper_sequence | 0.9677 | 0.9288 | 987.7 |
| document_sequence | 0.7457 | 0.4696 | 186.5 |

Mordor：

| 关系 | Gate mean | Active utility | 转移有效观测 |
|---|---:|---:|---:|
| provenance_process | 0.8815 | 0.7618 | 949.6 |
| provenance_host | 0.9267 | 0.8420 | 405.7 |
| provenance_network | 0.6330 | **0（未成熟，被剪掉）** | 7.6 |

这符合安全语义：当前 Mordor 分片中的 process/host 链路具有可复用信息，而 network 边样本太少，算法没有让它干预决策。

## Safe fallback 是否真的工作

CASIE 1,974 个测试事件中只有 57 个允许图介入：

- 1,334 次：图没有降低不确定性，退回 CaMVo；
- 322 次：基础投票足够确定，受保护；
- 246 次：没有成熟且有益的边；
- 15 次：图消息与基础票差异异常；
- 57 次：通过全部保护条件。

Mordor 626 个测试事件中 39 个允许图介入，其余绝大多数退回 CaMVo。增益并非来自全局标签偏置，而是集中在少数基础票不确定、且关系证据成熟的轮次。

## 统一验证选择的结果

被选配置主要参数：

- `max_graph_blend=0.20`；
- `gate_activation_threshold=0.45`；
- `protect_base_margin=0.20`；
- `require_entropy_reduction=true`；
- `dynamic_entity_edges=false`。

最后一项值得强调：代码支持在线实体候选边，但 validation 发现“只因为共享实体而新建边”没有增益，因此本次冻结配置主动关闭了它。当前结果依然是动态图：已有候选边的关系门会随反馈激活、衰减、剪枝或重新激活。将来 OpTC 提供更完整 process/network 实体后，可重新验证在线候选边。

## Claim boundary

这是一项可信的算法开发结果，还不是最终顶会主表：

- CASIE 使用在途矩阵的 complete-case 快照，可能受 provider 完成顺序影响；
- Mordor 只有 913 个完整事件、13 个测试簇；
- Mordor 的 benign 是未命中精确恶意时间戳的匹配控制，并非人工逐条确认；
- 线上模式假设审计标签在预测后到达；需要补充反馈延迟和稀疏反馈消融；
- 最终结果应在完整 CASIE 33,940 单元和未参与开发的 OpTC provenance 分片上冻结复核。

机器可读结果：`artifacts/safe_adaptive_graph/RESULTS.json`。
