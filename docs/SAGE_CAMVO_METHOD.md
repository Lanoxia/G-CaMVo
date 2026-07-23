# SAGE-CaMVo：可演化、可剪枝、可安全回退的在线攻击图投票

SAGE-CaMVo（**Safe Adaptive Graph Evolution for CaMVo**）是对旧 TRACE 图平滑的替代方案。它不再假设“相邻告警应当同标签”，而是把图本身作为在线学习对象：候选边可以由 provenance 数据产生，也可以由过去出现的主机、进程、网络实体主动提出；每种关系的标签转移规律和实际效用随审计反馈更新，低价值边会失活，环境变化后又能重新激活。

实现位于 `src/camvo/adaptive_graph_router.py`。

## 1. 保留 CaMVo，不另起炉灶

第 `t` 轮仍严格执行原 CaMVo：

1. 对 `n` 个模型计算 contextual-bandit 可靠性；
2. 在看到本轮回答之前选择一个满足置信度要求的最低成本子集 `S_t`；
3. 调用 `S_t` 中的多个模型；
4. 使用 CaMVo 权重 `ω_i,t = μ_i,t-1 q_i,t(x_t)` 进行离散加权投票；
5. 用模型是否同意本轮模型共识更新 LinUCB/Beta 状态。

图头不会改变模型能力奖励，也不会把算法改造成“每次只问一个模型”。

基础投票分布为：

```text
p0_t(y) = sum_{i in S_t} ω_i,t 1[y_i,t = y] / sum_{i in S_t} ω_i,t.
```

因此，没有可信图证据时，SAGE-CaMVo 的标签、模型子集和费用与 CaMVo 完全相同。

## 2. 在线演化图

### 2.1 候选边

当前节点 `t` 只允许连接到已经处理的节点 `u < t`。候选集合来自两条通道：

- **seed provenance edges**：CASIE 的同 hopper 顺序边、文章顺序边；Mordor/OpTC 的 host、process、network 因果边；
- **online entity proposer**：从有限大小的历史记忆中，为重复 document、hopper、host、process、network、user 实体提出最近父节点。

在线 proposer 只是候选生成器，不能直接影响答案。是否真正激活由验证集选择和下面的关系效用门控决定。本次双数据集验证自动关闭了仅依赖实体重复的新增边，因为它没有带来验证增益；这是一项数据驱动的剪枝结果，而不是代码不支持动态边。

### 2.2 关系特定转移

对每种关系 `r` 学习 Dirichlet 平滑的标签转移：

```text
T_r[a,b] ∝ count(parent_label=a, child_label=b, relation=r) + prior.
```

父节点分布 `p_u` 产生的子节点消息为：

```text
m_{u→t,r} = p_u T_r.
```

这允许学习 `Phishing → Databreach` 一类异标签攻击阶段转移，而不是把父标签直接复制给子节点。

### 2.3 可衰减的关系效用门

每种关系维护 Beta gate，记录它的转移消息是否比当前类别先验更能解释后续审计标签。用保守下界决定激活：

```text
LCB(g_r) = mean(g_r) - z sqrt(var(g_r))
a_r = [LCB(g_r) - τ]_+ / (1-τ).
```

计数使用指数衰减。因此：

- 新关系先观察，不立即干预；
- 有效关系逐渐激活；
- 连续无效关系被主动剪掉；
- 攻击模式变化后，旧证据逐渐遗忘，关系可以重新评估和恢复。

聚合图消息：

```text
pG_t = sum_{u,r} w_ut a_r p_u T_r / sum_{u,r} w_ut a_r.
```

## 3. Safe fallback：图必须证明自己值得介入

即使关系已经成熟，图消息仍需通过五个只依赖当前可见信息的保护条件：

1. 图后验具有最小类别间隔；
2. 如果基础票已经很确定，图不能轻易推翻它；
3. 图与基础票的 Jensen–Shannon divergence 不得异常大；
4. 图消息必须降低熵，而不是增加不确定性；
5. 至少有一类关系的保守效用下界超过激活阈值。

通过后使用有界凸组合：

```text
alpha_t <= alpha_max
p_t = (1-alpha_t) p0_t + alpha_t pG_t.
```

否则严格返回 `p0_t`。这使图从“默认施加偏置”变成“默认不干预”。

## 4. 在线反馈与因果性

本轮预测完成后，SOC 分析员或事后审计标签到达，调用：

```python
router.observe_graph_feedback(item, gold_label)
```

它不会调用任何 LLM，只更新：

- 父标签到子标签的关系转移矩阵；
- 关系效用 Beta gate；
- 类别先验；
- 当前节点的审计状态。

测试采用 one-step-delayed prequential 协议：第 `t` 个标签只能影响 `t+1` 及以后，不能回头修改第 `t` 个预测。跨 calibration/validation/test 时清空节点历史，但保留此前合法学到的关系参数。

## 5. 理论性质

### 5.1 CaMVo 退化一致性

如果不存在成熟边、保护条件失败或 `alpha_max=0`，则 `p_t=p0_t`，SAGE-CaMVo 返回与同状态 CaMVo 相同的标签；模型选择与费用始终不受图头影响。

### 5.2 有界单轮扰动

因为使用凸组合且 `alpha_t≤alpha_max`：

```text
||p_t-p0_t||_1 ≤ 2 alpha_max.
```

图对基础投票的单轮影响有确定上界，避免 product-of-experts 的概率爆炸。

### 5.3 非平稳适应

Beta gate 的有效样本采用 `N_t = ρN_{t-1}+w_t`。当 `ρ<1` 时旧攻击阶段的影响指数衰减，算法追踪的是近期关系效用，而不是永久固定的全局图。

### 5.4 计算开销

若最多保留 `K` 个父节点、每条边最多 `R` 个关系、类别数为 `C`，图头每轮开销为 `O(KRC²)`；`K` 和总边权均有硬上限。它不增加 LLM 调用，因此实际成本由 CaMVo 子集完全决定。

## 6. 与旧 TRACE 的本质区别

| 组件 | 旧 Continuous TRACE | SAGE-CaMVo |
|---|---|---|
| 基础投票 | 供应商置信度软票 | 原始 CaMVo 离散加权票 |
| 图结构 | 固定邻接 | 候选生成 + 在线激活/剪枝 |
| 标签传播 | 偏向同标签平滑 | 关系特定转移矩阵 |
| 图强度 | 固定 λ | 每轮、每关系自适应 |
| 失效处理 | 仍然传播 | 精确退回 CaMVo |
| 非平稳性 | 无遗忘 | 指数衰减与重新激活 |
| 费用 | 可能改变路由语义 | 与 CaMVo 完全相同 |

## 7. 复现

```bash
PYTHONPATH=src:. python3 scripts/evaluate_safe_adaptive_graph.py \
  --casie-snapshot artifacts/casie_live_snapshot_20260723/CASIE_RESPONSES_24470_SNAPSHOT.tar.gz \
  --mordor-bundle data/derived/mordor_offline_bundle \
  --bootstrap-iterations 2000 \
  --json-out artifacts/safe_adaptive_graph/RESULTS.json
```

该命令只读取已经缓存的真实模型回答，`provider_calls=0`。
