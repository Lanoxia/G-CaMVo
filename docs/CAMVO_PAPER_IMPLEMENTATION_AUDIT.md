# CaMVo 原论文实现审计

审计对象：NeurIPS 2025 *Cost-aware LLM-based Online Dataset Annotation*，重点为正文
Algorithm 1、公式 (2)–(4)、Appendix B 的 Beta 参数估计，以及 Appendix G 的 CCaMVo。
原论文没有发布代码，因此“复现”只能以论文伪代码和公式为规范，不能声称逐行复现作者代码。

## 一轮 CaMVo 到底做什么

对第 `t` 个输入 `x_t`：

1. 先得到上下文向量 `e_t = Emb(x_t)`，此时还没有调用本轮 LLM。
2. 对每个模型 `i`，用历史共识奖励拟合的 LinUCB 计算
   `q_i,t = e_t^T A_i^{-1} b_i` 和 LCB
   `theta_i,t = q_i,t - alpha sqrt(e_t^T A_i^{-1} e_t)`。
3. Beta mixture 计算 `Est_i(theta_i,t)`；公式 (3) 再把它向 `0.5` 做随
   `log(t)` 变化的 Laplace 平滑，得到用于 Oracle 的正确率下界 `L_i,t`。
4. 投票权重不是 `L_i,t`，而是
   `omega_i,t = mu_i,t-1 * q_i,t`。其中 `mu` 是模型过去与系统共识标签一致的比例。
5. Oracle 在所有满足 `|A| >= k_min` 且多数票正确置信度至少为 `delta` 的子集中，
   选成本最低者；若没有可行子集，就选全部模型。
6. 一次性查询整个子集，然后用 `omega` 做一次加权多数投票。
7. 若子集大小大于 1，则令 `r_i,t = 1[y_i,t = y_hat_t]`，更新
   `A_i`、`b_i`、`mu_i` 和 Beta 参数。没有 ground truth；单模型轮不更新。

所以 CaMVo 不是“根据当前答案逐个加模型”的 cascade，也不是“用金标签在线训练”的
监督式 router。它优化的是相对于全模型多数票的置信度与成本，而不是保证相对于真实标签的
绝对准确率。

## 代码逐项核对

| 论文组件 | 仓库实现 | 审计结论 |
|---|---|---|
| 每模型 `A=lambda_L I, b=0` | `algorithm/linucb.py` | 一致 |
| `q`、LCB 和裁剪到 `[0,1]` | `LinUCBArm.score` | 一致；数值上用 epsilon 避免端点 |
| `Est_i(theta)` 的两类 Beta mixture | `algorithm/calibration.py` | 一致；使用在线 Welford moments |
| 公式 (3) 的 `log(t+1)` 平滑 | `laplace_smooth` | 一致 |
| `omega=mu*q` | `router.py::_score_models` | 一致；额外设极小正权重以避免初始全零 |
| 公式 (2) 的精确多数票置信度 | `exact_majority_confidence` | 一致 |
| 论文实验的 Beta-CDF 近似 | `beta_cdf_confidence` | 一致；默认路由配置未必启用 |
| 最低成本可行子集、无解取全体 | `ExhaustiveSubsetOracle` | 一致，小模型池穷举得到全局最优 |
| 先选子集，再看当前回答 | `CaMVoRouter.route` | 一致 |
| `r=1[模型票=系统共识]` | `CaMVoRouter.route` | 一致 |
| `|A|=1` 不更新 | `update_single_model_rounds=False` | 默认一致 |
| Appendix G 相关性估计与 Gaussian copula | `ccamvo_router.py` | 已实现论文 Algorithms 3/4 |

## 必须明确披露的工程差异

1. **Mordor 主实验是 audited warm start。** 固定 calibration split 的金标签用于
   `observe_complete_feedback`，之后 validation/test 才冻结。这符合拥有历史已处置事件的
   SOC 场景，但不等同于论文“从零开始、全程无 ground truth”的协议。
2. **上下文嵌入不同。** 原论文实验用 384 维 `all-MiniLM-L6-v2`；当前离线 Mordor
   使用 64 维 hashing embedding，以保证无下载、确定性复现。算法骨架相同，表征能力不同。
3. **默认置信度计算不同。** 论文实验为了效率使用 Beta-CDF 近似；当前 Mordor warm-start
   配置默认使用公式 (2) 的精确枚举。四模型池下精确枚举很便宜，但数值结果不能冒充论文设置。
4. **初始化保护不同。** 论文没有完整说明 `mu` 在零观测时如何初始化。仓库采用 Beta(1,1)
   式共识先验、最小正投票权重和可配置 warm-up，防止零权重/未定义概率。设置
   `warmup_rounds=0` 更接近伪代码。
5. **成本口径不同。** 原论文实验假设只输出一个标签 token，并忽略输出 token；仓库按供应商
   实际 input/output token 价格计费。应在表格中写清楚口径。
6. **TRACE 修改了最终票面。** 当 `label_vote_regularization>0` 时，模型票仍按 `omega=mu*q`
   聚合，但会加入由过去父节点产生的结构化伪票。因此应称为“保留 CaMVo 子集选择与在线更新
   骨架的图正则扩展”，不应称为“原始 CaMVo 投票完全不变”。

## TRACE 可以合法改什么

我们保留三个不变量：本轮回答出现前一次性选择完整子集；被选模型平行投票；用共识奖励更新
CaMVo 状态。图只允许进入两个位置：

- 在 Oracle 前，用过去父节点证据保守修正每个模型的 `L_i,t`；
- 在投票时，把过去父节点形成的标签分布变成有界伪票。

图不能读取未来孩子、不能固定绑定某个供应商模型，也不能先看当前便宜模型答案再决定是否问
下一个模型，否则就变成另一类 cascade/router，不再是 CaMVo 的直接扩展。

## 对当前结果的正确命名

- `camvo_cold_start_paper_skeleton`：无金标签 warm start、Beta-CDF、`warmup=0`；仍因 hashing
  embedding 和初始化保护而只能称“论文骨架近似”。
- `camvo`：audited warm-start CaMVo，是 TRACE 的同协议基线。
- `causal_trace_camvo`：同样 audited warm start，随后只用过去图信息的 TRACE 扩展。

比较 TRACE 与 CaMVo 时必须使用相同初始化和相同成本口径；冷启动结果用于说明与原论文协议的
关系，不能和 warm-start TRACE 直接计算因果增益。
