# Causal State-Switch G-CaMVo：方法、验证与诚实结论

## 一句话结论

我们已经把原始 CaMVo 改造成一个**模型无关、严格因果、带审计反馈的图状态路由器**。
在 913 条冻结的真实 Adams/Mordor 模型响应上，它将 Macro-F1 从 CaMVo 的
`0.3981` 提升到 `0.9003`，也高于 DCR-G-CaMVo 的 `0.8707`；但它与“不调用模型，
直接沿用同主机最近一次已审计标签”的强基线完全持平。因此当前结果证明了
**时序状态比独立投票重要**，还没有证明多模型投票能比简单状态持续做得更好。

这条负面边界必须保留。否则很容易把 Mordor 标签的长连续区间误写成图算法的贡献。

## 1. 为什么原 CaMVo 在这里失效

原 CaMVo 在第 \(t\) 轮先为 \(n\) 个模型估计可靠性，从中选择 \(m\) 个模型，
再做加权投票并用“是否同意聚合结果”更新 bandit。这个机制隐含地把样本看成条件独立的
在线任务。Mordor 的事件却有明显的阶段持续性：同一主机在一段时间内往往维持 benign
或 malicious 状态，真正困难的是少数**状态边界**。

现有四模型在恶意类上的投票高度相关且普遍保守。最佳单模型的 malicious recall 只有
`0.1798`，多数票只有 `0.0570`。CaMVo 的共识奖励会进一步奖励这种共同保守行为，
所以 CaMVo k=2 的 malicious recall 只有 `0.0746`。

## 2. DCR-G-CaMVo：先修复模型协同

DCR 表示 Diversity、Causality、Reliability。它仍然在看到当前轮投票前选择模型子集，
但把原 CaMVo 的单一一致性权重改成三部分：

1. **类条件可靠性**：只用已审计标签学习每个模型的混淆矩阵
   \(P(V_i=v\mid Y=y)\)；
2. **错误多样性**：学习模型两两错误相关性，惩罚同时选择重复犯错的模型；
3. **因果图先验**：只读取当前事件之前、已经审计过的父节点，不看未来节点。

子集 \(S\) 的选择目标为

\[
U_t(S)=\sum_{i\in S} I(Y;V_i)
-\lambda_r\sum_{i<j\in S}\rho^+_{ij}\sqrt{I_iI_j}
-\lambda_c\log(1+C(S)/C_{\min})
+\lambda_q\bar q(S).
\]

选出子集后，用混淆矩阵构造可靠性后验：

\[
P(Y=y\mid v_S)\propto P(Y=y)
\prod_{i\in S}P(V_i=v_i\mid Y=y)^{\eta_i},
\]

其中 \(\eta_i\) 会按与已接纳模型的正错误相关性折扣。最后与 past-only 图后验做有界
safe fusion；图证据不足、冲突过大或越界时退回模型后验。

## 3. State-Switch G-CaMVo：把图变成动态安全状态

对实体 \(e\)（本实验中为 hostname），令最近一次已经到达的审计状态为
\(H_{e,t^-}\)。算法从历史审计事件学习：

\[
P(Y_t\mid H_{e,t^-})
\]

以及对每个候选模型子集 \(S\) 的 vote-pattern 条件分布：

\[
P(Y_t\mid H_{e,t^-},V_{S,t}=v_S).
\]

### 3.1 每轮仍严格遵循 CaMVo

1. 当前事件到达，但当前模型答案仍不可见；
2. 根据历史 vote-pattern 的条件互信息选择大小为 \(m\) 的子集：

\[
S_t=\arg\max_{|S|=m}
I(Y_t;V_{S,t}\mid H_{e,t^-})-\lambda_c C(S);
\]

3. 只调用 \(S_t\) 中的模型；
4. 用历史转移先验平滑当前 vote pattern：

\[
\hat p_t(y)=
\frac{N(H_{t^-},v_S,y)+\kappa P(y\mid H_{t^-})}
{N(H_{t^-},v_S)+\kappa};
\]

5. 只有当备选状态的后验不低于阈值 \(\tau\)，且 pattern support 足够时才切换；
   否则保持 \(H_{e,t^-}\)；
6. 输出后才允许审计标签进入转移表、vote-pattern 表和图历史。

因此它不是“先看所有模型再挑最好答案”，也不是把测试标签提前写入邻居；它仍是
CaMVo 的 pre-vote subset selection，只是把优化对象从独立样本正确率改成了状态切换信息。

## 4. 实验协议

- 数据：913 个 complete-case Mordor 事件，四个 Adams 真模型冻结响应；
- 新增 provider 调用：0；
- 外层：5 折 grouped CV；内层：3 折参数选择；
- 分组：hostname × 5-minute causal block，超过 48 事件按时间顺序切块；
- bootstrap：保留更粗的 hostname × 30-minute incident cluster，共 23 簇；
- 外层标签不参与内层参数选择；
- 测试流中的审计标签在本轮预测之后到达，只影响未来节点；
- 同时测量 0/10/25/50/100% 反馈率和 0/1/5/20 事件反馈延迟。

## 5. 主结果

| 方法 | Accuracy | Macro-F1 | Malicious Recall | 平均模型数 | 代理总成本 USD |
|---|---:|---:|---:|---:|---:|
| 最佳训练折单模型 | 0.5509 | 0.4791 | 0.1798 | 1.000 | 0.1485 |
| 四模型多数票 | 0.5115 | 0.3843 | 0.0570 | 4.000 | 10.2582 |
| CaMVo k=2 | 0.5148 | 0.3981 | 0.0746 | 3.552 | 8.3990 |
| SAGE-CaMVo | 0.5170 | 0.3994 | 0.0746 | 3.552 | 8.3990 |
| DCR-G-CaMVo k=2 | 0.8708 | 0.8707 | 0.8596 | 2.000 | 1.2505 |
| **State-Switch G-CaMVo k=2** | **0.9003** | **0.9003** | **0.8904** | **2.000** | 1.3046 |
| 主机最近状态持续 + 单模型冷启动 | **0.9003** | **0.9003** | **0.8904** | 0.036 | **0.0051** |

State-Switch 相对 CaMVo 的 Macro-F1 差为 `+0.50222`，23 个 incident cluster 的
95% bootstrap CI 为 `[+0.31484,+0.61432]`。相对 DCR 的差为 `+0.02958`，
CI 为 `[-0.00692,+0.05941]`，尚未达到双侧显著。相对主机持续基线的差严格为 `0`。

## 6. 反馈敏感度

| 审计反馈率 | State-Switch Macro-F1 | DCR Macro-F1 |
|---:|---:|---:|
| 0% | 0.4791 | 0.4900 |
| 10% | 0.7536 | 0.5449 |
| 25% | 0.8422 | 0.6292 |
| 50% | 0.8828 | 0.7359 |
| 100% | 0.9003 | 0.8707 |

| 反馈延迟 | State-Switch Macro-F1 | DCR Macro-F1 |
|---:|---:|---:|
| 1 个事件 | 0.8663 | 0.6212 |
| 5 个事件 | 0.7947 | 0.5616 |
| 20 个事件 | 0.6621 | 0.5150 |

这说明当前性能高度依赖及时人工审计。不能把 100% 即时反馈数字描述成完全自治 SOC。

## 7. 这项结果现在能证明什么

可以说：

- 原 CaMVo 的独立投票假设不适合这个攻击流；
- 类条件可靠性、错误多样性和 past-only 图状态能显著改善同一协议下的预测；
- 严格在线状态记忆对 Mordor 非常重要；
- 新实现是供应商无关的，不依赖 DeepSeek V4 Flash；
- 算法在反馈稀疏和延迟下呈现可解释的退化曲线。

不能说：

- State-Switch 已经超过所有合理基线；
- 多模型投票已经成功检测攻击阶段边界；
- 该结论已经泛化到 CASIE、OpTC 其他场景或真实 SOC；
- `0.9003` 是一个完全自治、无标签在线系统的结果。

## 8. 为什么当前二元投票没有纠正边界

5 折中内层选择的阈值都收敛到保守配置。最终 State-Switch 与状态持续基线逐事件预测
完全相同，说明当前四个模型的 `benign/malicious` 标签组合没有提供可泛化的切换证据。
最可能原因是：

1. 攻击边界极少，普通事件占据 vote-pattern 统计；
2. 四模型共享保守偏差，错误不够互补；
3. 收集阶段丢掉了 rationale 中的进程、用户、命令、ATT&CK tactic 和置信度结构；
4. 当前标签把多阶段攻击压成二元分类，信息瓶颈太强。

## 9. 下一版应冻结的研究假设

不要继续对同一 913 条外层结果试阈值。下一版应在 CASIE/OpTC 的独立数据到达前冻结：

1. 每个模型输出统一结构：`label, confidence, entities, tactic, evidence, novelty`；
2. 在不增加模型数量的前提下，用连续 log-likelihood ratio 而不是二元 vote pattern；
3. 显式预测 `stay/switch`，主任务和边界任务同时报告；
4. 加入 episode-level 指标：首次恶意检测延迟、每条攻击链召回、每主机误报；
5. 将主机、进程、用户、文件、网络连接作为多类型状态，而不是只有 hostname；
6. 预注册相对 host persistence、parent persistence、single、CaMVo、DCR 的比较。

只有独立数据上的状态边界指标改善，才能证明多模型图投票确实贡献了新信息。

## 10. 复现

```bash
python -m pip install -e .
python -m pytest tests/test_dcr_graph_router.py tests/test_state_switch_router.py -q

PYTHONPATH=src:scripts python scripts/evaluate_dcr_group_cv.py
```

默认命令复用 `RESULTS.json` 中已经保存的逐折内层搜索审计，再用正式部署路由器重放
外层 OOF，约两分钟完成。若要从零重算全部内层超参数网格，运行：

```bash
PYTHONPATH=src:scripts python scripts/evaluate_dcr_group_cv.py \
  --refresh-inner-search
```

主要文件：

- `src/camvo/dcr_graph_router.py`：DCR 子集选择和可靠性聚合；
- `src/camvo/state_switch_router.py`：可部署的因果状态切换路由器；
- `scripts/evaluate_dcr_group_cv.py`：嵌套分组 CV、强基线、敏感度和 bootstrap；
- `artifacts/dcr_group_cv/RESULTS.json`：完整机器可读结果；
- `artifacts/dcr_group_cv/oof_records.csv`：所有方法的外层 OOF 逐事件预测；
- `docs/DCR_GCAMVO_GROUP_CV_RESULTS_2026-07-23.md`：自动生成结果表。
