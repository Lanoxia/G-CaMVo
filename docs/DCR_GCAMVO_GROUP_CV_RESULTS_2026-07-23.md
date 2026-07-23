# Causal G-CaMVo：Mordor 嵌套分组交叉验证

## 协议

- 完整冻结事件：913；新增模型调用：0；
- 外层 5 折、内层 3 折；
- 内层选择来源：cached_nested_cv_audit；
- 分组：hostname:5-minute causal block, sequentially capped at 48 events; bootstrap retains the coarser hostname:30-minute incident cluster；
- 外层测试标签不参与内层参数选择；测试流反馈在预测后到达，只影响未来节点。

## 外层 OOF 主结果

| 方法 | Accuracy | Macro-F1 | 恶意 P/R/F1 | 平均模型数 | 总成本 USD | 图介入率 |
|---|---:|---:|---:|---:|---:|---:|
| single::dify/deepseek-v4-flash | 0.5509 | 0.4791 | 0.6949/0.1798/0.2857 | 1.000 | 0.1485 | 0.00% |
| best_train_single | 0.5509 | 0.4791 | 0.6949/0.1798/0.2857 | 1.000 | 0.1485 | 0.00% |
| majority_vote | 0.5115 | 0.3843 | 0.6190/0.0570/0.1044 | 4.000 | 10.2582 | 0.00% |
| calibration_weighted_vote | 0.5159 | 0.4016 | 0.6207/0.0789/0.1401 | 4.000 | 10.2582 | 0.00% |
| host_persistence_best_single | 0.9003 | 0.9003 | 0.9083/0.8904/0.8992 | 0.036 | 0.0051 | 96.39% |
| parent_persistence_best_single | 0.8839 | 0.8837 | 0.9167/0.8443/0.8790 | 0.094 | 0.0142 | 90.58% |
| camvo_k1 | 0.5148 | 0.3981 | 0.6182/0.0746/0.1331 | 3.418 | 7.9955 | 0.00% |
| camvo_k2 | 0.5148 | 0.3981 | 0.6182/0.0746/0.1331 | 3.552 | 8.3990 | 0.00% |
| sage | 0.5170 | 0.3994 | 0.6415/0.0746/0.1336 | 3.552 | 8.3990 | 3.61% |
| dcr_k1_full | 0.8719 | 0.8718 | 0.8792/0.8618/0.8704 | 1.001 | 0.1498 | 96.39% |
| dcr_k2_reliability_only | 0.5509 | 0.4791 | 0.6949/0.1798/0.2857 | 2.000 | 1.2505 | 0.00% |
| dcr_k2_diversity_no_graph | 0.5509 | 0.4791 | 0.6949/0.1798/0.2857 | 2.000 | 1.2505 | 0.00% |
| dcr_k2_full | 0.8708 | 0.8707 | 0.8789/0.8596/0.8692 | 2.000 | 1.2505 | 96.28% |
| state_switch_gcamvo_k2 | 0.9003 | 0.9003 | 0.9083/0.8904/0.8992 | 2.000 | 1.3046 | 96.71% |

## DCR-full 配对 cluster bootstrap

| 对照 | Macro-F1 差值 | 95% CI | DCR 更优概率 |
|---|---:|---:|---:|
| vs_camvo_k1 | +0.47264 | [+0.29767, +0.57025] | 100.00% |
| vs_camvo_k2 | +0.47264 | [+0.29767, +0.57025] | 100.00% |
| vs_sage | +0.47137 | [+0.29767, +0.56967] | 100.00% |
| vs_best_train_single | +0.39162 | [+0.11549, +0.51350] | 100.00% |
| vs_host_persistence | -0.02958 | [-0.05856, +0.00561] | 4.50% |

## State-Switch G-CaMVo 配对 cluster bootstrap

| 对照 | Macro-F1 差值 | 95% CI | State-Switch 更优概率 |
|---|---:|---:|---:|
| vs_host_persistence | +0.00000 | [+0.00000, +0.00000] | 0.00% |
| vs_dcr_k2 | +0.02958 | [-0.00692, +0.05941] | 95.10% |
| vs_camvo_k2 | +0.50222 | [+0.31484, +0.61432] | 100.00% |
| vs_best_train_single | +0.42120 | [+0.13407, +0.55672] | 100.00% |

## 审计反馈稀疏度（DCR k=2）

| 反馈率 | Accuracy | Macro-F1 | 恶意 Recall | 图介入率 |
|---:|---:|---:|---:|---:|
| 0% | 0.5542 | 0.4900 | 0.1996 | 6.35% |
| 10% | 0.5915 | 0.5449 | 0.2719 | 17.20% |
| 25% | 0.6517 | 0.6292 | 0.4057 | 33.73% |
| 50% | 0.7415 | 0.7359 | 0.5965 | 61.12% |
| 100% | 0.8708 | 0.8707 | 0.8596 | 96.28% |

## 审计反馈延迟（DCR k=2，100% 最终到达）

| 延迟事件数 | Accuracy | Macro-F1 | 恶意 Recall | 图介入率 |
|---:|---:|---:|---:|---:|
| 0 | 0.8708 | 0.8707 | 0.8596 | 96.28% |
| 1 | 0.6440 | 0.6212 | 0.3991 | 42.83% |
| 5 | 0.5991 | 0.5616 | 0.3070 | 23.99% |
| 20 | 0.5685 | 0.5150 | 0.2368 | 14.57% |

## State-Switch 反馈敏感度

| 条件 | Accuracy | Macro-F1 | 恶意 Recall |
|---|---:|---:|---:|
| 反馈率 0% | 0.5509 | 0.4791 | 0.1798 |
| 反馈率 10% | 0.7547 | 0.7536 | 0.6908 |
| 反馈率 25% | 0.8423 | 0.8422 | 0.8575 |
| 反馈率 50% | 0.8828 | 0.8828 | 0.8925 |
| 反馈率 100% | 0.9003 | 0.9003 | 0.8904 |
| 延迟 1 | 0.8664 | 0.8663 | 0.8509 |
| 延迟 5 | 0.7952 | 0.7947 | 0.7500 |
| 延迟 20 | 0.6725 | 0.6621 | 0.4978 |

## 外层折审计

| Fold | Train | Test | Test groups | Benign | Malicious |
|---:|---:|---:|---:|---:|---:|
| 0 | 728 | 185 | 9 | 99 | 86 |
| 1 | 731 | 182 | 9 | 87 | 95 |
| 2 | 732 | 181 | 8 | 88 | 93 |
| 3 | 730 | 183 | 9 | 94 | 89 |
| 4 | 731 | 182 | 8 | 89 | 93 |

## 结论边界

Exploratory nested grouped cross-validation on one Mordor scenario. All LLM responses are frozen real Adams outputs and provider calls are zero. Outer-fold labels are unseen during inner selection; within an outer test stream, audited feedback arrives after each prediction and affects only future nodes. This repairs the prior single-split instability but remains method development, not external confirmation.

本表的作用是修复旧 Mordor 单次划分的失衡并完成方法开发；最终主结论仍需在冻结后的未见 CASIE 文档或独立 OpTC 场景确认。
