# Complete CASIE frozen-response causal routing suite

## Protocol

- Complete source matrix: 33,940 responses for 8,485 events;
- Leakage-controlled cohort: 7,985 events from 977 whole documents;
- Excluded prior development material: 500 events from 23 documents;
- Outer evaluation: 5 document-grouped folds;
- Frozen hyperparameters transferred from the earlier Mordor/SAGE development;
- Test feedback is post-prediction and past-only; provider calls: 0.

## Main OOF table

| Method | Accuracy | Macro-F1 | Macro-recall | Avg. models | Proxy cost USD | Graph/state rate |
|---|---:|---:|---:|---:|---:|---:|
| single::dify/deepseek-v4-flash | 0.8967 | 0.8981 | 0.9072 | 1.000 | 0.5442 | 0.00% |
| single::dify/deepseek-v4-pro | 0.8852 | 0.8869 | 0.8984 | 1.000 | 17.1890 | 0.00% |
| single::dify/glm-5.2-fp8 | 0.9176 | 0.9185 | 0.9256 | 1.000 | 25.5799 | 0.00% |
| single::dify/minimax-m3 | 0.8733 | 0.8743 | 0.8825 | 1.000 | 6.0945 | 0.00% |
| best_train_single | 0.9176 | 0.9185 | 0.9256 | 1.000 | 25.5799 | 0.00% |
| majority_vote | 0.9133 | 0.9143 | 0.9215 | 4.000 | 49.4077 | 0.00% |
| calibration_weighted_vote | 0.9085 | 0.9096 | 0.9178 | 4.000 | 49.4077 | 0.00% |
| document_persistence_best_single | 0.7403 | 0.7390 | 0.7391 | 0.122 | 2.8553 | 87.76% |
| parent_persistence_best_single | 0.7403 | 0.7390 | 0.7391 | 0.122 | 2.8553 | 87.76% |
| camvo_k1 | 0.9066 | 0.9077 | 0.9155 | 1.951 | 22.8588 | 0.00% |
| camvo_k2 | 0.9064 | 0.9075 | 0.9154 | 2.439 | 23.9771 | 0.00% |
| sage | 0.9091 | 0.9101 | 0.9176 | 2.439 | 23.9771 | 2.47% |
| dcr_k1_frozen | 0.9239 | 0.9247 | 0.9309 | 2.000 | 26.1241 | 1.30% |
| dcr_k2_reliability_only | 0.9145 | 0.9155 | 0.9223 | 4.000 | 49.4077 | 0.00% |
| dcr_k2_diversity_no_graph | 0.9176 | 0.9185 | 0.9256 | 2.000 | 26.1241 | 0.00% |
| dcr_k2_frozen | 0.9239 | 0.9247 | 0.9309 | 2.000 | 26.1241 | 1.30% |
| state_switch_gcamvo_k2_frozen | 0.9250 | 0.9258 | 0.9305 | 2.000 | 25.8544 | 87.76% |

## SAGE paired document-cluster bootstrap

| Reference | Δ Macro-F1 | 95% CI | P(Δ>0) |
|---|---:|---:|---:|
| vs_camvo_k2 | +0.00255 | [+0.00148, +0.00378] | 100.00% |

## DCR k=2 paired document-cluster bootstrap

| Reference | Δ Macro-F1 | 95% CI | P(Δ>0) |
|---|---:|---:|---:|
| vs_best_train_single | +0.00615 | [+0.00423, +0.00835] | 100.00% |
| vs_camvo_k2 | +0.01716 | [+0.01272, +0.02168] | 100.00% |
| vs_dcr_k2_diversity_no_graph | +0.00615 | [+0.00423, +0.00835] | 100.00% |
| vs_dcr_k2_reliability_only | +0.00923 | [+0.00488, +0.01374] | 100.00% |
| vs_sage | +0.01461 | [+0.01021, +0.01895] | 100.00% |
| vs_single::dify/glm-5.2-fp8 | +0.00615 | [+0.00423, +0.00835] | 100.00% |
| vs_state_switch_gcamvo_k2_frozen | -0.00111 | [-0.00431, +0.00191] | 24.55% |

## State-Switch paired document-cluster bootstrap

| Reference | Δ Macro-F1 | 95% CI | P(Δ>0) |
|---|---:|---:|---:|
| vs_best_train_single | +0.00726 | [+0.00342, +0.01117] | 99.90% |
| vs_camvo_k2 | +0.01827 | [+0.01404, +0.02271] | 100.00% |
| vs_dcr_k2_frozen | +0.00111 | [-0.00202, +0.00422] | 76.00% |
| vs_document_persistence_best_single | +0.18678 | [+0.17558, +0.19869] | 100.00% |
| vs_parent_persistence_best_single | +0.18678 | [+0.17558, +0.19869] | 100.00% |

## Feedback-rate sensitivity (Macro-F1)

| Feedback rate | Persistence | DCR k=2 | State-Switch k=2 |
|---:|---:|---:|---:|
| 0% | 0.9185 | 0.9201 | 0.9201 |
| 10% | 0.8197 | 0.9201 | 0.9243 |
| 25% | 0.7650 | 0.9205 | 0.9246 |
| 50% | 0.7429 | 0.9231 | 0.9254 |
| 100% | 0.7390 | 0.9247 | 0.9258 |

## Feedback-delay sensitivity (Macro-F1)

| Delay (events) | Persistence | DCR k=2 | State-Switch k=2 |
|---:|---:|---:|---:|
| 0 | 0.7390 | 0.9247 | 0.9258 |
| 1 | 0.7183 | 0.9213 | 0.9245 |
| 5 | 0.8084 | 0.9202 | 0.9249 |
| 20 | 0.9177 | 0.9202 | 0.9202 |

## Claim boundary

Complete 33,940-cell real-model CASIE response matrix with zero new provider calls. OOF predictions use whole-document grouped folds and post-prediction past-only feedback. The 23 documents seen during earlier CASIE method development are excluded. CASIE supplies article sequence relations rather than host/process provenance, so independent OpTC confirmation is still required for SOC attack-chain claims.
