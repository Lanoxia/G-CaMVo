# G-CaMVo frozen real-model protocol

## 1. Purpose

This protocol turns the engineering PoC into a defensible preliminary research result. It prevents three common failure modes: tuning on test labels, comparing methods on different stochastic model outputs, and selecting the “strongest” model from an unverified prior.

## 2. Immutable response matrix

For every sampled item `x` and model `m`, the provider is queried at most once under a versioned prompt contract. The normalized tuple

```text
(label, confidence, rationale, input_tokens, output_tokens)
```

is stored in the content-addressed cache. Only a small allow-list of research metadata is persisted; authorization headers, keys, cookies, and arbitrary provider payloads are discarded. After collection, every policy receives a `FrozenResponseMatrixClient`; policy evaluation has no network capability.

The manifest contains item IDs, model IDs, a sample SHA-256, and completeness counts. A changed model pool or dataset sample cannot silently reuse the manifest.

## 3. Leakage-resistant split

CASIE is split by whole `document_id`, never by event mention. OpTC uses host × source × 30-minute incident windows when a higher-level scenario ID is unavailable. A deterministic iterative stratifier assigns rare-label groups first, then uses local move/swap repair to approximate item and label proportions without breaking groups. Default fractions are:

- calibration: 20%;
- validation: 10%;
- frozen test: 70%.

Each partition receives a SHA-256 digest of its item-ID set in the final report.

## 4. Calibration

Calibration labels are used for four declared purposes only:

1. estimate each model's accuracy and Macro-F1;
2. choose the best-single fallback model;
3. warm-start contextual reliability, confidence calibration, CCaMVo correlation, and causal transition statistics;
4. create Laplace-smoothed vote priors `(correct + 1) / (n + 2)`.

No test label contributes to these states.

## 5. Validation-only policy selection

TRACE candidates use a predeclared risk × causal-graph-strength grid; G-CaMVo uses a predeclared Laplacian regularization grid including zero:

```text
0.20, 0.10, 0.05, 0.03, 0.02, 0.01
graph strength: 0.0, 0.25, 0.5, 0.8, 1.2
G-CaMVo lambda: 0.0, 0.25, 0.5, 1.0, 2.0, 4.0
```

The reference is the highest-quality validation method among calibrated best single, full ensemble, CaMVo, CCaMVo, and G-CaMVo. A TRACE candidate is feasible only if all guards hold:

```text
Macro-F1 >= reference Macro-F1 - 0.01
Macro-recall >= reference Macro-recall - 0.01
abstention rate <= 0.05
Wilson95(selective error) <= Wilson95(reference error) + 0.02
```

Among feasible candidates, the minimum proxy-cost candidate is frozen. If none is feasible, the report explicitly records fallback selection of the highest-quality validation candidate; it does not disguise the failed guard.

## 6. Required baselines

- cheapest single;
- calibration-selected best single;
- fixed cheapest subset;
- Algorithm 2 online weighted majority;
- full calibrated ensemble;
- CaMVo;
- CCaMVo, following paper Appendix G Algorithms 3 and 4;
- G-CaMVo;
- validation-selected calibrated G-CaMVo and a frozen lambda=0 graph ablation;
- calibration-warm-started CaMVo, CCaMVo, and G-CaMVo (stronger label-matched controls);
- TRACE-GCaMVo.

The methods named `camvo` and `ccamvo` retain the paper's unlabeled, fully online update rule. Separate `calibrated_*` methods receive exactly the same audited calibration split as TRACE, preventing TRACE from obtaining an unreported supervision advantage. CCaMVo uses Welford-style online reward correlation, nearest-PSD projection, and deterministic Gaussian-copula Monte Carlo subset confidence.

## 7. Frozen test and uncertainty

The selected TRACE configuration is evaluated once on test. For TRACE versus best single, full ensemble, CaMVo, CCaMVo, and G-CaMVo, the report gives:

- test Macro-F1 delta;
- proxy-cost saving;
- paired cluster bootstrap mean and 95% interval;
- probability the candidate is better under document-cluster resampling.

The test report also includes coverage/abstention, graph-use rate, average decision risk, model selection rates, and latency proxies.

As a post-freeze robustness analysis, five predeclared seeds permute whole document blocks while preserving within-document causal order. No parameter is reselected. The report gives Macro-F1 and proxy-cost mean/std for original CaMVo, calibrated CaMVo, validation-tuned calibrated G-CaMVo, and TRACE-GCaMVo.

## 8. Claim boundary

CASIE validates real-model cost-aware event-subtype routing. It is not a provenance-based incident detection benchmark. A paper claim about security incident detection, timeline reconstruction, or mitigation requires the same protocol on real OpTC attack and benign-period telemetry. Proxy cost is a counterfactual deployment measure when Adams quota is free; it is not an API invoice. One frozen response per model/item controls cross-method randomness but does not estimate generation-level variance.
