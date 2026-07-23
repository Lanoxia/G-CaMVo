# TRACE-GCaMVo: a structure-aware, calibrated escalation policy for SOC routing

## Executive summary

TRACE-GCaMVo is the next research version of G-CaMVo. It retains CaMVo's online,
cost-aware model selection, but changes the decision unit from a preselected
majority-vote subset to a **sequential escalation process**. The router starts
with the model expected to provide the most information per unit cost, updates a
posterior over security labels after every response, and stops once estimated
decision risk is below a severity-dependent tolerance.

The graph is not used as an unconditional same-label smoother. Provenance and
document graphs can be heterophilic: a benign parent can lead to a malicious
child, and different stages of one incident need not have the same label.
TRACE therefore learns relation-specific label transitions, uses only past
neighbors, and blocks graph propagation until the learned transition is mature
and informative.

The design addresses three failures observed in the first Adams/CASIE PoC:

1. row-wise stratified sampling destroyed the graph (28 of 30 nodes were
   isolated);
2. hand-written model quality priors did not match observed model performance;
3. the confidence-threshold subset oracle escalated to the full four-model pool
   on 29 of 30 items.

## 1. Research question

Given a stream of security events connected by typed temporal or provenance
relations, choose a variable-length sequence of LLM calls for every event so as
to minimize cost and latency while keeping incident decision risk below a
user-defined tolerance. The policy must be causal, auditable, robust to
correlated LLM errors, and capable of abstaining when the available model pool
cannot resolve an event safely.

Let:

- \(G_t=(V_t,E_t)\) be the graph revealed up to round \(t\);
- \(x_v\) be the observable context for node \(v\);
- \(y_v\in\{1,\ldots,M\}\) be its unknown label;
- \(K\) be the model pool;
- \(c_i(x_v)\) be the estimated cost of querying model \(i\);
- \(r_{iv}\) be model \(i\)'s response;
- \(A_v=(i_1,\ldots,i_{J_v})\) be the ordered queried sequence.

The deployment objective is a constrained risk-cost problem:

\[
\min_\pi \; \mathbb E_\pi\!\left[\sum_v\sum_{i\in A_v}c_i(x_v)\right]
\quad\text{s.t.}\quad
\Pr(\hat y_v\neq y_v\mid\mathcal F_v)\le \varepsilon_{s(v)},
\]

where \(\mathcal F_v\) is the information available when node \(v\) is routed
and \(\varepsilon_{s(v)}\) is the risk tolerance for its severity tier.

This is not yet claimed as a distribution-free safety guarantee. In the current
implementation the constraint is an estimated posterior-risk constraint; a
future audited/conformal layer is required before using the word "guarantee" in
a deployment claim.

## 2. Why the first G-CaMVo formulation needs refinement

The two-page G-CaMVo draft made the correct high-level move: security instances
are not i.i.d., and graph structure should influence model allocation. Its
Laplacian objective,

\[
\sum_v(r_{vi}-\theta_i^\top x_v)^2+
\lambda_1\lVert\theta_i\rVert^2+
\lambda_2\sum_{(u,v)\in E}w_{uv}
(\theta_i^\top x_u-\theta_i^\top x_v)^2,
\]

is a useful starting hypothesis, but it silently assumes that connected nodes
should have similar model-correctness signals. That assumption is unsafe for a
typed attack graph. It also uses a batch graph matrix in a setting described as
online, and the claim that querying one junction necessarily reduces
uncertainty at unqueried leaves requires additional feedback and smoothness
assumptions.

TRACE keeps the insight but makes four changes:

- causal past-neighbor messages replace a batch Laplacian over future nodes;
- learned transition potentials replace unconditional same-label smoothing;
- sequential expected-value-of-information escalation replaces one-shot subset
  enumeration;
- correlation-aware evidence discounting replaces conditional independence.

## 3. TRACE-GCaMVo algorithm

### 3.1 Structure-preserving stream construction

Rows are never sampled independently. CASIE is sampled by source document and
ordered by event offset. OpTC is sampled by incident/time window and ordered by
event time. Calibration, validation, and test partitions split whole documents
or incidents.

Every run records node count, edge count, isolates, component sizes, mean degree
and, for evaluation only, edge label agreement. This is a precondition check:
if the graph has no usable edges, a graph-method comparison is invalid.

### 3.2 Online model reliability

Each model has three reliability signals:

1. a contextual LinUCB estimate from CaMVo;
2. a symmetric Beta posterior over global agreement/correctness;
3. a histogram-Beta calibrator for the confidence reported by the model.

All models receive the same weak initial prior. The prior therefore does not
hard-code a strong/weak model order. During deployment, analyst feedback can be
supplied through a separate audited feedback field. Without audited feedback,
updates use confidence-weighted ensemble pseudo-labels and must be described as
label-free self-calibration, not ground-truth calibration.

### 3.3 Causal, relation-gated graph prior

For a current node \(v\), only already-routed neighbors
\(N^-(v)\) are visible. For relation type \(\tau\), TRACE maintains a smoothed
transition matrix \(\Pi_\tau(y_u,y_v)\). A neighbor belief \(q_u\) proposes

\[
m_{u\rightarrow v}(y)=\sum_{y_u}q_u(y_u)\Pi_\tau(y_u,y).
\]

The initial node posterior is

\[
p_v^{(0)}(y)\propto p_0(y)
\prod_{u\in N^-(v)}m_{u\rightarrow v}(y)^{\lambda_g\gamma_{uv}},
\]

where \(\gamma_{uv}\) combines the bounded edge weight, neighbor confidence,
transition maturity and message informativeness. A relation is blocked when it
has too few observations or its transition is close to uniform. Consequently,
an uncertain or uninformative graph reduces exactly to the non-graph router.

### 3.4 Correlation-aware vote likelihood

Let \(a_{iv}\) be the calibrated probability that model \(i\) is correct on
node \(v\). TRACE tracks pairwise prediction agreement on co-queried items. If
the selected set is \(A\), model \(i\)'s evidence factor is

\[
d_i(A)=\max\left(d_{\min},
1-\lambda_{\mathrm{corr}}
\max_{j\in A}\widehat{\rho}_{ij}\right).
\]

The effective accuracy is shrunk toward random guessing:

\[
\widetilde a_{iv}=\frac1M+d_i(A)
\left(a_{iv}-\frac1M\right).
\]

Under a symmetric error model, observing response \(r_i\) gives likelihood

\[
P(r_i\mid y)=
\begin{cases}
\widetilde a_{iv},&r_i=y,\\
(1-\widetilde a_{iv})/(M-1),&r_i\ne y.
\end{cases}
\]

This discount is essential. Without it, two correlated cheap models can agree
on the same wrong answer and create a falsely confident posterior.

### 3.5 Expected information gain per cost

For each unqueried model, TRACE computes one-step expected information gain

\[
\operatorname{EIG}_i=
H(p_v)-\mathbb E_{R_i\sim p(R_i\mid p_v)}[H(p_v\mid R_i)].
\]

The next model is

\[
i^*=\arg\max_{i\notin A_v}
\frac{d_i(A_v)\operatorname{EIG}_i}{c_i(x_v)}.
\]

This is the optimal next action for the stated one-step entropy-reduction
surrogate. It is a myopic value-of-information policy, not a proof of global
optimality over all future query sequences.

### 3.6 Stopping and abstention

After each response, TRACE updates \(p_v\) and stops when

\[
|A_v|\ge k_{\min}
\quad\text{and}\quad
1-\max_y p_v(y)\le\varepsilon_{s(v)}.
\]

If the full model pool is exhausted and the estimated risk is still too high,
the event is marked for abstention/human review. The predicted label is retained
for offline metric calculation, but a production SOC integration should not
silently convert an abstention into an automated mitigation.

## 4. Algorithm sketch

```text
for event v in causal order:
    x_v <- embed(v)
    p <- gated_graph_prior(previous_neighbors(v))
    estimate per-model reliability and cost
    selected <- []

    while unqueried models remain:
        i <- argmax expected_information_gain(i, p) * diversity(i) / cost(i)
        r_i, self_confidence_i <- query(i, v)
        a_i <- online_calibrate(i, x_v, self_confidence_i)
        a_i <- shrink_toward_random(a_i, pairwise_redundancy(i, selected))
        p <- BayesUpdate(p, r_i, a_i)
        selected.append(i)
        if len(selected) >= k_min and 1 - max(p) <= risk_tolerance(v):
            break

    abstain <- 1 - max(p) > risk_tolerance(v)
    update reliability, confidence bins, pairwise redundancy, graph transitions
    emit label, posterior, trace, cost, and abstention flag
```

## 5. What can and cannot currently be proved

### Causality

The implementation has a direct no-future-leakage property: the graph memory
contains only nodes committed in earlier routing rounds. A future neighbor
cannot influence the current node.

### Risk interpretation

If \(p_v\) is a calibrated posterior, the stopping rule bounds conditional
0-1 Bayes risk by \(\varepsilon_{s(v)}\). The important premise is posterior
calibration. Pseudo-label feedback, graph misspecification and distribution
shift can violate it; therefore current `risk_tolerance` values are operating
parameters, not certified probabilities.

### Bounded graph influence

Because every edge and the total neighbor mass are capped, the change in any
pairwise log-odds caused by graph messages is bounded by the product of graph
strength, accepted neighbor mass and the maximum message log-ratio. This does
not prove that graph evidence is correct, but it prevents an unbounded cascade.

### Regret roadmap

A future formal regret result should state all of the following explicitly:

- a realizable or bounded-misspecification model for contextual reliability;
- delayed or audited feedback assumptions;
- a bound on transition-estimation error;
- a calibration-error term;
- a cost/risk comparator policy.

Under standard stochastic linear-bandit assumptions, the LinUCB component can
inherit a \(\widetilde O(d\sqrt T)\)-type term per model. It is not currently
valid to claim that adding an arbitrary provenance Laplacian automatically
improves this bound. Laplacian graph-bandit gains require an explicit smoothness
assumption, as in [Yang, Toni and Dong (AISTATS 2020)](https://proceedings.mlr.press/v108/yang20c.html).

## 6. Relation to closest work

- **CaMVo** supplies the online reliability learning and cost-aware ensemble
  motivation, but selects a subset before seeing responses and relies primarily
  on conditional independence.
- **Laplacian-regularized graph bandits** show how graph smoothness can improve
  regret when the graph actually links similar preference parameters; TRACE
  does not assume every provenance edge satisfies this condition.
- **Directed graph learning** finds that direction can matter under
  heterophily; TRACE operationalizes this as past-to-current transitions rather
  than symmetrizing future and past evidence. See
  [Rossi et al. (LoG 2024)](https://proceedings.mlr.press/v231/rossi24a.html).
- **Selective blocking under heterophily** motivates not propagating across
  uncertain edges. TRACE's maturity/informativeness gate follows the same
  safety intuition. See
  [Choi et al. (UAI 2025)](https://proceedings.mlr.press/v286/choi25a.html).
- **Conformal LLM routing** provides a path toward distribution-free violation
  control once an audited calibration set is available. TRACE currently exposes
  the required abstention and calibration interfaces but does not yet implement
  this guarantee. See
  [Uddin and Bauer (ACL SRW 2026)](https://aclanthology.org/2026.acl-srw.70/).

## 7. Falsifiable hypotheses

H1. Under graph-preserving sampling, TRACE reaches a quality-equivalent point
with materially lower cost than full ensemble and original CaMVo.

H2. Correlation correction contributes more than raw graph smoothing when model
outputs share systematic errors.

H3. Relation-gated graph evidence helps on high-assortativity relations and
automatically turns off on weak or heterophilic relations.

H4. The benefit is larger on OpTC incident reconstruction than on CASIE event
subtype classification because OpTC requires multi-event context.

H5. Audited calibration reduces abstention miscalibration and makes risk
tolerance portable across model pools and datasets.

## 8. Required real-model experiment

The next real-provider run should use document-grouped calibration/validation/
test splits. All model responses must be cached once per item, and every policy
must be replayed against the identical response matrix. The validation split is
used to choose risk tolerance; the test split is evaluated once. Report:

- Macro-F1 and per-class F1;
- cost, latency, average calls, full-pool rate and abstention rate;
- document-cluster bootstrap intervals;
- reliability diagrams/ECE for the stopping posterior;
- graph diagnostics and relation-wise ablations;
- provider failure rate and cache completeness.

CASIE remains the clean annotation benchmark. OpTC should be the primary SOC
case study once real benign-period telemetry is available.
