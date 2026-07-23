# OTRF Security Datasets / Mordor: G-CaMVo research plan

## Decision

Use Mordor as the immediate provenance/correlation experiment after CASIE, but
separate two evidence levels:

1. **Public exact-label benchmark sample (primary immediate experiment).** The
   public Cyber Defense Benchmark sample contains 155,350 real Mordor-derived
   Windows log rows, 3,912 flag records (3,769 unique malicious timestamps),
   three attack chains, 18 attack steps and nine MITRE ATT&CK tactics.
2. **Original APT29 Day 1 + Day 2 compound logs (schema and external-validity
   study).** The complete archives contain 783,367 rows and four hosts, but no
   record has an explicit malicious/benign label. Archive membership must not be
   treated as event-level ground truth.

This design gives us exact scoring now without falsely labelling every APT29
background event as malicious. The original APT29 data remains useful for graph
coverage and portability checks.

## Verified local data

### Original OTRF APT29

| Scenario | Rows | Uncompressed bytes | SHA-256 of official ZIP |
|---|---:|---:|---|
| Day 1 | 196,081 | 385,334,029 | `98a073140860560d70080ace9142961be4f64b4862bae892d62d0f254d0fdbe5` |
| Day 2 | 587,286 | 1,714,987,031 | `377f8cba5db95a453a3ee8bd19f493efafc23724541482a4da99da28ee4665f9` |
| Total | 783,367 | 2,100,321,060 | — |

Strict streaming audit results:

- zero malformed records and zero blank rows;
- hosts: UTICA 492,268; SCRANTON 197,538; NEWYORK 53,142; NASHUA 40,419;
- 139,528 records expose process-to-process identifiers suitable for graph
  construction;
- 10,390 records expose source/destination network identifiers;
- zero records expose an explicit event-level label field;
- every record is tagged `mordorDataset`, which denotes dataset membership, not
  maliciousness.

The official emulation workbook has 31 Day-1 action/setup rows and 34 Day-2
rows. It supplies stage, ATT&CK technique, step, description, operator command,
user, source and target. It does not supply a timestamp for every action.

### Public exact-label sample

| Quantity | Count |
|---|---:|
| Log rows | 155,350 |
| Flag records | 3,912 |
| Unique malicious timestamps | 3,769 |
| Log rows at flagged timestamps | 8,658 |
| Attack chains | 3 |
| Attack steps | 18 |
| ATT&CK tactics | 9 |

The implemented balanced task contains 7,538 timestamp groups (3,769 positive
and 3,769 matched controls). Its telemetry-only graph has 13,993 undirected
edges, no isolated nodes, one connected component and mean degree 3.713. The
edge label-agreement rate is 91.47%; this is an evaluation diagnostic, not an
input feature. Machine-readable preflight results are in
`artifacts/mordor_cdb_full_preflight.json`.

The deterministic 1,000-item Adams pilot retains 500 positive and 500 matched
negative timestamp groups. It has 1,661 graph edges, zero isolated nodes, four
components, mean degree 3.322 and 84.83% edge label agreement.

The benchmark keeps `sample_flags.json` hidden from the hunting agent. Each flag
contains an exact timestamp, chain index, step index, narrative-step identifiers
and relevance. This is suitable for deterministic evaluation.

## Proposed task

Treat a unique timestamp plus its same-host local neighborhood as one routing
item. The model contract is structured:

```json
{
  "label": "malicious | benign | abstain",
  "confidence": 0.0,
  "tactics": ["TA0000"],
  "evidence": ["field=value"],
  "escalate": false
}
```

The graph contains only edges derivable without test labels:

- parent-process and process-access GUID links;
- same-host temporal adjacency;
- shared source/destination network endpoints;
- shared process image, account or service identity with bounded degree;
- attack-chain/step labels are used only for scoring, never for graph creation or
  prompting.

G-CaMVo propagates calibrated uncertainty over this graph and buys another model
only when the estimated risk remains above the validation-selected threshold.
TRACE-G-CaMVo additionally allows abstention/escalation for low-evidence regions.

## Fair baselines

- cheapest single model;
- calibration-selected best single model;
- full four-model ensemble;
- online weighted majority;
- original CaMVo;
- CCaMVo;
- text-only contextual router;
- G-CaMVo with graph strength set to zero;
- G-CaMVo and TRACE-G-CaMVo.

Every method must reuse the same frozen model-response matrix. Candidate
generation, split assignment and graph edges are frozen before test evaluation.

## Split and metrics

Do not randomly split individual log rows. Use attack-chain or host-time blocks,
remove cross-partition edges and fit calibration parameters outside the test
partition. With only the public sample, report this explicitly as a bounded
three-chain case study rather than cross-environment generalization.

Primary metrics:

- unique malicious-timestamp recall and precision;
- tactic-balanced coverage;
- cost per recovered narrative step;
- proxy dollars and average models per item;
- time to first true positive;
- false alerts per 1,000 non-flagged timestamp groups;
- graph ablation and paired block-bootstrap confidence intervals.

## Claim boundary

This experiment can support claims about cost-aware multi-model threat-event
detection and attack-step coverage. It cannot by itself prove automated
mitigation effectiveness. Original APT29 logs are semi-labelled; only the exact
flag sample or another defensibly generated ground truth may be used for test
metrics.

## Local paths

- Original data: `data/raw/otrf/apt29/day1/` and `data/raw/otrf/apt29/day2/`
- Original metadata/emulation plan:
  `data/raw/otrf-security-datasets-meta/datasets/compound/apt29/`
- Public exact-label sample: `data/raw/cyber-defense-benchmark-sample/datasets/`
- Full audit: `artifacts/mordor_apt29_full_audit.json`
- Audit command: `camvo-mordor-audit` or
  `python -m camvo.security.mordor_cli`
- One-command Adams pilot (run after CASIE completes):
  `bash RUN_FORMAL_MORDOR_ADAMS.sh`
- Full exact-label response matrix after a successful pilot:
  `G_CAMVO_MORDOR_ITEMS=all bash RUN_FORMAL_MORDOR_ADAMS.sh`

## Sources

- https://github.com/OTRF/Security-Datasets
- https://github.com/OTRF/Security-Datasets/tree/master/datasets/compound/apt29
- https://github.com/simbianai/cyber_defense_benchmark
- https://arxiv.org/abs/2604.19533
