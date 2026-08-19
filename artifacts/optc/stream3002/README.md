# OpTC Stream3002 reproducibility release

This directory contains the public, credential-free reproducibility record for
the frozen OpTC provenance-stream evaluation used in G-CaMVo.

## Scope

- **16,902,846** source telemetry events were processed locally into strictly
  past-only provenance state.
- **3,002** chronological online checkpoints were evaluated.
- Five models produced a complete **15,010-cell** frozen response matrix
  (`3,002 x 5`); all cells pass the frozen JSON schema gate.
- Security reconstruction is evaluated against **76 observable official
  red-team steps** and **73 official adjacent transitions**.
- A separate campaign-held-out triage audit contains **207 non-overlapping host
  windows**: 35 attack and 172 background windows.

The 3,002 checkpoints are the routing, cost, latency, and convergence units.
The 76 official steps and 73 transitions are the attack-reconstruction units.
The 207 windows are an auxiliary binary-triage audit. These denominators are
reported separately and are never added together.

## Frozen headline result

| Method | Balanced accuracy | Official steps | Adjusted timeline | Transition F1 | Cost / 3,002 | Avg. models | Saving vs full |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strongest single model (GLM 5.2) | 60.9% | 49/76 | 34.1% | 11.8% | $31.13 | 1.00 | 58.0% |
| Online-weighted full vote | **65.9%** | 49/76 | **35.0%** | **14.0%** | $74.03 | 5.00 | 0.0% |
| CaMVo (`delta=0.80`, `k_min=3`) | 64.8% | 49/76 | **35.0%** | **14.0%** | **$22.66** | **3.01** | **69.4%** |

Balanced accuracy gives attack and background windows equal weight. The raw
class counts are retained in `results/optc_main_results.csv`. CaMVo and the
online full-panel baseline use the same strictly past reliability state;
CaMVo selects its subset before current responses are observed.

## Directory map

```text
figures/          Paper-ready PDF figures and PNG previews
protocol/         Frozen task definition, denominators, leakage rules, and claims
results/          Main tables, uncertainty analyses, ablations, and schema audit
reproducibility/  Frozen facts, route trace, and compact reproducibility record
scripts/          Offline analysis and package-generation scripts
```

The compressed route trace contains routing decisions and aggregate model
outputs needed for offline replay. Raw OpTC telemetry, prompts, API credentials,
provider logs, and uncompressed private response caches are intentionally
excluded.

## Reproduce the offline analysis

The statistical tables can be regenerated directly from the released audit
inputs without provider calls:

```bash
python artifacts/optc/stream3002/scripts/analyze_optc_ppt_aligned_statistics.py \
  --output-dir /tmp/optc-stream3002-recomputed
```

The campaign-reconstruction script additionally requires the locally processed
OpTC telemetry described in the protocol. No provider call is needed to inspect
or recompute the published tables, route trace, uncertainty estimates, or
figures.

Start with:

- `protocol/OPTC_PPT_ALIGNED_FINAL_PROTOCOL_20260819.md`
- `results/optc_main_results.csv`
- `results/schema_gate.json`
- `figures/01_OpTC_PPT_Aligned_Accuracy_Cost_Table.pdf`
- `figures/02_OpTC_PPT_Aligned_Online_Trajectory.pdf`

## Claim boundary

The release supports a cost-aware online provenance-analysis claim: across all
3,002 chronological checkpoints, CaMVo retains the full panel's official-step,
timeline, and transition scores at substantially lower replay-policy cost. It
does not treat model responses as independent labeled examples, and it does not
claim that the auxiliary host-window audit replaces official attack-chain
evaluation.
