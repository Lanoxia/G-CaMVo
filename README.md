# G-CaMVo

Graph-aware, cost-aware online LLM subset voting for dependent cyber-security events.

This repository is an independent research implementation built from the published
CaMVo formulation in *Cost-aware LLM-based Online Dataset Annotation* (NeurIPS
2025). It is not the authors' official implementation.

## Motivation

CaMVo selects a subset of \(m\) models from a pool of \(n\), aggregates their
weighted votes, and updates model reliability online. G-CaMVo preserves this
core routing and voting structure while introducing past-only graph evidence for
security events that are sequential or causally dependent.

The current implementation contains:

- the original CaMVo routing backbone;
- CCaMVo-style correlated-output confidence;
- SAGE-CaMVo with an online, decayed relation graph and safe fallback;
- DCR G-CaMVo with class-conditional reliability, error diversity, cost, and
  bounded causal graph fusion;
- State-Switch G-CaMVo with explicit online entity-state memory;
- CASIE, Mordor, and DARPA OpTC data adapters;
- group-aware evaluation, paired cluster bootstrap, ablations, and feedback
  delay/rate stress tests;
- generic OpenAI-compatible and Dify workflow interfaces.

## Current full-CASIE result

The complete frozen response matrix contains 8,485 CASIE events evaluated by
four real LLM endpoints (33,940 model/event responses). The primary
leakage-controlled suite removes 23 documents used during development and
evaluates 7,985 events from 977 documents with five document-grouped out-of-fold
splits.

| Method | Macro-F1 | Average models |
|---|---:|---:|
| CaMVo k=2 | 0.9075 | 2.439 |
| SAGE-CaMVo | 0.9101 | 2.439 |
| Best single model | 0.9185 | 1.000 |
| DCR G-CaMVo | **0.9247** | 2.000 |
| State-Switch G-CaMVo | **0.9258** | 2.000 |

DCR improves over CaMVo by +0.01716 Macro-F1 with a paired
document-cluster bootstrap 95% CI of `[+0.01272, +0.02168]`. Relative to the
matched diversity/no-graph ablation, DCR improves by +0.00615 with CI
`[+0.00423, +0.00835]`, providing evidence that graph information contributes
independently.

These figures are aggregate results from frozen model outputs. Raw provider
responses are intentionally not included.

## Reproduce the algorithmic framework

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
PYTHONPATH=src python -m unittest discover -s tests -v
```

No API key is required for unit tests or simulated experiments. To connect
models, copy `config/api_keys.env.example` to a local untracked file and bind a
provider through the generic interfaces in `src/camvo/llms/`.

## Research claim boundary

CASIE exposes article/event order, not a full host-process-file-network
provenance graph. The current evidence supports dependent-event routing and a
graph-specific gain within CASIE. DARPA OpTC is the planned provenance-rich
external validation.

Because the public OpTC release is approximately one terabyte, the first
experiment uses red-team-ground-truth-centered attack windows, affected hosts,
one- to two-hop causal neighborhoods, and time-matched benign windows. Coverage
is expanded only after a positive incident-cluster confidence interval.

## Repository map

```text
src/camvo/                 Core routers, model adapters, data and metrics
scripts/                   Offline frozen-matrix and dataset utilities
tests/                     Unit and integration tests
docs/                      Method, protocol, result, and claim-boundary notes
config/                    Credential-free configuration templates
dify/                      Portable multi-model workflow template
data/README.md             Dataset sources and leakage rules
```

## Data and confidentiality

The repository contains code and aggregate statistics only. It excludes:

- API keys, user tokens, and internal endpoints;
- company-specific workflow identifiers;
- raw provider responses and prompts;
- downloaded CASIE, Mordor, and OpTC data;
- runtime logs, budgets, checkpoints, and local notebooks.

## References

- CaMVo: https://proceedings.neurips.cc/paper_files/paper/2025/file/054e9f9a286671ababa3213d6e59c1c2-Paper-Conference.pdf
- DARPA Transparent Computing: https://www.darpa.mil/research/programs/transparent-computing
- OpTC data release: https://github.com/FiveDirections/OpTC-data

