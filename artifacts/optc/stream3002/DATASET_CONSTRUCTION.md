# Constructing the frozen OpTC Stream3002 benchmark

This document specifies how the public OpTC source files were transformed into
the frozen 3,002-checkpoint benchmark.  It separates **input construction**,
**evaluator-only ground truth**, and **post-freeze scoring** so that official
labels or model answers cannot influence candidate generation.

The raw OpTC archives are not redistributed here.  Obtain them from the public
OpTC release and place them at the relative paths listed in
`construction/config/optc_core_acquisition_v1.tsv`.  The companion SHA-256 file
identifies the exact 23 archives used by the experiment.

## 1. Source scope

The frozen scope covers the three documented red-team campaigns and matched
benign captures for the same host buckets:

| Campaign | Local time range (UTC-04:00) | Core hosts |
|---|---|---|
| Day 1, Plain PowerShell Empire | 2019-09-23 11:20–15:35 | SYSCLIENT0201, SYSCLIENT0402, SYSCLIENT0660 |
| Day 2, Custom PowerShell Empire | 2019-09-24 10:25–15:35 | SYSCLIENT0005, SYSCLIENT0501, SYSCLIENT0811, SYSCLIENT0974 |
| Day 3, Malicious Upgrade | 2019-09-25 10:25–14:30 | SYSCLIENT0051, SYSCLIENT0351 |

The exact case manifest is
`construction/config/optc_core_cases_v1.json`.  The acquisition inventory is
`construction/config/optc_core_acquisition_v1.tsv`; it records day, capture
class, host bucket, public source identifier, byte count, and required local
path.  Verify the files before processing:

```bash
sha256sum -c \
  artifacts/optc/stream3002/construction/config/optc_core_acquisition_v1.sha256
```

The original run read approximately 46 GiB of compressed source files and
indexed **16,902,846** in-scope events.

## 2. Construction stages

### E2 — label-blind canonical telemetry stage

`build_optc_stream_stage.py` reads only telemetry plus the acquisition and case
manifests.  It does not accept a ground-truth or label argument.

Each retained event is normalized to a common schema containing corpus, day,
event ID, timestamp, host, actor, object, object type, relation/action, process
fields, principal, and a fixed allowlist of telemetry properties.  Events are
stored in canonical `(corpus, timestamp, event_id)` order.  Per-file progress
is checkpointed in SQLite so the multi-gigabyte scan is resumable.

Expected gate:

- 23 source files audited;
- 16,902,846 in-scope events indexed;
- zero canonical ordering reversals in the staged database;
- zero future edges materialized;
- output fingerprint recorded in `e2_parser_report.json`.

### E3 — deterministic checkpoint candidate generation

`generate_optc_candidates.py` scans the E2 stage without reading official
labels or model responses.  A checkpoint is emitted when an already observed
event satisfies one of five trigger families:

1. remote execution or session activity;
2. private cross-host network activity on a fixed remote-service port set;
3. sensitive file or registry change;
4. a new task/process root;
5. periodic sampling of an active causal component.

The trigger selection is deterministic.  The frozen parameters are:

| Parameter | Value |
|---|---:|
| Per-component checkpoint cap | 6 |
| Per host-hour-trigger cap | 8 |
| Same-trigger cooldown | 60 seconds |
| Active-component periodic interval | 600 seconds |
| Minimum events before periodic sampling | 20 |

The resulting **3,002 checkpoints** consist of:

| Trigger | Count |
|---|---:|
| Active task periodic | 929 |
| Cross-host connection | 22 |
| New task root | 447 |
| Remote execution/session | 805 |
| Sensitive object change | 799 |

The source-capture totals are 1,429 attack-capture checkpoints and 1,573
benign-capture checkpoints.  These capture identifiers are never provider
labels: the model must still decide from prompt-visible evidence.

### E4 — physically separated evaluator-only gold

`build_optc_evaluator_gold.py` is the first stage allowed to read the official
red-team ground-truth PDF and the public exact-event label file.  Its SQLite
output is kept outside candidate generation and prompt-visible data.

It records:

- 101 documented red-team steps;
- the 76-step observability audit used by the final paper evaluation;
- 73 adjacent official step transitions;
- exact positive event identifiers used only for evaluator joins.

E4 cannot add, remove, or reorder E3 checkpoints.

### E5 — strictly past-only evidence bundles

`build_optc_evidence_bundles.py` expands every E3 anchor through actor/object
relations that were observed no later than that anchor.  The frozen bundle
budget is:

| Parameter | Value |
|---|---:|
| Lookback | 900 seconds |
| Provenance hops | 2 |
| Maximum queried events per hop | 2,000 |
| Maximum nodes | 30 |
| Maximum evidence edges | 40 |
| Maximum estimated prompt tokens | 4,000 |

Events are ranked deterministically by causal distance, anchor status,
relation priority, recency, and event ID.  Duplicate actor–relation–object
tuples are merged.  Raw UUIDs are replaced by local opaque node IDs (`N01`,
...) and evidence IDs (`E01`, ...).  All timestamps become offsets relative to
the current checkpoint.

The attack and benign captures have non-comparable collection dates.  Their
within-capture ranks are therefore merged into an opaque monotone replay
sequence.  This sequence preserves order within each capture without exposing
the original capture date to the model or router.

E5 writes four separated products:

- `prompt_visible/checkpoints.jsonl`: provider-visible evidence bundles;
- `router/checkpoint_features.jsonl`: label-blind hashes and replay position;
- `evaluator/checkpoint_map.jsonl`: private evaluator joins;
- `evaluator/splits.jsonl`: Day 1 calibration, Day 2 validation, Day 3 test.

The E5 gate requires 3,002 bundles, zero future evidence, zero forbidden label
metadata, zero raw UUIDs, zero absolute dates in rendered prompts, and zero
source-group overlap across splits.

### E6/E7 — prompt freeze and response matrix

`construction/prompt.py` is the exact frozen prompt and response validator.
The provider-visible schema requires verdict, confidence, analyst-review flag,
stage, evidence references, timeline, entities, attack-path edges, missing
evidence, recommended next step, and rationale.

Before provider collection, the prompt version, prompt hash, checkpoint order,
model pool, and expected `3,002 x 5 = 15,010` cells were frozen.  The released
`results/schema_gate.json` confirms that all 15,010 stored responses passed the
same schema and evidence-ID validation.

## 3. Rebuild commands

Install the repository and PDF dependency in a clean Python 3.10+ environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . pypdf
```

After placing the 23 archives, the zero-provider-call construction is:

```bash
python artifacts/optc/stream3002/construction/rebuild_stream3002.py \
  --ground-truth-pdf data/raw/optc-metadata/OpTCRedTeamGroundTruth.pdf \
  --exact-labels data/raw/optc-labels/labels.csv \
  --output-dir data/processed/optc_stream3002_rebuild
```

The command runs E2 through E5 and stops before any model provider call.  It
checks the frozen cardinalities and leakage gates after every stage.  Use
`--skip-stage` only when the existing stage database and its report have
already passed E2 validation.

## 4. What is and is not released

Released:

- exact source inventory, byte counts, and SHA-256 hashes;
- campaign/host/time scope;
- E2–E5 construction code and frozen parameters;
- exact prompt and response validator;
- aggregate results, audit inputs, route trace, figures, and offline analysis.

Not redistributed:

- third-party raw OpTC archives and official PDF;
- API credentials or provider endpoints tied to a user account;
- uncompressed private provider-response caches;
- raw prompts containing public-dataset telemetry values.

These exclusions prevent republishing third-party data or secrets.  They do
not change the deterministic benchmark definition.

## 5. Statistical-unit boundary

The construction yields several related but non-interchangeable units:

- 3,002 checkpoints: online routing, model use, cost, latency, and escalation;
- 15,010 cells: five frozen model outputs per checkpoint, not independent gold
  examples;
- 76 observable official steps and 73 transitions: attack reconstruction;
- 207 non-overlapping host windows: auxiliary attack/background triage;
- 1,306 one-minute windows: scale sensitivity only.

These denominators must be reported separately and must never be added.
