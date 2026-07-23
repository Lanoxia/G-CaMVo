# Dataset registry

Raw and processed datasets are intentionally excluded from Git. This directory
tracks only provenance, download instructions, and task definitions.

## CASIE

- Official source: <https://github.com/Ebiquity/CASIE>
- Local raw path: `data/raw/casie/`
- License/terms: follow the upstream repository and cite the CASIE publication.
- Upstream corpus: 1,000 source/annotation pairs.
- Local PoC task: classify each annotated event mention into one of five
  subtypes: `Databreach`, `Phishing`, `Ransom`, `DiscoverVulnerability`, or
  `PatchVulnerability`.
- Leakage rule: the gold subtype is stored in `AnnotationItem.metadata` only
  for simulation/evaluation. `CaMVoRouter` never reads it.
- Split rule: every event from one `document_id` must remain in the same
  calibration/validation/test partition. Use `camvo.security.splits.grouped_split`.
- Graph rule: the implemented same-hopper/same-document adjacency is a text
  relation baseline, not a host provenance graph.

Download:

```bash
git clone --depth 1 https://github.com/Ebiquity/CASIE.git data/raw/casie
```

Validate the full parser without running any model simulation:

```bash
PYTHONPATH=src python -c \
  "from camvo.security import load_casie_event_items; \
  print(load_casie_event_items('data/raw/casie/data').stats)"
```

Citation:

> Taneeya Satyapanich, Francis Ferraro, and Tim Finin. CASIE: Extracting
> Cybersecurity Event Information from Text. AAAI 2020.

## DARPA OpTC

- Official metadata and ground truth: <https://github.com/FiveDirections/OpTC-data>
- Full telemetry link: documented in the official metadata repository.
- Corrected 2026 research mirror (DOI `10.57745/UXCWOC`):
  <https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/UXCWOC>
- Scale: approximately one terabyte compressed; do not clone it blindly.
- Local metadata path: `data/raw/optc-metadata/`
- First bounded scenario: Day 3, "Malicious Upgrade", 2019-09-25,
  `SYSCLIENT0051` and `SYSCLIENT0351`.
- Curated machine-readable ground truth: `config/optc_scenarios.json`.
- Important clock rule: the public ground-truth PDF does not state its UTC
  offset. Community event labels encode `-04:00`; the audit CLI still requires
  this offset to be supplied explicitly so the alignment assumption is visible.

Metadata download:

```bash
git clone --depth 1 \
  https://github.com/FiveDirections/OpTC-data.git \
  data/raw/optc-metadata
```

Bounded Day-3 raw-data download (official Drive file IDs, resumable and
size-verified):

```bash
bash scripts/download_optc_day3_subset.sh
```

The pinned manifest is `config/optc_day3_subset_files.tsv`.  It downloads the
two 2019-09-25 evaluation shards containing host 0051 and host 0351, plus one
independent 20--23 September benign chunk for each corresponding host range.
The four compressed files total 6,054,377,369 bytes (about 5.64 GiB).  This is
the smallest current repository recipe that supports both the two-host attack
chain and defensible negative controls; downloading only attack-period files is
not sufficient for a binary detection experiment.

The corrected mirror packages whole dates as very large TAR files (for
example, 2019-09-25 is about 63 GB). The bounded Day-3 task needs only two
attack hosts plus defensible benign-period controls, so the repository does
not automatically download a full-day archive. Prefer the official Drive's
day/host shards when quota permits; otherwise stage the corrected TAR on
storage with at least 150 GB free and extract only required host files.

After placing only the selected eCAR shards under `data/raw/optc-day3/`, audit
them with an explicitly verified clock offset:

```bash
PYTHONPATH=src python -m camvo.security.optc_cli \
  --input data/raw/optc-day3 \
  --scenario-id optc-day3-malicious-upgrade \
  --utc-offset-minutes <VERIFIED_OFFSET> \
  --json-out artifacts/optc_day3_audit.json
```

The parser streams JSONL/NDJSON and gzip variants, normalizes hostnames,
deduplicates event IDs, records malformed rows, builds a heterogeneous
provenance graph, and creates a sparse event-correlation graph. Raw telemetry
and any derived data remain ignored by Git.

The repository-level Adams entrypoint performs a stricter zero-provider
preflight and then launches the frozen four-model protocol:

```bash
bash RUN_FORMAL_OPTC_ADAMS.sh
```

It refuses to read credentials or call a model until attack shards, benign
shards, labels, scenario alignment, sample schemas, and free space pass the
readiness gate. The machine-readable preflight is written under
`artifacts/adams_formal_optc_day3_<N>/preflight.json`.

Community event-level positive labels:

```bash
git clone \
  https://gist.github.com/hamelin/7e1f2be6f2d6f9f645de60f73bd45b1a.git \
  data/raw/optc-labels
```

These rows are a best-effort list of attack-related events and join raw eCAR
one-to-one on `id`. They are positive-only: a missing ID must remain unknown
unless a separate defensible benign sampling rule supplies the negative label.

```bash
PYTHONPATH=src python -m camvo.security.optc_label_cli \
  --json-out artifacts/optc_day3_labels_audit.json
```

### Executable task variants

The no-key fallback uses real positive IDs/entity topology and explicitly
synthetic action/object-matched benign controls:

```bash
PYTHONPATH=src python -m camvo.security.experiment_cli \
  --dataset optc-label-graph \
  --max-items 1200 \
  --json-out artifacts/optc_six_strategy.json
```

This is marked `simulation_only=true` and cannot support a real detection
claim. It exists because the public Google Drive raw shards can temporarily
return quota errors.

Once the bounded attack and benign shards are present, run the real-data
builder with simulated models before spending on APIs:

```bash
PYTHONPATH=src python -m camvo.security.experiment_cli \
  --dataset optc-real \
  --attack-path data/raw/optc-day3/attack \
  --benign-path data/raw/optc-benign \
  --optc-labels data/raw/optc-labels/labels.csv \
  --max-items 1200 \
  --json-out artifacts/optc_real_simulated_models.json
```

The real builder scans the complete input with deterministic bottom-k hash
sampling. Positives must join the label CSV. Negatives come only from the
official benign collection period and are matched to the positive
`(object_type, action)` distribution. Reports retain a warning that this
design may contain temporal/domain shift.

For a formal test, group whole attack scenarios or host-time blocks; remove
cross-partition graph edges and keep only previously observed neighbors during
online routing. One Day 3 scenario by itself does not demonstrate cross-scenario
generalization.

## OTRF/Mordor

OTRF/Mordor is now the immediate second dataset after CASIE. The complete APT29
Day 1 and Day 2 host-log archives and official emulation-plan metadata have been
downloaded locally (under ignored `data/raw/` paths). A strict audit found
783,367 valid NDJSON records, four hosts, zero parse failures, 139,528 records
with process-link fields and 10,390 with network-link fields.

Important label boundary: the original compound archives intentionally include
background activity and contain no explicit per-record malicious label. Do not
label every archive record positive. For deterministic evaluation, use the
public Cyber Defense Benchmark sample under
`data/raw/cyber-defense-benchmark-sample/datasets/`, which supplies 3,912 hidden
flag records over 155,350 Mordor-derived log rows. See
`docs/OTRF_MORDOR_DATASET_PLAN_2026-07-21.md`.

Mordor complements but does not replace the main OpTC provenance result. It is
smaller and easier to reproduce, while OpTC remains the stronger test of native
provenance and DARPA-specific claims.
