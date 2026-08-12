# Mordor frozen benchmarks and CaMVo online replay

This release records the exact Mordor-derived datasets and replay artifacts used in the G-CaMVo experiments discussed in August 2026.

## Upstream source

- Dataset repository: [OTRF/Security-Datasets](https://github.com/OTRF/Security-Datasets)
- Frozen upstream commit: `d9d40ef123d2c87d5d3df28c96bcab4f0faccc87`
- ATT&CK scoring taxonomy: Enterprise ATT&CK 8.2
- The complete upstream snapshot is not duplicated here. Each reproducibility package records the upstream-relative paths and SHA-256 identities of the official files used.

## Packages

### `Mordor_Stream114_Primary101_Reproducibility_v1.zip`

The exact reproducibility bundle for the earlier 114-round chronological stream:

- 114 chronological bundles;
- 101 primary tactic-level scored decisions;
- 13 additional stream-context decisions;
- five-model frozen response matrix (570 response cells);
- model-visible inputs and evaluator-only labels in separate directories;
- source-file inventory and hashes, frozen ATT&CK taxonomy, price snapshot, replay code, and traces;
- complete internal SHA-256 manifest.

SHA-256: `2ee5c986b18f3d060822611547b9c0b22b2d1f5fad04111f1a429f6a6af3d13e`

### `Mordor_Atomic112_Tactic_Results_20260811.zip`

The frozen decision-level extension:

- 112 chronological test decisions from 32 official source groups;
- source-group split and frozen test manifest;
- five individual-model results, uniform full vote, calibration-weighted full vote, online-learned full vote, and CaMVo;
- source-group bootstrap intervals and paired exact tests;
- exact CSV/JSON results, decision traces, analysis code, paper table, and figures;
- complete internal SHA-256 manifest.

SHA-256: `4c05151f90e8e0ed558e4710bf616026fbd84e72f949060aaaa80d4db721bbb3`

### `Mordor_Atomic112_Reproducibility_v2.zip`

The self-contained audit and replay bundle for the Atomic112 extension:

- all 171 frozen decision anchors and source-group-preserving calibration,
  validation, and test partitions (29/30/112 decisions);
- exact model-visible inputs and evaluator-only labels in separate directories;
- five complete frozen response files (171 rows each; 855 response cells);
- the separate compound-61 stress manifest and excluded-source audit;
- frozen price snapshot, Enterprise ATT&CK 8.2 taxonomy, and upstream OTRF
  commit/path/SHA-256 provenance;
- CaMVo source code, package-local replay script, results, traces, tables, and
  figures;
- package-wide SHA-256 manifest and reproduction instructions.

This is the package to use for independent Atomic112 auditing and replay. The
smaller Atomic112 results archive above is retained as a convenient
paper/professor-facing results package.

SHA-256: `16ee1ba7098771b1721c60955dc1f51a22262b09169a1c648634d0621fc0050c`

## Atomic-112 headline result

| Method | Correct / 112 | Top-1 tactic accuracy | Estimated cost / 1K decisions | Average models / decision |
|---|---:|---:|---:|---:|
| Strongest individual model (GLM 5.2) | 79/112 | 70.5% | $6.31 | 1.00 |
| Five-model full vote with online-learned weights | 81/112 | 72.3% | $13.18 | 5.00 |
| CaMVo (`delta=0.80`, `k_min=3`) | 80/112 | 71.4% | $5.75 | 3.32 |

CaMVo finishes one correct decision below online full vote while reducing estimated cost by 56.4%. The paired accuracy differences are not statistically significant at this scale; the supported claim is accuracy preservation at lower cost, not statistically significant accuracy superiority.

## Sample-count boundary

The 101 records in the first package are the primary scored decisions in the original 114-round benchmark, not the entirety of the raw Mordor archive. The 112 records in the second package are chronological decision anchors derived from 32 official source groups already represented in the earlier benchmark. They increase decision-level coverage but are not 112 newly independent official scenarios and are not a prospective source-unseen holdout. Both decision count and source-group count must be reported.

## Privacy and integrity

The packages exclude API keys, internal endpoints, local absolute paths, runtime logs, and professor correspondence. Package contents are covered by SHA-256 manifests and were scanned before publication.
