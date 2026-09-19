# GLiNER2.5 completion audit

This is a requirement-to-evidence ledger, **not a completion certificate**.
The task is to implement `PROMPT-gliner2.5.md`, preserve v2 behavior, publish
usable Rust inference and model bundles, and push the verified changes to GitHub.
An unexecuted script, model-skipping test, or planned artifact does not pass a gate.

## Requirement map

| Prompt requirement | Required artifact / evidence | Current status |
| --- | --- | --- |
| Read-first 1,6: specification and context | `gliner2-vs-gliner2.5.md`, `jev-and-gliner.md`, `agentic-use-cases.md` | Read; recovered from sibling research worktree |
| Read-first 2,3: current implementation/exporters | `src/{pipeline,extractor,decode,spans,embeddings,text,schema,classification,adapters}.rs`; existing encoder/extractor/classifier export/validate scripts | Read; validators currently only smoke-test shapes |
| Read-first 4: real pinned upstream | clone, commit, boundary/inference/processor/splitter/LoRA source inspection | Clone pinned at `d7c727458bf6929bc9ef5ee04e13c3f717a7c455`; Astra source review completed, corrections in plan |
| Read-first 5: checkpoint configs and release | revision-pinned configs for small/base/multi; release post | Retrieved all three configs and release post |
| Model roles; Phase A before B | Astra plan/reviews; Sol implementation against plan | Astra Phase A approved; Sol M0 implementation started against full plan |
| Phase A checkpoint comparison | `PLAN-gliner2.5.md`: revisions, graph differences, tokenizer/splitter | Verified configs in `docs/checkpoints/`, source-reviewed defaults and encoder widths in plan |
| Phase A graph and Rust mappings | plan covers all named upstream classes/functions and two-graph decision | Approved; extra explicit sparse graph required by actual source |
| Phase A public API | plan for `Extractor`, `SpanPipeline`, compatibility alias, `BoundaryPipeline`, `AutoPipeline::from_dir` | Approved enum abstraction; preserve low-level extractor name |
| Phase A goldens | ~30-case corpus design, >=3 long cases, edge/non-ASCII/CJK/tasks, tensors, tolerance | Designed in plan; actual fixtures still pending |
| Phase A risks | plan resolves every specification risk; maintain current risk list | Source-driven decisions documented; runtime experiments remain mandatory |
| M0 dispatch | `config.json` architecture default span, unknown/version errors; auto loader tests | M0: `src/config.rs`, `src/pipeline.rs`, `tests/{config,pipeline_api}.rs`; boundary loading explicitly unavailable until later milestones |
| M0 v2 compatibility | `SpanPipeline` and old alias; existing methods/examples unchanged | M0: source reviewed; model-backed suite and six tutorial result snapshots preserved |
| M0 preprocessing | config max_len word truncation, default none for v2 / 4096 boundary; query marker enumeration including `[R]` | M0: `src/preprocessing.rs`, `src/embeddings.rs`, unit tests and model-backed `tests/truncation_e2e.rs` |
| M0 artifact paths | consistent crate-root defaults and explicit overrides; no silent model test skips with artifacts | M0: `tests/common/mod.rs`, `GLINER2_TEST_ROOT`, strict `GLINER2_REQUIRE_MODELS=1`; model run has zero skips |
| M1 Python environment | uv `scripts/export/env`, pinned gliner2/torch/dependencies; ignored venv | M1: pyproject + uv.lock, exact Git pin, locked install and source/VCS checks verified |
| M1 golden generator | `scripts/parity/gen_boundary_goldens.py`; actual upstream checkpoint inference | M1: all30 public-oracle cases generated and independently reproduced (corrected upstream AutoExtractor loader) |
| M1 corpus/tensors | `.npz` + JSON: hidden/text/query, marginals/projections, pool indices/mask/compat, pair/null/count, final spans/confidence | M1: complete applicable stage arrays verified, 30 identical regenerated NPZ hashes; 3 long cases included |
| M1 fixture size | ignored full `fixtures/gliner2.5-base-v1/`, committed representative subset <=2 MB, reproducibility hashes | M1: full125MB ignored; subset1,786,618bytes including manifest; byte hashes and source identity checked |
| M1 common exports | real fp32 encoder/classifier export + numerical validation, state-dict key verification | M1: parent34 encoder cases/20 classifier comparisons pass original tolerance; recursive fp32/finite/opset17 audit; `docs/evidence/m1.json` |
| M2 marginals graph | `scripts/export/export_boundary_marginals.py`, `validate_boundary_marginals.py`, dynamic L/Q, actual ORT comparisons | Accepted: parent24 head/6 bypass + dynamic/B2 checks, recursive fp32/finite audit; prefix-only adjudication + gated scorer sensitivity; `docs/evidence/m2.json` |
| M2 Rust | `src/boundary/marginals.rs`, named I/O/threading consistent with orp/ort, committed-subset parity test | Accepted: subset/full30 corpus, pre-native L0/Q0 guards, no erroneous combined-length4096 cap; full model/no-model gates pass |
| M3 pool | `src/boundary/pool.rs`, top-k/quota/rank/dedup/order, exact indices/mask/compat for every fixture | Accepted:24 frozen real +9 synthetic bit-exact, debug/release; actual ORT-to-pool exact discrete parity on24 cases; `docs/evidence/m3.json` |
| M3 ties | root-cause evidence for any tie exception and Astra-written adjudication; no silent tolerance weakening | No exception needed: source-faithful stable sorts/tie rules and fixed AArch64 reduction grouping; exact frozen compatibility preserved |
| M4 scorer | `export_boundary_scorer.py`, explicit indices/mask/compat, null/count/candidate states; actual dynamic ORT tests | Accepted:24 real + dynamic/B2 cases, typed Rust preflight/runtime, actual marginal→pool→scorer integration; original score tolerances; `docs/evidence/m4.json` |
| M4 decoder | `src/boundary/decode.rs`: temperature/threshold/abstention/all overlap policies, half-open spans, Unicode offsets/order/dtypes | Accepted: upstream-generated synthetic vectors and real-stage parity, weighted interval scheduling/ties, strict null gate and UTF8-safe original offsets; boundary-specific tokenizer/classifier selection |
| M4 API/parity | entity + classification + combined extract; exact labels/spans, <=1e-3 confidence error on all fixtures | Accepted: exact21 eligible corpus IDs +2 independently regenerated mixed-task cases; final maximum confidence error5.7816505e-6, exact labels/text/order/offsets; later task families explicitly rejected |
| M4 tutorials | run tutorials 1 and 2 against base 2.5 | Accepted: parent ran original full tutorial_1_classification/tutorial_2_ner on2.5; same tutorials on v2 retain byte-identical baseline results |
| M5 records | real `RecordHead.forward_group` export, `decode_group` anchor/natural/legacy, all `extract_json*` APIs | Pending |
| M5 gate | structure goldens and tutorial 3 on base 2.5 | Pending |
| M6 relations | typed argument pair generator incl. caps, sparse biaffine scorer export, all `extract_relations*` APIs | Pending |
| M6 gate | relation goldens and tutorial 6 on base 2.5 | Pending |
| M7 all bundles | `export_all_2_5.sh`; small/base/multi complete graphs + config/tokenizer + manifest | Pending |
| M7 manifest | architecture/version, upstream commit, HF revision, opset, ort; fp32 finite masks, no weights in git | Pending |
| M7 publication | upload tested bundles to `codesoda/gliner2-onnx`, verify remote download; ask if credentials missing | Pending |
| M7 downloaders | `scripts/download_models.py`, `examples/download_models.rs`; fresh complete bundle download | Pending |
| M7 user docs | README "GLiNER2 vs 2.5: which to load", standalone consumer build/run via Git dependency | Pending |
| M7 latency | `RESULTS-gliner2.5.md`: measured CPU base v2/2.5 50/500/3000-word table, reproducible command/hardware | Pending |
| M7 CI/spec | CI skips absent models, checks real unit/committed-fixture coverage; specification §4/§5 updated to shipped state | Pending |
| M8 optional features | only after M7: attributes, constrained classification, JointIE, long chunking, each own gate | Optional; not started; do not advertise as shipped |
| Explicit-span primitive | public `score_explicit_spans`, equivalence tests and example | Pending (retain even if optional helpers deferred) |
| Every milestone gates | `cargo fmt`, `cargo clippy --all-targets -- -D warnings`, `cargo test` without AND with models | M0: fmt/clippy/no-model and strict-model suite pass; see progress evidence. Later milestones pending |
| Every milestone records | `PROGRESS-gliner2.5.md`: done/measured/next/questions; descriptive git commit; Astra review | Phase A entry/review complete; implementation gates pending |
| v2 invariant | actual pre/post byte-identical model outputs; all existing examples/tests still pass | M0: six tutorials' full results identical (timing/path log lines excluded), four-case snapshot identical SHA256; all existing model tests pass. Recheck later changes |
| Runtime/build constraints | no Python runtime/build dependency, ort rc.9 retained unless separately approved, no large fixtures/weights committed | Existing code satisfies; recheck all changes |
| Final GitHub state | clean committed branch, pushed SHA equality, CI result, documented installable revision | Pending |

## Baseline observations

- Branch `quick-mantis`, base commit `2883a73`, clean before document recovery.
- GitHub origin `https://github.com/codesoda/gliner2-rs.git`; authentication available.
- `cargo test` passes without model artifacts, including skips. This does **not**
  prove model correctness. Full local output: `/tmp/gliner25-work/baseline-test.log`.
- `cargo clippy --all-targets -- -D warnings` fails on 10 pre-existing library
  diagnostics (borrow/get-first/single-match/too-many-arguments/repeat/default).
- Existing ONNX-only v2 bundles are under `/Users/chrisraethke/projects/gliners2/onnx`.
  Downloaded tokenizer/config from `fastino/gliner2-base-v1` revision
  `79c3a777abc572b4767922f3916cf63fb5754df2` into `/tmp/gliner25-work/v2-tokenizer`.
  Public hosted v2 bundle lacks tokenizer/config files: fresh-consumer usability
  needs fixing.
- Baseline `2883a73` archived to `/tmp/gliner25-work/baseline/crate`, with its
  parent-layout model fixtures populated: **all existing tests pass, no SKIP**.
  Output: `/tmp/gliner25-work/baseline-model-test.log`. Cleaned the package build
  before running, because shared Cargo target reuse initially embedded the wrong
  `CARGO_MANIFEST_DIR` (that initial skipped run is not counted as evidence).
- Captured pre-change v2 formatted outputs on four English/Unicode/multi-entity
  and sentiment cases in `/tmp/gliner25-work/v2-before.txt`, using temporary
  `baseline/crate/examples/v2_snapshot.rs`. Compare byte-for-byte after M0.
  This is representative regression evidence, not exhaustive parity proof.
- Source documentation recovered from ctx-confirmed research worktree, not invented:
  Pi session `cac67d4e`, event `2ef7426e`, provider session
  `01a0b3fc-98b9-76ea-b503-c9a00fb83967`. Current files/remote sources are authoritative.
