# GLiNER2.5 progress

The active goal is **not complete**. See `AUDIT-gliner2.5.md` for the full
prompt-to-evidence checklist. Model-free test success is not model parity.

## Phase A — plan approved (historical entry)

### Done

- Recovered the requested prompt and its three companion research documents
  from the sibling research worktree; the task branch previously lacked docs/.
- Read current Rust pipeline, embeddings/schema/tokenization/decoding/adapters,
  exporter/validator scripts and context documents.
- Cloned upstream and pinned `d7c727458bf6929bc9ef5ee04e13c3f717a7c455`.
  Astra reviewed real boundary, processor, runtime and decoder source.
- Retrieved immutable small/base/multi model revisions and committed their
  actual configs. Compared encoder dimensions and safetensors headers.
- Authored `PLAN-gliner2.5.md` before implementation, including source-derived
  corrections: extra explicit sparse scorer, stable shared-pool sort, record
  legacy semantics, choice-prefix truncation, whitespace default for multi,
  directional relations and byte-offset policy.
- Started Sol M0 implementation against the full approved plan; no boundary
  inference support is being represented as finished at the scaffold stage.

### Measured

- Baseline no-model `cargo test`: passes with model tests skipped.
- Baseline strict Clippy: 10 pre-existing library diagnostics; fix mechanically
  in M0 without changing public APIs or v2 outputs.
- Baseline archived at `2883a73`: existing tests with actual v2 model artifacts
  and pinned tokenizer all pass with **zero SKIP messages**.
- Captured four representative pre-change v2 outputs (including Unicode and
  combined classification/entities) for byte-for-byte M0 comparison.
- All three boundary_head configs are identical. Small hidden width384;
  base/multi768. All actual checkpoint tensor dtypes F32. Verified classifier
  keys `.0` and `.3`; encoder configs' float16 metadata is not authoritative.
- Existing public ONNX repository lacks tokenizer/config files; fresh-consumer
  validation must not rely on developer-local metadata.

### Next

- Finish/review M0: architecture dispatch + compatibility aliases, synchronized
  token/offset truncation before choice prefix, extraction query enumeration,
  consistent test fixture paths/strict model tests, lint and formatting hygiene.
- Run all M0 gates; compare baseline snapshot and audit actual code before commit.
- Independent M1 Python artifact preparation is now running while M0 finishes;
  no M1 gate will be accepted before M0 review. No export success or stage-wise
  parity is claimed yet.
- Reference-loader investigation found `GLiNER2.from_pretrained` is deliberately
  span-only in pinned upstream and fails boundary checkpoints (missing max_width).
  `AutoExtractor.from_pretrained` succeeds with the real pinned base2.5 checkpoint
  and returns Alice/Acme/Paris with spans/confidence. Plan now documents this
  required correction. Experimental Python3.12 environment used torch2.8.0,
  transformers4.57.6, onnx1.17.0, ort1.20.1; upstream falls back to eager encoder.
- Planning/source documents are committed as `65646aa` and pushed to
  `origin/quick-mantis`. This is a progress branch, not a usable 2.5 release.

### Open questions / gates

- Exact floating-point compatibility parity across torch and Rust needs
  measurement; Astra has approved **no** tolerance or pool-order exception.
- Exporter dependency compatibility (upstream declares transformers<5 despite
  checkpoint metadata5.8.0), finite-mask attention lowering and long-sequence
  memory require actual experiments in M1/M2.
- HF publication prerequisite checked after M1: no credential is configured.
  Public downloads work; M7 upload to codesoda/gliner2-onnx will require local
  authentication with a repository-authorized write token. User was informed;
  interactive clarification UI failed, so authentication is not assumed.
  GitHub authentication/push works. Implementation continues meanwhile.
- Optional M8 helpers are not implemented or advertised. Public explicit-span
  primitive remains in required deliverables because downstream use needs it.

## M0 — accepted after source review and independent gates

### Done

- Sol implemented config parsing and high-level enum dispatch; existing
  `Gliner2Pipeline` aliases `SpanPipeline` and preserves its complete API.
  Root `Extractor` aliases the enum without changing low-level extractor API.
- Boundary constructor rejects unsupported inference clearly at this scaffold
  milestone. No false boundary-to-span dispatch or placeholder predictions.
- Shared preprocessing caps original words and offsets together; choice prefix
  is prepended afterward and never truncated. Raw and both classification paths
  apply the cap. No default cap on legacy span models.
- Added per-marker boundary query gathering with schema/field/kind metadata and
  validated shapes. Legacy embedding gather unchanged.
- Unified test fixture root with an explicit override and strict missing-model
  failure mode. Fixed pre-existing lint issues; broad Rust diff is mostly rustfmt.
- Astra parent reviewed actual source, added real model cap-propagation test,
  and independently reran gates and v2 tutorial comparisons.

### Measured

- `cargo fmt --all -- --check`: pass.
- `cargo clippy --all-targets -- -D warnings`: pass.
- No-model `cargo test -- --nocapture`: 42 tests pass, 12 explicit artifact skips.
- Strict model suite: 42 tests pass, **zero skips**.
- Tutorials1–6 run with actual v2 ONNX artifacts. All result text, including
  confidences, is byte-identical to baseline2883a73 after excluding only timing
  lines (model load line also contains the artifact path).
- Four-case English/Unicode combined-extraction snapshot is byte-identical
  without normalization. Evidence/hashes: `docs/evidence/m0.json`.
- Local command logs: `/tmp/gliner25-work/m0-review-{clippy,no-models,models}.log`,
  `/tmp/gliner25-work/m0-tutorial-comparison.txt`.

### Next / open

- Finish M1 pinned Python environment, upstream goldens, encoder/classifier
  exports and actual numerical validation; review all generated evidence.
- M2–M7 and public explicit scoring remain unimplemented/unverified. Goal is
  still open. M0 acceptance does not constitute a working 2.5 release.

## M1 — generated and independently reproduced (initial review entry)

### Done / measured

- Sol generated all30 real upstream cases with no oracle errors, including
  1000/2000/3000-word texts (1014/2012/3014 subword input lengths).
- Parent independently reran the full upstream corpus into a separate directory:
  **all30 NPZ files are byte-identical**, and Python/UTF-8 formatted outputs
  match for all30 cases. This is actual repeated inference, not merely checking
  that NPZ serialization is deterministic.
- Parent numerically validated encoder ONNX on all30 corpus inputs plus four
  generated variable-length/batch cases. Maximum absolute error6.444007e-5;
  existing atol1e-4/rtol1e-3 passes. Large relative maxima occur near zero.
- Classifier validation on generated row counts and real classification states:
  maximum absolute error2.861023e-6; same original tolerances pass.
- Full fixtures125MB remain ignored. Proposed committed subset1,785,134bytes
  covers Unicode/entity, classificationQ0, and relation inputs.
- Parent reran Rust fmt/clippy/no-model and strict-v2-model suites successfully.
  Independent logs/reports are `/tmp/gliner25-work/m1-parent-*`.

### Review gaps being fixed before accepting M1

- Verifier must reject duplicate/missing manifest cases and incorrect array/stage
  metadata, not trust the case_count field or caller-controlled invocation flags.
- Generator must verify source-model hashes and installed upstream commit rather
  than stamping constant provenance for arbitrary --model-dir; actual seed must
  propagate into per-case provenance.
- Partial export manifests must identify Rust ort crate version, not only Python
  ORT; subset size limit must include its final manifest. Add tampering tests.
- These tooling gaps do not invalidate the independently reproduced actual
  tensors, but must be resolved before M1 is accepted as reproducible tooling.

### Next

- Sol is implementing the targeted review fixes. M2 marginal graph and Rust
  loader preparation has started against the verified tensors; M3 pure Rust
  pool preparation is separate. Gates/commits remain ordered after review.
- A parent preflight demonstrated scalar f32 compatibility reduction differs
  from PyTorch SIMD by at most2.861023e-6 on the observed selected candidates.
  M3 must measure actual Rust behavior and prove exact discrete pool identities;
  **no tolerance change or discrete-order exception is approved yet**.

## M1 — accepted after review fixes

- Hardened verifier now checks unique complete case coverage, actual array/stage
  presence, routing equality, metadata and source provenance. Eight model-free
  tampering/provenance tests pass. Validator explicitly limits its claim to
  structural integrity; independent full oracle reproduction is separate.
- Generator validates source-file hashes and installed Git revision; seed
  propagation and record-stage metadata fixed. Source hashes, ort crate version
  and complete subset size are recorded. All shared exporter/reference loads
  now require the pinned upstream installation.
- Parent reran the hardened verifier/unit tests/lock check and source identity
  checks. Repairs changed only JSON provenance descriptions, not the independently
  reproduced30 NPZ files or formatted output values.
- Both graphs independently pass recursive finite-value/FP32/opset17 audit and
  onnx.checker. Parent numerical results and coverage are recorded in
  `docs/evidence/m1.json`. No tolerance relaxation was needed.
- `scripts/parity/README.md` now gives fresh-checkpoint download, pinned setup,
  generation, structural validation, actual reproduction and ONNX validation
  commands. Runtime Python dependency remains absent.
- Next: review M2 actual marginal graph/loader parity, then M3 candidate pool.
  Work on those is in progress and will be committed separately only after gates.
  All later feature/bundle/publication requirements remain open.

## M3 — independent source review and frozen-stage verification underway

- Sol ported literal shared-pool stable sorts/quota priorities/dedup/padding and
  reproduced the pinned PyTorch2.8 AArch64 four-lane cascade reduction, instead
  of accepting the earlier scalar-fold error. Parent inspected the actual code.
- Parent ran all24 applicable real fixtures and nine synthetic upstream vectors
  in both debug and optimized builds: indices/order/masks, compatibility and
  proposal logits all bit-exact. **No tolerance relaxation needed.**
- Synthetic generator independently reproduced the committed32,444-byte JSON
  vector file. Parent removed its developer-specific upstream checkout default;
  regeneration now uses the same locked, verified installed upstream as M1.
- Parent added rejection/tests for arithmetic overflow from finite input values
  and clarified diagnostics. Final rerun awaits M2's in-progress Rust wrapper:
  a concurrent mid-edit build hit its PostProcessor error-type mismatch, sent
  to the owning agent rather than modifying its file concurrently.
- M3 is not yet accepted/committed: M2 must finish first. Reference-platform/
  reduction decisions and the observed Unicode splitter difference are in the plan.

## M2/M3 — resumed parent review

- Added a Rust pre-inference L=0 guard: ORT1.20.1 was observed to segfault on
  that low-level graph input. Empty public text will normalize to `"."` upstream.
  Removed the incorrect graph-level4096 limit: original words are capped before
  choice prefixes, so a larger graph input is legal. Three model-free regression
  tests cover L=0 rejection, L=4097 validation and classification-only bypass.
- Replaced M3 test's source-file inclusion with the public boundary module.
  Added actual ORT-marginals-to-Rust-pool integration across all24 applicable
  full fixtures. **All indices/order/masks match exactly in debug and release**;
  compatibility/proposal floats pass the original1e-4/1e-3 numerical gate.
  Frozen-input bit-exact and nine synthetic cases still pass, including the
  newly added finite-input overflow rejection tests.
- Parent fmt and strict Clippy pass; the committed marginal subset also passes
  with the actual graph. Logs: `/tmp/gliner25-work/m3-ort-pool-parent.log`,
  `m3-ort-pool-parent-release.log`, `m2-m3-parent-clippy.log`,
  `m2-parent-subset.log` in the same directory.
- M2 remains blocked on centered inside-prefix drift for1000/2000/3000-word
  cases (maximum absolute errors approximately0.00109/0.00164/0.00314).
  No tolerance relaxation has been approved. A read-only numerical investigation
  is measuring restored downstream interval evidence and scorer sensitivity;
  all other marginal outputs already pass the original tolerances.
- No milestone acceptance, release tag, or completion is claimed by these checks.

## M2 adjudication / M4 preparation

- Parent independently reran the completed numerical diagnostic after reading
  its complete script. Across24 fixtures/5,116 valid candidate-query pairs,
  unchanged upstream scorer reproduces saved pair logits bit-exactly; substituting
  all ORT marginals changes pair logits by at most6.103515625e-5 and confidences
  by8.642673492431641e-7. No top-1 or0.5-threshold changes. Parent log:
  `/tmp/gliner25-work/m2-parent-drift.log`.
- Mean restoration and sqrt(length) normalization explain the smaller downstream
  effect. Rebuilding mean/cumsum alone does not fix strict raw-prefix parity.
  Astra approved only coordinate-aware prefix atol `1e-4 + 1.1e-6*i`, retaining
  rtol1e-3 and all other original gates. Rationale and limits are in the plan.
  Targeted implementation is adding repeatable sensitivity gates, strict-prefix
  diagnostic mode, full Rust head-corpus coverage and unit tests. M2 still awaits
  those gates and review; this is not an acceptance entry.
- Separate Sol preparations produced the M4 shared-scorer export/validator and
  pure Rust decoder/oracle vectors. Parent reviewed scorer wrapper and complete
  validator, then independently reran all24 frozen cases: pair maximum error
  9.3460083e-5, candidate states9.536743e-7, null1.4305115e-6,
  count4.172325e-7; original tolerances pass. Dynamic/B2 tests pass. Raw Q0
  Python probe works; C0 crashes and requires caller bypass. Logs/report:
  `/tmp/gliner25-work/m4-parent-scorer.{log,json}`.
- Parent reviewed complete pure decoder against the actual upstream overlap and
  entity decode source, independently passed its debug/release tests, and
  regenerated its9,729-byte oracle vector file byte-identically. Typed Rust
  scorer/integrated-head tests and boundary-only tokenizer preparation are
  delegated next. Preparations remain outside M2/M3 commit scope. No boundary
  high-level pipeline is usable yet.
- Background agent result delivery was not available for several completed IDs;
  parent recovered their final text from local output artifacts and inspected
  actual files. Do not infer active work from a stale agent ID.

## M2 — accepted after independent numerical and safety gates

- Parent reviewed the hardened comparison/sensitivity validators and independently
  reran them. All24 head/6 bypass fixtures plus dynamic/B2 cases pass the approved
  prefix-only gate;199 original raw-prefix failures remain explicitly reported.
  The separate sensitivity validator enforces original learned-score tolerances,
  confidence bounds, exact saved baseline and unchanged selection decisions.
- Six Python comparison regressions pass. Rust full marginal test passes24 head
  and6 bypass cases. L0 rejects before native inference; the isolated Python
  probe records the unsupported raw graph crash instead of claiming L0 support.
- Parent fmt and strict Clippy pass. Review workspace suite (including pending
  M3/M4 preparations):77 tests pass without models (20 explicit skips) and77
  pass with strict v2+boundary artifacts and full fixtures (**zero skips**).
  The combined artifact root initially lacked its fixture symlink; that run
  correctly failed, was repaired, and the complete strict run then passed.
- Evidence and graph hash: `docs/evidence/m2.json`. M2 commit scope excludes
  the pool and all M4 preparations. Next accept/commit M3 separately, then
  integrate M4 high-level entities/classification and end-to-end parity.
  No usable2.5 pipeline or release tag is claimed yet.

## M3 — accepted after frozen and actual-ORT integration gates

- Parent final debug/release reruns pass all6 pool tests, including24 real
  frozen cases with bit-exact floats/indices/order/masks and nine synthetic
  upstream cases. Actual ORT marginal outputs preserve exact candidate selection
  on all24 cases; floating proposal/compatibility values pass original numerical
  tolerances. No tie, order or frozen-compatibility exception was needed.
- Parent regenerated the32,444-byte synthetic vectors byte-identically using
  the pinned upstream. Source review confirms stable endpoint selection,
  start-major pairing, literal quota reservation/dedup and zero padding.
- Same unchanged workspace fmt/strict Clippy/no-model and strict all-model gates
  pass; final pool tests rerun after M2 acceptance. Evidence:
  `docs/evidence/m3.json`. Commit includes only pool/module/vector/test changes
  and this acceptance record; M4 code remains separately uncommitted.
- Next: integrate reviewed scorer/decoder with boundary preprocessing and
  source-faithful classification selection, replace the high-level scaffold,
  and prove formatted entity/classification parity. M4–M7 and explicit sparse
  scoring remain open; do not describe this branch as a finished2.5 release.

## M4 — integrated pipeline under parent review

- Sol implemented entity/classification AutoPipeline dispatch with one mixed-task
  encoder pass, Q0 bypass, adapters and explicit errors for pending JSON/relations.
  Parent read complete pipeline/config and helper implementations. Boundary-only
  classification preserves schema order and first-label ties; v2 stays unchanged.
- Parent verified scorer/classifier/preprocessor/decoder tests after replacing
  temporary source-file test inclusion with actual public-module imports.
  Added a post-activation finite check for temperature-scaling overflow.
  Regenerated token vectors byte-identically (9,664bytes); minimal examples run.
- Review found two test coverage gaps: missing classification results could pass,
  and full-corpus coverage only required >=2 cases. Targeted fixes require exact
  21-case coverage, all expected task keys, true mixed-task oracle fixtures,
  stronger adapter restore checks, and config underflow/graph-flag validation.
- Original full tutorials1(classification)/2(NER), not just the new small examples,
  must run against2.5. A separate change is adding architecture-aware example
  loading while preserving their v2 bodies/results and baseline comparisons.
- Independent M5 record and explicit sparse scorer exports are preparation only,
  outside the M4 scope. No M4/M5 acceptance or completion is claimed yet.

## M4 — accepted after hardened end-to-end and tutorial gates

- Parent reviewed the full pipeline/config/scorer/decoder/preprocessor and repaired
  coverage/robustness gaps. Tests now enforce all21 eligible corpus IDs plus two
  separately generated mixed-task cases, and cannot pass by dropping classifiers.
  Parent independently regenerated the mixed fixture byte-identically.
- Parent actual encoder→marginals→Rust pool→scorer→formatted output comparisons
  pass with exact labels/text/order/UTF8 offsets. Across49 confidence comparisons,
  maximum absolute error is5.7816505e-6 (unchanged1e-3 gate).
- Parent recursively audited the shared scorer graph:30 fp32 finite initializers,
  12 fp32 finite constants, checker passes. All24 learned-scorer frozen cases pass
  original tolerances; public Rust helper tests and complete head-chain tests pass.
- The original full classification/NER tutorials now use an architecture-aware
  loader without changing their bodies. Parent ran both against2.5 and v2.
  v2 result payloads remain byte-identical to baseline, excluding only model-load
  and inference-timing lines; hashes are in `docs/evidence/m4.json`.
- Parent final fmt/strict all-target/all-feature Clippy pass. Review workspace:
  87 tests pass without models (24 explicit skips),87 pass with strict v2+boundary
  artifacts/full fixtures (zero skips). These counts include three uncommitted
  M5 assignment preparation tests, which are excluded from the M4 commit.
- Record metadata will use additive boundary sidecar APIs to preserve existing
  public schema struct literals. M5 record/explicit graphs and assignment helper
  remain preparations; JSON/records/relations deliberately error until integrated.
  M5–M7, public explicit scoring, model publication and release remain open.

## M5 preparation — low-level components verified, high-level gate still open

- Parent independently exported both new graphs to a separate directory and
  reproduced their hashes exactly. The record graph passes four real and four
  synthetic original `forward_group` comparisons; the separate explicit sparse
  scorer passes24 full-head cases, choice captures and dynamic/invalid cases.
  Original tolerances remain unchanged; no shared-scorer substitution was used.
- Assignment/record/explicit runtime vectors and the17-case record decoder
  fixture regenerate byte-identically. Typed helpers now compile through actual
  library modules rather than private source includes. Parent debug/release
  assignment+record tests pass; record decoder unit/integration tests pass8/8
  in each profile after correcting heap-fallback ordering and rejecting
  temperature-scaling overflow. LLVM and GLiNER2 source notices are included.
- Encoding/marginal/shared-candidate stages were factored out without adding new
  high-level behavior. Parent old-runtime workspace baseline passes102 tests
  with zero skips and reproduces all six original v2 tutorial result payloads.
- Evidence: `docs/evidence/m5-preparation.json`. This is not M5 acceptance:
  JSON/record/choice formatting orchestration, public explicit scoring,
  structure end-to-end parity and tutorial3 remain open.

## User-approved direct ORT migration — implemented, acceptance issue open

- User explicitly approved removing ORP and using the latest published ORT.
  Dependency pins now resolve to ORT/ort-sys2.0.0-rc.13; native build information
  confirms ONNX Runtime1.28.0 (`rel-1.28.0`, commit `da9b5e3`). ORP/composable
  are no longer active dependencies. Rust1.88 is the declared dependency floor;
  local tests use1.95 (not an independent MSRV validation).
- A private per-session mutex preserves public `infer(&self)` methods. Session
  settings retain CPU/four intra-op threads/Level3 optimization and schema-name
  checks. The raw-tensor bridge preserves public ndarray0.16 types and logical
  row-major order for nonstandard arrays; owned outputs stay safe after unlock.
- Parent strict all-model/full-fixture suite passes110 tests with zero skips;
  no-model suite passes110 with29 explicit skip messages. Formatting, strict
  all-target/all-feature Clippy, actual native-version checks and shared-session
  concurrent inference tests pass. Full boundary tutorials1/2 also run.
- Five v2 tutorials retain byte-identical payloads. Tutorial1 has exactly one
  changed confidence: technology0.8674003481864929 →0.8674008846282959
  (absolute5.364418029785156e-7, nine f32 ULPs). Labels, ordering and all other
  displayed values match. This still fails the strict byte invariant; no
  rounding or tolerance change was applied. Cross-feed diagnosis now localizes
  the change to native encoder execution: identical inputs, deterministic repeated
  runs, identical classifier output for identical old/new label embeddings in
  both runtimes. Encoder maximum absolute drift is1.9073486328125e-5.
  Optimization Disable/Level1/Level2/Level3, deterministic compute and disabling
  prepacking did not restore old encoder bytes; no production setting was changed.
- The proposed exception is absolute v2 confidence drift <=1e-6, retaining exact
  labels/text/spans/order and all other gates. This requires explicit user
  approval; it has not been adopted. The structured confirmation tool failed
  (`Cannot read properties of undefined (reading 'cancelled')`), so the next
  required input is the user's confidence-policy decision.
- Evidence: `docs/evidence/ort-migration.json`. Migration acceptance, commit/tag
  and the overall release goal are not claimed complete.

## Direct ORT migration — confidence exception approved

- User explicitly replied **Approved** to the <=1e-6 absolute v2 confidence
  exception. Labels, text, spans, ordering, structure and non-confidence values
  remain exact; no output rounding/bias or other tolerance changes are allowed.
- Parent reviewed the completed stage diagnostic: old/new inputs are identical;
  encoder hidden states differ deterministically, and both classifiers are
  bit-identical for either set of frozen embeddings. The exact divergent native
  operator is not claimed. Tested session knobs do not restore old bytes.
- An executable fail-closed regression comparator and frozen original tutorial
  payloads are being added before final acceptance/commit. All broader M5–M7
  and release requirements remain in force.

## Direct ORT migration — accepted after approved regression gate

- The fail-closed comparator now pins all six original v2 payload hashes and
  permits drift only in34 recognized confidence slots. Non-confidence bytes,
  case IDs, labels, strings, coordinates, ordering and even line endings stay
  exact. NaN/Inf, tolerance increases, reference tampering and missing/extra
  cases fail. No runtime output is rewritten. Parent ran15 unit tests; CI now
  includes those model-free checks.
- Parent exercised the comparator's real `--run` path, rebuilding/running all
  six original tutorials. It passes the user-approved1e-6 gate with exactly one
  changed value (5.364418029785156e-7); the other five payloads remain byte-exact.
  Frozen reference text plus manifest totals100,006bytes.
- To separate this commit from M5 preparations, parent archived HEAD and overlaid
  only the migration source files, with a distinct Cargo target directory.
  This isolated migration scope passes91 strict-model tests with zero skips,
  91 no-model tests (25 explicit skip messages), formatting and strict Clippy.
  A missing full-fixture symlink correctly failed the initial strict run; after
  repairing artifact setup the entire suite passed. Working-tree superset
  checks additionally pass52 optimized head/pipeline/helper tests and the smoke
  example. Public ndarray0.16 APIs and per-session synchronization are preserved.
- Evidence: `docs/evidence/ort-migration.json`. This accepts the runtime change,
  not M5–M7, the public explicit-span API, bundle publication or a release tag.

## M5 continuation — typed metadata prepared

- The additive `RecordMetadata` sidecar selects natural/latent/anchorless modes
  without changing existing public schema struct fields. Compilation routes
  Content queries by schema/field identity, including noncontiguous global query
  IDs; rejects unknown/ambiguous metadata and invalid anchors; preserves legacy
  behavior for unannotated structures. Defaults and explicit cardinality/exclusive
  overrides follow upstream.
- Parent read the complete implementation and verified12 unit tests in debug and
  release, plus library Clippy. A fresh debug rerun again passes12/12. This is
  preparation only: metadata is not yet wired to public extraction methods.
- Ragged candidate/seed preparation is now implemented and parent-verified with9
  unit tests: uncapped natural anchors, field-major duplicate latent seeds and
  anchorless context, empty-native bypasses, and ragged assignment trimming.
  Runtime configuration adds record temperature and rejects incompatible baked
  record dimensions; parent5 config tests pass.
- Legacy JSON now shares one encoder invocation with entities/classification,
  emits a single instance, and scores choice prefixes through the separate
  explicit graph. Parent inspected the changes and ran11 strict model tests
  (4 legacy JSON plus7 existing pipeline tests), zero skips; original legacy
  structure output and M4 fixtures pass. All-target/all-feature Clippy passes.
- Unicode choice tables reproduce byte-identically. Parent independently compared
  Rust and Python output for all1,112,064 Unicode scalar values (lowercase,
  casefold, IGNORECASE representatives and word membership):23,222,132 bytes
  exact. Separately,600 deterministic literal-ownership cases exactly match the
  untouched upstream static method after converting offsets to UTF8 bytes.
  Parent corrected nearest-choice ranking to count Python codepoints rather
  than byte gaps;14 helper tests pass, including a multilingual counterexample.
- These are local, uncommitted preparations, not M5 acceptance. Full record-mode
  orchestration/formatting, all four annotated structure goldens, and the public
  explicit primitive still need their final gates before the release work.

## Direct ORT follow-up — project MSRV verified and corrected

- An isolated build of accepted commit a3ecbfa exposed an incorrect Rust1.88
  project-floor declaration. ORT itself permits1.88, but locked hf-hub/Xet
  dependencies reject1.88;1.89 then fails on Xet's use of
  `str::floor_char_boundary`, stabilized in1.91. No dependencies were downgraded.
- Rust1.91 now passes91 strict model tests with zero skips and91 no-model tests
  (25 explicit artifact skips), plus all six executable v2 tutorial regressions.
  The sole approved confidence delta remains5.364418029785156e-7. The isolated
  tutorial runner first rejected missing config/tokenizer links; after supplying
  the pinned files it passed in full. The corrected manifest also builds all
  tests on1.91; isolated formatting and1.91 all-target/all-feature Clippy with
  warnings denied also pass.
- Cargo/README now declare1.91. Evidence: `docs/evidence/msrv-verification.json`.
  Verification excludes uncommitted M5 work and is local macOS arm64 evidence;
  this follow-up is not a completed GLiNER2.5 release or new tag.

## M5 — records and JSON extraction accepted

- Existing JSON methods now perform boundary legacy single-instance extraction;
  additive typed-sidecar methods support natural, latent and anchorless records
  on BoundaryPipeline and AutoPipeline. Unannotated schemas keep legacy behavior;
  span architecture rejects record metadata explicitly. Existing schema struct
  fields and v2 methods are unchanged.
- Real ragged head orchestration, assignment/exclusivity/cardinality decoding,
  field validators, calibrated candidate/assignment confidence, source-order
  literal choice ownership and fallback formatting are wired into the single
  joint encoder path. Separate explicit weights are used for choice scoring.
  Source notices accompany the GLiNER2, libc++ and Unicode adaptations.
- Parent corrected codepoint-versus-byte proximity and synthetic-suffix mapping,
  and removed a spurious unsupported-path test skip by using the existing split
  v2 tokenizer/ONNX fixture layout. No numerical tolerance was loosened.
- All5 frozen structure cases pass exact shape/text/order/UTF8-coordinate parity.
  Record maximum confidence error is3.6805868e-6 (original allowed1e-3), identical
  in debug/release and on declared Rust1.91. Parent suites:162 strict model tests
  with zero skips on1.95 and1.91;162 no-model tests with36 artifact skip messages;
  75 optimized helper/head/pipeline tests with zero skips; fmt and all-target,
  all-feature Clippy pass. Full tutorial3 runs on2.5 in release mode.
- Fresh six-v2-tutorial comparator passes, with only the previously approved
  5.364418e-7 confidence change;15 comparator unit tests pass. Committed/intended
  fixture total is1,980,912bytes, below the2MB limit. Full fixtures and graphs
  remain ignored.
- Evidence: `docs/evidence/m5.json`, backed by preparation/export and Unicode
  verification records. Public explicit-span scoring, M6 relations, M7 bundles,
  publication/readback, external consumer proof and a release tag remain open.

## Public explicit-span API — accepted

- Added ordered label/span scoring on BoundaryPipeline and AutoPipeline using
  the separate sparse explicit scorer, plus typed byte-offset results and a
  multiline Unicode example. Span models reject the API explicitly.
- Parent reviewed preprocessing/preflight, duplicate/query routing, single-pass
  encoding/marginals and independent upstream oracle generation. Invalid UTF-8,
  partial words and truncated spans fail before encoder work; empty axes bypass
  native heads. No proposal, threshold, abstention or overlap filtering is used.
- Parent regenerated the 3,593-byte fixture with the pinned untouched upstream
  model: SHA256 `1462e45ad80346ef73a4c4569cd166ef7d6d9c8c950a4b23dd994835c8904f74`,
  identical to the submitted artifact. The indexed fixture total is1,984,505bytes.
- Parent targeted verification passed65 library tests,3 explicit pipeline tests
  with actual v2/boundary graphs, and1 example unit test, zero skips. Formatting
  and staged whitespace checks passed. The isolated accepted-scope snapshot
  passed178 strict tests with zero skips,178 no-model tests with38 artifact skip
  messages, all-target/all-feature Clippy, the release example, Rust1.91 all-target
  check and20 selected explicit tests. Six v2 tutorials pass with only the same
  approved5.364418e-7 confidence difference.
- Parent verified all eight staged code/test/fixture overlays match the tested
  snapshot byte-for-byte and independently checked the command logs and counts.
  Evidence: `docs/evidence/public-explicit-spans.json`. M6 remains under
  implementation/review and M7 remains open; this is not a release or new tag.

## M6 — relations accepted

- Implemented typed proposals, learned relation graph/runtime, source-faithful
  mention/edge deduplication and all existing relation APIs in the joint prompt.
  Query routing is head/tail-specific; relation scores are calibrated separately,
  and original UTF8 bytes are retained while canonical/nearest metrics count
  Python codepoints. Tutorial6 now loads either architecture.
- Review found a real discrete-order bug in the initial proposal sigmoid:
  algebraically stable negative evaluation changed a source f32 tie. The local
  correction reproduces source scalar algebra and pinned SLEEF vector arithmetic
  with layout-aware block/tail handling. No rounding bias, global decoder change
  or relaxed tie tolerance was used. Parent independently compared actual-source
  arithmetic on12,033,560 vector and2,000,001 scalar values with zero bit failures;
  source constants/order/license were checked against the immutable SLEEF commit.
- Parent regenerated relation graph and oracle vectors independently. Default
  graph hash is unchanged (`cf6d94f492c74fe3e3ffcfc613beffd3f02cd9b0099594d3e4601b74d20bc40f`).
  An exporter assertion incorrectly assumed a masked last row even for a P=1
  trace; parent restricted the assertion to masked rows and validated the P=1
  re-export across the full dynamic/real suite. Original tolerances pass.
- Parent corrected auxiliary fixture placement after the M4 corpus enumerator
  rightly rejected non-corpus JSON files at its root; generated relation vectors
  now live in ignored subdirectories, without weakening corpus validation.
- Final parent gates:217 strict tests with zero skips on both Rust1.95 and1.91,
  217 no-model tests with47 artifact skips,101 optimized scoped tests, fmt,
  all-target/all-feature Clippy, and actual boundary tutorial6 pass. Four relation
  outputs have exact keys/text/order/UTF8 bounds and maximum confidence
  error7.748604e-7, identical in debug/release/MSRV. The fresh six-v2-tutorial gate
  retains the same sole5.364418e-7 confidence drift.
- Two additional independently regenerated upstream cases prove Unicode relation
  endpoints and all four task families together with a choice prefix/description.
  Every non-confidence value matches exactly; maximum confidence error8.940697e-7.
  The committed real employment oracle runs even without full fixture opt-in;
  strict full mode still requires all four relation-only IDs. Config Cartesian
  overflow is rejected before loading models. Missing strict decode data fails.
- The committed development sigmoid harness extracts actual Rust arithmetic;
  its14,037,113 comparisons pass, with traversal-mirror limitations clearly stated
  and actual public-API layout tests separate. Parent reviewed the real source,
  notices, tests, independently regenerated artifacts and measured all gates.
  Evidence: `docs/evidence/m6.json`. M7 bundles/publication/readback/consumer/CI/tag
  remain open; M6 acceptance is not a completed release.

## M7 — integration in progress (not accepted)

- Parallel implementation produced complete-bundle export/validation tooling,
  explicit Rust/Python downloaders, manifest validation, benchmark and downstream
  consumer harnesses, documentation and the MSRV/stable CI matrix.
- Three fresh staged bundles now contain all seven graphs and 14 checked runtime/
  notice files each. Source encoder widths are independently checked: small384,
  base768 and multi768. Every manifest remains `exported-unvalidated` with
  `release_ready: false`; export success is not numerical acceptance.
- Parent integration review found and corrected mismatched graph-signature
  schemas, venv subprocess symlink resolution, and oracle metadata confusing
  upstream-added punctuation with the caller's original text. A model-free
  regression covers empty, Unicode and punctuation-normalized inputs. Original
  M0–M6 fixtures and tolerance rules are unchanged.
- Review also found missing synthetic-export CI dependencies, incomplete benchmark
  evidence and an unbound downstream HF-revision claim. These are integration
  work, not waived release gates. The model-free CI now exposes artifact skips.
- Hugging Face authentication is available through the user-provided shell
  environment; no secret is stored in this repository. Write permission and
  publication/readback are not yet proven. Independent per-checkpoint source/
  ONNX/native checks, final benchmark, clean downloads, external consumer,
  remote CI and a matching release tag remain required.
- Independent small validation exposed an actual public-output discrepancy:
  upstream retains requested relation keys with empty lists, while Rust omitted
  them after all candidates were rejected. Corrected boundary formatting (not
  numeric decoding or the comparator) and added suppressed/empty-input coverage.
  Fresh small and base aggregate reports now pass all seven Python/ONNX stages
  plus native exact-output checks on30 corpus+mixed+explicit cases each.
- Multi generation passed all32 source cases; validation is blocked at the
  long3000 centered prefix: endpoint abs error0.0071811676 exceeds the existing
  gate0.00340269073. No tolerance change is accepted. Source-ordered fp32 mean
  reduction is being investigated; small/base success is not relabeled for multi.
- User reported repeated macOS Python crash dialogs. Diagnostic reports and
  validation logs identify isolated unsupported-axis ORT1.20.1 probes (marginals
  L=0, scorer C=0), not ordinary supported-input inference. Those probes are being
  made explicitly opt-in; public Rust guards and supported-axis parity remain
  mandatory. Historical actual crash observations are retained, not hidden.
