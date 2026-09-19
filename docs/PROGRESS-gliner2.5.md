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
- HF publication credentials have not been checked. Ask for input if absent
  before the M7 upload; GitHub authentication is available.
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
