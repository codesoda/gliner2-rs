# M7 working bundle contract

This is the coordination contract for M7 implementation, not release acceptance.
Follow `PLAN-gliner2.5.md`; M0–M6 and explicit scoring are accepted at34321d1.
All lanes have disjoint file ownership. No lane commits, uploads, tags, modifies
upstream/oracle environments, weakens tolerances, or claims overall completion.

## Bundle and manifest

Bundle names: `gliner2.5-{small,base,multi}-v1`. Each contains config.json,
tokenizer.json, tokenizer_config.json, encoder_config/config.json, original model
card as SOURCE_MODEL_CARD.md, LICENSE (Apache2 text), NOTICE (original source/model
identities and explicit fp32/ONNX conversion notice), and seven graphs:

- encoder.onnx
- classifier.onnx
- boundary_marginals.onnx
- boundary_scorer.onnx
- boundary_explicit_scorer.onnx
- boundary_records.onnx
- boundary_relations.onnx

`export_manifest.json` is JSON with:

- manifest_version:1
- architecture:"boundary", architecture_version:1
- status:"exported-unvalidated" or "validated"
- release_ready:false until parent accepts actual per-checkpoint validation
- hf_model, hf_revision, gliner2_commit (immutable original source identities)
- opset:17, precision:"fp32"
- ort_crate_version:"2.0.0-rc.13", native_onnx_runtime:"1.28.0"
- validation_onnxruntime:"1.20.1" (Python reference environment, not native Rust)
- dependencies (actual pinned export versions), source_file_sha256
- files: map of safe bundle-relative file names to {bytes:positive integer,
  sha256:64 lowercase hex}. Include every runtime/notice file above, but not the
  manifest itself. Optional additional metadata must be explicitly listed.
- graphs: map of the exact seven graph names to actual input/output names,
  dtypes and shapes/signatures; graph file checksums are in files.
- validation: absent/null until measured evidence is available; then reference
  actual source/ONNX/native reports with hashes and the exact model/graph hashes.

Exporters produce unvalidated manifests. Do not manufacture validation status
from export success or base-model results for another checkpoint. Parent will
adjudicate promotion after reviewing reports. Rust/Python public downloaders
accept only validated/release_ready boundary bundles; validation development can
load graphs directly via the existing pipeline before promotion.

Downloaders reject unsupported architecture/version, partial state, missing
required files, invalid checksums/sizes and unsafe manifest paths (absolute,
traversal, Windows separators/prefixes). Ensure resolved files remain inside the
bundle directory. Verify files with streaming SHA256. Model acquisition remains
explicit, not an automatic build or inference operation. Do not add Python to
Rust build/runtime.

Preserve selectors base/large for v2; add 2.5-small/2.5-base/2.5-multi (also full
bundle names if convenient). `all` explicitly means all five; explain increased
download scope. Support destination override and immutable HF repository revision
selection. v2 bundles currently lack tokenizer/config: retrieve original Fastino
metadata at verified immutable revisions, collocate it with ONNX files, and keep
legacy split-layout loading as fallback. Do not apply boundary manifests to v2.

## Pins and proof

Upstream GLiNER2 d7c727458bf6929bc9ef5ee04e13c3f717a7c455.
Small f1e4d8fdd6fe328f45dee6aca3e6a07c9db4296e; base
78cea040597df251eedefa9d7ee2a756af39fe64; multi
235cf92d6d4318da9bfca0d08975c8fa7250d13b.

Export lane provides `bundle_profiles.PROFILES` keyed by small/base/multi, with
fields model_id, revision, bundle_name, source_sha256; and
`verify_source(profile_name, model_dir) -> dict[str,str]` returning verified
source-file hashes. Other lanes may code against this fixed interface before the
file appears. Scripts expose --model small|base|multi|all (where applicable),
--model-dir (single checkpoint), --bundle-dir/--out-dir, --fixture-dir and
--report-json as relevant; document exact CLI in their completion reports.

Source snapshots are cached locally. Base source hashes are in export/common.py;
small/multi verified source hashes and snapshot locations are recorded at
/tmp/gliner25-work/m7-source-downloads.json. Prove source identity from content,
not a directory name. Preserve existing base-only golden provenance checks.
A separate per-bundle oracle generator can reuse unchanged hook/schema helpers,
but must record its actual verified model identity, never label small/multi as base.

## Parallel lanes

1. Export: NEW scripts/export/bundle_profiles.py, export_bundle_2_5.py,
   export_all_2_5.sh, test_bundle_export.py; ignored output bundles only.
2. Validation: NEW scripts/parity/gen_bundle_validation_vectors.py,
   scripts/export/validate_bundle_2_5.py, tests/bundle_parity.rs and optional new
   validation helper modules under distinct bundle_validation names.
3. Rust download/validation: NEW src/bundle.rs, tests/bundle.rs; edit src/lib.rs,
   Cargo.toml/Cargo.lock and examples/download_models.rs. Sole dependency owner.
4. Python download/metadata: scripts/download_models.py, NEW
   scripts/test_download_models.py, examples/common/mod.rs. Share verified v2
   metadata pins by a small NEW docs/checkpoints/v2-metadata-pins.json consumed
   by the Rust lane; don't edit Rust downloader yourself.
5. Benchmark: NEW examples/benchmark_gliner2.rs, scripts/benchmark_gliner2.sh,
   docs/BENCHMARK-gliner2.5.md. Defer authoritative timing until heavy lanes idle.
6. CI: .github/workflows/ci.yml only. Required fmt/Clippy/locked tests, Rust1.91
   and stable, branch/PR/manual triggers. No tags or workflow dispatch yet.
7. Documentation: README.md, docs/gliner2-vs-gliner2.5.md,
   docs/RESULTS-gliner2.5.md. No invented benchmark/remote publication claims.
8. Consumer: NEW scripts/verify_external_consumer.sh and optional NEW
   scripts/consumer-smoke/ template files; no root Cargo or downloader edits.

Parent owns PLAN/PROGRESS/AUDIT/evidence, manifest promotion, final version,
commits/pushes/publication, actual CI and release gates. Never cargo fmt all shared
files from a lane; format only owned files. Full integration tests run after the
implementation barrier. Record failures honestly instead of changing other lanes.

## Resource and acceptance policy

Machine has18GB RAM. Parallelize implementation and lightweight checks; export
and model-reference validation sequentially, and benchmark alone. Full weights,
large vectors and reports stay ignored/outside git. Committed fixtures remain
below2,000,000bytes (currently1,984,505).

Base seven local graphs already passed M0–M6. Preserve their current copies while
exporting to a separate staging root. Each checkpoint must pass independent
source/ONNX/native validation (including dynamic shapes and Unicode); do not
reuse base weights, tokenizer, dimensions or reference outputs for small/multi.
No tolerance changes: original stage1e-4+1e-3*abs(ref), documented M2 prefix rule,
exact discrete outputs, final confidence1e-3, v2 confidence-only drift<=1e-6.

Final proof: all three validated bundles, published HF revision and clean readback,
working Rust/Python downloaders, real CPU50/500/3000-word benchmark, remote CI,
pushed-source external Rust consumer without Python, then matching version/tag.
Missing local HF write credentials are a publication blocker, not permission to
claim completion. Never print/request tokens in chat.
