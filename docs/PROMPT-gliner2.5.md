# Agent prompt — add GLiNER2.5 (boundary architecture) support to `gliner2-rs`

Copy everything below the line into a fresh agent session started at the repo
root of `gliner2-rs`.

---

You are extending **`gliner2-rs`**, a Rust crate (ONNX Runtime via `ort`/`orp`)
that runs Fastino's **GLiNER2** span-architecture models (`gliner2-base-v1`,
`gliner2-large-v1`) for entity extraction, JSON/structure extraction,
relations and classification. Your job is to add support for **GLiNER2.5**
(`fastino/gliner2.5-{small,base,multi}-v1`), which keeps the same DeBERTa-v3
encoder, prompt layout and classifier MLP but replaces the extraction head with
a new **boundary architecture** (`config.json: architecture = "boundary"`),
while keeping every existing v2 code path and test working.

Working directory: the crate root (`Cargo.toml`, `src/`, `tests/`, `examples/`,
`scripts/export/`, `docs/`). Models live outside git: `models/<name>/`
(HF checkpoint) and `onnx/<name>/` (exports); tests skip when artifacts are
missing (see `tests/entities_e2e.rs`) — keep that convention. Note
`model_root()` in tests uses the *parent* of `CARGO_MANIFEST_DIR` while the
README says `./onnx`; check which is true on disk and make them consistent
early.

## Read first, in this order
1. `docs/gliner2-vs-gliner2.5.md` — the requirements doc: what is shared, what
   changed (full boundary-head pipeline), config values, file-by-file mapping
   onto this crate, the recommended two-graph ONNX split, phased plan, risks.
2. `src/pipeline.rs`, `src/extractor.rs`, `src/decode.rs`, `src/spans.rs`,
   `src/embeddings.rs`, `src/text.rs`, `src/schema.rs`, `src/classification.rs`,
   `src/adapters.rs` — the current v2 implementation you are generalising.
3. `scripts/export/export_encoder.py`, `export_extractor_padded.py`,
   `export_classifier.py` and their `validate_*.py` twins — the export/validate
   pattern to follow.
4. Upstream `fastino-ai/GLiNER2` (clone it; pin the commit you use):
   `gliner2/models/boundary/{model.py,heads.py,encoding.py,pool.py,proposal.py,scoring.py,records.py,relations.py,engine.py}`,
   `gliner2/inference/{candidate_decoder.py,overlap.py,runtime.py,chunking.py}`,
   `gliner2/processor.py` (special tokens ~L281, splitter + `max_len` ~L575),
   `gliner2/processing/word_splitter.py`, `gliner2/training/lora.py`.
5. HF `fastino/gliner2.5-base-v1/config.json` (`architecture "boundary"`,
   `architecture_version 1`, `boundary_dim 128`, `pair_dim 128`,
   `content_dim 64`, `boundary_attention_layers 2` (window 128, 4 heads),
   `candidate_pool "shared"`, `pool_boundary_top_k 32`, `pool_size 192`,
   `min_pool_per_query 8`, abstention/count_head/records/relations enabled,
   `overlap_policy "flat"`, `max_len 4096`, `MASK_LOGIT = -1e4`) and the
   release post <https://fastino.ai/blog/gliner2-5>.
6. `docs/jev-and-gliner.md` and `docs/agentic-use-cases.md` — why 2.5 matters
   (long context, explicit-span scoring, attributes, relations); context only.

## Model roles
- **GPT Astra 6**: planning, design review, adjudicating open questions.
  Produce `docs/PLAN-gliner2.5.md` before code; re-review at every gate,
  especially parity failures (decide "fix Rust" vs "fix export" vs "accept
  tolerance" with a written reason).
- **GPT 5.6 Sol**: implementation (Python export scripts, Rust code, tests,
  benchmarks), milestone by milestone against the plan.
- Run as separate subagents where the harness supports it; hand the
  implementer the plan file, not a summary.

## Non-negotiables
- The v2 path (`Gliner2Pipeline` on `gliner2-base-v1`) must keep passing every
  existing test and example, byte-identical outputs. 2.5 is **additive**.
- Parity with upstream Python is proved **stage by stage** with golden
  fixtures, not by eyeballing end-to-end output.
- ONNX exports use fp32 weights (checkpoints are fp16 — cast at export) and a
  finite mask sentinel (`-1e4`), never `-inf`.
- Pin the upstream `gliner2` commit and model revisions used for goldens and
  record them in the exported bundle (`export_manifest.json`:
  `architecture`, `architecture_version`, `gliner2_commit`, `hf_revision`,
  opset, ort version).
- No new Python at inference time. Python is only for export, validation and
  golden generation (`uv`-managed venv under `scripts/`, never committed).

## Phase A — plan (Astra)
Write `docs/PLAN-gliner2.5.md` containing:
- Confirmed upstream commit + which classes/functions map to which exported
  graph and which Rust module; confirm `candidate_pool == "shared"` for all
  three 2.5 checkpoints and whether `small`/`multi` differ in any config
  value that changes the graph (e.g. `boundary_attention_layers`, dims,
  tokenizer / `CharLevelSplitter` for `multi`).
- Decision on the ONNX split (default: `boundary_marginals.onnx` +
  `boundary_scorer.onnx` with the candidate pool in Rust; alternative:
  single graph with topk/argsort/unique inside — state why not).
- Public API shape: `Extractor` trait (or enum) over the existing public
  methods; `SpanPipeline` (= current `Gliner2Pipeline`, kept as a type alias
  for compatibility), `BoundaryPipeline`, `AutoPipeline::from_dir` dispatching
  on `config.json`.
- Golden fixture design: corpus (~30 texts: short NER, sentiment/intent,
  JSON structures, relations, ≥3 texts of 1–3k words, non-ASCII/CJK,
  empty/edge cases), per-stage tensors to dump, tolerances per stage.
- Resolution for every item in `docs/gliner2-vs-gliner2.5.md` §5 "Risks".
Do not start Phase B until the plan answers all of the above.

## Phase B — implement (Sol), milestone gates
M0 **Dispatch + hygiene** (no new weights): read `config.json.architecture`
   (default `"span"`); introduce the trait + `AutoPipeline`; move
   `Gliner2Pipeline` body to `SpanPipeline`, alias the old name; apply
   `max_len` word truncation in shared preprocessing (v2 default = none unless
   config has it; 2.5 = 4096); add the `[R]` marker + one-query-per-marker
   enumeration to `src/embeddings.rs` behind the trait. All existing tests
   green.
M1 **Python side + goldens**: `scripts/export/env` (uv) with pinned `gliner2`
   and `torch`; `scripts/parity/gen_boundary_goldens.py` runs upstream
   `GLiNER2.from_pretrained("fastino/gliner2.5-base-v1")` on the fixture corpus
   and dumps, per text and task: encoder `last_hidden_state`, `text_states`,
   `query_states`, boundary marginals (`start_logits`, `end_logits`,
   `inside_prefix`, projections), pool `indices/mask/compat`, `pair_logits`,
   `null_logits`, count outputs, and the final formatted result with
   `include_spans=True, include_confidence=True`. Save as `.npz` + JSON under
   `fixtures/gliner2.5-base-v1/` (gitignored, plus a small committed subset
   ≤ 2 MB). Also export/validate `encoder.onnx` and `classifier.onnx` for
   2.5 with the existing scripts (check classifier state-dict key names).
M2 **`boundary_marginals.onnx`**: `export_boundary_marginals.py` wrapping
   `BoundaryEncoder` + `BoundaryQueryHead` (+ pool projections). Dynamic axes
   for L and Q. `validate_boundary_marginals.py` compares against goldens
   (fp32, atol ~1e-4 rel 1e-3; record actual). Rust `src/boundary/marginals.rs`
   loads it; Rust test asserts parity on the committed subset.
M3 **Candidate pool in Rust** (`src/boundary/pool.rs`): max over queries,
   top-32 starts/ends, Cartesian pairing, compat via projections, per-query
   quota 8 with rank bonus, global fill, dedup to 192, **stable tie-breaking
   identical to torch** (`torch.topk`/`argsort(stable=True)` semantics — read
   `pool.py` carefully, mirror the order of operations). Gate: pool
   `indices/mask/compat` exactly equal to goldens on every fixture. If exact
   equality is impossible due to float ties, Astra decides the tolerance rule
   (e.g. set-equality of the top-192 + logit-gap threshold) and documents it.
M4 **`boundary_scorer.onnx` + decode → entities & classification**:
   `export_boundary_scorer.py` wrapping `SharedPoolScorer` with explicit
   `indices [C,2]`, `mask [C]`, `compat [C]`, `null_projection`, `count_head`
   (mirrors upstream `score_explicit_spans`); outputs `pair_logits [Q,C]`,
   `candidate_states [C,H]`, `null_logits [Q]`, count logits.
   `src/boundary/decode.rs`: sigmoid + threshold, abstention, overlap
   policies (`flat`, `disallow`, others in `overlap.py`), half-open boundary
   → char offsets (be explicit about byte vs char; match Python on non-ASCII
   fixtures or document the mapping), stable ordering, `dtype` list/str.
   `BoundaryPipeline::{extract_entities, classify_*, extract}` for entities +
   classification only. Gate: final entity/classification output equals the
   golden formatted result (spans, labels; confidences within 1e-3) on all
   fixtures; `examples/tutorial_1_classification.rs` and `tutorial_2_ner.rs`
   run against `onnx/gliner2.5-base-v1`.
M5 **Records** (JSON structures): export `RecordHead` (`forward_group`),
   port `decode_group` (anchor / natural mode, legacy fallback),
   `extract_json*` on `BoundaryPipeline`. Gate: golden parity on structure
   fixtures; `tutorial_3_json_extraction.rs` runs on 2.5.
M6 **Relations**: port `TypedRelationPairGenerator` (thresholded heads/tails,
   pair cap) and export `SparseRelationScorer` (biaffine).
   `extract_relations*` on `BoundaryPipeline`. Gate: golden parity;
   `tutorial_6_relation_extraction.rs` on 2.5.
M7 **Bundles + docs + CI**: `export_all_2_5.sh` producing
   `onnx/gliner2.5-{small,base,multi}-v1/` with `export_manifest.json`;
   upload to HF `codesoda/gliner2-onnx` (ask before uploading if credentials
   are missing); extend `download_models.{py,rs}` with the new bundles; README
   section "GLiNER2 vs 2.5: which to load"; benchmark table (latency: base v2
   vs 2.5 on 50-word / 500-word / 3000-word inputs, CPU) in
   `docs/RESULTS-gliner2.5.md`; CI keeps skip-when-no-models. Update
   `docs/gliner2-vs-gliner2.5.md` §4/§5 to reflect what shipped.
M8 **Optional library features** (only after M7, each its own gate): span
   attributes (`AttributeGroup` via the scorer graph on explicit spans),
   constrained classification (exact/beam over per-task probabilities, pure
   Rust), `JointIE` beam search, `*_long` chunking with overlap merge. Also
   expose `score_explicit_spans` publicly — it is the primitive the
   compaction/reranking use cases in `docs/agentic-use-cases.md` need.

## Rules
- Every milestone ends with `cargo fmt`, `cargo clippy --all-targets -D
  warnings`, `cargo test` (with and without model artifacts present), an
  entry in `docs/PROGRESS-gliner2.5.md` (done / measured / next / open
  questions) and a git commit. Small commits, descriptive messages.
- Parity failures are investigated to root cause before any tolerance is
  loosened; write the reason down.
- Read the real upstream source and `ort` crate source rather than guessing
  op support; if an op fails to export or run under the pinned `ort`
  (`=2.0.0-rc.9`), prefer moving that step into Rust over upgrading `ort`
  (an `ort` upgrade is a separate, Astra-approved decision).
- Keep `orp`/`ort` session handling consistent with the existing encoder /
  extractor code (same threading and input naming conventions).
- Do not add Python dependencies to the Rust build; do not commit model
  weights or large fixtures.
- Keep `docs/gliner2-vs-gliner2.5.md` §8-style risk list current: add
  anything you discover (e.g. DeBERTa position buckets at 4096 words, fp16
  overflow, `multi` tokenizer differences).

Start with Phase A. When `docs/PLAN-gliner2.5.md` is written, summarise the
key decisions (ONNX split, trait shape, fixture design, risk resolutions) in a
few lines, then proceed to M0 without waiting for confirmation.
