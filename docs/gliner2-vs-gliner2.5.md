# GLiNER2 vs GLiNER2.5 — differences, and what it takes for `gliner2-rs` to support both

_Last reviewed: September 2026, against upstream `fastino-ai/GLiNER2` `main` and
`fastino/gliner2.5-base-v1` `config.json`._

## TL;DR

- **Same encoder, same prompt format, same classifier head; entirely new
  extraction head.** GLiNER2.5 keeps DeBERTa-v3, the `[P] [E] [C] [R] [L] [SEP]`
  schema prompt, the word splitter and the `[L]`-token classification MLP. It
  replaces the fixed-width span grid (`span_rep` + count MLP + CountLSTM) with a
  **boundary** head: per-query start/end/inside logits, a sparse candidate pool,
  and a reranker. No `max_width`, no `[L, W]` score grid, no 20-class count head.
- **There is no GLiNER2.5 paper.** The only arXiv paper is GLiNER2
  (2507.18546). GLiNER2.5 is documented in Fastino's blog post and in the code
  under `gliner2/models/boundary/`. The HF card cites the 2.0 paper.
- **`gliner2-rs` today implements the v2 span head exactly** (`spans.rs`,
  `extractor.rs`, `decode.rs`, `extractor_padded.onnx`). None of that transfers.
  Tokenizer, prompt formatting, encoder export, embedding gather, classifier
  export, schema/JSON/result types and the adapter mechanism all do.
- **Effort:** ~1–2 weeks for v2.5 entities + classification (ONNX export of the
  boundary head is the hard part), ~1–2 more for records + relations, ~4–6 weeks
  for full parity incl. attributes / JointIE / constrained classification /
  chunking. Additive to the v2 path; no rewrite of what exists.

## Sources

| What | Where |
| --- | --- |
| GLiNER2 paper | https://arxiv.org/abs/2507.18546 |
| GLiNER2.5 release post | https://fastino.ai/blog/gliner2-5-span-free-information-extraction |
| Model card | https://huggingface.co/fastino/gliner2.5-base-v1 |
| Checkpoint config | `https://huggingface.co/fastino/gliner2.5-base-v1/raw/main/config.json` |
| Upstream code | `gliner2/models/span/` (v2), `gliner2/models/boundary/` (v2.5), `gliner2/processing/`, `gliner2/inference/` |

## Implementation source review (pinned follow-up)

The implementation plan is [`PLAN-gliner2.5.md`](./PLAN-gliner2.5.md), with
requirement evidence tracked separately in [`AUDIT-gliner2.5.md`](./AUDIT-gliner2.5.md).
At upstream commit `d7c727458bf6929bc9ef5ee04e13c3f717a7c455`, source/header
inspection corrected several assumptions in the original research below:

- Actual released checkpoint tensors are **F32**, despite encoder configs
  declaring float16. Export still explicitly casts/validates fp32.
- Shared-pool endpoint selection uses **stable sort**, not unstable torch.topk.
  Pool output is padded to 192; quotas use reserved rank priorities and final
  tie-breaking by encoded span key. See plan for exact ordering.
- **Explicit-span scoring uses a separate sparse learned scorer**, even on
  shared-pool models. The two shared-inference graphs are insufficient for
  JSON choice fields or public explicit scoring; a third graph is required.
- Legacy JSON uses a single boundary-decoded record, not the v2 count/grid
  path. Natural record mode can have more than 32 anchor instances.
- `multi` defaults to whitespace splitting; character splitting is an explicit
  option. Public upstream inference defaults to no word cap unless supplied;
  the Rust task explicitly requests the config cap (4096 for these models).
- The boundary window attention implementation and encoder still use dense
  attention operations; do not interpret bounded candidate selection as
  linear end-to-end memory/time.
- `[R]` is already included in the crate's legacy embedding gather. What is
  missing is explicit query routing and boundary-specific relation handling.

These corrections override the older descriptions below. They are verified
source findings, **not claims that Rust 2.5 inference has shipped**. M0 is in
progress; all mandatory model/export/parity/publication gates remain tracked.

## 1. What is shared

| Component | GLiNER2 | GLiNER2.5 | Notes |
| --- | --- | --- | --- |
| Encoder | DeBERTa-v3 base / mDeBERTa / xsmall | same family | 2.5 family: small (74M, xsmall), base (194M), multi (287M, mDeBERTa) |
| Token pooling | first sub-token | first sub-token | `token_pooling: "first"` |
| Special tokens | `[P] [E] [C] [L] [SEP]` (+`[R]` in newer 2.x) | `[P] [E] [C] [R] [L] [SEP]` | `processor.py:281-285` |
| Word splitter | `WhitespaceTokenSplitter` regex | identical regex | `src/text.rs` already matches byte-for-byte; 2.5 adds optional `CharLevelSplitter` for CJK |
| Prompt layout | `[P] task ([E] a [E] b) [SEP] text…`, multiple schemas concatenated | same | `format_input_with_mapping` in `src/schema.rs` is reusable |
| Classification head | `create_mlp(H → 2H → 1)` on each `[L]` embedding | identical shape | `classifier.onnx` export reusable; state-dict key names still to be verified |
| Public API shape | `extract_entities / classify_text / extract_json / extract_relations / extract(schema)` | same + `_long` variants, `Classifier`, `JointIE`, `AttributeGroup` | `AutoExtractor.from_pretrained` dispatches on `config.architecture` |
| LoRA | `apply_lora` on encoder | `apply_lora` on encoder | crate's "swap merged `encoder.onnx`" approach still applies |

## 2. What changed: the extraction head

### GLiNER2 (span architecture) — what this crate implements

```
text_emb [L,H] ─┐
                ├─ span_rep(text, spans[L×W]) ──► span_rep [L, W, H]
spans_idx ──────┘
[P] emb ────────── count_pred MLP ──────────► count_logits [20]
[C]/[E] embs ───── count_embed (GRU over 20 count steps + cross-field transformer)
                                            ──► struct_proj [20, F, H]
einsum("lkd,pmd->pmlk") ────────────────────► span_scores [20, F, L, W]
```

Decode: for each (instance, field), sigmoid every cell of the `[L, W]` grid,
threshold, greedy non-overlap. Instances = argmax of `count_logits`.
Relations are a 2-field structure (head, tail). Hard limits: `max_width`
(crate: 8 words), 20 instances, `MAX_FIELDS` baked into the ONNX (crate: 64).

### GLiNER2.5 (boundary architecture)

Relevant `config.json` values for `gliner2.5-base-v1`:

```
boundary_dim 128, pair_dim 128, content_dim 64
boundary_attention_layers 2, heads 4, window 128, refinement_layers 1
candidate_pool "shared", pool_boundary_top_k 32, pool_size 192, min_pool_per_query 8
enable_span_content true, enable_rotary_endpoints true, use_inside_evidence true
enable_abstention true, enable_count_head true
enable_records true (record_dim 128, record_instance_queries 32)
enable_relations true (biaffine, heads/tails per type 32, pair_cap 64)
overlap_policy "flat", max_len 4096
```

Pipeline (`BoundaryHead.forward`, `pool.py`, `heads.py`, `encoding.py`):

1. **Query routing** (`_encode_core`): every `[E]`/`[C]`/`[R]` marker embedding
   becomes one *query* vector `[Q, H]`. Text words become `text_states [L, H]`.
2. **BoundaryEncoder**: builds `L+1` boundary states from (left token, right
   token) pairs with learned BOS/EOS, projects to 128-d, runs 2 windowed
   self-attention layers + 1 SwiGLU block → `boundary_states [L+1, 128]`.
3. **BoundaryQueryHead**: scaled dot products give, per query,
   `start_logits [Q, L+1]`, `end_logits [Q, L+1]`, `inside_logits [Q, L]`, plus a
   centred prefix-sum of inside logits (`inside_prefix [Q, L+1]`, `inside_mean`).
4. **DocumentCandidatePool** (shared across queries): max over queries of
   start/end logits → top-32 starts × top-32 ends → Cartesian pairs with
   `end > start` → compat = projected dot product → score = compat + start + end
   marginal → per-query quota of 8 best pairs, then global fill, dedup → ≤192
   candidates `indices [C, 2]` (half-open word spans). Uses `topk`, stable
   `argsort`, dedup: data-dependent ops.
5. **SharedPoolScorer**: candidate feature = start proj + end proj +
   `Linear(3)` over `(log1p(len), len/L, rsqrt(len))` + prior proj + span content
   (mean-pooled value projection via prefix sums) → LayerNorm → FiLM-conditioned
   MLP per query + dot product + gathered start/end marginals + inside evidence
   → `pair_logits [Q, C]`.
6. **Auxiliary heads**: `null_projection` (abstention prob per query, threshold
   0.5) and `count_head` (log-rate per query, used for adaptive thresholding —
   off in this checkpoint).
7. **Decode** (`inference/candidate_decoder.py`): `sigmoid(logit /
   pair_temperature) ≥ threshold` → per-query overlap policy (default `flat` =
   weighted interval scheduling; alternatives exist) → char offsets
   `start_map[s], end_map[e-1]` → stable sort.
8. **Records** (`RecordHead`, `records.py`): anchor-based instance queries (32)
   over `candidate_states` (a `Linear(2·128 → H)` over the pooled candidate
   endpoints); field assignment logits per instance; `decode_group` builds
   records. Replaces count MLP + CountLSTM. Supports `mode="natural"` with an
   anchor field; legacy structures still decode via the span-style path.
9. **Relations** (`relations.py`): `TypedRelationPairGenerator` picks top heads /
   tails per type from the same candidate pool (argument threshold 0.2, cap 64
   pairs) → biaffine `SparseRelationScorer`; relation query = concat of the head
   and tail `[R]` marker states.
10. **Library-only additions** (no new weights): span attributes via
    `score_explicit_spans`, `Classifier` with constraint-aware exact/beam
    decoding, `JointIE` (typed entity–relation graph search), `*_long` chunking
    with overlap merge.

### Consequences

| | GLiNER2 | GLiNER2.5 |
| --- | --- | --- |
| Max span length | `max_width` words (8 here); longer spans never scored | any length within the window |
| Context | encoder limit, no chunking helper | trained to 4096 words + library chunking |
| Compute vs length | O(L·W) span grid | linear in L for fixed budget |
| Recall guarantee | every span ≤ W is scored | a gold span can be dropped at proposal time (pool of 192) |
| Instance count | 20-class MLP | anchor-based record decoding |
| Relations | independent head/tail spans | pooled candidates, biaffine, optional joint decoding |
| Determinism / debuggability | dense grid, trivial | top-k/dedup with tie-break rules |
| Upstream status | "stable" | README calls boundary "experimental", `architecture_version: 1` |

### Published benchmarks (macro F1, from the release post)

| Dataset | 2.5 Multi | 2.5 Base | GLiNER2 Multi | GLiNER2 Base |
| --- | ---: | ---: | ---: | ---: |
| Overall (16 tasks) | **56.17** | 54.87 | 56.09 | 53.34 |
| Classification avg | **72.44** | 69.86 | 70.32 | 68.89 |
| Extraction avg | 46.40 | 45.88 | **47.56** | 44.01 |
| xnli | **62.30** | 54.49 | 37.55 | 49.01 |
| few_nerd | 52.37 | **55.14** | 51.49 | 47.22 |
| ronec (multilingual NER) | **40.13** | 37.01 | 38.86 | 31.55 |
| crossner_politics | 55.26 | 56.41 | 62.47 | **66.52** |
| crossner_ai | 45.60 | 50.69 | 50.31 | **52.12** |
| imdb | 85.96 | 88.10 | 89.42 | **89.70** |
| clinc_oos (intent) | 61.32 | 62.20 | 62.59 | **63.62** |
| multilingual_sentiment | 79.42 | 63.14 | **81.30** | 57.57 |

Read: 2.5 is a different model with different (fully synthetic) training data,
not a strict upgrade. Big wins on NLI, general/multilingual NER, long spans;
small losses on several CrossNER splits, sentiment and intent.

## 3. Which to pick

**GLiNER2.5** when you need: spans longer than ~8–12 words; whole documents;
relations you can trust as a graph; per-span attributes; constrained multi-task
classification; NLI-style classification; non-English NER.

**GLiNER2** when: you need it to run in this crate today; the task is short-text
NER / sentiment / topic / intent (v2 is as good or better on the published
numbers); you depend on span-architecture fine-tunes (GLiGuard, PII models, your
own LoRAs — weights do not transfer); you want a dense, deterministic head.

They are not mutually exclusive: encoder, prompt format and classifier head are
shared, so one crate can serve both.

## 4. Mapping onto `gliner2-rs`

### Reusable as-is or nearly

| File | Status |
| --- | --- |
| `src/text.rs` | ✅ identical regex to upstream `WhitespaceTokenSplitter`; byte offsets vs Python char offsets is pre-existing and internally consistent |
| `src/tokenizer.rs`, `src/schema.rs` | ✅ same prompt layout |
| `src/embeddings.rs` | ⚠️ needs `[R]` marker + "one query per marker" enumeration |
| `src/encoder.rs`, `scripts/export/export_encoder.py` | ✅ same DeBERTa; verify fp16 → fp32 cast and 4096-word sequences |
| `src/classifier.rs`, `src/classification.rs`, `export_classifier.py` | ✅ same MLP shape |
| `src/schema_spec.rs`, `src/json.rs`, `src/validators.rs`, `src/entities.rs`, `src/relations.rs`, `src/structures.rs` (types) | ✅ |
| `src/adapters.rs`, `export_adapter_encoder.py` | ✅ |

### Not reusable for 2.5

`src/spans.rs`, `src/extractor.rs`, `src/decode.rs` (grid decode), the count
loop in `extract_structures_with_specs`, relation-as-structure in
`extract_relations_with_specs`, `export_extractor*.py`, and the
`max_width` / `extractor_max_fields` assumptions in `pipeline.rs`.

### Gaps that are not 2.5-specific but 2.5 exposes

- `max_len` word truncation (`processor.py:580`) is not applied anywhere in the
  crate.
- `CharLevelSplitter` not ported (only matters for CJK with the multi model).

## 5. Upgrade plan

### Phase 0 — dispatch and scaffolding (½–1 day)

- Read `config.json`; branch on `architecture` (`"span"` default when missing).
- Introduce an `Extractor` trait (or enum) over the public methods; move the
  current `Gliner2Pipeline` body to `SpanPipeline`; add `BoundaryPipeline`;
  add an `AutoExtractor`-style constructor.
- Apply `max_len` word truncation in the shared preprocessing.

### Phase 1 — boundary head export + entities + classification (1–2 weeks)

Upstream ships no ONNX exporter. The `export_mode` flag only removes a block
loop in the *per-query* proposer, which this checkpoint doesn't use
(`candidate_pool: "shared"`). Recommended split:

1. `boundary_marginals.onnx` — `BoundaryEncoder` + `BoundaryQueryHead`.
   Inputs `text_states [L,H]`, `query_states [Q,H]`; outputs
   `boundary_states [L+1,128]`, `start_logits`, `end_logits`, `inside_prefix`,
   `inside_mean`, plus the pool's projected `start_all`/`end_all` [L+1,128].
   Pure linear/attention/cumsum — exports cleanly. Note masks use a finite
   sentinel (`MASK_LOGIT = -1e4`) not `-inf`.
2. **Candidate pool in Rust** (`pool.rs`): max over queries, top-32 starts/ends,
   Cartesian pairing, compat via the exported projections, per-query quota (8)
   with rank bonus, global fill, dedup to 192. Replicate stable-sort
   tie-breaking so parity tests pass.
3. `boundary_scorer.onnx` — `SharedPoolScorer` given explicit
   `indices [C,2]`, `mask [C]`, `compat [C]`, plus `null_projection` and
   `count_head`. Outputs `pair_logits [Q,C]`, `candidate_states [C,H]`
   (needed for records), `null_logits [Q]`. Mirrors upstream's own
   `score_explicit_spans` path, so it is a sanctioned graph shape.
4. Decoder (`boundary_decode.rs`): sigmoid/threshold, overlap policies
   (`flat` WIS + others in `inference/overlap.py`), abstention, half-open
   boundary → char offsets, stable ordering.
5. Golden-parity tests against Python outputs per task; keep the CI
   skip-when-no-models pattern.

Alternative: export the whole head as one graph with `topk`/`argsort`/`unique`
inside. Faster to try, but ORT support for stable argsort + dedup with dynamic
shapes is the risk; the split above avoids it.

### Phase 2 — records and relations (1–2 weeks)

- Export `RecordHead` (`forward_group` path) and port `decode_group`
  (~200 lines) including anchor / natural mode and legacy structure fallback.
- Port `TypedRelationPairGenerator` (thresholded top heads/tails, pair cap) and
  export `SparseRelationScorer`.

### Phase 3 — library features (optional, incremental)

- Span attributes (`AttributeGroup`, `score_explicit_spans` reuse of the
  scorer graph).
- Constrained classification (exact/beam search over per-task probabilities —
  pure Rust, no new weights).
- `JointIE` beam search with typed endpoints / uniqueness / no-cycle rules.
- `*_long` chunking with overlap merge policies.

### Risks / things to confirm early

- Actual checkpoint headers contain F32 weights; encoder metadata says float16.
  Cast and inspect exports. Preserve trained relative-position bucket settings;
  4096 words does not mean resizing position embeddings to 4096. Long sequence
  numerical/memory tests remain required.
- Classifier keys confirmed: `classifier.0.{weight,bias}` and
  `classifier.3.{weight,bias}`. Numerical export parity remains required.
- Byte vs char offsets: Rust retains UTF-8 byte offsets, golden comparisons
  explicitly convert Python code-point offsets. Unicode/CJK tests remain required.
- Upstream is pinned in the plan, but manifests and verified exports are pending.
- Additional explicit sparse scorer is required for JSON choices and explicit
  scoring; exporting only the shared scorer is not a parity implementation.
- Scalar exclusive record assignment requires global matching, not greedy fill;
  ragged inference differs from dense training. Natural anchors are not capped
  at 32. Relations require directional 2H queries and text-state endpoint gathers.
- Cross-framework floating reduction may change compatibility values or pool
  cutoffs; exact discrete parity remains required. No tolerance exception yet.
- Encoder-only adapter switching cannot represent head-targeted LoRA; reject
  unsupported adapter bundles instead of silently ignoring adapted heads.
- Existing hosted v2 bundles omit tokenizer/config files. Fresh installation
  must be verified independently of this developer machine's model directories.
- Current upstream declares transformers<5 while checkpoint metadata names
  5.8.0. Pin a tested export dependency set and record any compatibility issues.
