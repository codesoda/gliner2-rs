# GLiNER2.5 implementation plan

Owner: GPT Astra 6 (planning/gate decisions); GPT 5.6 Sol (implementation).
Scope: mandatory M0–M7 plus public explicit-span scoring. M8 helper APIs are
optional follow-ups, not implied by ordinary boundary-model support. No claim of
completion until `AUDIT-gliner2.5.md` has evidence for every mandatory requirement.

**Phase A status: approved for M0.** Astra upstream source review completed;
critical corrections below supersede the research prompt's inaccurate assumptions.
No tolerance loosening has been approved.

## 1. Pins and confirmed configuration

Upstream: `https://github.com/fastino-ai/GLiNER2`, commit
`d7c727458bf6929bc9ef5ee04e13c3f717a7c455`. Development clone is outside git.

| HF repository | Immutable revision | Encoder |
| --- | --- | --- |
| fastino/gliner2.5-small-v1 | f1e4d8fdd6fe328f45dee6aca3e6a07c9db4296e | microsoft/deberta-v3-xsmall |
| fastino/gliner2.5-base-v1 | 78cea040597df251eedefa9d7ee2a756af39fe64 | microsoft/deberta-v3-base |
| fastino/gliner2.5-multi-v1 | 235cf92d6d4318da9bfca0d08975c8fa7250d13b | microsoft/mdeberta-v3-base |

Actual checkpoint configs are committed as `docs/checkpoints/{small,base,multi}-config.json`.
All three `boundary_head` dictionaries are identical: shared pool, boundary/pair
128, content 64, attention layers 2/heads 4/window 128, refinement 1, pool top-k
32/size 192/quota 8, candidate attention 0, abstention/count/records/relations on,
adaptive threshold off, temperatures 1, max_len 4096, token_pooling first.
They also enable rotary endpoints, endpoint difference features, 8-head pair
compatibility, directional relation states, and biaffine content. These are
**required graph behavior**, not optional advanced library features.
The only observed model-config difference is model_name. Retrieved encoder
configs confirm hidden width 384 for small and 768 for base/multi, each with
12 layers, max_position_embeddings 512, position_buckets 256, relative_attention
true and pos_att_type [p2c,c2p]. Export each checkpoint with its actual hidden
width and tokenizer; never substitute the base tokenizer for multi. Upstream `resolve_word_splitter(None)` selects whitespace for all models.
Character splitting is an explicit option, regex `[A-Za-z0-9@._\\-+]+|\\S`, not
model-name dispatch. architecture_version must be 1. Safetensors header review
finds F32 tensors in all three actual checkpoints despite encoder config metadata
saying float16; always cast and verify rather than relying on either assumption.

Runtime stays `ort = =2.0.0-rc.9` / corresponding ort-sys (ONNX Runtime 1.20).
Export opset 17 with fp32 weights, CPU validation and finite `-1e4` masks.
No upgrade of ort is approved. Actual unsupported operations are redesigned or
moved to Rust, with evidence recorded, rather than hiding errors behind an upgrade.

## 2. Public API and backward compatibility

- Rename existing implementation to `pipeline::SpanPipeline`; preserve
  `pub type Gliner2Pipeline = SpanPipeline`. Existing public constructors,
  methods and result types remain available and unchanged.
- `AutoPipeline::from_dir(bundle)` reads config, dispatches span/boundary and
  loads only that architecture's files. Absent architecture defaults to span;
  malformed config, unknown architecture/version, unsupported graph variants,
  and missing required files return contextual errors before inference.
- Use a public dispatch enum as the prompt-permitted `Extractor` abstraction
  (root alias `Extractor = AutoPipeline`). Preserve low-level
  `extractor::Extractor`, which is already the v2 ONNX head wrapper; do not
  rename it or confuse it with the high-level API.
- The enum exposes the existing high-level methods (including classification,
  entities, JSON, relations, combined schema extraction and adapters) and delegates
  to the relevant pipeline. Do not silently route boundary models through v2.
  During M0, an explicitly unsupported boundary constructor is acceptable
  scaffolding, but is not a completed boundary implementation or release.
- `BoundaryPipeline` owns tokenizer, encoder, classifier, marginal/scorer/record/
  relation sessions. Reuse `orp::Model`, `RuntimeParameters::default()` (4 CPU
  threads), and named tensor I/O as in encoder/extractor. Validate shapes at
  module boundaries; return errors rather than indexing panics.
- Keep the old schema/result types. Extend schemas additively for record modes
  and overlap options if required; preserve defaults for existing callers.
- Byte offsets remain the crate-wide contract, including boundary output, so
  `text[start..end]` is safe. Golden comparison explicitly converts Python
  Unicode code-point offsets to UTF-8 byte offsets; labels/text/order must match.
  Do not pretend byte positions equal Python positions on non-ASCII input.
- Add query-routing metadata: exactly one query per `[E]`, `[C]`, `[R]` marker,
  in schema order, retaining schema and field identity. `[L]` is classification
  only, `[P]` is not an extraction query. Existing legacy marker gathering
  already includes `[R]`; preserve it and add tests rather than duplicating it.
- Shared word preprocessing applies config `max_len` before encoding. Missing
  max_len means no v2 truncation; boundary defaults to 4096. Offset vectors and
  text tokens must be truncated together (otherwise v2 falls back to joined
  strings with zero offsets). Cover raw inference and both classification paths,
  as well as structure choice-prefix accounting. Upstream truncates original
  words BEFORE prepending choice-prefix words. Do not truncate the prefix or
  truncate the already-combined token sequence a second time. Upstream public
  calls default to no cap despite config.max_len; goldens must pass explicit
  max_len=4096 to match this task's requested Rust default policy.

## 3. Graph split and module map

Use two extraction graphs with Rust candidate selection. A monolithic graph is
rejected because top-k ordering, stable sorting, variable-sized dedup and query
quota selection are control-heavy and must be independently testable.

| Upstream / existing source | Artifact / responsibility |
| --- | --- |
| encoder module; `processor.py` format/pool mapping | existing `encoder.onnx`, `src/{tokenizer,schema,text,embeddings}.rs` |
| classifier MLP | existing `classifier.onnx`, `src/{classifier,classification}.rs` |
| boundary encoding/query heads and pool endpoint projections | `boundary_marginals.onnx`, `src/boundary/marginals.rs` |
| DocumentCandidatePool selection | `src/boundary/pool.rs`, pure Rust |
| SharedPoolScorer, null/count heads and candidate-state projection | `boundary_scorer.onnx`, `src/boundary/scorer.rs` |
| boundary_proposer.score_explicit_pairs + SparseBoundaryPairScorer | **additional** `boundary_explicit_scorer.onnx`, same Rust scorer module; mandatory for M5 choice fields and public explicit scoring |
| candidate_decoder + overlap + runtime formatting | `src/boundary/decode.rs`, reuse result types |
| RecordHead.forward_group | `boundary_records.onnx`, `src/boundary/records.rs` loader + Rust decode_group |
| TypedRelationPairGenerator | `src/boundary/relations.rs`, Rust top arguments + capped pairs |
| SparseRelationScorer | `boundary_relations.onnx`, same Rust module loader |
| model._encode_core / engine orchestration | `src/boundary/pipeline.rs` |
| training/lora.py | existing merged encoder swap; no Python inference dependency |
| chunking / attributes / constraints / JointIE | M8 only, distinct tested helpers |

Marginal inputs: text `[L,H]`, queries `[Q,H]`; outputs include boundary states
`[L+1,D]`, start/end `[Q,L+1]`, inside logits/prefix/mean and all pool projections
needed to preserve compatibility arithmetic. Scorer takes explicit half-open
indices `[C,2]`, mask `[C]`, compat `[C]` and required marginal/text/query tensors;
returns pair `[Q,C]`, candidate `[C,H]`, null `[Q]`, count and any directional
candidate states needed for records/relations. Dynamic L/Q/C are required.
Retain batch-one axes in the export contracts: text `[1,L,H]`/mask `[1,L]`,
queries `[1,Q,H]`/mask `[1,Q]`; corresponding outputs include boundary mask,
inside_prefix_mean `[1,Q,1]`. Pool projections are specifically
`shared_pool_builder.start_projection/end_projection`. Inside prefixes are
centered fp32 with separately restored mean. Pool outputs are PADDED to 192.
Shared scorer transposes its internal `[B,C,Q]` result. Its 128-wide feature
vectors are NOT record states: use `candidate_encoder` on concatenated raw
boundary endpoints to get H-wide states and zero invalid rows.

**Required correction to the prompt:** upstream `score_explicit_spans` invokes
the sparse proposer/scorer even for shared-pool checkpoints, not SharedPoolScorer.
It has separate learned weights and extra rotary/gate/eight-head-compatibility/
endpoint-difference/content/length/inside terms. Export it separately by M5
(choices depend on it). Accept per-query indices `[1,Q,C,2]` and masks. The
ordinary two-graph split remains correct, but cannot implement explicit scoring.

Records: unannotated JSON uses one legacy instance with boundary-decoded fields,
NOT the v2 count/grid fallback. Explicit record metadata enables natural (one
instance per valid anchor, not capped at 32), latent (field-major candidate
context including duplicates), or anchorless (32 learned queries) modes.
Export inference `forward_group`, not the dense training path; Rust handles
ragged gathers and metadata. Assignment output includes ABSENT column zero.
Scalar exclusive assignment needs global min-cost matching; list assignment uses
strongest instance, confidence=min(candidate,assignment). Preserve required
fields, source-anchor ordering, content dedup and literal-choice association.

Relations: two `[R]` queries head/tail, concatenated into `[1,R,2H]`. Pair
proposals use raw sigmoid logits >=0.2, max 32 endpoints each, stable
probability/start/end/index ordering, max 64 pairs, excluding self spans by
policy. The relation graph receives ENCODER TEXT states (despite an upstream
parameter called boundary_states), relation states, relation IDs and four
endpoint arrays. Gather start and end-1; preserve content-gated biaffine terms.
Port engine containing-mention canonicalization, coordinate/semantic dedup,
nearest-occurrence and token-subset rules, not just sigmoid/threshold.

## 4. Golden design and acceptance rules

Commit a deterministic, human-readable ~30-case corpus under `scripts/parity/`:
8 short NER (duplicates, overlapping/nested mentions, >8-word spans included),
5 sentiment/intent/multi-task classification, 5 JSON records (choice fields,
list/str, natural anchor and legacy), 4 relations, 3 non-ASCII/CJK,
3 long texts (1000, 2000, 3000 words), 2 empty/punctuation/empty-schema edges.
Cases may include several task schemas; record their IDs and actual word counts.

`gen_boundary_goldens.py` must run the pinned upstream model with eval/no-grad,
CPU fp32 and deterministic seeds. It may hook upstream functions, not replace
upstream selection or decoding with the implementation under test. Dump NPZ plus
JSON per case: tokenization/maps, encoder last_hidden_state, gathered text/query
states, all marginal/projection tensors, pool indices/mask/compat, pair/null/count,
record/relation tensors as relevant, final formatted result with both options true.
Record upstream/model/dependency pins, corpus hash and file hashes.

Full fixtures live in ignored `fixtures/gliner2.5-base-v1/`; a representative
committed subset <=2 MB includes numerical tensors and exact decoder/pool edge
cases. CI must actually exercise the subset without heavyweight checkpoints;
ONNX-dependent tests may skip missing models, pure Rust parity tests may not.
Include ties, no queries/candidates, truncation, Unicode offsets and all overlap
policies as targeted synthetic cases in addition to real checkpoint goldens.

Initial numerical gates: encoder/marginals/scorer fp32 atol 1e-4, rtol 1e-3;
report observed maximum errors, including long cases. Pool indices/order/mask
must be exact; compat exact with golden projection inputs, then measure drift
when fed exported projections separately. Final labels, text, span ordering and
mapped offsets exact, confidences <=1e-3. Never round confidence values to hide
an error. Record all upstream empty-input behavior, not arbitrary empty tensors.
Only Astra can approve a changed tolerance after root cause and sensitivity tests;
no tie exception is pre-approved. No finite-output claim without checking arrays.

## 5. Risk resolutions and verification

1. **fp16 checkpoints:** cast the whole model to float32 before tracing; assert
   parameter/output dtypes and finite values. Existing exporters omit this and
   existing validators only report shapes: add actual PyTorch-vs-ORT assertions.
2. **4096 words / DeBERTa positions:** word limit is not a subword limit. Export
   with dynamic token sequence axes; test actual short/long schemas including
   4096-word boundary and record memory/time. Do not clamp subwords to 512 or
   state linear end-to-end encoder complexity. Relative bucket ops must survive
   tracing for variable sequences; investigate any exporter specialization.
   All three use position_biased_input=false; there is no absolute position
   table to resize. Do NOT change max_position_embeddings512 to4096: logarithmic
   relative buckets saturate, preserving trained numerics. Boundary window
   attention also builds a dense NxN mask (distance<=128), so memory can be
   quadratic even though candidate selection is bounded.
3. **Classifier keys:** verified `classifier.0.{weight,bias}` and
   `classifier.3.{weight,bias}` (Linear/ReLU/Dropout/Linear). Load through upstream
   model, assert dimensions in M1, compare raw and temperature-adjusted logits.
4. **Unicode:** byte-offset public contract plus explicit oracle conversion as
   above. Preserve original text before lowercase normalization; test combining
   marks, emoji and CJK. Determine optional character splitter from source.
5. **Experimental upstream:** immutable commit + three immutable HF revisions;
   validate architecture_version and supported flags. Every complete bundle has
   `export_manifest.json` with architecture/version/commit/revision/opset/ort,
   dependency versions, graph signatures and checksums. Downloaders validate it.
6. **Top-k ties:** verified shared selection uses stable descending torch.sort,
   NOT unstable topk. Tie break endpoints by index; pairs are start-major. Quota
   ranks have priorities 5000+quota down to 5001, flattened query-major, then
   global pairs. Stable score-desc/key-asc dedup keeps first key occurrence;
   final stable priority-desc order breaks ties by span key. Invalid slots are
   [0,0], compat zero, score -1e4; pad to 192. Reproduce the exact sequence.
   Floating compatibility reduction can differ between torch/Rust SIMD; keep
   the original exact gate until measured evidence justifies an explicit change.
   No set-equality or pool-order exception is approved.
7. **ONNX attention/export:** preserve full graph-affecting flags (directional,
   rotary, content, 8-head compatibility). Dynamic-shape multi-length tests are
   mandatory, not single dummy-input export success. Rust wrappers follow ort
   rc.9 APIs actually installed; Python ORT uses matching runtime generation.
8. **Records/relations coupling:** preserve candidate identity/order and query
   routing across heads; natural/legacy schema paths must have separate fixtures.
9. **v2 regression:** do not replace v2 weights/algorithms. Capture representative
   pre-change outputs and compare bytes after hygiene changes; run all existing
   tests with real models in addition to skip-mode CI. Correct crate-root paths
   and allow an explicit test artifact-root override; strict test mode prevents
   a missing-file typo from being reported as a model-enabled success.
10. **Fresh install:** current hosted v2 bundles omit tokenizer/config. Full 2.5
    bundles must contain all runtime files; fix downloaders to obtain v2 metadata
    too. Test a clean external consumer pinned to pushed GitHub SHA, no Python.
11. **Resources/publication:** weights and full fixtures stay ignored; use shared
    caches/symlinks locally. Validate bundles before HF upload. Missing credentials
    require user input, not a fabricated upload; push small code commits to the
    existing GitHub repository without force-push.

## 6. Execution and review gates

Each M0–M7 ends with `cargo fmt`, `cargo clippy --all-targets -- -D warnings`,
`cargo test` without and with actual artifacts, relevant Python validation and
model tutorials, an Astra review, a `PROGRESS-gliner2.5.md` entry and descriptive
commit. No later milestone is called complete while an earlier gate is open.
The correct Clippy invocation includes `--` before `-D warnings`.

- M0: config/dispatch/alias, preprocessing/query routing, fixture-path hygiene;
  fix pre-existing lint issues mechanically without changing v2 semantics.
- M1: pinned uv environment, real goldens, common exports and validators.
- M2: marginals graph + Rust loader + numerical parity.
- M3: pure Rust pool + full exact oracle comparisons.
- M4: scorer/decoder/entities/classification; tutorials 1–2 and formatted parity.
- M5: record graph/decode/API; structure parity and tutorial 3.
- M6: pair generator/relation graph/decode/API; parity and tutorial 6.
- M7: three complete tested bundles (including the additional explicit scorer),
  upload/read-back/downloaders, user docs,
  `RESULTS-gliner2.5.md` CPU 50/500/3000-word benchmark (warmups/repetitions,
  hardware/thread/model hashes stated), CI, specification update and external
  Rust consumer test. Build public explicit-span scoring on the separate sparse
  explicit scorer, NOT on the M4 shared scorer.
- M8: optional helpers only if M7 complete; document implemented scope honestly.

Final audit explicitly covers every row in `AUDIT-gliner2.5.md`, actual model
artifacts, remote GitHub SHA/CI, HF downloads and downstream consumer execution.
A green no-model test run, a completed checklist file, or a push alone is not
completion. If blocked, report exact failed command/evidence and required next
input while leaving the active goal incomplete.
