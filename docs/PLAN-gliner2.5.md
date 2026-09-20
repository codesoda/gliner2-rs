# GLiNER2.5 implementation plan

Owner: GPT Astra 6 (planning/gate decisions); GPT 5.6 Sol (implementation).
Scope: mandatory M0–M7 plus public explicit-span scoring. M8 helper APIs are
optional follow-ups, not implied by ordinary boundary-model support. No claim of
completion until `AUDIT-gliner2.5.md` has evidence for every mandatory requirement.

**Phase A status: approved for M0.** Astra upstream source review completed;
critical corrections below supersede the research prompt's inaccurate assumptions.
Astra has approved one evidence-backed marginal-prefix comparison adjustment;
see the numerical adjudication below. All other numerical/discrete gates remain
unchanged.

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

**User-approved runtime migration (after M4):** remove ORP and use direct
`ort = =2.0.0-rc.13` / matching ort-sys, the latest published versions verified
against the live crates.io registry. This supersedes the original rc.9 freeze
in the implementation prompt. rc.13 targets ONNX Runtime 1.28 and requires
Rust 1.88 itself, but the locked Hugging Face/Xet dependency graph requires
Rust 1.91 (`str::floor_char_boundary`). The declared project floor is corrected
to 1.91 after actual 1.88/1.89 failures and a successful isolated 1.91 model,
no-model and six-tutorial gate. The usual local compiler is 1.95.
M0–M4 historical evidence used rc.9 /
native 1.20 and is not retroactively relabeled. The pinned Python oracle remains
unchanged; new Rust-runtime comparisons must independently satisfy its gates.
Keep public `infer(&self)` signatures via per-session synchronization because
newer ORT `Session::run` requires mutable access. Preserve CPU defaults, four
intra-op threads, Level3 graph optimization, exact input-name sets and required
output-name subsets. Own/copy outputs before releasing the session guard.
The migration must pass model-free and strict model tests, numerical parity,
v2 tutorial regression checks, formatting and Clippy before acceptance.
**User-approved exception:** v2 confidence values may differ from the old native
runtime by at most1e-6 absolute (finite f32, without rounding or bias). Labels,
text, spans, ordering, result structure and all non-confidence values remain
exact. All other stage tolerances are unchanged. This replaces byte identity
only for recognized confidence values across the runtime migration. No speedup
is implied; benchmark separately.
The exception follows deterministic native encoder drift: max1.9073486328125e-5
in hidden states for the isolated tutorial case, but bit-identical classifier
results when cross-fed the same old/new embeddings. Six full v2 tutorials have
one displayed confidence difference of5.364418029785156e-7; the other five
payloads match byte-for-byte. Multiple optimization/determinism/prepacking
experiments did not restore old encoder bytes. The exact first divergent native
operator is not claimed. See `evidence/ort-migration.json` for diagnostic evidence.
Export remains opset 17, fp32, with finite `-1e4` masks and no Python inference.

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
  relation sessions. Use a small private direct-ORT session helper (4 CPU
  threads, Level3 optimization) and named tensor I/O as in encoder/extractor.
  Validate shapes at module boundaries; return errors rather than indexing
  panics. ORP was removed in the accepted user-approved migration at a3ecbfa.
- Keep the old schema/result types. Extend schemas additively for record modes
  and overlap options if required; preserve defaults for existing callers.
  Existing SchemaSpec/StructureSpec/JsonSchema have public fields and support
  struct literals, so adding required fields would break Rust source compatibility.
  Prefer typed boundary record metadata as a sidecar keyed by structure name,
  accepted by additive `extract_with_records` / `extract_json_with_records`
  methods. Existing extraction methods remain unannotated legacy-record mode.
  Do not hide record metadata in global state or overload label strings.
- M5 record orchestration must use upstream's ragged `forward_group`, not its
  dense training variant: natural seeds include every live anchor candidate,
  latent seeds concatenate all fields (including duplicate spans), and anchorless
  attention uses the same field-major duplicated context. Trim padded graph
  assignment columns before decoding so padding cannot alter softmax outcomes.
- Choice lookup has three distinct Unicode operations: prefix lookup uses
  lowercase equality, formatted-surface matching uses full casefold, and literal
  ownership uses Python `re.IGNORECASE` with Python word boundaries. They are not
  interchangeable (for example, sharp-s folds to `ss` but is not an IGNORECASE
  literal match for `ss`). Prepare compact pinned Python3.12/Unicode15 data in
  Rust source, generated only by export tooling, to avoid a hidden inference
  dependency or host Rust/regex Unicode-version drift. Verify literal ownership
  against the unchanged upstream method and convert its character offsets to
  the public UTF-8 byte contract. No additional committed tensor fixtures needed.
- The mandatory high-level explicit primitive will be
  `BoundaryPipeline::score_explicit_spans(text, labels, spans)`, where labels are
  ordered entity-style extraction queries and spans are ordered half-open UTF-8
  byte pairs in the original text. Return one typed group per label and one
  typed score per supplied span, including original text/bounds, raw logit and
  pair-temperature-calibrated confidence. Preserve duplicate labels/spans and
  caller order; no candidate pool, threshold, abstention, overlap or dedup step.
  Encode all labels once, run marginals once, then the separate explicit scorer
  with the same supplied spans for every query. The existing low-level
  `ExplicitInput` remains available for query-specific tensor candidates.
  Require nonempty, valid UTF-8 spans exactly aligned with retained word-token
  boundaries: reject truncation, partial-word spans and synthetic punctuation
  rather than silently snapping coordinates. Empty label/candidate axes bypass
  native heads. Add AutoPipeline forwarding with an explicit unsupported error
  on span architecture (no emulation through v2). Prove byte/token mapping,
  duplicate/order preservation and separate-head parity, and provide a Rust
  line-scoring example with whitespace-trimmed line bounds. This API does not
  promise constrained probabilities summing to one or optional M8 features.
- Byte offsets remain the crate-wide contract, including boundary output, so
  `text[start..end]` is safe. Golden comparison explicitly converts Python
  Unicode code-point offsets to UTF-8 byte offsets; labels/text/order must match.
  Do not pretend byte positions equal Python positions on non-ASCII input.
  Selection metrics must still use Python codepoint distances/lengths: record
  choice proximity and M6 relation canonical-mention/nearest-occurrence ranking
  can change if calculated in bytes. Convert to bytes only for public coordinates
  or count codepoints in the corresponding validated original-text slices.
- Boundary classification decoding needs its own source-faithful selection:
  upstream runtime._extract_classification_result retains multi-label hits in
  schema order (not confidence order), and argmax ties/fallback select the first
  label. Existing Rust decode_classification sorts multi-label hits by confidence
  and selects the last equal maximum. Do not change that legacy v2 function;
  add boundary-specific selection while reusing public result types and the
  classifier ONNX wrapper. Apply classification_temperature before activation.
- Boundary combined extraction must encode all tasks together in one prompt,
  in upstream processor order: structures, entities, relations, classifications.
  Do not reuse SpanPipeline::extract_internal's separate task-family passes;
  that would change encoder context and break combined-schema parity. Preserve
  this prompt/routing order even while a later task-family decoder is pending.
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
| TypedRelationPairGenerator | `src/boundary/relation_pairs.rs`, pure Rust typed top arguments + capped pairs |
| SparseRelationScorer | `boundary_relations.onnx`, `src/boundary/relations.rs` loader over encoder text states and concatenated head/tail query states |
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
strongest instance, confidence=min(candidate,assignment). The pinned environment
has no SciPy, so the upstream internal shortest-augmenting Hungarian branch is
normative (it promotes f32-derived costs to f64 and uses strict ascending-column
loop ties). New oracle generators must not silently switch to the optional SciPy
branch. Preserve required
fields, source-anchor ordering, content dedup and literal-choice association.

Relations: two `[R]` queries head/tail, concatenated into `[1,R,2H]`. Pair
proposals use raw sigmoid logits >=0.2, max 32 endpoints each, stable
probability/start/end/index ordering, max 64 pairs, excluding self spans by
policy. The relation graph receives ENCODER TEXT states (despite an upstream
parameter called boundary_states), relation states, relation IDs and four
endpoint arrays. Gather start and end-1; preserve content-gated biaffine terms.
Port engine containing-mention canonicalization, coordinate/semantic dedup,
nearest-occurrence and token-subset rules, not just sigmoid/threshold. Relation
queries are the two dedicated `[R]` states for each relation schema, never the
entity-query IDs. Proposal selection uses the raw shared scorer output before
entity abstention/overlap/threshold decoding. Do not exclude choice-prefix spans
until after relation scoring; upstream filters them while mapping final edges.
Final confidence is sigmoid(relation logit / relation temperature), not multiplied
by endpoint probabilities. Per-relation thresholds override the request threshold.
Group duplicate canonical relation names before deduplication; descriptions affect
prompt context, not public keys. Empty proposals bypass the native relation head.

## 4. Golden design and acceptance rules

Commit a deterministic, human-readable ~30-case corpus under `scripts/parity/`:
8 short NER (duplicates, overlapping/nested mentions, >8-word spans included),
5 sentiment/intent/multi-task classification, 5 JSON records (choice fields,
list/str, natural anchor and legacy), 4 relations, 3 non-ASCII/CJK,
3 long texts (1000, 2000, 3000 words), 2 empty/punctuation/empty-schema edges.
Cases may include several task schemas; record their IDs and actual word counts.

`gen_boundary_goldens.py` must run the pinned upstream model with eval/no-grad,
CPU fp32 and deterministic seeds. **Measured API correction:** the prompt's
`GLiNER2.from_pretrained` call fails on boundary checkpoints at this commit
(`AttributeError: ExtractorConfig has no max_width`), because GLiNER2 deliberately
remains span-only. Use `AutoExtractor.from_pretrained` or `BoundaryExtractor`;
never monkey-patch the oracle. A real base-checkpoint smoke test succeeded with
AutoExtractor, torch2.8.0, transformers4.57.6 and CPU fp32. Upstream automatically
falls back from unsupported DeBERTa SDPA to eager on this transformers version;
record eager explicitly in the environment/manifest and parity runs. It may hook upstream functions, not replace
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

### M2 numerical adjudication — centered prefix only

Parent source review and independent rerun of the read-only sensitivity diagnostic
confirmed mean-reduction/per-token ORT rounding accumulates in centered prefix
coordinates. Merely rebuilding mean/cumsum in PyTorch from ORT logits still leaves
54 coordinates outside the original gate; higher-precision accumulation alone is
not sufficient. Across24 fixtures/12,640 prefix coordinates,199 fail the original
1e-4/1e-3 gate, exclusively in1000/2000/3000-word cases. Maximum raw drift is
0.00313687325. The smallest observed absolute envelope coefficient is
1.0344171086582568e-6 per coordinate (on top of1e-4).

Astra approves **only for ONNX `inside_prefix` comparisons**:
`abs(actual-ref) <= 1e-4 + 1.1e-6*i + 1e-3*abs(ref)`, where `i` is the boundary
coordinate, not the whole sequence length. Keep the original strict comparison
available for diagnostics; untouched oracle/wrapper comparisons retain it.
This is an empirical CPUfp32/ORT1.20 reference contract, not a universal bound.
Each other checkpoint/provider must independently pass validation; no automatic
further increase is authorized.

The decision is conditioned on unchanged downstream gates: on all5,116 actual
valid candidate/query pairs, restored and square-root-normalized inside evidence
changes by at most6.103515625e-5; substituting all ORT marginal outputs into the
unchanged learned scorer changes pair logits by at most the same amount and
confidence by8.642673492431641e-7. The original scorer is bit-exact to saved pair
logits; no top-1 or0.5-threshold crossings were observed. Actual ORT-to-Rust pool
selection is exact on all24 fixtures in debug/release. Scorer atol1e-4/rtol1e-3,
final confidence1e-3, and exact discrete selection remain unchanged.

Low-level marginal runtime contract is L>=1,Q>=1: Rust rejects L=0 before ORT
(the raw graph crashes on this input), and classification-only Q=0 bypasses the
head. Public empty text normalizes to `.`. Validation must record these limits,
not claim raw empty-dimension graph support.

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
   Observed M1 oracle splits decomposed `Café` into `cafe` and the combining
   mark. Rust regex `\\w` includes combining marks whereas Python's does not;
   boundary preprocessing must handle that difference without changing legacy
   span behavior. Parent source review of processor._collate_batch additionally
   confirms a trailing `.` is appended whenever nonempty input does not end in
   `.`, `!`, or `?`; empty input becomes `.`. Boundary encoder preprocessing
   must reproduce that normalization (the actual empty fixture has L=1), while
   public byte offsets must remain slice-safe for the caller's original text.
   Do not return a span extending into synthetic punctuation. Python `\\s` also
   recognizes U+001C–U+001F and case-insensitive `[a-z]` recognizes İ/ı/ſ/K;
   account for these boundary-tokenizer details without changing the v2 splitter.
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
   M3 investigation found PyTorch2.8 AArch64 uses four-lane/four-way cascade
   reduction. The Rust implementation deliberately fixes that operation grouping
   on every host rather than using host-native SIMD width. Parent debug/release
   tests matched all24 applicable frozen real cases bit-for-bit, including compat
   and proposal floats. This preserves the original exact gate with no tolerance
   exception. The reference corpus was generated on macOS arm64, CPUfp32, one
   thread: newly generated native x86 PyTorch tensors may differ numerically.
   Record that reference-platform contract in final bundle/validation docs; do
   not claim bit identity to arbitrary hardware-native oracle regeneration.
   Actual ORT-marginals-to-pool integration now matches indices/order/masks on
   all24 applicable real fixtures in debug and release; compatibility/proposal
   floats pass the original numerical gate. No set-equality or pool-order
   exception is approved.
7. **ONNX attention/export:** preserve full graph-affecting flags (directional,
   rotary, content, 8-head compatibility). Dynamic-shape multi-length tests are
   mandatory, not single dummy-input export success. Rust wrappers must follow
   the actual installed rc.13 APIs after migration. Python validation remains
   on its pinned 1.20.1 environment; record both versions explicitly and rerun
   all affected Rust model parity rather than assuming cross-version identity.
8. **Records/relations coupling:** preserve candidate identity/order and query
   routing across heads; natural/legacy schema paths must have separate fixtures.
9. **v2 regression:** do not replace v2 weights/algorithms. Capture representative
   pre-change outputs and compare bytes after hygiene changes; run all existing
   tests with real models in addition to skip-mode CI. The approved direct-ORT
   migration exception above applies only to confidence comparisons (<=1e-6);
   preserve exact non-confidence output and do not round values. Correct crate-root paths
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
  Proposal sigmoid uses raw pair logits (no pair temperature), then stable
  probability/start/end/flattened-index ordering,32 endpoints per side and64
  head-major/tail-minor product-ranked pairs per relation. Self-span exclusion
  compares exact coordinates. No discrete tie exception is pre-approved.
  Relation graph input is encoder text states, not128-wide boundary states;
  relation queries concatenate head then tail into2H channels. The original
  scorer's `float(max(length,1))` must not freeze the distance denominator during
  export: validate dynamic L/B/R/P against the untouched module. Masked relation
  scores are0.0 in source, not the extraction mask sentinel.
  Postprocessing must preserve source insertion order through canonical mention,
  exact-coordinate and semantic deduplication; then remove strict token-subset
  edges and sort by head/tail position and confidence. Canonical mention lengths
  and nearest-occurrence gaps use Python codepoint coordinates, not byte widths.
  Semantic text uses pinned full casefold plus Python whitespace splitting
  (including U+001C..U+001F, unlike ordinary Rust whitespace).
- M7: three complete tested bundles (including the additional explicit scorer),
  with original license notices and an explicit ONNX/fp32 conversion notice.
  The README model cards at all three pinned HF revisions declare Apache-2.0
  (small `f1e4d8fdd6fe328f45dee6aca3e6a07c9db4296e`, base
  `78cea040597df251eedefa9d7ee2a756af39fe64`, multi
  `235cf92d6d4318da9bfca0d08975c8fa7250d13b`).
  Preserve these source/model pins in publication metadata. Continue with
  upload/read-back/downloaders, user docs,
  `RESULTS-gliner2.5.md` CPU 50/500/3000-word benchmark (warmups/repetitions,
  hardware/thread/model hashes stated), CI, specification update and external
  Rust consumer test. Build public explicit-span scoring on the separate sparse
  explicit scorer, NOT on the M4 shared scorer.
- M8: optional helpers only if M7 complete; document implemented scope honestly.

### M7 packaging implementation constraints

The current exporters intentionally write partial manifests (some still record
historical rc.9/ORT1.20 metadata); those files are not suitable for publication.
A final bundle builder must enumerate all seven graphs: encoder, classifier,
boundary marginals, shared scorer, explicit scorer, records and relations. It
must derive actual signatures/checksums, include tokenizer/config files and
source/model licenses, and distinguish Python validation ORT1.20.1 from Rust
rc.13/native1.28. Source provenance must be proven by file hashes for each of the
three immutable revisions, not inferred from a directory name. The base-only
frozen corpus/provenance checks must remain fail-closed; extending generation to
small/multi requires explicit verified model identity, never relabeling base
outputs. Each new checkpoint needs its own oracle/ONNX/native validation.

Both opt-in downloaders must recognize the three new model names, download a
complete bundle, and validate its required file set and recorded checksums.
Treat manifest paths as untrusted relative paths: reject absolute/traversal paths
and unsupported architecture/version, and do not call a partial manifest a
usable bundle. Existing v2 downloads must also acquire tokenizer/config so a
fresh Rust consumer needs neither a local Python checkpoint nor an existing
HF cache. `examples/common/mod.rs` currently reads architecture from a complete
ONNX bundle but still constructs SpanPipeline using the separate `models/`
directory; correct that selection for colocated v2 metadata while preserving the
legacy split-layout fallback. Add model-free malformed/missing/hash-mismatch
tests, then validate actual published downloads in a clean destination. Keep upload authentication
outside code and logs; missing local credentials remain an explicit blocker.

Final audit explicitly covers every row in `AUDIT-gliner2.5.md`, actual model
artifacts, remote GitHub SHA/CI, HF downloads and downstream consumer execution.
A green no-model test run, a completed checklist file, or a push alone is not
completion. If blocked, report exact failed command/evidence and required next
input while leaving the active goal incomplete.
