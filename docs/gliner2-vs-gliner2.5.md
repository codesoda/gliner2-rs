# GLiNER2 vs GLiNER2.5

This document explains which architecture to load and maps the GLiNER2.5 work to
its current implementation state. The authoritative implementation and release
gates remain [`PLAN-gliner2.5.md`](PLAN-gliner2.5.md) and
[`AUDIT-gliner2.5.md`](AUDIT-gliner2.5.md).

> **Release status:** the GLiNER2.5 implementation through M6, plus public
> explicit-span scoring, has been accepted in the development branch. M7 bundle
> construction, per-checkpoint validation, publication, download readback,
> benchmark measurement, CI and an external-consumer test are still in progress.
> There is not yet a released GLiNER2.5 bundle or release tag to recommend.

## Which model should I choose?

Choose by architecture and checkpoint behavior, not by treating 2.5 as an
in-place upgrade:

| Choose | When it is the better fit |
| --- | --- |
| **GLiNER2 (`span`)** | You need the currently published `gliner2-base-v1` or `gliner2-large-v1` ONNX artifacts; depend on a v2 fine-tune or encoder adapter; want the established fixed-width dense span head; or your short-text NER/classification workload already performs well on v2. |
| **GLiNER2.5 (`boundary`)** | You need candidates beyond v2's fixed span width, boundary record formation, the learned sparse relation head, multilingual 2.5 checkpoints, or direct scoring of caller-supplied spans. Wait for validated bundles if you need a published artifact rather than a development checkout. |

The model families have different extraction heads and training data. Published
upstream results show task-dependent gains and losses; “2.5” does not mean every
v2 workload gets higher quality. Weights for a v2 extraction head do not transfer
to a boundary head.

Checkpoint choices planned for the first 2.5 bundles are:

| Bundle | Upstream checkpoint | Encoder | Typical reason to choose it |
| --- | --- | --- | --- |
| `gliner2.5-small-v1` | `fastino/gliner2.5-small-v1` | DeBERTa-v3 xsmall, width 384 | Lowest footprint; must still pass independent validation before publication. |
| `gliner2.5-base-v1` | `fastino/gliner2.5-base-v1` | DeBERTa-v3 base, width 768 | Reference implementation and accepted M0–M6 parity work. |
| `gliner2.5-multi-v1` | `fastino/gliner2.5-multi-v1` | mDeBERTa-v3 base, width 768 | Multilingual tokenizer/checkpoint; never substitute the base tokenizer. |

All three default to whitespace word splitting. Character splitting is an
explicit boundary-pipeline option, not automatic dispatch for the multi model.

## Architecture differences

The two families share DeBERTa-style encoders, schema prompt markers, first
sub-token pooling and a classification MLP. Their extraction paths differ:

| Area | GLiNER2 (`span`) | GLiNER2.5 (`boundary`) |
| --- | --- | --- |
| Candidate representation | Dense `[word, max_width]` span grid | Start/end/inside marginals, then a sparse shared pool of at most 192 candidates |
| Span length | Limited by exported `max_width` | Not limited by v2's width grid; still limited by retained input and model/resource constraints |
| Entity scoring | Legacy extractor graph | Boundary marginals + Rust pool selection + shared scorer |
| Records | Count/grid structure path | Legacy single-record path or typed natural/latent/anchorless record metadata and record head |
| Relations | Relation represented through the v2 structure path | Typed head/tail proposals and separate content-gated biaffine relation graph |
| Explicit spans | No equivalent learned boundary head | Separate sparse explicit scorer for caller-provided spans |
| Default word cap | No cap when absent from config | Checkpoint config defaults to 4096 words |

A fixed candidate budget does **not** make the whole 2.5 pipeline linear: the
encoder and boundary attention contain dense operations, and long inputs can be
quadratic in memory/time. The 4096 setting is a word cap, not a promise that every
4096-word request fits every machine.

## Rust API selection

Use `AutoPipeline::from_dir` when application code may load either architecture:

```rust,ignore
use gliner2_rs::pipeline::AutoPipeline;

let pipeline = AutoPipeline::from_dir("/path/to/one/complete/bundle")?;
```

`config.json` selects `span` or `boundary`; a missing architecture retains the
legacy span default. Unknown architectures/versions and missing required graph
files are errors rather than fallback to the other architecture.

For source compatibility, `Gliner2Pipeline` remains an alias for `SpanPipeline`,
and the crate-root `Extractor` alias refers to `AutoPipeline`. This is distinct
from the existing low-level `extractor::Extractor` v2 ONNX wrapper.

The high-level entity, classification, JSON, relation, combined-schema and
adapter methods delegate to the selected implementation. Boundary combined
extraction uses one prompt in structures/entities/relations/classifications
order; it does not run separate v2 task-family passes.

### Coordinates and overlap

Public spans are half-open UTF-8 **byte** ranges into the caller's original
text, so `&text[start..end]` is valid. Python reference offsets are Unicode
code-point offsets and are converted during parity checks; byte and code-point
positions are not interchangeable on Unicode text. Internal nearest-occurrence
and relation ranking preserve Python code-point semantics before exposing bytes.
Synthetic terminal punctuation used by boundary preprocessing is never returned
as caller text.

Boundary overlap policies are `Allow`, `Nested`, `Disallow` (the checkpoint's
`flat` policy) and `Longest`, configurable with
`AutoPipeline::set_boundary_overlap_policy`. This option is intentionally
rejected for span models instead of changing v2 behavior.

### Records

Existing JSON/schema methods retain their old signatures. Boundary record
formation is additive through a sidecar keyed by structure name:

```rust,ignore
use gliner2_rs::boundary::record_schema::{RecordConfig, RecordMetadata};

let mut records = RecordMetadata::new();
records.insert("people".into(), RecordConfig::natural("name"));
let result = pipeline.extract_json_with_records(
    text,
    &schema,
    &records,
    0.5,
    true,  // confidence
    true,  // spans
)?;
```

`RecordConfig::natural(anchor)`, `latent()` and `anchorless()` select the three
record modes. Per-field cardinality/exclusivity overrides use
`RecordFieldOptions`. Structures omitted from the sidecar keep legacy boundary
JSON behavior. Span/v2 models explicitly reject record metadata; it is not
stored globally or encoded into label strings.

### Relations

The existing `extract_relations*` and combined-schema methods work on both
architectures but dispatch to different heads. In a boundary model, endpoints
come from raw shared-scorer candidates, typed relation queries use distinct head
and tail states, and the final confidence is the temperature-calibrated learned
relation score—not a product of endpoint probabilities. The implementation also
ports upstream canonical-mention, coordinate/semantic deduplication,
nearest-occurrence and strict token-subset behavior.

### Explicit-span scoring

`BoundaryPipeline::score_explicit_spans` and the matching `AutoPipeline` method
score each supplied span for each supplied label with the separate learned sparse
explicit scorer:

```rust,ignore
let scores = pipeline.score_explicit_spans(
    "Alice joined Acme.",
    &["person".into(), "organization".into()],
    &[[0, 5], [13, 17]],
)?;
```

Bounds must be nonempty, half-open UTF-8 bytes exactly aligned to retained word
boundaries. Invalid UTF-8 boundaries, partial words, truncation and wholly
synthetic punctuation are errors, not silently snapped spans. Labels and spans,
including duplicates, preserve caller order. The result reports source text,
bounds, raw logit and pair-temperature-calibrated confidence.

This API deliberately performs no candidate pooling, thresholding, abstention,
overlap handling or deduplication. Scores are independent sigmoid values and are
not constrained to sum to one. Empty labels return no groups; empty spans return
one empty group per label without calling native heads. Span/v2 models return an
unsupported error rather than emulating this with their grid head.

## Complete bundle contract (M7 development)

A usable 2.5 bundle is not “an encoder plus a head.” It must colocate tokenizer
and configuration metadata, notices, and exactly these seven graphs:

1. `encoder.onnx`
2. `classifier.onnx`
3. `boundary_marginals.onnx`
4. `boundary_scorer.onnx`
5. `boundary_explicit_scorer.onnx`
6. `boundary_records.onnx`
7. `boundary_relations.onnx`

It also needs `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`encoder_config/config.json`, `SOURCE_MODEL_CARD.md`, `LICENSE`, `NOTICE`, and
`export_manifest.json`. The notice identifies the source/model and the fp32/ONNX
conversion. The manifest records immutable source/model
identities, fp32/opset 17 conversion, actual graph signatures, sizes and SHA-256
checksums. A downloader must reject partial/unvalidated bundles, unsafe relative
paths, unsupported architecture/version, and size/hash mismatches.

The M7 downloader interface is being extended with selectors `2.5-small`,
`2.5-base` and `2.5-multi` in addition to v2 `base` and `large`; `all` will mean
all five and can be a large download. Full bundle names may also be accepted.
Destination override and immutable repository revision selection are part of the
M7 contract. These selectors describe the interface under integration, **not a
claim that validated 2.5 artifacts are already hosted**.

M7 also colocates the previously omitted v2 tokenizer/config metadata with v2
ONNX files so a fresh consumer does not need a Python checkpoint or warm HF
cache. Legacy split `models/` + `onnx/` loading remains a compatibility fallback.
Boundary manifests are not applied retroactively to v2 bundles.

## Runtime and numerical scope

Rust inference uses `ort = 2.0.0-rc.13` directly with native ONNX Runtime 1.28,
four intra-op threads and Level3 graph optimization. The project minimum is Rust
1.91 because of the locked Hugging Face/Xet dependency graph. Python is not used
by the Rust build or inference runtime.

Accepted numerical evidence is CPU fp32 on the pinned base checkpoint. Python
reference generation uses ONNX Runtime 1.20.1 and pinned Torch 2.8.0; Rust native
checks use ORT 1.28. Stage comparisons retain `1e-4 + 1e-3*abs(reference)` except
the explicitly documented centered-prefix coordinate envelope. Discrete pool and
formatting outputs remain exact; final confidence tolerance is `1e-3`. The v2
runtime migration permits only finite confidence drift up to `1e-6`, with all
non-confidence output exact. These are tested reference contracts, not universal
cross-provider, cross-hardware error guarantees.

Small and multi need independent source, ONNX and native validation. Base results
must not be relabeled as evidence for those checkpoints. See
[`RESULTS-gliner2.5.md`](RESULTS-gliner2.5.md) for measured parity evidence and
pending benchmark fields.

## Implementation requirement map

“Accepted” below means the development gate was reviewed; it does not mean M7
publication or a tagged release exists.

| Milestone / original requirement | Development status | User-visible mapping |
| --- | --- | --- |
| M0 dispatch, compatibility and preprocessing | Accepted | `SpanPipeline`, compatibility alias, `AutoPipeline`, architecture/version checks, query routing, max-length and artifact-root hygiene |
| M1 pinned environment, corpus and common graphs | Accepted for base | Deterministic 30-case reference corpus, encoder/classifier fp32 validation and immutable pins |
| M2 marginals | Accepted for base | Dynamic boundary marginal graph/wrapper, guarded empty axes and centered-prefix numerical rule |
| M3 candidate selection | Accepted for base | Pure Rust stable top-k/quota/dedup/padding with exact discrete reference ordering |
| M4 shared scorer/entities/classification | Accepted for base | Entity/classification/combined extraction, boundary overlap and source-faithful classification selection |
| M5 records and JSON | Accepted for base | Record graph, legacy JSON plus typed natural/latent/anchorless sidecars and choice scoring via explicit head |
| M6 relations | Accepted for base | Typed proposals, relation graph, all relation APIs and upstream postprocessing semantics |
| Public explicit-span primitive | Accepted for base | Ordered byte-span API, separate learned graph, preflight errors and multiline example |
| Direct ORT migration | Accepted | rc.13/native 1.28/Rust 1.91, no ORP/Python runtime; v2 confidence-only exception as above |
| M7 bundles/publication/download/benchmark/CI/consumer | In development | Seven-graph small/base/multi bundles, manifests, selectors, v2 metadata colocation and release proof are pending |
| M8 optional helpers | Unsupported/not started | Attributes, constrained classification, JointIE and long-document chunk/merge helpers are not promised by ordinary boundary support |

## Immutable sources

- Upstream GLiNER2 source: `d7c727458bf6929bc9ef5ee04e13c3f717a7c455`
- small: `fastino/gliner2.5-small-v1` at
  `f1e4d8fdd6fe328f45dee6aca3e6a07c9db4296e`
- base: `fastino/gliner2.5-base-v1` at
  `78cea040597df251eedefa9d7ee2a756af39fe64`
- multi: `fastino/gliner2.5-multi-v1` at
  `235cf92d6d4318da9bfca0d08975c8fa7250d13b`

Relevant upstream references include the
[GLiNER2 paper](https://arxiv.org/abs/2507.18546), the
[GLiNER2.5 release post](https://fastino.ai/blog/gliner2-5-span-free-information-extraction),
and code under `gliner2/models/{span,boundary}` at the pinned commit. There is no
separate GLiNER2.5 paper; the model cards refer to the GLiNER2 paper.
