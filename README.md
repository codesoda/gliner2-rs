# gliner2-rs

[![CI](https://github.com/codesoda/gliner2-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/codesoda/gliner2-rs/actions/workflows/ci.yml)

Rust inference for [GLiNER2](https://github.com/fastino-ai/GLiNER2) via ONNX
Runtime — a Rust crate that runs entity extraction, classification, structured
records and relations from ONNX exports without a Python build or inference
dependency. The development branch supports both the legacy GLiNER2 span
architecture and the GLiNER2.5 boundary architecture. Validated small/base/multi
2.5 bundles are published; the v0.2.0 release tag is still pending the final M7
benchmark and release gates. Python export/reference tooling is
included for development only.

## Using this from your own Rust project

This crate isn't published to crates.io — depend on it directly from GitHub.
Pin to a tag rather than a branch so your build doesn't move out from under
you; see [Tags & versioning](#tags--versioning) for what's available.

```toml
# Cargo.toml
[dependencies]
gliner2-rs = { git = "https://github.com/codesoda/gliner2-rs", tag = "v0.1.0" }
```

You'll also need the ONNX model files locally — they're not part of the
crate (see [Models](#models) below). Point your code at wherever you
downloaded them:

```rust
use gliner2_rs::pipeline::Gliner2Pipeline;

fn main() -> anyhow::Result<()> {
    let pipeline = Gliner2Pipeline::new(
        "path/to/gliner2-base-v1",                 // model_dir (tokenizer files)
        "path/to/gliner2-base-v1/encoder.onnx",     // encoder ONNX
        "path/to/gliner2-base-v1/extractor_padded.onnx", // extractor ONNX
    )?;

    let labels = vec!["person".to_string(), "organization".to_string()];
    let matches = pipeline.extract_entities(
        "Alice joined Acme Corp last week.",
        &labels,
        0.5, // confidence threshold
    )?;

    for m in matches {
        println!("{}: {:?}", m.label, m.spans);
    }
    Ok(())
}
```

See `examples/` in this repo (`tutorial_1_classification.rs` through
`tutorial_11_adapter_switching.rs`) for classification, structured/JSON
extraction, relation extraction, validators, and LoRA adapters.

### Runtime compatibility

The current branch uses `ort` **2.0.0-rc.13** directly (native ONNX Runtime
**1.28**), without ORP. It requires Rust **1.91+** (the locked Hugging Face/Xet
transitive dependencies require newer APIs than ORT's own Rust 1.88 floor).
Existing inference methods and public `ndarray` **0.16** types are retained.
Each model session serializes its own inference calls; independent sessions can
run concurrently.

The runtime upgrade can change floating-point confidences slightly. The v2
regression gate permits at most **1e-6 absolute confidence drift**, while labels,
text, spans, ordering and all non-confidence output remain exact. No outputs are
rounded to meet this gate. See [migration evidence](docs/evidence/ort-migration.json).

To run all six original v2 tutorials and check their frozen reference outputs:

```bash
python3 scripts/parity/validate_v2_tutorials.py --run \
  --model /path/to/onnx/gliner2-base-v1 \
  --actual-dir /tmp/gliner2-v2-regression --actual-prefix current
```

The tutorials use tokenizer/config files colocated with the ONNX graphs, or
fall back to the legacy `models/gliner2-base-v1/` directory. Python is used only
for this development-time regression harness, not Rust inference or the Rust build.

### Tags & versioning

Tagged releases live at `vX.Y.Z` and match the version in `Cargo.toml`.
Check [existing tags](https://github.com/codesoda/gliner2-rs/tags) or
`git ls-remote --tags https://github.com/codesoda/gliner2-rs` for what actually
exists. Do not infer that unreleased branch APIs are present in an older tag.
To pin a specific commit instead of a tag, use `rev = "<sha>"` in place of
`tag = "..."`. The parent M7 release process will create a matching tag only
after bundles, remote CI and external-consumer evidence pass.

### CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) provides the
model-free build/test surface. The Rust 1.91 + stable and Python jobs passed at
[`9332af6`](https://github.com/codesoda/gliner2-rs/actions/runs/35563422282).
No model weights are required for ordinary CI: model-dependent tests detect absent artifacts and
emit explicit skips rather than downloading multi-gigabyte files. Consequently,
a green no-model run is not evidence that every counted test executed inference.

Model tests resolve artifacts from the crate root by default. To test a separate
artifact tree and make missing files fail instead of skip, run:

```bash
GLINER2_TEST_ROOT=/path/to/artifact-root GLINER2_REQUIRE_MODELS=1 cargo test
```

The root must contain `models/gliner2-base-v1/` and
`onnx/gliner2-base-v1/`.

## Layout

- `src/` — the `gliner2-rs` crate: loads ONNX exports and runs inference directly
  through [`ort`](https://crates.io/crates/ort), without ORP.
- `examples/` — tutorials covering classification, NER, JSON extraction,
  relations, adapters, and training data prep.
- `tests/` — integration tests.
- `scripts/export/` — Python development tools to export/validate v2 and the
  seven-graph GLiNER2.5 boundary bundle; Python is not used by Rust inference.
- `scripts/download_models.py` — fetches the ONNX exports from Hugging Face.

## Models

Published ONNX exports include v2 `gliner2-base-v1` / `gliner2-large-v1` and
validated GLiNER2.5 `gliner2.5-{small,base,multi}-v1` bundles:

👉 https://huggingface.co/codesoda/gliner2-onnx

Downloading is manual and opt-in; model acquisition is never part of `cargo
build` or inference. Select either published v2 model explicitly:

**Rust (no Python required):**

```bash
cargo run --example download_models -- --model base
cargo run --example download_models -- --model large
```

**Python:**

```bash
pip install huggingface_hub
python3 scripts/download_models.py --model base
python3 scripts/download_models.py --model large
```

### GLiNER2.5 downloads (current branch; not v0.1.0)

Selectors are `base`, `large`, `2.5-small`, `2.5-base`, `2.5-multi` and `all`.
Boundary and `all` downloads require an explicit immutable publication revision:

```bash
HF_REV=27310cd26099a387b9936a1e13b03d6a0700baf2
cargo run --release --example download_models -- \
  --model 2.5-base --dest ./onnx --revision "$HF_REV"
# Alternative Python development tool:
python3 scripts/download_models.py --model 2.5-base --dest ./onnx --revision "$HF_REV"
```

Both actual downloaders independently fetched and verified all five profiles at
this revision: identical 64-file inventories and SHA-256 hashes, totaling
5,150,060,191 bytes per destination. `all` is opt-in and downloads all five models,
not just the boundary family. See [results](docs/RESULTS-gliner2.5.md).

A complete boundary bundle is a single directory containing `config.json`,
`tokenizer.json`, `tokenizer_config.json`, `encoder_config/config.json`,
`SOURCE_MODEL_CARD.md`, `LICENSE`, `NOTICE`, `export_manifest.json`, and all
seven graphs:

```text
encoder.onnx
classifier.onnx
boundary_marginals.onnx
boundary_scorer.onnx
boundary_explicit_scorer.onnx
boundary_records.onnx
boundary_relations.onnx
```

The downloaders accept only boundary manifests marked `validated` and
`release_ready`, then check the architecture/version, required file set,
streaming SHA-256 hashes and byte sizes. Unsafe absolute/traversal/Windows-style
manifest paths and partial bundles are rejected. Export success alone does not
promote a bundle.

M7 also downloads and colocates the tokenizer/config metadata omitted from the
old hosted v2 directories. This makes a fresh Rust consumer independent of a
Python checkpoint and preexisting Hugging Face cache. The examples retain their
legacy split `models/` + `onnx/` fallback. Boundary manifests are not used to
reinterpret v2 bundles.

## Rust usage

```bash
cargo run --release --example tutorial_1_classification
```

Most examples accept `--model <path-to-onnx-dir>` to point at a specific
downloaded model bundle (see `examples/common/mod.rs`).

### GLiNER2.5 boundary API (unreleased)

M0–M6 and public explicit-span scoring passed their development gates. Fresh
small/base/multi bundles each passed seven source/ONNX stages and 32 native cases,
and have been published and independently downloaded. A clean remote-Git
consumer built and ran both architectures on Rust 1.91 with Python-named
execution denied ([proof](docs/evidence/m7-consumer.json)). Final benchmarking
still gates the release. The existing `v0.1.0` tag does not contain this API;
do not pin a nonexistent release tag.

Use architecture-aware loading when either family may be selected:

```rust,ignore
use gliner2_rs::pipeline::AutoPipeline;

let pipeline = AutoPipeline::from_dir("/path/to/one/complete/bundle")?;
let entities = pipeline.extract_entities(
    "Alice joined Acme Corp.",
    &["person".into(), "organization".into()],
    0.5,
)?;
```

`config.json` dispatches `span` or `boundary`; malformed/unsupported config and
missing required graphs are errors. `Gliner2Pipeline` remains a compatibility
alias for `SpanPipeline`, while the crate-root `Extractor` alias selects
`AutoPipeline`. The high-level classification, entity, JSON, relation, combined
schema and adapter methods delegate to the loaded architecture without routing a
boundary model through the v2 extraction head.

All public coordinates are half-open UTF-8 **byte offsets into the original
text**, so `&text[start..end]` is safe. Python reference character offsets are
converted explicitly. Synthetic punctuation added during boundary preprocessing
is not exposed as caller text.

#### Records and relations

Existing schema/JSON signatures remain source-compatible. Boundary callers can
opt into natural, latent or anchorless record formation with a typed
`RecordMetadata` sidecar through `extract_with_records` or
`extract_json_with_records`; structures omitted from the sidecar retain legacy
record behavior. Span/v2 models explicitly reject this boundary-only metadata.

The existing `extract_relations*` APIs dispatch to the 2.5 typed proposal and
learned biaffine relation head for boundary models. Relation confidence comes
from the calibrated relation logit, not a product of entity probabilities. The
boundary implementation preserves canonical mention selection, deduplication,
nearest occurrence and token-subset filtering, including Python code-point
semantics before converting output to UTF-8 bytes.

#### Explicit-span scoring

`BoundaryPipeline::score_explicit_spans`, also available through `AutoPipeline`,
uses the separate learned sparse scorer—not the ordinary shared candidate-pool
scorer:

```rust,ignore
let text = "Alice joined Acme.";
let labels = vec!["person".to_owned(), "organization".to_owned()];
let scores = pipeline.score_explicit_spans(text, &labels, &[[0, 5], [13, 17]])?;
```

Spans must be nonempty, retained word-aligned byte ranges. Invalid UTF-8
boundaries, partial words, truncation and synthetic punctuation return errors
rather than being snapped. Labels and spans preserve order and duplicates. Each
result contains original text/bounds, raw logit and pair-temperature-calibrated
confidence. This path performs no candidate pooling, thresholding, abstention,
overlap resolution or deduplication; independent sigmoid confidences need not sum
to one. Empty axes bypass native heads, and span/v2 models return an unsupported
error rather than emulating this method.

For a multiline Unicode example with trimmed line bounds:

```bash
cargo run --release --example explicit_span_scoring -- \
  --model /path/to/complete/gliner2.5-base-v1
```

Optional M8 helpers—attributes, constrained classification, JointIE and
long-document chunk/merge APIs—are not implemented or implied by ordinary
boundary-model support. See [GLiNER2 vs GLiNER2.5](docs/gliner2-vs-gliner2.5.md)
and the honest pending measurements in
[GLiNER2.5 results](docs/RESULTS-gliner2.5.md).

## Exporting your own ONNX models

See `scripts/export/export_encoder.py`, `scripts/export/export_extractor_padded.py`,
and `scripts/export/export_classifier.py`, each paired with a `validate_*.py`
script to sanity-check the export against the original PyTorch model. Run
them from the repo root, e.g.:

```bash
python3 scripts/export/export_encoder.py --model-dir models/gliner2-base-v1 --out-dir onnx/gliner2-base-v1
```
