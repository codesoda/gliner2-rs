# gliner2-rs

[![CI](https://github.com/codesoda/gliner2-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/codesoda/gliner2-rs/actions/workflows/ci.yml)

Rust inference for [GLiNER2](https://github.com/fastino-ai/GLiNER2) via ONNX
Runtime — a Rust crate that runs entity extraction, classification, and
structured extraction from ONNX exports of the Python `gliner2` models,
without a Python dependency at inference time. Also includes the Python
export scripts used to produce those ONNX files.

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
**1.28**), without ORP. It requires Rust **1.88+**. Existing inference methods
and public `ndarray` **0.16** types are retained. Each model session serializes
its own inference calls; independent sessions can run concurrently.

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

The tutorials also require tokenizer/config files under
`models/gliner2-base-v1/`. Python is used only for this development-time
regression harness, not Rust inference or the Rust build.

### Tags & versioning

Tagged releases live at `vX.Y.Z` and match the version in `Cargo.toml`.
Check [existing tags](https://github.com/codesoda/gliner2-rs/tags) or
`git ls-remote --tags https://github.com/codesoda/gliner2-rs` for what's
available. Every tag is built and tested by [CI](.github/workflows/ci.yml)
before/after being pushed. To pin a specific commit instead of a tag, use
`rev = "<sha>"` in place of `tag = "..."`.

### CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) builds the crate
(lib, tests, examples), then runs `cargo test` on every push to `main`,
every `v*` tag, and every pull request. No model weights are required —
model-dependent tests detect a missing `onnx/`/`models/` directory and skip
themselves, so CI stays fast without pulling multi-gigabyte ONNX files.

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
- `scripts/export/` — Python scripts to export/validate GLiNER2 components
  (encoder, extractor, classifier) to ONNX.
- `scripts/download_models.py` — fetches the ONNX exports from Hugging Face.

## Models

The ONNX exports (`gliner2-base-v1`, `gliner2-large-v1`) are hosted on
Hugging Face rather than committed here, since they're large binaries:

👉 https://huggingface.co/codesoda/gliner2-onnx

Downloading them is a manual, opt-in step — nothing in this crate fetches
models automatically. Pick one:

**Rust (no Python required):**

```bash
cargo run --example download_models -- --model all     # both models
cargo run --example download_models -- --model base    # just gliner2-base-v1
```

**Python:**

```bash
pip install huggingface_hub
python3 scripts/download_models.py          # both models
python3 scripts/download_models.py --model base
```

Either way, this populates `./onnx/gliner2-base-v1/` and
`./onnx/gliner2-large-v1/` (each with `encoder.onnx`,
`extractor.onnx`/`extractor_padded.onnx`, and `classifier.onnx`), which is
the layout the examples expect.

## Rust usage

```bash
cargo run --release --example tutorial_1_classification
```

Most examples accept `--model <path-to-onnx-dir>` to point at a specific
downloaded model bundle (see `examples/common/mod.rs`).

## Exporting your own ONNX models

See `scripts/export/export_encoder.py`, `scripts/export/export_extractor_padded.py`,
and `scripts/export/export_classifier.py`, each paired with a `validate_*.py`
script to sanity-check the export against the original PyTorch model. Run
them from the repo root, e.g.:

```bash
python3 scripts/export/export_encoder.py --model-dir models/gliner2-base-v1 --out-dir onnx/gliner2-base-v1
```
