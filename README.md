# gliner2-rs

Rust inference for [GLiNER2](https://github.com/fastino-ai/GLiNER2) via ONNX
Runtime — a Rust crate that runs entity extraction, classification, and
structured extraction from ONNX exports of the Python `gliner2` models,
without a Python dependency at inference time. Also includes the Python
export scripts used to produce those ONNX files.

## Layout

- `src/` — the `gliner2-rs` crate: loads ONNX exports and runs inference via
  [`orp`](https://crates.io/crates/orp) / [`ort`](https://crates.io/crates/ort).
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
