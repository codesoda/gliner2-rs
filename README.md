# gliners2

Rust inference for [GLiNER2](https://github.com/fastino-ai/GLiNER2) via ONNX
Runtime — export scripts to convert the Python `gliner2` models to ONNX, plus
a Rust crate (`gliner2-rs`) that runs entity extraction, classification, and
structured extraction from those exports without a Python dependency at
inference time.

## Layout

- `export/` — Python scripts to export/validate GLiNER2 components
  (encoder, extractor, classifier) to ONNX.
- `gliner2-rs/` — Rust crate for loading the ONNX exports and running
  inference (via [`orp`](https://crates.io/crates/orp) /
  [`ort`](https://crates.io/crates/ort)). See its `examples/` for tutorials
  covering classification, NER, JSON extraction, relations, adapters, and
  training data prep.
- `scripts/download_models.py` — fetches the ONNX exports from Hugging Face.

## Models

The ONNX exports (`gliner2-base-v1`, `gliner2-large-v1`) are hosted on
Hugging Face rather than committed here, since they're large binaries:

👉 https://huggingface.co/codesoda/gliner2-onnx

```bash
pip install huggingface_hub
python3 scripts/download_models.py          # both models
python3 scripts/download_models.py --model base
```

This populates `./onnx/gliner2-base-v1/` and `./onnx/gliner2-large-v1/`
(each with `encoder.onnx`, `extractor.onnx`/`extractor_padded.onnx`, and
`classifier.onnx`), which is the layout `gliner2-rs`'s examples expect.

## Rust usage

```bash
cd gliner2-rs
cargo run --release --example tutorial_1_classification
```

Most examples accept `--model <path-to-onnx-dir>` to point at a specific
downloaded model bundle (see `examples/common/mod.rs`).

## Exporting your own ONNX models

See `export/export_encoder.py`, `export/export_extractor_padded.py`, and
`export/export_classifier.py`, each paired with a `validate_*.py` script to
sanity-check the export against the original PyTorch model.
