# GLiNER2.5 parity fixtures

`boundary_corpus.json` is the deterministic 30-case M1 corpus.
`gen_boundary_goldens.py` loads the pinned boundary checkpoint with
`AutoExtractor`, runs the public extraction API at `max_len=4096`, and observes
intermediate tensors with forward hooks. It never replaces proposal, scoring,
or decoding behavior.

Run from the repository root with the pinned uv environment. The checkpoint
is downloaded explicitly (about 0.8 GB); no inference code downloads it:

```bash
uv sync --project scripts/export/env --locked --python 3.12
PY=scripts/export/env/.venv/bin/python
MODEL="$($PY -c 'from huggingface_hub import snapshot_download; print(snapshot_download("fastino/gliner2.5-base-v1", revision="78cea040597df251eedefa9d7ee2a756af39fe64", allow_patterns=["*.json", "*.safetensors", "encoder_config/*", "*.model"]))')"
"$PY" scripts/parity/gen_boundary_goldens.py --model-dir "$MODEL"
"$PY" scripts/parity/validate_boundary_goldens.py
"$PY" -m unittest discover -s scripts/parity -p 'test_*.py'
```

Generation verifies the installed upstream Git commit and the checkpoint source
hashes. The validator checks structural integrity, complete corpus coverage,
finite tensors, routing consistency, hashes, and subset size; it does **not**
independently rerun inference. To check reproducibility, generate into a second
directory using `--out-dir <second-dir> --no-subset` and compare the per-case NPZ
hashes and formatted outputs. A numerical ONNX comparison is a separate gate:

```bash
"$PY" scripts/export/export_encoder.py --model-dir "$MODEL" --out-dir onnx/gliner2.5-base-v1
"$PY" scripts/export/export_classifier.py --model-dir "$MODEL" --out-dir onnx/gliner2.5-base-v1
"$PY" scripts/export/validate_encoder.py --model-dir "$MODEL" --onnx-path onnx/gliner2.5-base-v1/encoder.onnx --golden-dir fixtures/gliner2.5-base-v1
"$PY" scripts/export/validate_classifier.py --model-dir "$MODEL" --onnx-path onnx/gliner2.5-base-v1/classifier.onnx --golden-dir fixtures/gliner2.5-base-v1
```

These two graphs are only an incomplete M1 bundle, not a usable boundary extractor.

The complete output is generated under the ignored
`fixtures/gliner2.5-base-v1/`. The separately tracked
`fixtures/gliner2.5-base-v1-subset/` contains real entity, classification,
relation, and Unicode tensors and is constrained to 2 MiB. JSON stores both
upstream Unicode-code-point offsets and converted UTF-8 byte offsets.
