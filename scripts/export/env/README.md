# GLiNER2 export environment

This uv project pins the Python reference/export stack, including the exact
upstream GLiNER2 Git revision. Create or refresh it without downloading the
already-cached checkpoint:

```bash
cd scripts/export/env
UV_LINK_MODE=copy uv sync --locked --python 3.12
uv run python ../export_encoder.py \
  --model-dir ~/.cache/huggingface/hub/models--fastino--gliner2.5-base-v1/snapshots/78cea040597df251eedefa9d7ee2a756af39fe64 \
  --out-dir ../../../onnx/gliner2.5-base-v1
```

The `.venv` is intentionally ignored. `uv.lock` is committed and must be
updated only deliberately. Exports run on CPU in fp32 with eager attention,
fixed seeds, and opset 17.
