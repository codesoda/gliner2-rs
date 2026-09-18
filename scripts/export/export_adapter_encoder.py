#!/usr/bin/env python3
"""
Export an adapter-specific `encoder.onnx` for GLiNER2 LoRA adapters.

Why?
- ONNX Runtime inference sessions have static weights.
- To "swap LoRA adapters" at runtime in Rust, we export a *separate* ONNX encoder that already
  includes the adapter weights (either merged into the base weights, or represented as LoRA ops
  in the exported graph depending on GLiNER2 internals).

This script:
1) loads the base GLiNER2 model (local path or HF id),
2) loads a LoRA adapter directory via `model.load_adapter(...)`,
3) exports `model.encoder` to `OUT_DIR/encoder.onnx`,
4) copies `adapter_config.json` into `OUT_DIR` (if present) so Rust can read metadata.

Example:
    python export/export_adapter_encoder.py \
      --base models/gliner2-base-v1 \
      --adapter ./adapters/legal_adapter/final \
      --out ./adapters/legal
"""

from __future__ import annotations

import argparse
import os
import shutil

import torch
from gliner2 import GLiNER2


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a GLiNER2 LoRA adapter encoder to ONNX.")
    parser.add_argument(
        "--base",
        default="models/gliner2-base-v1",
        help="Base model path or HF repo id (default: models/gliner2-base-v1).",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Adapter directory (typically .../final/) containing adapter weights/config.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for adapter bundle (writes encoder.onnx; copies adapter_config.json if present).",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.out, exist_ok=True)

    device = args.device
    print(f"Loading base model: {args.base}")
    model = GLiNER2.from_pretrained(args.base).to(device)
    model.eval()

    print(f"Loading adapter: {args.adapter}")
    model.load_adapter(args.adapter)

    wrapper = EncoderWrapper(model.encoder).to(device).eval()

    batch, seq_len = args.batch, args.seq_len
    dummy_ids = torch.ones((batch, seq_len), dtype=torch.long, device=device)
    dummy_mask = torch.ones((batch, seq_len), dtype=torch.long, device=device)

    out_path = os.path.join(args.out, "encoder.onnx")
    print(f"Exporting encoder ONNX to: {out_path}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_ids, dummy_mask),
            out_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "last_hidden_state": {0: "batch", 1: "seq"},
            },
            opset_version=args.opset,
        )

    # Copy adapter_config.json if it exists so Rust can read `lora_r`.
    src_cfg = os.path.join(args.adapter, "adapter_config.json")
    if os.path.exists(src_cfg):
        dst_cfg = os.path.join(args.out, "adapter_config.json")
        shutil.copy2(src_cfg, dst_cfg)
        print(f"Copied adapter config to: {dst_cfg}")
    else:
        print("NOTE: adapter_config.json not found; Rust will still load encoder.onnx, but lora_r will be unknown.")

    print("Done.")


if __name__ == "__main__":
    main()

