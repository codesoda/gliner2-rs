#!/usr/bin/env python3
"""Export the shared GLiNER2 encoder with dynamic batch/sequence axes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import (
    assert_finite,
    configure_determinism,
    copy_runtime_metadata,
    load_reference_model,
    normalize_onnx_mask_constants,
    update_partial_manifest,
)


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/gliner2-base-v1")
    parser.add_argument("--out-dir", default="onnx/gliner2-base-v1")
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.opset != 17:
        raise ValueError("GLiNER2 bundles require opset 17")
    configure_determinism(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, _ = load_reference_model(args.model_dir, device=args.device)
    wrapper = EncoderWrapper(model.encoder).float().eval()
    dummy_ids = torch.ones(
        (args.batch, args.seq_len), dtype=torch.long, device=args.device
    )
    dummy_mask = torch.ones_like(dummy_ids)
    with torch.inference_mode():
        assert_finite("encoder export probe", wrapper(dummy_ids, dummy_mask))

    out_path = out_dir / "encoder.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        str(out_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    mask_audit = normalize_onnx_mask_constants(out_path)
    copy_runtime_metadata(args.model_dir, out_dir)
    manifest = update_partial_manifest(
        out_dir,
        args.model_dir,
        opset=args.opset,
        graph_name="encoder.onnx",
        graph_signature={
            "inputs": {
                "input_ids": ["batch", "sequence"],
                "attention_mask": ["batch", "sequence"],
            },
            "outputs": {
                "last_hidden_state": ["batch", "sequence", model.hidden_size]
            },
            "mask_audit": mask_audit,
        },
    )
    print(f"saved {out_path}")
    print(f"updated incomplete M1 manifest {manifest}")


if __name__ == "__main__":
    main()
