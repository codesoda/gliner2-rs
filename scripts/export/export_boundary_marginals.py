#!/usr/bin/env python3
"""Export the GLiNER2.5 boundary marginal graph with finite attention masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from boundary_marginals import (
    OUTPUT_NAMES,
    BoundaryMarginalGraph,
    assert_wrapper_matches_oracle,
    oracle_outputs,
)
from common import (
    BASE_HF_REVISION,
    assert_finite,
    configure_determinism,
    copy_runtime_metadata,
    load_reference_model,
    normalize_onnx_mask_constants,
    update_partial_manifest,
)

DEFAULT_MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--fastino--gliner2.5-base-v1"
    / "snapshots"
    / BASE_HF_REVISION
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument(
        "--out-dir", default="onnx/gliner2.5-base-v1"
    )
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--text-length", type=int, default=33)
    parser.add_argument("--query-count", type=int, default=3)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def validate_config(config: dict) -> None:
    if config.get("architecture") != "boundary":
        raise ValueError("boundary marginals require architecture='boundary'")
    if config.get("architecture_version") != 1:
        raise ValueError("boundary marginals require architecture_version=1")
    head = config.get("boundary_head", {})
    if head.get("candidate_pool") != "shared":
        raise ValueError("boundary marginals require candidate_pool='shared'")
    if head.get("boundary_dim") != 128:
        raise ValueError("this bundle contract requires boundary_dim=128")


def main() -> None:
    args = parse_args()
    if args.opset != 17:
        raise ValueError("GLiNER2 bundles require opset 17")
    if args.batch < 1 or args.text_length < 1 or args.query_count < 1:
        raise ValueError("export trace dimensions must all be positive")
    configure_determinism(args.seed)

    model, config = load_reference_model(args.model_dir, device=args.device)
    validate_config(config)
    wrapper = BoundaryMarginalGraph(model).float().eval()
    hidden_size = int(model.hidden_size)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    text_states = torch.randn(
        args.batch,
        args.text_length,
        hidden_size,
        generator=generator,
        dtype=torch.float32,
    )
    text_mask = torch.ones(args.batch, args.text_length, dtype=torch.bool)
    query_states = torch.randn(
        args.batch,
        args.query_count,
        hidden_size,
        generator=generator,
        dtype=torch.float32,
    )
    query_mask = torch.ones(args.batch, args.query_count, dtype=torch.bool)

    with torch.inference_mode():
        expected = oracle_outputs(
            model, text_states, text_mask, query_states, query_mask
        )
        actual = wrapper(text_states, text_mask, query_states, query_mask)
        assert_wrapper_matches_oracle(OUTPUT_NAMES, expected, actual)
        for name, value in zip(OUTPUT_NAMES, actual):
            if value.dtype != torch.bool:
                assert_finite(f"{name} export probe", value)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "boundary_marginals.onnx"
    torch.onnx.export(
        wrapper,
        (text_states, text_mask, query_states, query_mask),
        str(out_path),
        input_names=["text_states", "text_mask", "query_states", "query_mask"],
        output_names=list(OUTPUT_NAMES),
        dynamic_axes={
            "text_states": {0: "batch", 1: "text_length"},
            "text_mask": {0: "batch", 1: "text_length"},
            "query_states": {0: "batch", 1: "query_count"},
            "query_mask": {0: "batch", 1: "query_count"},
            "boundary_states": {0: "batch", 1: "boundary_count"},
            "boundary_mask": {0: "batch", 1: "boundary_count"},
            "start_logits": {
                0: "batch",
                1: "query_count",
                2: "boundary_count",
            },
            "end_logits": {
                0: "batch",
                1: "query_count",
                2: "boundary_count",
            },
            "inside_logits": {
                0: "batch",
                1: "query_count",
                2: "text_length",
            },
            "inside_prefix": {
                0: "batch",
                1: "query_count",
                2: "boundary_count",
            },
            "inside_prefix_mean": {0: "batch", 1: "query_count"},
            "start_all": {0: "batch", 1: "boundary_count"},
            "end_all": {0: "batch", 1: "boundary_count"},
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
        graph_name=out_path.name,
        graph_signature={
            "inputs": {
                "text_states": ["batch", "text_length", hidden_size],
                "text_mask": ["batch", "text_length"],
                "query_states": ["batch", "query_count", hidden_size],
                "query_mask": ["batch", "query_count"],
            },
            "outputs": {
                "boundary_states": ["batch", "boundary_count", 128],
                "boundary_mask": ["batch", "boundary_count"],
                "start_logits": ["batch", "query_count", "boundary_count"],
                "end_logits": ["batch", "query_count", "boundary_count"],
                "inside_logits": ["batch", "query_count", "text_length"],
                "inside_prefix": ["batch", "query_count", "boundary_count"],
                "inside_prefix_mean": ["batch", "query_count", 1],
                "start_all": ["batch", "boundary_count", 128],
                "end_all": ["batch", "boundary_count", 128],
            },
            "attention": "explicit-finite-mask-matmul-softmax",
            "mask_audit": mask_audit,
        },
    )
    print(f"saved {out_path}")
    print(f"updated incomplete manifest {manifest}")


if __name__ == "__main__":
    main()
