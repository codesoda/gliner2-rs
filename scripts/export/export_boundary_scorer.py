#!/usr/bin/env python3
"""Export the GLiNER2.5 shared boundary scorer from explicit pool inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from boundary_scorer import (
    INPUT_NAMES,
    OUTPUT_NAMES,
    BoundaryScorerGraph,
    assert_wrapper_matches_oracle,
    oracle_outputs,
)
from common import (
    BASE_HF_REVISION,
    assert_finite,
    configure_determinism,
    load_reference_model,
    normalize_onnx_mask_constants,
    sha256_file,
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
    parser.add_argument("--out-dir", default="onnx/gliner2.5-base-v1")
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--text-length", type=int, default=33)
    parser.add_argument("--query-count", type=int, default=3)
    parser.add_argument("--candidate-count", type=int, default=17)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def validate_config(config: dict) -> None:
    if config.get("architecture") != "boundary":
        raise ValueError("boundary scorer requires architecture='boundary'")
    if config.get("architecture_version") != 1:
        raise ValueError("boundary scorer requires architecture_version=1")
    head = config.get("boundary_head", {})
    required = {
        "candidate_pool": "shared",
        "boundary_dim": 128,
        "pair_dim": 128,
        "candidate_attention_layers": 0,
        "candidate_attention_heads": 4,
        "query_attention_layers": 0,
        "enable_span_content": True,
        "content_dim": 64,
        "content_soft_max_pool": False,
        "use_inside_evidence": True,
        "enable_abstention": True,
        "enable_count_head": True,
        "enable_records": True,
    }
    mismatches = [
        f"{name}={head.get(name)!r}, expected {expected!r}"
        for name, expected in required.items()
        if head.get(name) != expected
    ]
    if mismatches:
        raise ValueError("unsupported boundary scorer flags: " + "; ".join(mismatches))


def trace_inputs(
    *, seed: int, batch: int, length: int, queries: int, candidates: int, hidden: int
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    boundary_count = length + 1
    boundary_states = torch.randn(
        batch, boundary_count, 128, generator=generator, dtype=torch.float32
    )
    text_states = torch.randn(
        batch, length, hidden, generator=generator, dtype=torch.float32
    )
    text_mask = torch.ones(batch, length, dtype=torch.bool)
    query_states = torch.randn(
        batch, queries, hidden, generator=generator, dtype=torch.float32
    )
    query_mask = torch.ones(batch, queries, dtype=torch.bool)
    start_logits = torch.randn(
        batch, queries, boundary_count, generator=generator, dtype=torch.float32
    )
    end_logits = torch.randn(
        batch, queries, boundary_count, generator=generator, dtype=torch.float32
    )
    inside_values = torch.randn(
        batch, queries, length, generator=generator, dtype=torch.float32
    )
    inside_prefix_mean = inside_values.mean(dim=-1, keepdim=True)
    centered = inside_values - inside_prefix_mean
    inside_prefix = torch.cat(
        (torch.zeros(batch, queries, 1), centered.cumsum(dim=-1)), dim=-1
    )

    flat = torch.arange(candidates, dtype=torch.long)
    starts = (flat % max(length, 1)).view(1, candidates).expand(batch, -1)
    widths = (flat % 5 + 1).view(1, candidates).expand(batch, -1)
    ends = (starts + widths).clamp(max=length)
    starts = torch.minimum(starts, (ends - 1).clamp_min(0))
    candidate_indices = torch.stack((starts, ends), dim=-1)
    candidate_mask = torch.ones(batch, candidates, dtype=torch.bool)
    candidate_compat = torch.randn(
        batch, candidates, generator=generator, dtype=torch.float32
    )
    return (
        boundary_states,
        text_states,
        text_mask,
        query_states,
        query_mask,
        start_logits,
        end_logits,
        inside_prefix,
        inside_prefix_mean,
        candidate_indices,
        candidate_mask,
        candidate_compat,
    )


def graph_signature(hidden_size: int, mask_audit: dict) -> dict:
    return {
        "inputs": {
            "boundary_states": ["batch", "boundary_count", 128],
            "text_states": ["batch", "text_length", hidden_size],
            "text_mask": ["batch", "text_length"],
            "query_states": ["batch", "query_count", hidden_size],
            "query_mask": ["batch", "query_count"],
            "start_logits": ["batch", "query_count", "boundary_count"],
            "end_logits": ["batch", "query_count", "boundary_count"],
            "inside_prefix": ["batch", "query_count", "boundary_count"],
            "inside_prefix_mean": ["batch", "query_count", 1],
            "candidate_indices": ["batch", "candidate_count", 2],
            "candidate_mask": ["batch", "candidate_count"],
            "candidate_compat": ["batch", "candidate_count"],
        },
        "outputs": {
            "pair_logits": ["batch", "query_count", "candidate_count"],
            "candidate_states": ["batch", "candidate_count", hidden_size],
            "null_logits": ["batch", "query_count"],
            "count_log_rates": ["batch", "query_count"],
        },
        "candidate_pool": "explicit-query-agnostic-document-pool",
        "pair_score_order": "query-major",
        "candidate_state_source": "candidate_encoder(raw_boundary_start,end)",
        "mask_logit": -1.0e4,
        "mask_audit": mask_audit,
    }


def main() -> None:
    args = parse_args()
    if args.opset != 17:
        raise ValueError("GLiNER2 bundles require opset 17")
    if min(args.batch, args.text_length, args.query_count, args.candidate_count) < 1:
        raise ValueError("export trace dimensions must all be positive")
    configure_determinism(args.seed)
    model, config = load_reference_model(args.model_dir, device=args.device)
    validate_config(config)
    hidden_size = int(model.hidden_size)
    wrapper = BoundaryScorerGraph(model).float().eval()
    inputs = trace_inputs(
        seed=args.seed,
        batch=args.batch,
        length=args.text_length,
        queries=args.query_count,
        candidates=args.candidate_count,
        hidden=hidden_size,
    )
    with torch.inference_mode():
        expected = oracle_outputs(model, *inputs)
        actual = wrapper(*inputs)
    assert_wrapper_matches_oracle(OUTPUT_NAMES, expected, actual)
    for name, value in zip(OUTPUT_NAMES, actual):
        assert_finite(f"{name} export probe", value)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "boundary_scorer.onnx"
    dynamic_axes = {
        "boundary_states": {0: "batch", 1: "boundary_count"},
        "text_states": {0: "batch", 1: "text_length"},
        "text_mask": {0: "batch", 1: "text_length"},
        "query_states": {0: "batch", 1: "query_count"},
        "query_mask": {0: "batch", 1: "query_count"},
        "start_logits": {0: "batch", 1: "query_count", 2: "boundary_count"},
        "end_logits": {0: "batch", 1: "query_count", 2: "boundary_count"},
        "inside_prefix": {0: "batch", 1: "query_count", 2: "boundary_count"},
        "inside_prefix_mean": {0: "batch", 1: "query_count"},
        "candidate_indices": {0: "batch", 1: "candidate_count"},
        "candidate_mask": {0: "batch", 1: "candidate_count"},
        "candidate_compat": {0: "batch", 1: "candidate_count"},
        "pair_logits": {0: "batch", 1: "query_count", 2: "candidate_count"},
        "candidate_states": {0: "batch", 1: "candidate_count"},
        "null_logits": {0: "batch", 1: "query_count"},
        "count_log_rates": {0: "batch", 1: "query_count"},
    }
    torch.onnx.export(
        wrapper,
        inputs,
        str(out_path),
        input_names=list(INPUT_NAMES),
        output_names=list(OUTPUT_NAMES),
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    mask_audit = normalize_onnx_mask_constants(out_path)
    signature = graph_signature(hidden_size, mask_audit)
    proposed_manifest_entry = {
        "boundary_scorer.onnx": {
            **signature,
            "sha256": sha256_file(out_path),
            "bytes": out_path.stat().st_size,
        }
    }
    print(f"saved {out_path}")
    print("export_manifest.json was not modified")
    print("proposed manifest graph entry:")
    print(json.dumps(proposed_manifest_entry, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
