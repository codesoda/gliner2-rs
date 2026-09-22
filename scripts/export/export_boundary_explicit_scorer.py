#!/usr/bin/env python3
"""Export the mandatory learned explicit-span boundary scorer graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from boundary_explicit_scorer import (
    INPUT_NAMES,
    OUTPUT_NAMES,
    BoundaryExplicitScorerGraph,
    assert_output_parity,
    independently_composed_outputs,
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
    parser.add_argument("--candidate-count", type=int, default=7)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def validate_config(config: dict) -> None:
    if config.get("architecture") != "boundary":
        raise ValueError("explicit scorer requires architecture='boundary'")
    if config.get("architecture_version") != 1:
        raise ValueError("explicit scorer requires architecture_version=1")
    head = config.get("boundary_head", {})
    required = {
        "boundary_dim": 128,
        "pair_dim": 128,
        "enable_span_content": True,
        "content_dim": 64,
        "content_soft_max_pool": False,
        "use_inside_evidence": True,
        "enable_rotary_endpoints": True,
        "query_conditioned_inside_weight": True,
        "endpoint_difference_features": True,
        "reranker_endpoint_compat": True,
        "multihead_pair_compat_heads": 8,
    }
    mismatches = [
        f"{name}={head.get(name)!r}, expected {expected!r}"
        for name, expected in required.items()
        if head.get(name) != expected
    ]
    if mismatches:
        raise ValueError("unsupported explicit scorer flags: " + "; ".join(mismatches))


def trace_inputs(
    *, seed: int, batch: int, length: int, queries: int, candidates: int, hidden: int
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    boundaries = length + 1
    boundary_states = torch.randn(
        batch, boundaries, 128, generator=generator, dtype=torch.float32
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
        batch, queries, boundaries, generator=generator, dtype=torch.float32
    )
    end_logits = torch.randn(
        batch, queries, boundaries, generator=generator, dtype=torch.float32
    )
    inside_values = torch.randn(
        batch, queries, length, generator=generator, dtype=torch.float32
    )
    inside_prefix_mean = inside_values.mean(dim=-1, keepdim=True)
    inside_prefix = torch.cat(
        (
            torch.zeros(batch, queries, 1, dtype=torch.float32),
            (inside_values - inside_prefix_mean).cumsum(dim=-1),
        ),
        dim=-1,
    )

    flat = torch.arange(candidates, dtype=torch.long).reshape(1, 1, candidates)
    starts = (flat % length).expand(batch, queries, -1)
    ends = (starts + 1 + flat % 5).clamp(max=length)
    starts = torch.minimum(starts, ends - 1)
    candidate_indices = torch.stack((starts, ends), dim=-1)
    candidate_mask = torch.ones(batch, queries, candidates, dtype=torch.bool)
    if candidates > 1:
        candidate_mask[..., -1] = False
        candidate_indices[..., -1, 0] = -1
        candidate_indices[..., -1, 1] = length + 1
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
    )


def repair_legacy_dynamic_stack_axis(path: Path) -> dict[str, object]:
    """Repair the legacy exporter's wrong axis for the original length stack.

    ``continuous_length_features`` uses ``torch.stack(..., dim=-1)``.  With
    dynamic candidate axes the TorchScript ONNX exporter incorrectly resolves
    that negative axis to 2 instead of the runtime last axis 3.  This changes
    only the lowering metadata for that original operation; no scorer arithmetic
    or learned module is replaced.
    """
    graph = onnx.load(str(path))
    matches = [
        node
        for node in graph.graph.node
        if node.name == "/pair_scorer/Concat_19" and node.op_type == "Concat"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "expected one legacy-exported sparse-scorer length-feature stack, "
            f"found {len(matches)}"
        )
    node = matches[0]
    axis = next((item for item in node.attribute if item.name == "axis"), None)
    if axis is None or axis.i != 2:
        raise RuntimeError(
            f"unexpected exported length-feature stack axis: {None if axis is None else axis.i}"
        )
    # The three Unsqueeze axis constants are part of the same Stack lowering.
    # Their rank-3 inputs need a new final axis (3), then Concat joins that axis.
    constant_names = {
        f"/pair_scorer/Constant_{number}" for number in (58, 95, 96, 97)
    }
    constants = [
        item
        for item in graph.graph.node
        if item.name in constant_names and item.op_type == "Constant"
    ]
    if {item.name for item in constants} != constant_names:
        raise RuntimeError("could not identify all length-feature stack axis constants")
    for constant in constants:
        value = next(
            (item for item in constant.attribute if item.name == "value"), None
        )
        if value is None or not np.array_equal(
            numpy_helper.to_array(value.t), np.asarray([2], dtype=np.int64)
        ):
            raise RuntimeError(f"unexpected axis constant in {constant.name}")
        value.t.CopyFrom(numpy_helper.from_array(np.asarray([3], dtype=np.int64)))
    axis.i = 3

    rotary_constant_names = {
        "/pair_scorer/rotary/Constant_10",
        "/pair_scorer/rotary/Constant_11",
        "/pair_scorer/rotary_1/Constant_8",
        "/pair_scorer/rotary_1/Constant_9",
    }
    rotary_constants = [
        item
        for item in graph.graph.node
        if item.name in rotary_constant_names and item.op_type == "Constant"
    ]
    if {item.name for item in rotary_constants} != rotary_constant_names:
        raise RuntimeError("could not identify all sparse-scorer rotary stack axes")
    for constant in rotary_constants:
        value = next(
            (item for item in constant.attribute if item.name == "value"), None
        )
        if value is None or not np.array_equal(
            numpy_helper.to_array(value.t), np.asarray([2], dtype=np.int64)
        ):
            raise RuntimeError(f"unexpected rotary axis constant in {constant.name}")
        value.t.CopyFrom(numpy_helper.from_array(np.asarray([3], dtype=np.int64)))
    rotary_concat_names = {
        "/pair_scorer/rotary/Concat",
        "/pair_scorer/rotary_1/Concat",
    }
    rotary_concats = [
        item
        for item in graph.graph.node
        if item.name in rotary_concat_names and item.op_type == "Concat"
    ]
    if {item.name for item in rotary_concats} != rotary_concat_names:
        raise RuntimeError("could not identify both sparse-scorer rotary stacks")
    for rotary_concat in rotary_concats:
        rotary_axis = next(
            (item for item in rotary_concat.attribute if item.name == "axis"), None
        )
        if rotary_axis is None or rotary_axis.i != 2:
            raise RuntimeError(f"unexpected rotary stack axis in {rotary_concat.name}")
        rotary_axis.i = 3

    rotary_flatten_shape_names = {
        "/pair_scorer/rotary/Concat_1",
        "/pair_scorer/rotary_1/Concat_1",
    }
    rotary_flatten_shapes = [
        item
        for item in graph.graph.node
        if item.name in rotary_flatten_shape_names and item.op_type == "Concat"
    ]
    if {item.name for item in rotary_flatten_shapes} != rotary_flatten_shape_names:
        raise RuntimeError("could not identify both sparse-scorer rotary flatten shapes")
    for flatten_shape in rotary_flatten_shapes:
        if len(flatten_shape.input) != 3:
            raise RuntimeError(f"unexpected rotary flatten shape in {flatten_shape.name}")
        # flatten(-2) on the corrected rank-4 stack returns rank 3: preserve
        # B,N and merge the final pair-width axes.  The legacy trace retained
        # a stale fourth dimension because it inferred the wrong Stack rank.
        del flatten_shape.input[2]

    difference_matches = [
        item
        for item in graph.graph.node
        if item.name == "/pair_scorer/Concat_9" and item.op_type == "Concat"
    ]
    if len(difference_matches) != 1:
        raise RuntimeError("could not identify endpoint-difference concatenation")
    difference_axis = next(
        (
            item
            for item in difference_matches[0].attribute
            if item.name == "axis"
        ),
        None,
    )
    if difference_axis is None or difference_axis.i != 2:
        raise RuntimeError("unexpected endpoint-difference concatenation axis")
    difference_axis.i = 3

    head_width_matches = [
        item
        for item in graph.graph.node
        if item.name == "/pair_scorer/Constant_38" and item.op_type == "Constant"
    ]
    if len(head_width_matches) != 1:
        raise RuntimeError("could not identify multi-head compatibility width")
    head_width = next(
        (item for item in head_width_matches[0].attribute if item.name == "value"),
        None,
    )
    if head_width is None or not np.array_equal(
        numpy_helper.to_array(head_width.t), np.asarray([2], dtype=np.int64)
    ):
        raise RuntimeError("unexpected legacy-exported compatibility head width")
    # Corrected rotary output width is pair_dim=128, split over eight heads.
    head_width.t.CopyFrom(numpy_helper.from_array(np.asarray([16], dtype=np.int64)))

    squeeze_guard_names = {
        "/pair_scorer/Constant_39",
        "/pair_scorer/Constant_43",
        "/pair_scorer/Constant_79",
    }
    squeeze_guards = [
        item
        for item in graph.graph.node
        if item.name in squeeze_guard_names and item.op_type == "Constant"
    ]
    if {item.name for item in squeeze_guards} != squeeze_guard_names:
        raise RuntimeError("could not identify rank-4 final-axis squeeze guards")
    for guard in squeeze_guards:
        value = next((item for item in guard.attribute if item.name == "value"), None)
        if value is None or not np.array_equal(
            numpy_helper.to_array(value.t), np.asarray([2], dtype=np.int64)
        ):
            raise RuntimeError(f"unexpected squeeze guard axis in {guard.name}")
        value.t.CopyFrom(numpy_helper.from_array(np.asarray([3], dtype=np.int64)))

    nested_squeeze_names = {
        "/pair_scorer/Constant_41",
        "/pair_scorer/Constant_45",
        "/pair_scorer/Constant_81",
    }
    repaired_nested: set[str] = set()
    for control in graph.graph.node:
        if control.op_type != "If":
            continue
        for attribute in control.attribute:
            if attribute.type != onnx.AttributeProto.GRAPH:
                continue
            for nested in attribute.g.node:
                if nested.name not in nested_squeeze_names:
                    continue
                value = next(
                    (item for item in nested.attribute if item.name == "value"), None
                )
                if value is None or not np.array_equal(
                    numpy_helper.to_array(value.t), np.asarray([2], dtype=np.int64)
                ):
                    raise RuntimeError(f"unexpected nested squeeze axis in {nested.name}")
                value.t.CopyFrom(
                    numpy_helper.from_array(np.asarray([3], dtype=np.int64))
                )
                shape = attribute.g.output[0].type.tensor_type.shape
                del shape.dim[:]
                for dimension in ("batch", "query_count", "candidate_count"):
                    shape.dim.add().dim_param = dimension
                repaired_nested.add(nested.name)
    if repaired_nested != nested_squeeze_names:
        raise RuntimeError("could not identify all nested final-axis squeezes")

    onnx.save(graph, str(path))
    return {
        "operation": node.name,
        "legacy_exported_axis": 2,
        "runtime_axis": 3,
        "repaired_unsqueeze_constants": sorted(constant_names),
        "repaired_rotary_constants": sorted(rotary_constant_names),
        "repaired_rotary_concats": sorted(rotary_concat_names),
        "repaired_rotary_flatten_shapes": sorted(rotary_flatten_shape_names),
        "repaired_endpoint_difference_concat": "/pair_scorer/Concat_9",
        "repaired_multihead_width": {"legacy": 2, "runtime": 16},
        "repaired_squeeze_guards": sorted(squeeze_guard_names),
        "repaired_nested_squeezes": sorted(nested_squeeze_names),
        "reason": (
            "dynamic unsqueeze/torch.stack(dim=-1) legacy exporter correction "
            "for content-pool length and continuous length features"
        ),
    }


def graph_signature(hidden_size: int, mask_audit: dict, stack_repair: dict) -> dict:
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
            "candidate_indices": ["batch", "query_count", "candidate_count", 2],
            "candidate_mask": ["batch", "query_count", "candidate_count"],
        },
        "outputs": {
            "pair_logits": ["batch", "query_count", "candidate_count"],
            "compatibility": ["batch", "query_count", "candidate_count"],
            "legal_mask": ["batch", "query_count", "candidate_count"],
        },
        "path": "boundary_proposer.score_explicit_pairs+SparseBoundaryPairScorer",
        "marginals_added": "exactly-once-by-sparse-pair-scorer",
        "candidate_coordinates": "half-open-per-query",
        "mask_logit": -1.0e4,
        "runtime_contract": "batch>=1,text_length>=1,query_count>=1,candidate_count>=1",
        "mask_audit": mask_audit,
        "legacy_export_stack_repair": stack_repair,
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
    wrapper = BoundaryExplicitScorerGraph(model).float().eval()
    inputs = trace_inputs(
        seed=args.seed,
        batch=args.batch,
        length=args.text_length,
        queries=args.query_count,
        candidates=args.candidate_count,
        hidden=hidden_size,
    )
    with torch.inference_mode():
        expected = independently_composed_outputs(model, *inputs)
        actual = wrapper(*inputs)
    assert_output_parity(expected, actual)
    for name, value in zip(OUTPUT_NAMES, actual):
        if value.dtype != torch.bool:
            assert_finite(f"{name} export probe", value)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "boundary_explicit_scorer.onnx"
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
        "candidate_indices": {0: "batch", 1: "query_count", 2: "candidate_count"},
        "candidate_mask": {0: "batch", 1: "query_count", 2: "candidate_count"},
        "pair_logits": {0: "batch", 1: "query_count", 2: "candidate_count"},
        "compatibility": {0: "batch", 1: "query_count", 2: "candidate_count"},
        "legal_mask": {0: "batch", 1: "query_count", 2: "candidate_count"},
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
    stack_repair = repair_legacy_dynamic_stack_axis(out_path)
    mask_audit = normalize_onnx_mask_constants(out_path)
    proposed_manifest_entry = {
        out_path.name: {
            **graph_signature(hidden_size, mask_audit, stack_repair),
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
