#!/usr/bin/env python3
"""Export the learned GLiNER2.5 sparse relation scorer graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from boundary_relations import (
    INPUT_NAMES,
    OUTPUT_NAMES,
    BoundaryRelationsGraph,
    source_logits,
    validate_abi_inputs,
)
from common import (
    BASE_HF_REVISION,
    assert_finite,
    configure_determinism,
    load_reference_model,
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
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--text-length", type=int, default=31)
    parser.add_argument("--relation-count", type=int, default=3)
    parser.add_argument("--pair-count", type=int, default=67)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def validate_config(config: dict, wrapper: BoundaryRelationsGraph) -> None:
    if config.get("architecture") != "boundary":
        raise ValueError("relation scorer requires architecture='boundary'")
    if config.get("architecture_version") != 1:
        raise ValueError("relation scorer requires architecture_version=1")
    head = config.get("boundary_head", {})
    required = {
        "enable_relations": True,
        "directional_relation_states": True,
        "relation_biaffine_content": True,
        "relation_heads_per_type": 32,
        "relation_tails_per_type": 32,
        "relation_pair_cap": 64,
        "relation_argument_proposal_threshold": 0.2,
    }
    mismatches = [
        f"{name}={head.get(name)!r}, expected {expected!r}"
        for name, expected in required.items()
        if head.get(name) != expected
    ]
    if wrapper.relation_query_dim != 2 * wrapper.hidden_size:
        mismatches.append(
            f"loaded relation width={wrapper.relation_query_dim}, expected "
            f"{2 * wrapper.hidden_size}"
        )
    if mismatches:
        raise ValueError("unsupported boundary relation flags: " + "; ".join(mismatches))


def make_inputs(
    *, seed: int, batch: int, length: int, relations: int, pairs: int, hidden: int
) -> tuple[torch.Tensor, ...]:
    if min(batch, length, relations, pairs, hidden) < 1:
        raise ValueError("relation export dimensions must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    text = torch.randn(batch, length, hidden, generator=generator, dtype=torch.float32)
    relation = torch.randn(
        batch, relations, 2 * hidden, generator=generator, dtype=torch.float32
    )
    pair_ids = torch.arange(pairs, dtype=torch.int64)
    batch_index = pair_ids.remainder(batch)
    relation_index = pair_ids.remainder(relations)
    head_start = pair_ids.remainder(length)
    head_width = pair_ids.remainder(5) + 1
    head_end = torch.minimum(head_start + head_width, torch.tensor(length))
    head_start = torch.minimum(head_start, head_end - 1)
    tail_start = (length - 1 - pair_ids.remainder(length)).clamp_min(0)
    tail_width = pair_ids.remainder(7) + 1
    tail_end = torch.minimum(tail_start + tail_width, torch.tensor(length))
    tail_start = torch.minimum(tail_start, tail_end - 1)
    pair_mask = torch.ones(pairs, dtype=torch.bool)
    if pairs > 1:
        pair_mask[-1] = False
    result = (
        text,
        relation,
        batch_index,
        relation_index,
        head_start,
        head_end,
        tail_start,
        tail_end,
        pair_mask,
    )
    validate_abi_inputs(result)
    return result


def finite_graph_audit(path: Path) -> dict[str, object]:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    imports = {item.domain: item.version for item in model.opset_import}
    if imports.get("") != 17:
        raise AssertionError(f"expected default opset 17, found {imports.get('')}")
    if [value.name for value in model.graph.input] != list(INPUT_NAMES):
        raise AssertionError("relation graph input signature changed")
    if [value.name for value in model.graph.output] != list(OUTPUT_NAMES):
        raise AssertionError("relation graph output signature changed")

    counts = {"fp32_initializers": 0, "fp32_finite_float_constants": 0}
    for value_info in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        element_type = value_info.type.tensor_type.elem_type
        if element_type in (
            onnx.TensorProto.FLOAT16,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.BFLOAT16,
        ):
            raise AssertionError(
                f"ONNX value {value_info.name!r} has non-fp32 float type {element_type}"
            )
    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer)
        if array.dtype.kind in "fc":
            if array.dtype != np.float32:
                raise AssertionError(
                    f"initializer {initializer.name!r} is {array.dtype}, expected fp32"
                )
            if not np.isfinite(array).all():
                raise AssertionError(f"initializer {initializer.name!r} is non-finite")
            counts["fp32_initializers"] += 1
    for node in model.graph.node:
        for attribute in node.attribute:
            arrays: list[np.ndarray] = []
            if attribute.type == onnx.AttributeProto.TENSOR:
                arrays.append(numpy_helper.to_array(attribute.t))
            elif attribute.type == onnx.AttributeProto.TENSORS:
                arrays.extend(numpy_helper.to_array(value) for value in attribute.tensors)
            elif attribute.type == onnx.AttributeProto.FLOAT:
                arrays.append(np.asarray(attribute.f, dtype=np.float32))
            elif attribute.type == onnx.AttributeProto.FLOATS:
                arrays.append(np.asarray(attribute.floats, dtype=np.float32))
            for array in arrays:
                if array.dtype.kind in "fc":
                    if array.dtype != np.float32:
                        raise AssertionError(
                            f"constant {node.name or node.op_type!r} is {array.dtype}, expected fp32"
                        )
                    if not np.isfinite(array).all():
                        raise AssertionError(f"constant {node.name or node.op_type!r} is non-finite")
                    counts["fp32_finite_float_constants"] += 1
    return {
        **counts,
        "opset": imports[""],
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def main() -> None:
    args = parse_args()
    if args.opset != 17:
        raise ValueError("GLiNER2 bundles require opset 17")
    configure_determinism(args.seed)
    model, config = load_reference_model(args.model_dir, device=args.device)
    wrapper = BoundaryRelationsGraph(model).eval()
    validate_config(config, wrapper)
    inputs = make_inputs(
        seed=args.seed,
        batch=args.batch,
        length=args.text_length,
        relations=args.relation_count,
        pairs=args.pair_count,
        hidden=int(model.hidden_size),
    )
    with torch.inference_mode():
        original = source_logits(model.relation_scorer, inputs)
        wrapped = wrapper(*inputs)
    torch.testing.assert_close(wrapped, original, atol=1e-4, rtol=1e-3)
    assert_finite("original relation logits", original)
    assert_finite("wrapper relation logits", wrapped)
    masked = wrapped[~inputs[-1]]
    if bool((masked != 0.0).any()) or bool(torch.signbit(masked).any()):
        raise AssertionError("false pair_mask must produce positive zero")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "boundary_relations.onnx"
    dynamic_axes = {
        "text_states": {0: "batch", 1: "text_length"},
        "relation_query_states": {0: "batch", 1: "relation_count"},
        "batch_index": {0: "pair_count"},
        "relation_index": {0: "pair_count"},
        "head_start": {0: "pair_count"},
        "head_end": {0: "pair_count"},
        "tail_start": {0: "pair_count"},
        "tail_end": {0: "pair_count"},
        "pair_mask": {0: "pair_count"},
        "relation_logits": {0: "pair_count"},
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
    audit = finite_graph_audit(out_path)
    signature = {
        "inputs": {
            "text_states": ["batch", "text_length", wrapper.hidden_size],
            "relation_query_states": ["batch", "relation_count", 2 * wrapper.hidden_size],
            **{name: ["pair_count"] for name in INPUT_NAMES[2:]},
        },
        "outputs": {"relation_logits": ["pair_count"]},
        "coordinates": "half-open word indices; public ABI requires 0 <= start < end <= L",
        "routing": "public ABI requires valid batch/relation indices even when pair_mask=false",
        "masked_logit": 0.0,
        "empty_policy": "B/L/R/P must be positive; P=0 or R=0 caller bypasses ONNX",
        "text_state_semantics": "actual encoder text states, not boundary encoder states",
        "dynamic_length_divisor": True,
        "audit": audit,
    }
    proposed_manifest_entry = {
        "boundary_relations.onnx": {
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
