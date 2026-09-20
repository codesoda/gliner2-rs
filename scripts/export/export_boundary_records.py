#!/usr/bin/env python3
"""Export the dynamic-mode GLiNER2.5 inference record kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch

from boundary_records import (
    INPUT_NAMES,
    OUTPUT_NAMES,
    BoundaryRecordsGraph,
    validate_abi_inputs,
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
    parser.add_argument("--field-count", type=int, default=3)
    parser.add_argument("--candidate-count", type=int, default=7)
    parser.add_argument("--context-count", type=int, default=11)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def validate_config(config: dict, wrapper: BoundaryRecordsGraph) -> None:
    if config.get("architecture") != "boundary":
        raise ValueError("boundary records require architecture='boundary'")
    if config.get("architecture_version") != 1:
        raise ValueError("boundary records require architecture_version=1")
    head = config.get("boundary_head", {})
    required = {
        "candidate_pool": "shared",
        "enable_records": True,
        "record_dim": 128,
        "record_instance_queries": 32,
    }
    mismatches = [
        f"{name}={head.get(name)!r}, expected {expected!r}"
        for name, expected in required.items()
        if head.get(name) != expected
    ]
    if wrapper.record_dim != 128 or wrapper.instance_queries != 32:
        mismatches.append(
            f"loaded RecordHead dimensions={wrapper.record_dim}/{wrapper.instance_queries}, "
            "expected 128/32"
        )
    if mismatches:
        raise ValueError("unsupported boundary record flags: " + "; ".join(mismatches))


def export_inputs(
    *, seed: int, fields: int, candidates: int, context: int, seeds: int, hidden: int
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    field_query_states = torch.randn(fields, hidden, generator=generator)
    field_candidate_states = torch.randn(
        fields, candidates, hidden, generator=generator
    )
    field_candidate_mask = torch.ones(fields, candidates, dtype=torch.bool)
    if candidates > 1:
        field_candidate_mask[0, -1] = False
    context_states = torch.randn(context, hidden, generator=generator)
    context_mask = torch.ones(context, dtype=torch.bool)
    seed_states = torch.randn(seeds, hidden, generator=generator)
    seed_object_logits = torch.randn(seeds, generator=generator)
    mode = torch.tensor(0, dtype=torch.int64)
    result = (
        field_query_states,
        field_candidate_states,
        field_candidate_mask,
        context_states,
        context_mask,
        seed_states,
        seed_object_logits,
        mode,
    )
    validate_abi_inputs(result)
    return result


def count_control_flow(graph: onnx.GraphProto) -> int:
    count = 0
    for node in graph.node:
        if node.op_type == "If":
            count += 1
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                count += count_control_flow(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                count += sum(count_control_flow(item) for item in attribute.graphs)
    return count


def graph_signature(hidden_size: int, mask_audit: dict, if_nodes: int) -> dict:
    return {
        "inputs": {
            "field_query_states": ["field_count", hidden_size],
            "field_candidate_states": ["field_count", "candidate_count", hidden_size],
            "field_candidate_mask": ["field_count", "candidate_count"],
            "context_states": ["context_count", hidden_size],
            "context_mask": ["context_count"],
            "seed_states": ["seed_count", hidden_size],
            "seed_object_logits": ["seed_count"],
            "mode": [],
        },
        "outputs": {
            "instance_states": ["instance_count", hidden_size],
            "object_logits": ["instance_count"],
            "assignment_logits": [
                "field_count",
                "instance_count",
                "candidate_count_plus_absent",
            ],
        },
        "mode_values": {"natural": 0, "latent": 1, "anchorless": 2},
        "absent_column": 0,
        "assignment_scale": "none",
        "anchorless_instance_count": 32,
        "mask_logit": -1.0e4,
        "positive_dimension_contract": ["field_count", "candidate_count", "context_count", "seed_count"],
        "no_instance_policy": "caller bypass for natural/latent; anchorless still invokes graph",
        "onnx_if_nodes_recursive": if_nodes,
        "mask_audit": mask_audit,
    }


def main() -> None:
    args = parse_args()
    if args.opset != 17:
        raise ValueError("GLiNER2 bundles require opset 17")
    if min(
        args.field_count,
        args.candidate_count,
        args.context_count,
        args.seed_count,
    ) < 1:
        raise ValueError("record export dimensions must all be positive")
    configure_determinism(args.seed)
    model, config = load_reference_model(args.model_dir, device=args.device)
    wrapper = BoundaryRecordsGraph(model).eval()
    validate_config(config, wrapper)
    inputs = export_inputs(
        seed=args.seed,
        fields=args.field_count,
        candidates=args.candidate_count,
        context=args.context_count,
        seeds=args.seed_count,
        hidden=int(model.hidden_size),
    )
    scripted = torch.jit.script(wrapper)
    if str(scripted.graph).count("prim::If") < 2:
        raise AssertionError("TorchScript did not retain both dynamic mode branches")

    with torch.inference_mode():
        natural = scripted(*inputs)
        latent = scripted(*((*inputs[:-1], torch.tensor(1, dtype=torch.int64))))
        empty_context = list(inputs)
        empty_context[3] = torch.zeros(1, int(model.hidden_size), dtype=torch.float32)
        empty_context[4] = torch.zeros(1, dtype=torch.bool)
        empty_context[7] = torch.tensor(2, dtype=torch.int64)
        anchorless = scripted(*tuple(empty_context))
    if not torch.equal(natural[1], inputs[6]):
        raise AssertionError("natural object logits must be the original seed pair logits")
    if natural[0].shape[0] != args.seed_count or latent[0].shape[0] != args.seed_count:
        raise AssertionError("natural/latent graph unexpectedly capped seed instances")
    if anchorless[0].shape[0] != wrapper.instance_queries:
        raise AssertionError("anchorless graph did not emit all learned queries")
    if not torch.equal(anchorless[0], wrapper.instance_embed):
        raise AssertionError("empty anchorless context must return exact instance embeddings")
    for mode_name, values in (
        ("natural", natural),
        ("latent", latent),
        ("anchorless", anchorless),
    ):
        for name, value in zip(OUTPUT_NAMES, values):
            assert_finite(f"{mode_name}/{name}", value)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "boundary_records.onnx"
    dynamic_axes = {
        "field_query_states": {0: "field_count"},
        "field_candidate_states": {0: "field_count", 1: "candidate_count"},
        "field_candidate_mask": {0: "field_count", 1: "candidate_count"},
        "context_states": {0: "context_count"},
        "context_mask": {0: "context_count"},
        "seed_states": {0: "seed_count"},
        "seed_object_logits": {0: "seed_count"},
        "instance_states": {0: "instance_count"},
        "object_logits": {0: "instance_count"},
        "assignment_logits": {
            0: "field_count",
            1: "instance_count",
            2: "candidate_count_plus_absent",
        },
    }
    torch.onnx.export(
        scripted,
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
    graph = onnx.load(str(out_path))
    onnx.checker.check_model(graph)
    if_nodes = count_control_flow(graph.graph)
    if if_nodes < 2:
        raise AssertionError(
            "ONNX export specialized a mode branch; refusing a three-mode-incomplete graph"
        )
    signature = graph_signature(int(model.hidden_size), mask_audit, if_nodes)
    proposed_manifest_entry = {
        "boundary_records.onnx": {
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
