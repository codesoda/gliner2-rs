#!/usr/bin/env python3
"""Validate boundary_records.onnx against untouched RecordHead.forward_group."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper

SCRIPT_DIR = Path(__file__).resolve().parent
PARITY_DIR = SCRIPT_DIR.parent / "parity"
if str(PARITY_DIR) not in sys.path:
    sys.path.insert(0, str(PARITY_DIR))

from boundary_records import (  # noqa: E402
    INPUT_NAMES,
    OUTPUT_NAMES,
    BoundaryRecordsGraph,
    assert_wrapper_matches_group,
    padded_source_assignments,
    prepare_group_inputs,
)
from common import (  # noqa: E402
    BASE_HF_REVISION,
    MASK_LOGIT,
    compare_arrays,
    configure_determinism,
    load_reference_model,
)
from export_boundary_records import count_control_flow, validate_config  # noqa: E402
from gen_boundary_goldens import build_expected_batch, build_schema  # noqa: E402
from gliner2.models.outputs import CandidateTensorBatch  # noqa: E402
from gliner2.processing.records import (  # noqa: E402
    FieldCardinality,
    RecordFieldSpec,
    RecordSpec,
)

DEFAULT_MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--fastino--gliner2.5-base-v1"
    / "snapshots"
    / BASE_HF_REVISION
)
REAL_CASES = (
    "json_natural_people",
    "json_latent_products",
    "json_anchorless_list",
    "json_natural_choice",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument(
        "--onnx-path", default="onnx/gliner2.5-base-v1/boundary_records.onnx"
    )
    parser.add_argument("--golden-dir", default="fixtures/gliner2.5-base-v1")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--report-json")
    return parser.parse_args()


def iter_graphs(graph: onnx.GraphProto):
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from iter_graphs(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child in attribute.graphs:
                    yield from iter_graphs(child)


def graph_audit(path: Path) -> dict[str, int]:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    imports = {item.domain: item.version for item in model.opset_import}
    if imports.get("") != 17:
        raise AssertionError(f"expected default opset 17, found {imports.get('')}")

    counts: Counter[str] = Counter()
    for graph in iter_graphs(model.graph):
        counts["graphs_recursive"] += 1
        for value_info in (*graph.input, *graph.output, *graph.value_info):
            tensor_type = value_info.type.tensor_type
            if tensor_type.elem_type in (
                onnx.TensorProto.FLOAT16,
                onnx.TensorProto.DOUBLE,
                onnx.TensorProto.BFLOAT16,
            ):
                raise AssertionError(
                    f"ONNX value {value_info.name!r} has non-fp32 float type "
                    f"{tensor_type.elem_type}"
                )
        for initializer in graph.initializer:
            value = numpy_helper.to_array(initializer)
            if value.dtype.kind in "fc":
                counts["float_initializers"] += 1
                if value.dtype != np.float32:
                    raise AssertionError(
                        f"initializer {initializer.name!r} is {value.dtype}, expected fp32"
                    )
                if not np.isfinite(value).all():
                    raise AssertionError(f"initializer {initializer.name!r} is non-finite")
        for node in graph.node:
            counts[f"op_{node.op_type}"] += 1
            for attribute in node.attribute:
                arrays: list[np.ndarray] = []
                if attribute.type == onnx.AttributeProto.TENSOR:
                    arrays.append(numpy_helper.to_array(attribute.t))
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    arrays.extend(numpy_helper.to_array(item) for item in attribute.tensors)
                elif attribute.type == onnx.AttributeProto.FLOAT:
                    arrays.append(np.asarray(attribute.f, dtype=np.float32))
                elif attribute.type == onnx.AttributeProto.FLOATS:
                    arrays.append(np.asarray(attribute.floats, dtype=np.float32))
                for value in arrays:
                    if value.dtype.kind in "fc":
                        counts["float_constants"] += 1
                        if value.dtype != np.float32:
                            raise AssertionError(
                                f"node {node.name or node.op_type!r} has {value.dtype} "
                                "floating constants, expected fp32"
                            )
                        if not np.isfinite(value).all():
                            raise AssertionError(
                                f"node {node.name or node.op_type!r} has non-finite constants"
                            )
    if count_control_flow(model.graph) < 2:
        raise AssertionError("record graph does not retain both dynamic mode If nodes")
    return dict(sorted(counts.items()))


def fixture_candidates(data: np.lib.npyio.NpzFile) -> CandidateTensorBatch:
    proposal = (
        torch.from_numpy(data["boundary_head_0_proposal_logits"].copy()).float()
        if "boundary_head_0_proposal_logits" in data.files
        else None
    )
    return CandidateTensorBatch(
        indices=torch.from_numpy(data["boundary_head_0_indices"].copy()).long(),
        proposal_logits=proposal,
        pair_logits=torch.from_numpy(data["boundary_head_0_pair_logits"].copy()).float(),
        valid_mask=torch.from_numpy(data["boundary_head_0_valid_mask"].copy()).bool(),
        query_mask=torch.from_numpy(data["boundary_head_0_query_mask"].copy()).bool(),
        candidate_states=torch.from_numpy(
            data["boundary_head_0_candidate_states"].copy()
        ).float(),
    )


def source_group_and_states(
    head: torch.nn.Module,
    spec: RecordSpec,
    query_states: torch.Tensor,
    candidates: CandidateTensorBatch,
) -> tuple[Any, torch.Tensor]:
    captured: list[torch.Tensor] = []

    def capture_instance_input(_module, args, _output):
        captured.append(args[0].detach().clone())

    handle = head.inst_proj.register_forward_hook(capture_instance_input)
    try:
        with torch.inference_mode():
            group = head.forward_group(spec, query_states, candidates, 0)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise AssertionError(f"forward_group called inst_proj {len(captured)} times")
    return group, captured[0]


def numpy_inputs(inputs: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
    return {
        name: value.detach().cpu().numpy()
        for name, value in zip(INPUT_NAMES, inputs)
    }


def numpy_outputs(outputs: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
    return {
        name: value.detach().cpu().numpy()
        for name, value in zip(OUTPUT_NAMES, outputs)
    }


def expected_outputs(
    group: Any, instance_states: torch.Tensor, candidate_count: int
) -> dict[str, np.ndarray]:
    values = (
        instance_states,
        group.object_logits,
        padded_source_assignments(group.assign_logits, candidate_count),
    )
    return numpy_outputs(values)


def run_onnx(
    session: ort.InferenceSession, inputs: tuple[torch.Tensor, ...]
) -> dict[str, np.ndarray]:
    values = session.run(list(OUTPUT_NAMES), numpy_inputs(inputs))
    return dict(zip(OUTPUT_NAMES, values))


def update_stats(
    stats: dict[str, dict[str, float | int]], name: str, report: dict
) -> None:
    stage = stats.setdefault(name, {"comparisons": 0, "max_abs": 0.0, "max_rel": 0.0})
    stage["comparisons"] = int(stage["comparisons"]) + 1
    stage["max_abs"] = max(float(stage["max_abs"]), float(report["max_abs"]))
    stage["max_rel"] = max(float(stage["max_rel"]), float(report["max_rel"]))


def compare_outputs(
    label: str,
    expected: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    stats: dict[str, dict[str, float | int]],
    *,
    atol: float,
    rtol: float,
) -> None:
    for name in OUTPUT_NAMES:
        report = compare_arrays(
            f"{label}/{name}", expected[name], actual[name], atol=atol, rtol=rtol
        )
        update_stats(stats, name, report)


def assert_masks_and_shapes(
    label: str,
    inputs: tuple[torch.Tensor, ...],
    outputs: dict[str, np.ndarray],
    *,
    instance_queries: int,
) -> dict[str, int]:
    field_mask = inputs[2].cpu().numpy()
    mode = int(inputs[7].item())
    expected_instances = instance_queries if mode == 2 else inputs[5].shape[0]
    if outputs["instance_states"].shape[0] != expected_instances:
        raise AssertionError(f"{label}: wrong dynamic instance count")
    expected_shape = (field_mask.shape[0], expected_instances, field_mask.shape[1] + 1)
    if outputs["assignment_logits"].shape != expected_shape:
        raise AssertionError(
            f"{label}: assignment shape {outputs['assignment_logits'].shape} != {expected_shape}"
        )
    invalid = ~np.broadcast_to(
        field_mask[:, None, :],
        (field_mask.shape[0], expected_instances, field_mask.shape[1]),
    )
    candidate_logits = outputs["assignment_logits"][..., 1:]
    if invalid.any() and not np.array_equal(
        candidate_logits[invalid],
        np.full(np.count_nonzero(invalid), MASK_LOGIT, dtype=np.float32),
    ):
        raise AssertionError(f"{label}: invalid padded columns are not exactly -1e4")
    if not all(np.isfinite(value).all() for value in outputs.values()):
        raise AssertionError(f"{label}: ONNX output is non-finite")
    return {
        "invalid_assignment_columns": int(np.count_nonzero(invalid)),
        "instances": expected_instances,
    }


def synthetic_case(
    seed: int,
    *,
    mode: str,
    counts: tuple[int, ...],
    hidden: int,
    duplicate: bool = False,
) -> tuple[RecordSpec, torch.Tensor, CandidateTensorBatch]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    fields = len(counts)
    candidates = max(max(counts, default=0), 1)
    query_states = torch.randn(fields, hidden, generator=generator)
    states = torch.randn(1, fields, candidates, hidden, generator=generator)
    if duplicate and fields > 1 and counts[0] and counts[1]:
        states[0, 1, 0] = states[0, 0, 0]
    pair_logits = torch.randn(1, fields, candidates, generator=generator)
    valid = torch.zeros(1, fields, candidates, dtype=torch.bool)
    indices = torch.zeros(1, fields, candidates, 2, dtype=torch.int64)
    for field_index, count in enumerate(counts):
        valid[0, field_index, :count] = True
        if count:
            indices[0, field_index, :count, 0] = torch.arange(count)
            indices[0, field_index, :count, 1] = torch.arange(1, count + 1)
    candidates_batch = CandidateTensorBatch(
        indices=indices,
        proposal_logits=None,
        pair_logits=pair_logits,
        valid_mask=valid,
        query_mask=torch.ones(1, fields, dtype=torch.bool),
        candidate_states=states,
    )
    field_specs = tuple(
        RecordFieldSpec(
            query_id=index,
            name=f"field_{index}",
            role_index=index,
            cardinality=FieldCardinality.OPTIONAL_ONE,
            is_anchor=mode == "natural" and index == 0,
            exclusive=True,
        )
        for index in range(fields)
    )
    spec = RecordSpec(
        task_index=0,
        task_name=f"synthetic_{mode}",
        task_type="json_structures",
        mode=mode,
        fields=field_specs,
        anchor_query_id=0 if mode == "natural" else None,
    )
    return spec, query_states, candidates_batch


def main() -> None:
    args = parse_args()
    if not ort.__version__.startswith("1.20."):
        raise RuntimeError(
            f"validation requires ONNX Runtime 1.20.x, found {ort.__version__}"
        )
    configure_determinism(args.seed)
    onnx_path = Path(args.onnx_path)
    audit = graph_audit(onnx_path)
    model, config = load_reference_model(args.model_dir)
    wrapper = BoundaryRecordsGraph(model).eval()
    validate_config(config, wrapper)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if tuple(item.name for item in session.get_inputs()) != INPUT_NAMES:
        raise AssertionError("ONNX record input signature changed")
    if tuple(item.name for item in session.get_outputs()) != OUTPUT_NAMES:
        raise AssertionError("ONNX record output signature changed")

    golden_stats: dict[str, dict[str, float | int]] = {}
    synthetic_stats: dict[str, dict[str, float | int]] = {}
    counts: Counter[str] = Counter()
    mask_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    golden_dir = Path(args.golden_dir)

    for case_id in REAL_CASES:
        metadata = json.loads((golden_dir / f"{case_id}.json").read_text())
        schema = build_schema(model, metadata["schema_spec"])
        batch = build_expected_batch(model, metadata["text"], schema)
        if len(batch.record_specs[0]) != 1:
            raise AssertionError(f"{case_id}: expected exactly one actual RecordSpec")
        spec = next(iter(batch.record_specs[0].values()))
        with np.load(golden_dir / f"{case_id}.npz", allow_pickle=False) as data:
            query_states = torch.from_numpy(data["query_states"][0].copy()).float()
            candidates = fixture_candidates(data)
            captured_instance_states = torch.from_numpy(
                data["record_inst_proj_0_input"].copy()
            ).float()
        group, instance_states = source_group_and_states(
            model.record_decoder, spec, query_states, candidates
        )
        torch.testing.assert_close(
            instance_states,
            captured_instance_states,
            atol=args.atol,
            rtol=args.rtol,
            msg=lambda message: (
                f"{case_id}: live forward_group instance states differ from "
                f"captured record_inst_proj_0_input: {message}"
            ),
        )
        inputs = prepare_group_inputs(spec, query_states, candidates)
        with torch.inference_mode():
            wrapped = wrapper(*inputs)
        assert_wrapper_matches_group(
            group,
            instance_states,
            inputs,
            wrapped,
            atol=args.atol,
            rtol=args.rtol,
        )
        expected = expected_outputs(group, instance_states, inputs[1].shape[1])
        actual = run_onnx(session, inputs)
        compare_outputs(
            case_id,
            expected,
            actual,
            golden_stats,
            atol=args.atol,
            rtol=args.rtol,
        )
        mask_counts.update(
            assert_masks_and_shapes(
                case_id, inputs, actual, instance_queries=wrapper.instance_queries
            )
        )
        mode_counts[spec.mode] += 1
        counts["real_forward_group_cases"] += 1
        print(
            f"{case_id}: mode={spec.mode} F={inputs[0].shape[0]} "
            f"C={inputs[1].shape[1]} M={inputs[3].shape[0]} "
            f"seedNi={inputs[5].shape[0]} J={actual['instance_states'].shape[0]} ok",
            flush=True,
        )

    synthetic_specs = (
        ("natural_gt32", "natural", (41, 0, 7), False),
        ("latent_duplicates", "latent", (9, 2, 0, 5), True),
        ("anchorless_uneven", "anchorless", (0, 4), False),
        ("anchorless_empty", "anchorless", (0, 0, 0), False),
    )
    for index, (label, selected_mode, valid_counts, duplicate) in enumerate(synthetic_specs):
        spec, query_states, candidates = synthetic_case(
            args.seed + 100 + index,
            mode=selected_mode,
            counts=valid_counts,
            hidden=int(model.hidden_size),
            duplicate=duplicate,
        )
        group, instance_states = source_group_and_states(
            model.record_decoder, spec, query_states, candidates
        )
        inputs = prepare_group_inputs(spec, query_states, candidates)
        if label == "latent_duplicates":
            if inputs[5].shape[0] != sum(valid_counts):
                raise AssertionError("latent field-major seeds lost duplicate candidates")
            if not torch.equal(inputs[5][0], inputs[5][valid_counts[0]]):
                raise AssertionError("latent duplicate candidate identity was not retained")
        with torch.inference_mode():
            wrapped = wrapper(*inputs)
        assert_wrapper_matches_group(
            group,
            instance_states,
            inputs,
            wrapped,
            atol=args.atol,
            rtol=args.rtol,
        )
        expected = expected_outputs(group, instance_states, inputs[1].shape[1])
        actual = run_onnx(session, inputs)
        compare_outputs(
            label,
            expected,
            actual,
            synthetic_stats,
            atol=args.atol,
            rtol=args.rtol,
        )
        mask_counts.update(
            assert_masks_and_shapes(
                label, inputs, actual, instance_queries=wrapper.instance_queries
            )
        )
        if label == "anchorless_empty" and not np.array_equal(
            actual["instance_states"], model.record_decoder.instance_embed.detach().numpy()
        ):
            raise AssertionError("empty anchorless context did not return exact learned queries")
        if label == "natural_gt32" and actual["instance_states"].shape[0] != 41:
            raise AssertionError("natural instances were artificially capped at 32")
        mode_counts[selected_mode] += 1
        counts["synthetic_dynamic_cases"] += 1
        print(
            f"{label}: mode={selected_mode} F={inputs[0].shape[0]} "
            f"C={inputs[1].shape[1]} M={inputs[3].shape[0]} "
            f"seedNi={inputs[5].shape[0]} J={actual['instance_states'].shape[0]} ok",
            flush=True,
        )

    bypass_status: dict[str, str] = {}
    for index, selected_mode in enumerate(("natural", "latent")):
        spec, query_states, candidates = synthetic_case(
            args.seed + 200 + index,
            mode=selected_mode,
            counts=(0, 0),
            hidden=int(model.hidden_size),
        )
        try:
            prepare_group_inputs(spec, query_states, candidates)
        except ValueError as exc:
            bypass_status[selected_mode] = str(exc)
        else:
            raise AssertionError(f"{selected_mode} Ni=0 contract was not rejected")
        counts["no_instance_bypass_cases"] += 1

    if counts != Counter(
        {
            "real_forward_group_cases": 4,
            "synthetic_dynamic_cases": 4,
            "no_instance_bypass_cases": 2,
        }
    ):
        raise AssertionError(f"unexpected validation counts: {dict(counts)}")
    if not all(mode_counts[name] > 0 for name in ("natural", "latent", "anchorless")):
        raise AssertionError(f"not all modes ran through one session: {dict(mode_counts)}")

    report = {
        "onnx": str(onnx_path),
        "onnxruntime": ort.__version__,
        "atol": args.atol,
        "rtol": args.rtol,
        "graph_audit": audit,
        "case_counts": dict(sorted(counts.items())),
        "same_graph_dynamic_mode_runs": dict(sorted(mode_counts.items())),
        "mask_and_instance_checks": dict(sorted(mask_counts.items())),
        "zero_dimension_policy": (
            "F/C/M/seed-Ni are positive. Rust bypasses natural/latent Ni=0; "
            "anchorless empty context is represented by one zero row with mask=false."
        ),
        "no_instance_bypass_status": bypass_status,
        "onnx_vs_untouched_forward_group_real_per_output": golden_stats,
        "onnx_vs_untouched_forward_group_synthetic_per_output": synthetic_stats,
    }
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
