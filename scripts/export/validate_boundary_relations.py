#!/usr/bin/env python3
"""Validate boundary_relations.onnx against the untouched sparse scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from boundary_relations import (
    INPUT_NAMES,
    OUTPUT_NAMES,
    BoundaryRelationsGraph,
    source_logits,
    validate_abi_inputs,
)
from common import BASE_HF_REVISION, configure_determinism, load_reference_model
from export_boundary_relations import finite_graph_audit, validate_config

DEFAULT_MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--fastino--gliner2.5-base-v1"
    / "snapshots"
    / BASE_HF_REVISION
)
REAL_CASES = (
    "relation_employment",
    "relation_founded",
    "relation_location",
    "relation_multiple_types",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument(
        "--onnx-path", default="onnx/gliner2.5-base-v1/boundary_relations.onnx"
    )
    parser.add_argument("--golden-dir", default="fixtures/gliner2.5-base-v1")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--confidence-atol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--report-json")
    return parser.parse_args()


def fixture_inputs(data: np.lib.npyio.NpzFile) -> tuple[torch.Tensor, ...]:
    prefix = "relation_scorer_0_"
    fixture_names = {
        "relation_query_states": "relation_states",
    }
    return tuple(
        torch.from_numpy(data[prefix + fixture_names.get(name, name)].copy())
        for name in INPUT_NAMES
    )


def numpy_inputs(inputs: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
    return {
        name: value.detach().cpu().numpy()
        for name, value in zip(INPUT_NAMES, inputs)
    }


def run_onnx(
    session: ort.InferenceSession, inputs: tuple[torch.Tensor, ...]
) -> np.ndarray:
    return session.run(list(OUTPUT_NAMES), numpy_inputs(inputs))[0]


def synthetic_inputs(
    seed: int,
    *,
    batch: int,
    length: int,
    relations: int,
    pairs: int,
    hidden: int,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    text = torch.randn(batch, length, hidden, generator=generator)
    relation = torch.randn(batch, relations, 2 * hidden, generator=generator)
    batch_index = torch.arange(pairs, dtype=torch.int64).remainder(batch)
    relation_index = torch.arange(pairs, dtype=torch.int64).remainder(relations)
    head_start = torch.empty(pairs, dtype=torch.int64)
    head_end = torch.empty(pairs, dtype=torch.int64)
    tail_start = torch.empty(pairs, dtype=torch.int64)
    tail_end = torch.empty(pairs, dtype=torch.int64)
    for index in range(pairs):
        if index % 4 == 0:  # self span
            hs = ts = index % length
            he = te = min(length, hs + 1)
        elif index % 4 == 1:  # reverse textual order
            ts = index % length
            te = min(length, ts + 1)
            hs = max(ts, length - 1 - ts)
            he = min(length, hs + 1)
        elif index % 4 == 2:  # long head/content pooling
            hs, he = 0, length
            ts = max(0, length - 1)
            te = length
        else:
            hs = index % length
            he = min(length, hs + 1 + index % 7)
            hs = min(hs, he - 1)
            ts = (index * 3) % length
            te = min(length, ts + 1 + index % 5)
        head_start[index], head_end[index] = hs, he
        tail_start[index], tail_end[index] = ts, te
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


def compare(
    label: str,
    expected: np.ndarray,
    actual: np.ndarray,
    stats: dict[str, float | int],
    failures: list[str],
    *,
    atol: float,
    rtol: float,
    confidence_atol: float | None = None,
) -> None:
    stats["comparisons"] = int(stats.get("comparisons", 0)) + 1
    if expected.shape != actual.shape:
        failures.append(f"{label}: shape {actual.shape} != {expected.shape}")
        stats["failure_count"] = int(stats.get("failure_count", 0)) + 1
        return
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        failures.append(f"{label}: non-finite values")
        stats["failure_count"] = int(stats.get("failure_count", 0)) + 1
        return
    absolute = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    relative = absolute / np.maximum(np.abs(expected.astype(np.float64)), 1e-12)
    stats["max_abs"] = max(float(stats.get("max_abs", 0.0)), float(absolute.max(initial=0)))
    stats["max_rel"] = max(float(stats.get("max_rel", 0.0)), float(relative.max(initial=0)))
    allowed = atol + rtol * np.abs(expected.astype(np.float64))
    if not np.all(absolute <= allowed):
        flat = int(np.argmax(absolute - allowed))
        index = np.unravel_index(flat, absolute.shape)
        failures.append(
            f"{label}: mismatch at {index}, expected={expected[index]!r}, "
            f"actual={actual[index]!r}, abs={absolute[index]:.9g}, allowed={allowed[index]:.9g}"
        )
        stats["failure_count"] = int(stats.get("failure_count", 0)) + 1
    if confidence_atol is not None:
        expected_confidence = 1.0 / (1.0 + np.exp(-np.clip(expected, -80, 80)))
        actual_confidence = 1.0 / (1.0 + np.exp(-np.clip(actual, -80, 80)))
        confidence_error = np.abs(actual_confidence - expected_confidence)
        stats["max_confidence_abs"] = max(
            float(stats.get("max_confidence_abs", 0.0)),
            float(confidence_error.max(initial=0)),
        )
        if np.any(confidence_error > confidence_atol):
            failures.append(
                f"{label}: confidence max_abs={confidence_error.max():.9g} "
                f"> {confidence_atol}"
            )
            stats["failure_count"] = int(stats.get("failure_count", 0)) + 1


def main() -> None:
    args = parse_args()
    if ort.__version__ != "1.20.1":
        raise RuntimeError(
            f"validation requires ONNX Runtime 1.20.1, found {ort.__version__}"
        )
    configure_determinism(args.seed)
    onnx_path = Path(args.onnx_path)
    audit = finite_graph_audit(onnx_path)
    model, config = load_reference_model(args.model_dir)
    wrapper = BoundaryRelationsGraph(model).eval()
    validate_config(config, wrapper)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if tuple(value.name for value in session.get_inputs()) != INPUT_NAMES:
        raise AssertionError("ONNX relation input signature changed")
    if tuple(value.name for value in session.get_outputs()) != OUTPUT_NAMES:
        raise AssertionError("ONNX relation output signature changed")

    failures: list[str] = []
    frozen_stats: dict[str, float | int] = {"failure_count": 0}
    wrapper_stats: dict[str, float | int] = {"failure_count": 0}
    onnx_stats: dict[str, float | int] = {"failure_count": 0}
    case_reports: list[dict[str, object]] = []
    golden_dir = Path(args.golden_dir)

    for case_id in REAL_CASES:
        with np.load(golden_dir / f"{case_id}.npz", allow_pickle=False) as data:
            inputs = fixture_inputs(data)
            frozen = data["relation_scorer_0_logits"].copy()
        validate_abi_inputs(inputs)
        with torch.inference_mode():
            original = source_logits(model.relation_scorer, inputs)
            wrapped = wrapper(*inputs)
        original_np = original.detach().cpu().numpy()
        wrapped_np = wrapped.detach().cpu().numpy()
        actual = run_onnx(session, inputs)
        compare(
            f"{case_id}/untouched_vs_frozen",
            frozen,
            original_np,
            frozen_stats,
            failures,
            atol=args.atol,
            rtol=args.rtol,
        )
        compare(
            f"{case_id}/wrapper_vs_untouched",
            original_np,
            wrapped_np,
            wrapper_stats,
            failures,
            atol=args.atol,
            rtol=args.rtol,
        )
        compare(
            f"{case_id}/onnx_vs_untouched",
            original_np,
            actual,
            onnx_stats,
            failures,
            atol=args.atol,
            rtol=args.rtol,
            confidence_atol=args.confidence_atol,
        )
        mask = inputs[-1].numpy()
        if np.any(~mask) and not np.all(
            actual[~mask].view(np.uint32) == np.float32(0.0).view(np.uint32)
        ):
            failures.append(f"{case_id}: false pair_mask output is not exact positive zero")
        case_reports.append(
            {
                "case_id": case_id,
                "B": inputs[0].shape[0],
                "L": inputs[0].shape[1],
                "R": inputs[1].shape[1],
                "P": inputs[2].shape[0],
                "kind": "full_relation_golden",
            }
        )

    synthetic_shapes = (
        (1, 1, 1, 1),
        (1, 7, 3, 9),
        (2, 31, 1, 67),
        (2, 129, 3, 97),
    )
    for index, (batch, length, relations, pairs) in enumerate(synthetic_shapes):
        label = f"synthetic_B{batch}_L{length}_R{relations}_P{pairs}"
        inputs = synthetic_inputs(
            args.seed + 100 + index,
            batch=batch,
            length=length,
            relations=relations,
            pairs=pairs,
            hidden=int(model.hidden_size),
        )
        with torch.inference_mode():
            original = source_logits(model.relation_scorer, inputs)
            wrapped = wrapper(*inputs)
        original_np = original.detach().cpu().numpy()
        compare(
            f"{label}/wrapper_vs_untouched",
            original_np,
            wrapped.detach().cpu().numpy(),
            wrapper_stats,
            failures,
            atol=args.atol,
            rtol=args.rtol,
        )
        actual = run_onnx(session, inputs)
        compare(
            f"{label}/onnx_vs_untouched",
            original_np,
            actual,
            onnx_stats,
            failures,
            atol=args.atol,
            rtol=args.rtol,
            confidence_atol=args.confidence_atol,
        )
        mask = inputs[-1].numpy()
        if np.any(~mask) and not np.all(
            actual[~mask].view(np.uint32) == np.float32(0.0).view(np.uint32)
        ):
            failures.append(f"{label}: false pair_mask output is not exact positive zero")
        case_reports.append(
            {
                "case_id": label,
                "B": batch,
                "L": length,
                "R": relations,
                "P": pairs,
                "kind": "dynamic_shape_and_position_divisor",
            }
        )

    # The raw source scorer safely clamps invalid routing for gathers and masks
    # every such result to +0.0. This is graph hardening only: both Python ABI
    # validation and the public Rust wrapper reject these inputs before ORT.
    invalid_routing = list(
        synthetic_inputs(
            args.seed + 200,
            batch=2,
            length=7,
            relations=3,
            pairs=4,
            hidden=int(model.hidden_size),
        )
    )
    invalid_routing[2] = torch.tensor([-1, 2, -1, 2], dtype=torch.int64)
    invalid_routing[3] = torch.tensor([-1, 3, 3, -1], dtype=torch.int64)
    invalid_routing_inputs = tuple(invalid_routing)
    try:
        validate_abi_inputs(invalid_routing_inputs)
    except ValueError:
        pass
    else:
        failures.append("invalid routing unexpectedly passed the public Python ABI")
    with torch.inference_mode():
        invalid_original = source_logits(
            model.relation_scorer,
            invalid_routing_inputs,
            enforce_public_abi=False,
        )
        invalid_wrapped = wrapper(*invalid_routing_inputs)
    invalid_expected = invalid_original.detach().cpu().numpy()
    invalid_actual = run_onnx(session, invalid_routing_inputs)
    compare(
        "invalid_routing/wrapper_vs_untouched",
        invalid_expected,
        invalid_wrapped.detach().cpu().numpy(),
        wrapper_stats,
        failures,
        atol=args.atol,
        rtol=args.rtol,
    )
    compare(
        "invalid_routing/onnx_vs_untouched",
        invalid_expected,
        invalid_actual,
        onnx_stats,
        failures,
        atol=args.atol,
        rtol=args.rtol,
        confidence_atol=args.confidence_atol,
    )
    if not np.all(invalid_actual.view(np.uint32) == np.float32(0.0).view(np.uint32)):
        failures.append("invalid routing did not produce exact positive-zero logits")
    case_reports.append(
        {
            "case_id": "invalid_routing_raw_graph_probe",
            "B": 2,
            "L": 7,
            "R": 3,
            "P": 4,
            "kind": "raw_graph_safe_clamp_not_public_contract",
        }
    )

    failure_count = len(failures)
    report = {
        "onnx": str(onnx_path),
        "onnxruntime": ort.__version__,
        "atol": args.atol,
        "rtol": args.rtol,
        "confidence_atol": args.confidence_atol,
        "graph_audit": audit,
        "real_case_count": len(REAL_CASES),
        "synthetic_case_count": len(synthetic_shapes) + 1,
        "position_divisor_lengths": [shape[1] for shape in synthetic_shapes],
        "dynamic_axes_exercised": {
            "B": sorted({shape[0] for shape in synthetic_shapes}),
            "L": [shape[1] for shape in synthetic_shapes],
            "R": sorted({shape[2] for shape in synthetic_shapes}),
            "P": [shape[3] for shape in synthetic_shapes],
        },
        "untouched_vs_frozen": frozen_stats,
        "wrapper_vs_untouched": wrapper_stats,
        "onnx_vs_untouched": onnx_stats,
        "failure_count": failure_count,
        "failures": failures,
        "cases": case_reports,
        "empty_policy": "B/L/R/P >= 1; P=0 or no relation query caller bypasses ORT",
        "masked_policy": "valid coordinates/routing required; pair_mask=false returns +0.0",
    }
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failure_count:
        raise AssertionError(f"relation validation had {failure_count} failures")


if __name__ == "__main__":
    main()
