#!/usr/bin/env python3
"""Validate boundary_marginals.onnx against goldens and the untouched oracle."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from boundary_marginals import (
    OUTPUT_NAMES,
    BoundaryMarginalGraph,
    assert_wrapper_matches_oracle,
    oracle_outputs,
)
from common import (
    BASE_HF_REVISION,
    compare_arrays,
    configure_determinism,
    load_reference_model,
)

import onnx
from onnx import numpy_helper

DEFAULT_MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--fastino--gliner2.5-base-v1"
    / "snapshots"
    / BASE_HF_REVISION
)
PREFIX_COORDINATE_ATOL = 1.1e-6

GOLDEN_KEYS = {
    "boundary_states": "boundary_encoder_0_states",
    "boundary_mask": "boundary_encoder_0_mask",
    "start_logits": "marginals_0_start_logits",
    "end_logits": "marginals_0_end_logits",
    "inside_logits": "marginals_0_inside_logits",
    "inside_prefix": "marginals_0_inside_prefix",
    "inside_prefix_mean": "marginals_0_inside_prefix_mean",
    "start_all": "pool_start_projection_0_output",
    "end_all": "pool_end_projection_0_output",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument(
        "--onnx-path",
        default="onnx/gliner2.5-base-v1/boundary_marginals.onnx",
    )
    parser.add_argument("--golden-dir", default="fixtures/gliner2.5-base-v1")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--report-json")
    parser.add_argument(
        "--probe-unsupported-axes",
        action="store_true",
        help=(
            "opt in to isolated ORT L=0 diagnostics outside the supported contract; "
            "may trigger an OS crash dialog (supported Q=0 checks always run)"
        ),
    )
    parser.add_argument(
        "--strict-prefix",
        action="store_true",
        help=(
            "apply the original uniform atol to inside_prefix; this intentionally "
            "recovers the pre-acceptance long-prefix failure gate"
        ),
    )
    return parser.parse_args()


def finite_graph_audit(path: Path) -> dict[str, int]:
    graph = onnx.load(str(path))
    imports = {item.domain: item.version for item in graph.opset_import}
    if imports.get("") != 17:
        raise AssertionError(f"expected default opset 17, found {imports.get('')}")
    onnx.checker.check_model(graph)
    counts = {"finite_constants": 0, "finite_initializers": 0}

    def tensor(value: onnx.TensorProto, label: str, counter: str) -> None:
        array = numpy_helper.to_array(value)
        if array.dtype.kind in "fc":
            if array.dtype != np.float32:
                raise AssertionError(f"{label}: expected fp32, found {array.dtype}")
            if not np.isfinite(array).all():
                raise AssertionError(f"{label}: non-finite tensor")
            counts[counter] += 1

    def visit(subgraph: onnx.GraphProto) -> None:
        for initializer in subgraph.initializer:
            tensor(
                initializer, f"initializer {initializer.name!r}", "finite_initializers"
            )
        for node in subgraph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    tensor(attribute.t, f"node {node.name!r}", "finite_constants")
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    for value in attribute.tensors:
                        tensor(value, f"node {node.name!r}", "finite_constants")
                elif attribute.type == onnx.AttributeProto.GRAPH:
                    visit(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for child in attribute.graphs:
                        visit(child)

    visit(graph.graph)
    return counts


def torch_inputs(inputs: dict[str, np.ndarray]) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(inputs["text_states"]),
        torch.from_numpy(inputs["text_mask"]),
        torch.from_numpy(inputs["query_states"]),
        torch.from_numpy(inputs["query_mask"]),
    )


def numpy_outputs(outputs: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
    return {
        name: value.detach().cpu().numpy() for name, value in zip(OUTPUT_NAMES, outputs)
    }


def run_onnx(
    session: ort.InferenceSession, inputs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    values = session.run(list(OUTPUT_NAMES), inputs)
    return dict(zip(OUTPUT_NAMES, values))


def update_stats(
    stats: dict[str, dict[str, float | int]], name: str, report: dict
) -> None:
    stage = stats.setdefault(
        name,
        {"comparisons": 0, "max_abs": 0.0, "max_rel": 0.0},
    )
    stage["comparisons"] = int(stage["comparisons"]) + 1
    stage["max_abs"] = max(float(stage["max_abs"]), float(report["max_abs"]))
    stage["max_rel"] = max(float(stage["max_rel"]), float(report["max_rel"]))


def compare_onnx_array(
    name: str,
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    atol: float,
    rtol: float,
    prefix_coordinate_atol: float = 0.0,
) -> dict[str, float | int | str | list[int]]:
    """Compare one ONNX result, optionally using the approved prefix envelope.

    The extra absolute allowance is indexed by the last-axis prefix coordinate,
    not by the tensor's global sequence length. Coordinate zero therefore gets
    no extra allowance. This is an empirical inside-prefix accumulation bound,
    not a general fp32 precision claim.
    """
    if expected.shape != actual.shape:
        raise AssertionError(f"{name}: shape {actual.shape} != {expected.shape}")
    if not np.isfinite(expected).all():
        raise AssertionError(f"{name} expected contains non-finite values")
    if not np.isfinite(actual).all():
        raise AssertionError(f"{name} actual contains non-finite values")

    reference = expected.astype(np.float64)
    observed = actual.astype(np.float64)
    absolute = np.abs(observed - reference)
    relative = absolute / np.maximum(np.abs(reference), 1e-12)
    original_tolerance = atol + rtol * np.abs(reference)
    tolerance = original_tolerance
    if prefix_coordinate_atol:
        if expected.ndim == 0:
            raise AssertionError(f"{name}: prefix output must have a coordinate axis")
        coordinates = np.arange(expected.shape[-1], dtype=np.float64).reshape(
            (1,) * (expected.ndim - 1) + (expected.shape[-1],)
        )
        tolerance = original_tolerance + prefix_coordinate_atol * coordinates

    failures = absolute > tolerance
    report: dict[str, float | int | str | list[int]] = {
        "name": name,
        "shape": list(expected.shape),
        "max_abs": float(absolute.max(initial=0.0)),
        "max_rel": float(relative.max(initial=0.0)),
        "raw_original_failure_count": int(
            np.count_nonzero(absolute > original_tolerance)
        ),
    }
    if failures.any():
        excess = absolute - tolerance
        flat = int(np.argmax(excess))
        index = tuple(int(value) for value in np.unravel_index(flat, absolute.shape))
        raise AssertionError(
            f"{name}: numerical mismatch at {index}: expected={expected[index]!r} "
            f"actual={actual[index]!r}; abs={absolute[index]:.9g}, "
            f"tolerance={tolerance[index]:.9g}, max_abs={report['max_abs']:.9g}, "
            f"max_rel={report['max_rel']:.9g}"
        )
    return report


def compare_output_sets(
    label: str,
    expected: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    stats: dict[str, dict[str, float | int]],
    *,
    atol: float,
    rtol: float,
) -> None:
    """Apply the unchanged gate to non-ONNX comparisons."""
    for name in OUTPUT_NAMES:
        reference = expected[name]
        observed = actual[name]
        if reference.dtype == np.bool_:
            if reference.shape != observed.shape or not np.array_equal(
                reference, observed
            ):
                raise AssertionError(f"{label}/{name}: boolean output differs")
            report = {"max_abs": 0.0, "max_rel": 0.0}
        else:
            report = compare_arrays(
                f"{label}/{name}", reference, observed, atol=atol, rtol=rtol
            )
        update_stats(stats, name, report)


def compare_onnx_output_sets(
    label: str,
    expected: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    stats: dict[str, dict[str, float | int]],
    raw_original_failures: Counter[str],
    *,
    atol: float,
    rtol: float,
    strict_prefix: bool,
    failures: list[str],
) -> None:
    """Apply the coordinate envelope only to ONNX inside_prefix outputs."""
    for name in OUTPUT_NAMES:
        reference = expected[name]
        observed = actual[name]
        if reference.dtype == np.bool_:
            if reference.shape != observed.shape or not np.array_equal(
                reference, observed
            ):
                raise AssertionError(f"{label}/{name}: boolean output differs")
            report = {"max_abs": 0.0, "max_rel": 0.0, "raw_original_failure_count": 0}
        else:
            coordinate_atol = (
                PREFIX_COORDINATE_ATOL
                if name == "inside_prefix" and not strict_prefix
                else 0.0
            )
            try:
                report = compare_onnx_array(
                    f"{label}/{name}",
                    reference,
                    observed,
                    atol=atol,
                    rtol=rtol,
                    prefix_coordinate_atol=coordinate_atol,
                )
            except AssertionError as error:
                # Shape and finite failures are contract failures, not tolerance
                # observations, and must never be softened into report entries.
                if (
                    reference.shape != observed.shape
                    or not np.isfinite(reference).all()
                    or not np.isfinite(observed).all()
                ):
                    raise
                failures.append(str(error))
                absolute = np.abs(
                    observed.astype(np.float64) - reference.astype(np.float64)
                )
                relative = absolute / np.maximum(
                    np.abs(reference.astype(np.float64)), 1e-12
                )
                original_tolerance = atol + rtol * np.abs(reference.astype(np.float64))
                report = {
                    "max_abs": float(absolute.max(initial=0.0)),
                    "max_rel": float(relative.max(initial=0.0)),
                    "raw_original_failure_count": int(
                        np.count_nonzero(absolute > original_tolerance)
                    ),
                }
        raw_original_failures[name] += int(report["raw_original_failure_count"])
        update_stats(stats, name, report)


def live_outputs(
    model: torch.nn.Module,
    wrapper: BoundaryMarginalGraph,
    inputs: dict[str, np.ndarray],
    *,
    atol: float,
    rtol: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    tensors = torch_inputs(inputs)
    with torch.inference_mode():
        oracle = oracle_outputs(model, *tensors)
        wrapped = wrapper(*tensors)
    assert_wrapper_matches_oracle(OUTPUT_NAMES, oracle, wrapped, atol=atol, rtol=rtol)
    return numpy_outputs(oracle), numpy_outputs(wrapped)


def synthetic_inputs(
    seed: int, batch: int, length: int, queries: int, hidden: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    text_states = rng.standard_normal((batch, length, hidden), dtype=np.float32)
    query_states = rng.standard_normal((batch, queries, hidden), dtype=np.float32)
    text_mask = np.ones((batch, length), dtype=bool)
    query_mask = np.ones((batch, queries), dtype=bool)
    if length > 1:
        text_mask[0, -(max(1, length // 7)) :] = False
        if batch > 1:
            text_mask[1, max(1, length // 2) :] = False
    if queries > 1:
        query_mask[0, -1] = False
        if batch > 1:
            query_mask[1, 0] = False
    return {
        "text_states": text_states,
        "text_mask": text_mask,
        "query_states": query_states,
        "query_mask": query_mask,
    }


def main() -> None:
    args = parse_args()
    if not ort.__version__.startswith("1.20."):
        raise RuntimeError(
            f"validation requires ONNX Runtime 1.20.x, found {ort.__version__}"
        )
    configure_determinism(args.seed)
    onnx_path = Path(args.onnx_path)
    graph_audit = finite_graph_audit(onnx_path)
    model, config = load_reference_model(args.model_dir)
    if config.get("architecture") != "boundary":
        raise ValueError("expected a boundary checkpoint")
    wrapper = BoundaryMarginalGraph(model).float().eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    fixtures = sorted(Path(args.golden_dir).glob("*.npz"))
    if len(fixtures) != 30:
        raise AssertionError(
            f"full-corpus validation requires 30 NPZ fixtures, found {len(fixtures)}"
        )
    fixture_ids = {path.stem for path in fixtures}
    mandatory_long = {"long_1000_words", "long_2000_words", "long_3000_words"}
    if not mandatory_long <= fixture_ids:
        raise AssertionError(
            f"missing mandatory long cases: {sorted(mandatory_long - fixture_ids)}"
        )

    onnx_golden_stats: dict[str, dict[str, float | int]] = {}
    oracle_golden_stats: dict[str, dict[str, float | int]] = {}
    synthetic_stats: dict[str, dict[str, float | int]] = {}
    counts: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    raw_original_failures: Counter[str] = Counter()
    tolerance_failures: list[str] = []

    for fixture in fixtures:
        metadata_path = fixture.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        category = metadata["category"]
        categories[category] += 1
        with np.load(fixture, allow_pickle=False) as data:
            inputs = {
                "text_states": data["text_states"].astype(np.float32, copy=True),
                "text_mask": data["text_mask"].astype(bool, copy=True),
                "query_states": data["query_states"].astype(np.float32, copy=True),
                "query_mask": data["query_mask"].astype(bool, copy=True),
            }
            query_count = inputs["query_states"].shape[1]
            oracle, _wrapped = live_outputs(
                model,
                wrapper,
                inputs,
                atol=args.atol,
                rtol=args.rtol,
            )
            actual = run_onnx(session, inputs)
            if query_count == 0:
                counts["q0_graph_cases"] += 1
                counts[
                    (
                        "classification_q0_cases"
                        if category == "classification"
                        else "other_q0_cases"
                    )
                ] += 1
                compare_onnx_output_sets(
                    f"{fixture.stem}/q0-live-oracle",
                    oracle,
                    actual,
                    synthetic_stats,
                    raw_original_failures,
                    atol=args.atol,
                    rtol=args.rtol,
                    strict_prefix=args.strict_prefix,
                    failures=tolerance_failures,
                )
                for key in GOLDEN_KEYS.values():
                    if key in data:
                        raise AssertionError(
                            f"{fixture.stem}: Q=0 bypass unexpectedly has {key}"
                        )
            else:
                expected = {name: data[key].copy() for name, key in GOLDEN_KEYS.items()}
                compare_output_sets(
                    f"{fixture.stem}/oracle-golden",
                    expected,
                    oracle,
                    oracle_golden_stats,
                    atol=args.atol,
                    rtol=args.rtol,
                )
                compare_onnx_output_sets(
                    f"{fixture.stem}/onnx-golden",
                    expected,
                    actual,
                    onnx_golden_stats,
                    raw_original_failures,
                    atol=args.atol,
                    rtol=args.rtol,
                    strict_prefix=args.strict_prefix,
                    failures=tolerance_failures,
                )
                counts["golden_head_cases"] += 1
                if fixture.stem in mandatory_long:
                    counts["long_head_cases"] += 1
        print(
            f"{fixture.stem}: L={inputs['text_states'].shape[1]} "
            f"Q={inputs['query_states'].shape[1]} ok",
            flush=True,
        )
        del oracle, actual
        gc.collect()

    synthetic_shapes = ((1, 1), (2, 3), (33, 5), (129, 1), (257, 3))
    for index, (length, queries) in enumerate(synthetic_shapes):
        inputs = synthetic_inputs(
            args.seed + 100 + index, 1, length, queries, int(model.hidden_size)
        )
        oracle, _wrapped = live_outputs(
            model, wrapper, inputs, atol=args.atol, rtol=args.rtol
        )
        actual = run_onnx(session, inputs)
        compare_onnx_output_sets(
            f"synthetic-l{length}-q{queries}",
            oracle,
            actual,
            synthetic_stats,
            raw_original_failures,
            atol=args.atol,
            rtol=args.rtol,
            strict_prefix=args.strict_prefix,
            failures=tolerance_failures,
        )
        counts["synthetic_dynamic_cases"] += 1

    batch_inputs = synthetic_inputs(args.seed + 200, 2, 17, 3, int(model.hidden_size))
    oracle, _wrapped = live_outputs(
        model, wrapper, batch_inputs, atol=args.atol, rtol=args.rtol
    )
    actual = run_onnx(session, batch_inputs)
    compare_onnx_output_sets(
        "synthetic-b2-l17-q3",
        oracle,
        actual,
        synthetic_stats,
        raw_original_failures,
        atol=args.atol,
        rtol=args.rtol,
        strict_prefix=args.strict_prefix,
        failures=tolerance_failures,
    )
    counts["synthetic_batch_cases"] += 1

    q0_inputs = synthetic_inputs(args.seed + 201, 1, 2, 0, int(model.hidden_size))
    oracle, _wrapped = live_outputs(
        model, wrapper, q0_inputs, atol=args.atol, rtol=args.rtol
    )
    actual = run_onnx(session, q0_inputs)
    compare_onnx_output_sets(
        "synthetic-l2-q0",
        oracle,
        actual,
        synthetic_stats,
        raw_original_failures,
        atol=args.atol,
        rtol=args.rtol,
        strict_prefix=args.strict_prefix,
        failures=tolerance_failures,
    )
    counts["synthetic_q0_cases"] += 1

    l0_inputs = synthetic_inputs(args.seed + 202, 1, 0, 1, int(model.hidden_size))
    l0_oracle: dict[str, np.ndarray] | None = None
    try:
        l0_oracle, _wrapped = live_outputs(
            model, wrapper, l0_inputs, atol=args.atol, rtol=args.rtol
        )
        oracle_status = "available"
    except (RuntimeError, ValueError, IndexError) as exc:
        oracle_status = f"unsupported: {type(exc).__name__}: {exc}"

    # L=0 is outside the supported graph contract. Opt-in diagnostics use a
    # subprocess because ORT 1.20 may SIGSEGV and trigger an OS crash dialog.
    # Rust must reject L=0 before native inference; the high-level pipeline
    # normalizes empty user text to a single '.' token. Q=0 checks stay enabled.
    probe_status = "not-run-unsupported-contract"
    probe_returncode = None
    if args.probe_unsupported_axes:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            input_path = temporary_path / "input.npz"
            output_path = temporary_path / "output.npz"
            np.savez(input_path, **l0_inputs)
            child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import numpy as np, onnxruntime as ort, sys; "
                        "z=np.load(sys.argv[2]); "
                        "s=ort.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider']); "
                        "names=[o.name for o in s.get_outputs()]; "
                        "values=s.run(names,{k:z[k] for k in "
                        "['text_states','text_mask','query_states','query_mask']}); "
                        "np.savez(sys.argv[3],**dict(zip(names,values)))"
                    ),
                    str(onnx_path),
                    str(input_path),
                    str(output_path),
                ],
                check=False,
            )
            probe_returncode = child.returncode
            if child.returncode == -11:
                probe_status = "known-ort-1.20-sigsegv-observed"
            elif child.returncode != 0:
                raise AssertionError(
                    "unexpected L=0 isolated ORT probe failure: "
                    f"status {child.returncode}; only the known -11 SIGSEGV is tolerated"
                )
            else:
                if l0_oracle is None:
                    raise AssertionError(
                        "L=0 ORT probe returned outputs but the pinned oracle could not "
                        "provide a numerical reference"
                    )
                with np.load(output_path, allow_pickle=False) as values:
                    actual = {name: values[name].copy() for name in OUTPUT_NAMES}
                compare_onnx_output_sets(
                    "synthetic-l0-q1-isolated-unsupported-contract",
                    l0_oracle,
                    actual,
                    synthetic_stats,
                    raw_original_failures,
                    atol=args.atol,
                    rtol=args.rtol,
                    strict_prefix=args.strict_prefix,
                    failures=tolerance_failures,
                )
                probe_status = "returned-numerically-validated-outputs"

    l0_status = {
        "contract": "unsupported-L0",
        "oracle_status": oracle_status,
        "isolated_probe_status": probe_status,
        "isolated_probe_returncode": probe_returncode,
        "known_sigsegv_is_numerical_gate_failure": False,
        "rust_guard_requirement": (
            "MarginalModel must reject L=0 before calling ONNX Runtime"
        ),
        "pipeline_empty_text_policy": "normalize empty text to '.' so graph L>=1",
    }

    expected_counts = {
        "golden_head_cases": 24,
        "q0_graph_cases": 6,
        "classification_q0_cases": 5,
        "other_q0_cases": 1,
        "long_head_cases": 3,
        "synthetic_dynamic_cases": 5,
        "synthetic_batch_cases": 1,
        "synthetic_q0_cases": 1,
    }
    for name, expected in expected_counts.items():
        if counts[name] != expected:
            raise AssertionError(f"{name}: {counts[name]} != {expected}")

    report = {
        "onnx": str(onnx_path),
        "onnxruntime": ort.__version__,
        "atol": args.atol,
        "rtol": args.rtol,
        "inside_prefix_onnx_tolerance": {
            "mode": (
                "strict-original"
                if args.strict_prefix
                else "coordinate-accumulation-envelope"
            ),
            "rule": (
                f"atol(i) = {args.atol:.9g} + {PREFIX_COORDINATE_ATOL:.9g} * i; "
                f"rtol = {args.rtol:.9g}"
                if not args.strict_prefix
                else f"atol = {args.atol:.9g}; rtol = {args.rtol:.9g}"
            ),
            "scope": "ONNX inside_prefix comparisons only",
            "interpretation": (
                "empirical prefix coordinate accumulation bound; not a universal precision claim"
            ),
        },
        "graph_audit": graph_audit,
        "fixture_count": len(fixtures),
        "category_counts": dict(sorted(categories.items())),
        "case_counts": dict(sorted(counts.items())),
        "q0_policy": (
            "Python graph Q=0 probes are validated, but Rust rejects Q=0 because "
            "classification-only requests must bypass marginals"
        ),
        "l0_status": l0_status,
        "raw_original_tolerance_failure_count": sum(raw_original_failures.values()),
        "raw_original_tolerance_failures_per_output": dict(
            sorted(raw_original_failures.items())
        ),
        "tolerance_failure_count": len(tolerance_failures),
        "tolerance_failures": tolerance_failures,
        "onnx_vs_golden_per_output": onnx_golden_stats,
        "oracle_vs_golden_per_output": oracle_golden_stats,
        "onnx_vs_live_oracle_synthetic_per_output": synthetic_stats,
    }
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if tolerance_failures:
        raise AssertionError(
            f"{len(tolerance_failures)} ONNX output comparisons exceeded the active gate"
        )


if __name__ == "__main__":
    main()
