#!/usr/bin/env python3
"""Validate boundary_scorer.onnx against all frozen shared-head fixtures."""

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
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper

from boundary_scorer import (
    INPUT_NAMES,
    OUTPUT_NAMES,
    BoundaryScorerGraph,
    assert_wrapper_matches_oracle,
    oracle_outputs,
)
from common import (
    BASE_HF_REVISION,
    MASK_LOGIT,
    compare_arrays,
    configure_determinism,
    load_reference_model,
)
from export_boundary_scorer import validate_config

DEFAULT_MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--fastino--gliner2.5-base-v1"
    / "snapshots"
    / BASE_HF_REVISION
)
FIXTURE_INPUT_KEYS = {
    "boundary_states": "shared_scorer_0_boundary_states",
    "text_states": "marginals_0_text_states",
    "text_mask": "marginals_0_text_mask",
    "query_states": "shared_scorer_0_query_states",
    "query_mask": "shared_scorer_0_query_mask",
    "start_logits": "marginals_0_start_logits",
    "end_logits": "marginals_0_end_logits",
    "inside_prefix": "marginals_0_inside_prefix",
    "inside_prefix_mean": "marginals_0_inside_prefix_mean",
    "candidate_indices": "shared_scorer_0_indices",
    "candidate_mask": "shared_scorer_0_mask",
    "candidate_compat": "shared_scorer_0_compat",
}
FIXTURE_OUTPUT_KEYS = {
    "pair_logits": "boundary_head_0_pair_logits",
    "null_logits": "boundary_head_0_null_logits",
    "count_log_rates": "boundary_head_0_count_log_rates",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument(
        "--onnx-path", default="onnx/gliner2.5-base-v1/boundary_scorer.onnx"
    )
    parser.add_argument("--golden-dir", default="fixtures/gliner2.5-base-v1")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--report-json")
    return parser.parse_args()


def finite_graph_audit(path: Path) -> dict[str, int]:
    graph = onnx.load(str(path))
    imports = {item.domain: item.version for item in graph.opset_import}
    if imports.get("") != 17:
        raise AssertionError(f"expected default opset 17, found {imports.get('')}")
    constants = 0
    initializers = 0
    for node in graph.graph.node:
        for attribute in node.attribute:
            if attribute.type != onnx.AttributeProto.TENSOR:
                continue
            value = numpy_helper.to_array(attribute.t)
            if value.dtype.kind in "fc":
                constants += 1
                if not np.isfinite(value).all():
                    raise AssertionError(f"ONNX node {node.name!r} is non-finite")
    for initializer in graph.graph.initializer:
        value = numpy_helper.to_array(initializer)
        if value.dtype.kind in "fc":
            initializers += 1
            if value.dtype != np.float32:
                raise AssertionError(
                    f"ONNX initializer {initializer.name!r} is {value.dtype}, "
                    "expected fp32"
                )
            if not np.isfinite(value).all():
                raise AssertionError(
                    f"ONNX initializer {initializer.name!r} is non-finite"
                )
    return {"finite_constants": constants, "fp32_initializers": initializers}


def torch_inputs(inputs: dict[str, np.ndarray]) -> tuple[torch.Tensor, ...]:
    return tuple(torch.from_numpy(inputs[name]) for name in INPUT_NAMES)


def numpy_outputs(outputs: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
    return {
        name: value.detach().cpu().numpy()
        for name, value in zip(OUTPUT_NAMES, outputs)
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
        name, {"comparisons": 0, "max_abs": 0.0, "max_rel": 0.0}
    )
    stage["comparisons"] = int(stage["comparisons"]) + 1
    stage["max_abs"] = max(float(stage["max_abs"]), float(report["max_abs"]))
    stage["max_rel"] = max(float(stage["max_rel"]), float(report["max_rel"]))


def compare_output_sets(
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


def assert_mask_behavior(
    label: str,
    inputs: dict[str, np.ndarray],
    outputs: dict[str, np.ndarray],
) -> dict[str, int]:
    candidate_mask = inputs["candidate_mask"]
    query_mask = inputs["query_mask"]
    invalid_pairs = ~(query_mask[:, :, None] & candidate_mask[:, None, :])
    pair_logits = outputs["pair_logits"]
    if invalid_pairs.any() and not np.array_equal(
        pair_logits[invalid_pairs],
        np.full(np.count_nonzero(invalid_pairs), MASK_LOGIT, dtype=np.float32),
    ):
        observed = pair_logits[invalid_pairs]
        raise AssertionError(
            f"{label}: invalid pair logits are not exactly {MASK_LOGIT}: "
            f"range=[{observed.min()}, {observed.max()}]"
        )
    invalid_candidates = ~candidate_mask
    candidate_states = outputs["candidate_states"]
    if invalid_candidates.any() and not np.array_equal(
        candidate_states[invalid_candidates],
        np.zeros_like(candidate_states[invalid_candidates]),
    ):
        raise AssertionError(f"{label}: invalid candidate states are not exactly zero")
    return {
        "invalid_pairs": int(np.count_nonzero(invalid_pairs)),
        "invalid_candidate_rows": int(np.count_nonzero(invalid_candidates)),
    }


def fixture_inputs(data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    inputs: dict[str, np.ndarray] = {}
    for name, key in FIXTURE_INPUT_KEYS.items():
        value = data[key].copy()
        if name in {"text_mask", "query_mask", "candidate_mask"}:
            value = value.astype(bool, copy=False)
        elif name == "candidate_indices":
            value = value.astype(np.int64, copy=False)
        else:
            value = value.astype(np.float32, copy=False)
        inputs[name] = value
    return inputs


def fixture_outputs(data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    expected = {
        name: data[key].astype(np.float32, copy=True)
        for name, key in FIXTURE_OUTPUT_KEYS.items()
    }
    expanded = data["boundary_head_0_candidate_states"].astype(np.float32, copy=True)
    if expanded.shape[1] < 1:
        raise AssertionError("head fixture unexpectedly has no extraction queries")
    canonical = expanded[:, 0]
    if not np.array_equal(
        expanded, np.broadcast_to(canonical[:, None], expanded.shape)
    ):
        raise AssertionError("shared candidate states differ between query expansions")
    expected["candidate_states"] = canonical

    candidate_major = data[
        "shared_scorer_0_pair_logits_candidate_major"
    ].astype(np.float32, copy=False)
    if not np.array_equal(expected["pair_logits"], candidate_major.transpose(0, 2, 1)):
        raise AssertionError("frozen shared scorer/head pair-logit transpose differs")
    return expected


def live_outputs(
    model: torch.nn.Module,
    wrapper: BoundaryScorerGraph,
    inputs: dict[str, np.ndarray],
    *,
    atol: float,
    rtol: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    tensors = torch_inputs(inputs)
    with torch.inference_mode():
        oracle = oracle_outputs(model, *tensors)
        wrapped = wrapper(*tensors)
    assert_wrapper_matches_oracle(
        OUTPUT_NAMES, oracle, wrapped, atol=atol, rtol=rtol
    )
    return numpy_outputs(oracle), numpy_outputs(wrapped)


def synthetic_inputs(
    seed: int, *, batch: int, length: int, queries: int, candidates: int, hidden: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    boundary_count = length + 1
    text_states = rng.standard_normal((batch, length, hidden), dtype=np.float32)
    boundary_states = rng.standard_normal(
        (batch, boundary_count, 128), dtype=np.float32
    )
    query_states = rng.standard_normal((batch, queries, hidden), dtype=np.float32)
    text_mask = np.ones((batch, length), dtype=bool)
    query_mask = np.ones((batch, queries), dtype=bool)
    if length > 1:
        text_mask[0, -1] = False
        if batch > 1:
            text_mask[1, max(1, length // 2) :] = False
    if queries > 1:
        query_mask[0, -1] = False
        if batch > 1:
            query_mask[1, 0] = False
    start_logits = rng.standard_normal(
        (batch, queries, boundary_count), dtype=np.float32
    )
    end_logits = rng.standard_normal(
        (batch, queries, boundary_count), dtype=np.float32
    )
    inside_values = rng.standard_normal((batch, queries, length), dtype=np.float32)
    inside_prefix_mean = np.zeros((batch, queries, 1), dtype=np.float32)
    inside_prefix = np.zeros((batch, queries, boundary_count), dtype=np.float32)
    for batch_index in range(batch):
        valid_length = max(int(text_mask[batch_index].sum()), 1)
        if queries:
            mean = inside_values[batch_index, :, :valid_length].mean(
                axis=-1, keepdims=True
            )
            inside_prefix_mean[batch_index] = mean
            centered = inside_values[batch_index] - mean
            centered[:, valid_length:] = 0.0
            inside_prefix[batch_index, :, 1:] = np.cumsum(
                centered, axis=-1, dtype=np.float32
            )

    candidate_indices = np.zeros((batch, candidates, 2), dtype=np.int64)
    candidate_mask = np.ones((batch, candidates), dtype=bool)
    for batch_index in range(batch):
        valid_length = max(int(text_mask[batch_index].sum()), 1)
        for candidate in range(candidates):
            start = candidate % valid_length
            end = min(valid_length, start + 1 + candidate % 4)
            candidate_indices[batch_index, candidate] = (start, end)
        if candidates > 1:
            candidate_mask[batch_index, -1] = False
            candidate_indices[batch_index, -1] = (0, 0)
    candidate_compat = rng.standard_normal((batch, candidates), dtype=np.float32)
    candidate_compat[~candidate_mask] = 0.0
    return {
        "boundary_states": boundary_states,
        "text_states": text_states,
        "text_mask": text_mask,
        "query_states": query_states,
        "query_mask": query_mask,
        "start_logits": start_logits,
        "end_logits": end_logits,
        "inside_prefix": inside_prefix,
        "inside_prefix_mean": inside_prefix_mean,
        "candidate_indices": candidate_indices,
        "candidate_mask": candidate_mask,
        "candidate_compat": candidate_compat,
    }


def isolated_zero_probe(
    onnx_path: Path,
    inputs: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
    *,
    label: str,
    atol: float,
    rtol: float,
) -> str:
    """Keep known ORT native zero-dimension hazards outside this process."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        input_path = root / "inputs.npz"
        output_path = root / "outputs.npz"
        np.savez(input_path, **inputs)
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import numpy as np, onnxruntime as ort, sys; "
                    "z=np.load(sys.argv[2]); "
                    "s=ort.InferenceSession(sys.argv[1],"
                    "providers=['CPUExecutionProvider']); "
                    "names=[o.name for o in s.get_outputs()]; "
                    "v=s.run(names,{k:z[k] for k in "
                    + repr(list(INPUT_NAMES))
                    + "}); np.savez(sys.argv[3],**dict(zip(names,v)))"
                ),
                str(onnx_path),
                str(input_path),
                str(output_path),
            ],
            check=False,
        )
        if child.returncode != 0:
            return f"defined-caller-bypass (ORT process status {child.returncode})"
        with np.load(output_path, allow_pickle=False) as values:
            actual = {name: values[name].copy() for name in OUTPUT_NAMES}
        for name in OUTPUT_NAMES:
            compare_arrays(
                f"{label}/{name}",
                expected[name],
                actual[name],
                atol=atol,
                rtol=rtol,
            )
        assert_mask_behavior(label, inputs, actual)
        return "validated"


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
    validate_config(config)
    wrapper = BoundaryScorerGraph(model).float().eval()
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )

    fixtures = sorted(Path(args.golden_dir).glob("*.npz"))
    if len(fixtures) != 30:
        raise AssertionError(
            f"full-corpus validation requires 30 NPZ fixtures, found {len(fixtures)}"
        )
    head_fixtures: list[Path] = []
    for fixture in fixtures:
        with np.load(fixture, allow_pickle=False) as data:
            if "shared_scorer_0_pair_logits_candidate_major" in data.files:
                head_fixtures.append(fixture)
    if len(head_fixtures) != 24:
        raise AssertionError(
            f"expected all 24 actual shared-head cases, found {len(head_fixtures)}"
        )

    oracle_golden_stats: dict[str, dict[str, float | int]] = {}
    onnx_golden_stats: dict[str, dict[str, float | int]] = {}
    synthetic_stats: dict[str, dict[str, float | int]] = {}
    counts: Counter[str] = Counter()
    invalid_counts: Counter[str] = Counter()

    for fixture in head_fixtures:
        with np.load(fixture, allow_pickle=False) as data:
            inputs = fixture_inputs(data)
            expected = fixture_outputs(data)
        oracle, _wrapped = live_outputs(
            model, wrapper, inputs, atol=args.atol, rtol=args.rtol
        )
        compare_output_sets(
            f"{fixture.stem}/oracle-golden",
            expected,
            oracle,
            oracle_golden_stats,
            atol=args.atol,
            rtol=args.rtol,
        )
        actual = run_onnx(session, inputs)
        compare_output_sets(
            f"{fixture.stem}/onnx-golden",
            expected,
            actual,
            onnx_golden_stats,
            atol=args.atol,
            rtol=args.rtol,
        )
        behavior = assert_mask_behavior(fixture.stem, inputs, actual)
        invalid_counts.update(behavior)
        counts["golden_head_cases"] += 1
        if fixture.stem.startswith("long_"):
            counts["long_head_cases"] += 1
        print(
            f"{fixture.stem}: L={inputs['text_states'].shape[1]} "
            f"Q={inputs['query_states'].shape[1]} "
            f"C={inputs['candidate_indices'].shape[1]} ok",
            flush=True,
        )
        del oracle, actual, expected, inputs
        gc.collect()

    shapes = ((1, 1, 1), (2, 3, 2), (9, 2, 7), (23, 5, 31))
    for index, (length, queries, candidates) in enumerate(shapes):
        inputs = synthetic_inputs(
            args.seed + 100 + index,
            batch=1,
            length=length,
            queries=queries,
            candidates=candidates,
            hidden=int(model.hidden_size),
        )
        oracle, _wrapped = live_outputs(
            model, wrapper, inputs, atol=args.atol, rtol=args.rtol
        )
        actual = run_onnx(session, inputs)
        compare_output_sets(
            f"synthetic-l{length}-q{queries}-c{candidates}",
            oracle,
            actual,
            synthetic_stats,
            atol=args.atol,
            rtol=args.rtol,
        )
        behavior = assert_mask_behavior("synthetic", inputs, actual)
        invalid_counts.update(behavior)
        counts["synthetic_dynamic_cases"] += 1

    batch_inputs = synthetic_inputs(
        args.seed + 200,
        batch=2,
        length=17,
        queries=4,
        candidates=13,
        hidden=int(model.hidden_size),
    )
    oracle, _wrapped = live_outputs(
        model, wrapper, batch_inputs, atol=args.atol, rtol=args.rtol
    )
    actual = run_onnx(session, batch_inputs)
    compare_output_sets(
        "synthetic-b2-l17-q4-c13",
        oracle,
        actual,
        synthetic_stats,
        atol=args.atol,
        rtol=args.rtol,
    )
    behavior = assert_mask_behavior("synthetic-batch", batch_inputs, actual)
    invalid_counts.update(behavior)
    counts["synthetic_batch_cases"] += 1

    zero_status: dict[str, str] = {}
    for label, queries, candidates in (("q0", 0, 3), ("c0", 2, 0)):
        zero_inputs = synthetic_inputs(
            args.seed + 300 + len(zero_status),
            batch=1,
            length=3,
            queries=queries,
            candidates=candidates,
            hidden=int(model.hidden_size),
        )
        try:
            oracle, _wrapped = live_outputs(
                model, wrapper, zero_inputs, atol=args.atol, rtol=args.rtol
            )
        except (RuntimeError, ValueError, IndexError) as exc:
            zero_status[label] = (
                f"defined-caller-bypass (upstream {type(exc).__name__}: {exc})"
            )
        else:
            zero_status[label] = isolated_zero_probe(
                onnx_path,
                zero_inputs,
                oracle,
                label=f"synthetic-{label}",
                atol=args.atol,
                rtol=args.rtol,
            )
        counts[f"synthetic_{label}_probes"] += 1

    expected_counts = {
        "golden_head_cases": 24,
        "long_head_cases": 3,
        "synthetic_dynamic_cases": 4,
        "synthetic_batch_cases": 1,
        "synthetic_q0_probes": 1,
        "synthetic_c0_probes": 1,
    }
    for name, expected in expected_counts.items():
        if counts[name] != expected:
            raise AssertionError(f"{name}: {counts[name]} != {expected}")

    report = {
        "onnx": str(onnx_path),
        "onnxruntime": ort.__version__,
        "atol": args.atol,
        "rtol": args.rtol,
        "graph_audit": graph_audit,
        "fixture_count": len(fixtures),
        "case_counts": dict(sorted(counts.items())),
        "zero_dimension_policy": (
            "Q=0 or C=0 is bypassed by the caller unless the isolated ORT probe "
            "is validated; zero dimensions are never invoked in-process"
        ),
        "zero_dimension_status": zero_status,
        "invalid_mask_checks": dict(sorted(invalid_counts.items())),
        "oracle_vs_golden_per_output": oracle_golden_stats,
        "onnx_vs_golden_per_output": onnx_golden_stats,
        "onnx_vs_live_oracle_synthetic_per_output": synthetic_stats,
    }
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
