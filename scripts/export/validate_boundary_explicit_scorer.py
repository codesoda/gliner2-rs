#!/usr/bin/env python3
"""Validate the separate learned explicit-span scorer and M2 composition."""

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

from boundary_explicit_scorer import (
    INPUT_NAMES,
    OUTPUT_NAMES,
    BoundaryExplicitScorerGraph,
    full_method_outputs,
    independently_composed_outputs,
)
from common import (
    BASE_HF_REVISION,
    MASK_LOGIT,
    compare_arrays,
    configure_determinism,
    load_reference_model,
    sha256_file,
)
from export_boundary_explicit_scorer import validate_config

DEFAULT_MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--fastino--gliner2.5-base-v1"
    / "snapshots"
    / BASE_HF_REVISION
)
MARGINAL_OUTPUT_NAMES = (
    "boundary_states",
    "boundary_mask",
    "start_logits",
    "end_logits",
    "inside_logits",
    "inside_prefix",
    "inside_prefix_mean",
    "start_all",
    "end_all",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument(
        "--onnx-path",
        default="onnx/gliner2.5-base-v1/boundary_explicit_scorer.onnx",
    )
    parser.add_argument(
        "--marginals-onnx",
        default="onnx/gliner2.5-base-v1/boundary_marginals.onnx",
    )
    parser.add_argument("--golden-dir", default="fixtures/gliner2.5-base-v1")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--confidence-atol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--report-json")
    return parser.parse_args()


def finite_graph_audit(path: Path) -> dict[str, object]:
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    imports = {item.domain: item.version for item in graph.opset_import}
    if imports.get("") != 17:
        raise AssertionError(f"expected default opset 17, found {imports.get('')}")
    counts = {"finite_constants": 0, "fp32_initializers": 0}

    def inspect_tensor(value: onnx.TensorProto, label: str, key: str) -> None:
        array = numpy_helper.to_array(value)
        if array.dtype.kind not in "fc":
            return
        if not np.isfinite(array).all():
            raise AssertionError(f"{label} is non-finite")
        if key == "fp32_initializers" and array.dtype != np.float32:
            raise AssertionError(f"{label} is {array.dtype}, expected fp32")
        counts[key] += 1

    def visit(subgraph: onnx.GraphProto) -> None:
        for initializer in subgraph.initializer:
            inspect_tensor(
                initializer, f"initializer {initializer.name!r}", "fp32_initializers"
            )
        for node in subgraph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    inspect_tensor(
                        attribute.t, f"constant node {node.name!r}", "finite_constants"
                    )
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    for value in attribute.tensors:
                        inspect_tensor(
                            value,
                            f"constant-list node {node.name!r}",
                            "finite_constants",
                        )
                elif attribute.type == onnx.AttributeProto.GRAPH:
                    visit(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for child in attribute.graphs:
                        visit(child)

    visit(graph.graph)
    input_names = [value.name for value in graph.graph.input]
    output_names = [value.name for value in graph.graph.output]
    if input_names != list(INPUT_NAMES):
        raise AssertionError(f"graph inputs {input_names} != {list(INPUT_NAMES)}")
    if output_names != list(OUTPUT_NAMES):
        raise AssertionError(f"graph outputs {output_names} != {list(OUTPUT_NAMES)}")
    return {
        **counts,
        "opset": imports[""],
        "inputs": input_names,
        "outputs": output_names,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def torch_inputs(inputs: dict[str, np.ndarray]) -> tuple[torch.Tensor, ...]:
    return tuple(torch.from_numpy(inputs[name]) for name in INPUT_NAMES)


def numpy_outputs(outputs: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
    return {
        name: value.detach().cpu().numpy()
        for name, value in zip(OUTPUT_NAMES, outputs)
    }


def run_explicit(
    session: ort.InferenceSession, inputs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    values = session.run(list(OUTPUT_NAMES), inputs)
    return dict(zip(OUTPUT_NAMES, values))


def run_marginals(
    session: ort.InferenceSession,
    text_states: np.ndarray,
    text_mask: np.ndarray,
    query_states: np.ndarray,
    query_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    values = session.run(
        list(MARGINAL_OUTPUT_NAMES),
        {
            "text_states": text_states,
            "text_mask": text_mask,
            "query_states": query_states,
            "query_mask": query_mask,
        },
    )
    return dict(zip(MARGINAL_OUTPUT_NAMES, values))


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
    failures: list[str],
    *,
    atol: float,
    rtol: float,
    confidence_atol: float,
) -> None:
    for name in OUTPUT_NAMES:
        reference = expected[name]
        observed = actual[name]
        try:
            if reference.dtype == np.bool_:
                if reference.shape != observed.shape or not np.array_equal(
                    reference, observed
                ):
                    raise AssertionError(f"{label}/{name}: boolean output differs")
                report = {"max_abs": 0.0, "max_rel": 0.0}
            else:
                report = compare_arrays(
                    f"{label}/{name}",
                    reference,
                    observed,
                    atol=atol,
                    rtol=rtol,
                )
            update_stats(stats, name, report)
        except AssertionError as error:
            failures.append(str(error))
    if "pair_logits" in expected and "pair_logits" in actual:
        reference_confidence = 1.0 / (
            1.0 + np.exp(-np.clip(expected["pair_logits"], -80.0, 80.0))
        )
        actual_confidence = 1.0 / (
            1.0 + np.exp(-np.clip(actual["pair_logits"], -80.0, 80.0))
        )
        maximum = float(
            np.max(np.abs(reference_confidence - actual_confidence), initial=0.0)
        )
        confidence = stats.setdefault(
            "final_confidence",
            {"comparisons": 0, "max_abs": 0.0, "max_rel": 0.0},
        )
        confidence["comparisons"] = int(confidence["comparisons"]) + 1
        confidence["max_abs"] = max(float(confidence["max_abs"]), maximum)
        if maximum > confidence_atol:
            failures.append(
                f"{label}/final_confidence: max_abs={maximum:.9g} exceeds "
                f"{confidence_atol:.9g}"
            )


def assert_mask_contract(
    label: str,
    outputs: dict[str, np.ndarray],
    failures: list[str],
) -> dict[str, int]:
    legal = outputs["legal_mask"]
    invalid = ~legal
    if invalid.any():
        compatibility = outputs["compatibility"][invalid]
        pair_logits = outputs["pair_logits"][invalid]
        if not np.array_equal(compatibility, np.zeros_like(compatibility)):
            failures.append(f"{label}: invalid compatibility is not exactly zero")
        if not np.array_equal(
            pair_logits,
            np.full(pair_logits.shape, MASK_LOGIT, dtype=np.float32),
        ):
            failures.append(
                f"{label}: invalid pair logits are not exactly {MASK_LOGIT}"
            )
    return {"legal_pairs": int(legal.sum()), "invalid_pairs": int(invalid.sum())}


def frozen_inputs(
    data: np.lib.npyio.NpzFile,
    *,
    marginal_index: int,
    candidate_indices: np.ndarray,
    candidate_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    prefix = f"marginals_{marginal_index}_"
    return {
        "boundary_states": data[prefix + "boundary_states"].astype(
            np.float32, copy=True
        ),
        "text_states": data[prefix + "text_states"].astype(np.float32, copy=True),
        "text_mask": data[prefix + "text_mask"].astype(bool, copy=True),
        "query_states": data[prefix + "query_states"].astype(np.float32, copy=True),
        "query_mask": data[prefix + "query_mask"].astype(bool, copy=True),
        "start_logits": data[prefix + "start_logits"].astype(np.float32, copy=True),
        "end_logits": data[prefix + "end_logits"].astype(np.float32, copy=True),
        "inside_prefix": data[prefix + "inside_prefix"].astype(np.float32, copy=True),
        "inside_prefix_mean": data[prefix + "inside_prefix_mean"].astype(
            np.float32, copy=True
        ),
        "candidate_indices": candidate_indices.astype(np.int64, copy=True),
        "candidate_mask": candidate_mask.astype(bool, copy=True),
    }


def shared_explicit_inputs(data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    query_count = data["marginals_0_query_states"].shape[1]
    indices = np.repeat(
        data["shared_scorer_0_indices"][:, None, :, :], query_count, axis=1
    )
    mask = np.repeat(data["shared_scorer_0_mask"][:, None, :], query_count, axis=1)
    return frozen_inputs(
        data,
        marginal_index=0,
        candidate_indices=indices,
        candidate_mask=mask,
    )


def live_outputs(
    model: torch.nn.Module,
    wrapper: BoundaryExplicitScorerGraph,
    inputs: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    tensors = torch_inputs(inputs)
    with torch.inference_mode():
        composed = independently_composed_outputs(model, *tensors)
        wrapped = wrapper(*tensors)
        full = full_method_outputs(
            model,
            tensors[1],
            tensors[2],
            tensors[3],
            tensors[4],
            tensors[9],
            tensors[10],
        )
    return (
        numpy_outputs(composed),
        numpy_outputs(wrapped),
        full.detach().cpu().numpy(),
    )


def with_m2_marginals(
    base: dict[str, np.ndarray], marginal_outputs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    result = dict(base)
    for name in (
        "boundary_states",
        "start_logits",
        "end_logits",
        "inside_prefix",
        "inside_prefix_mean",
    ):
        result[name] = marginal_outputs[name]
    return result


def synthetic_case(
    model: torch.nn.Module,
    *,
    seed: int,
    batch: int,
    length: int,
    queries: int,
    candidates: int,
) -> dict[str, np.ndarray]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = int(model.hidden_size)
    text_states = torch.randn(batch, length, hidden, generator=generator)
    query_states = torch.randn(batch, queries, hidden, generator=generator)
    text_mask = torch.ones(batch, length, dtype=torch.bool)
    query_mask = torch.ones(batch, queries, dtype=torch.bool)
    if length > 1:
        text_mask[0, -1] = False
        if batch > 1:
            text_mask[1, max(1, length // 2) :] = False
    if queries > 1:
        query_mask[0, -1] = False
        if batch > 1:
            query_mask[1, 0] = False
    indices = torch.zeros(batch, queries, candidates, 2, dtype=torch.long)
    candidate_mask = torch.ones(batch, queries, candidates, dtype=torch.bool)
    for batch_index in range(batch):
        valid_length = max(int(text_mask[batch_index].sum()), 1)
        for query in range(queries):
            for candidate in range(candidates):
                start = candidate % valid_length
                end = min(valid_length, start + 1 + candidate % 4)
                indices[batch_index, query, candidate] = torch.tensor((start, end))
    with torch.inference_mode():
        encoding = model.boundary_head.boundary_encoder(text_states, text_mask)
        marginals = model.boundary_head.boundary_query_head(
            encoding.states,
            encoding.mask,
            text_states,
            text_mask,
            query_states,
            query_mask,
        )
    return {
        "boundary_states": encoding.states.numpy(),
        "text_states": text_states.numpy(),
        "text_mask": text_mask.numpy(),
        "query_states": query_states.numpy(),
        "query_mask": query_mask.numpy(),
        "start_logits": marginals.start_logits.numpy(),
        "end_logits": marginals.end_logits.numpy(),
        "inside_prefix": marginals.inside_prefix.numpy(),
        "inside_prefix_mean": marginals.inside_prefix_mean.numpy(),
        "candidate_indices": indices.numpy(),
        "candidate_mask": candidate_mask.numpy(),
    }


def invalid_case(model: torch.nn.Module, seed: int) -> dict[str, np.ndarray]:
    inputs = synthetic_case(
        model, seed=seed, batch=2, length=8, queries=2, candidates=7
    )
    indices = inputs["candidate_indices"]
    mask = inputs["candidate_mask"]
    # Negative, reversed, zero-width, out-of-range, supplied false, then legal.
    indices[:, :, 0] = (-1, 1)
    indices[:, :, 1] = (3, 2)
    indices[:, :, 2] = (2, 2)
    indices[:, :, 3] = (0, 99)
    indices[:, :, 4] = (0, 1)
    mask[:, :, 4] = False
    inputs["query_mask"][:, 1] = False
    return inputs


def expected_legal(inputs: dict[str, np.ndarray]) -> np.ndarray:
    lengths = inputs["text_mask"].sum(axis=1).reshape(-1, 1, 1)
    starts = inputs["candidate_indices"][..., 0]
    ends = inputs["candidate_indices"][..., 1]
    return (
        (starts >= 0)
        & (ends > starts)
        & (ends <= lengths)
        & inputs["query_mask"][:, :, None]
        & inputs["candidate_mask"]
    )


def isolated_zero_probe(
    onnx_path: Path, inputs: dict[str, np.ndarray], label: str
) -> dict[str, object]:
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
                    "import numpy as np,onnxruntime as ort,sys;"
                    "z=np.load(sys.argv[2]);"
                    "s=ort.InferenceSession(sys.argv[1],providers=['CPUExecutionProvider']);"
                    "n=[o.name for o in s.get_outputs()];"
                    "v=s.run(n,{k:z[k] for k in "
                    + repr(list(INPUT_NAMES))
                    + "});np.savez(sys.argv[3],**dict(zip(n,v)))"
                ),
                str(onnx_path),
                str(input_path),
                str(output_path),
            ],
            capture_output=True,
            check=False,
        )
        result: dict[str, object] = {
            "label": label,
            "contract": "rejected-before-ORT",
            "isolated_returncode": child.returncode,
        }
        if child.returncode == 0:
            with np.load(output_path, allow_pickle=False) as values:
                result["isolated_shapes"] = {
                    name: list(values[name].shape) for name in OUTPUT_NAMES
                }
                result["isolated_finite"] = all(
                    values[name].dtype == np.bool_ or np.isfinite(values[name]).all()
                    for name in OUTPUT_NAMES
                )
        else:
            result["isolated_status"] = "native-runtime-rejected-or-failed"
            result["stderr_tail"] = child.stderr.decode(errors="replace")[-500:]
        return result


def zero_inputs(
    base: dict[str, np.ndarray], *, zero: str, hidden: int
) -> dict[str, np.ndarray]:
    if zero == "q0":
        result = dict(base)
        for name in (
            "query_states",
            "query_mask",
            "start_logits",
            "end_logits",
            "inside_prefix",
            "inside_prefix_mean",
            "candidate_indices",
            "candidate_mask",
        ):
            result[name] = result[name][:, :0]
        return result
    if zero == "c0":
        result = dict(base)
        result["candidate_indices"] = result["candidate_indices"][:, :, :0]
        result["candidate_mask"] = result["candidate_mask"][:, :, :0]
        return result
    if zero == "l0":
        batch, queries = base["query_mask"].shape
        return {
            "boundary_states": np.zeros((batch, 1, 128), dtype=np.float32),
            "text_states": np.zeros((batch, 0, hidden), dtype=np.float32),
            "text_mask": np.zeros((batch, 0), dtype=bool),
            "query_states": base["query_states"],
            "query_mask": base["query_mask"],
            "start_logits": np.zeros((batch, queries, 1), dtype=np.float32),
            "end_logits": np.zeros((batch, queries, 1), dtype=np.float32),
            "inside_prefix": np.zeros((batch, queries, 1), dtype=np.float32),
            "inside_prefix_mean": np.zeros((batch, queries, 1), dtype=np.float32),
            "candidate_indices": base["candidate_indices"],
            "candidate_mask": base["candidate_mask"],
        }
    raise ValueError(zero)


def main() -> None:
    args = parse_args()
    if not ort.__version__.startswith("1.20."):
        raise RuntimeError(
            f"validation requires ONNX Runtime 1.20.x, found {ort.__version__}"
        )
    configure_determinism(args.seed)
    onnx_path = Path(args.onnx_path)
    marginals_path = Path(args.marginals_onnx)
    graph_audit = finite_graph_audit(onnx_path)
    model, config = load_reference_model(args.model_dir)
    validate_config(config)
    wrapper = BoundaryExplicitScorerGraph(model).float().eval()
    explicit_session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    marginal_session = ort.InferenceSession(
        str(marginals_path), providers=["CPUExecutionProvider"]
    )

    fixtures = sorted(Path(args.golden_dir).glob("*.npz"))
    if len(fixtures) != 30:
        raise AssertionError(
            f"full-corpus validation requires 30 NPZ fixtures, found {len(fixtures)}"
        )
    head_fixtures: list[Path] = []
    for fixture in fixtures:
        with np.load(fixture, allow_pickle=False) as data:
            if "shared_scorer_0_indices" in data.files:
                head_fixtures.append(fixture)
    if len(head_fixtures) != 24:
        raise AssertionError(
            f"expected all 24 actual shared-head fixtures, found {len(head_fixtures)}"
        )

    failures: list[str] = []
    counts: Counter[str] = Counter()
    masks: Counter[str] = Counter()
    frozen_stats: dict[str, dict[str, float | int]] = {}
    onnx_stats: dict[str, dict[str, float | int]] = {}
    full_stats: dict[str, dict[str, float | int]] = {}
    m2_stats: dict[str, dict[str, float | int]] = {}
    synthetic_stats: dict[str, dict[str, float | int]] = {}

    for fixture in head_fixtures:
        with np.load(fixture, allow_pickle=False) as data:
            inputs = shared_explicit_inputs(data)
        composed, wrapped, full = live_outputs(model, wrapper, inputs)
        compare_output_sets(
            f"{fixture.stem}/wrapper-vs-composed",
            composed,
            wrapped,
            frozen_stats,
            failures,
            atol=args.atol,
            rtol=args.rtol,
            confidence_atol=args.confidence_atol,
        )
        compare_output_sets(
            f"{fixture.stem}/frozen-marginals-vs-full",
            {**composed, "pair_logits": full},
            composed,
            full_stats,
            failures,
            atol=args.atol,
            rtol=args.rtol,
            confidence_atol=args.confidence_atol,
        )
        actual = run_explicit(explicit_session, inputs)
        compare_output_sets(
            f"{fixture.stem}/explicit-onnx-vs-composed",
            composed,
            actual,
            onnx_stats,
            failures,
            atol=args.atol,
            rtol=args.rtol,
            confidence_atol=args.confidence_atol,
        )
        masks.update(assert_mask_contract(fixture.stem, actual, failures))

        m2 = run_marginals(
            marginal_session,
            inputs["text_states"],
            inputs["text_mask"],
            inputs["query_states"],
            inputs["query_mask"],
        )
        composed_inputs = with_m2_marginals(inputs, m2)
        composed_m2, _wrapped_m2, _full_again = live_outputs(
            model, wrapper, composed_inputs
        )
        actual_m2 = run_explicit(explicit_session, composed_inputs)
        compare_output_sets(
            f"{fixture.stem}/m2-to-explicit-onnx-vs-full-upstream",
            {**composed_m2, "pair_logits": full},
            actual_m2,
            m2_stats,
            failures,
            atol=args.atol,
            rtol=args.rtol,
            confidence_atol=args.confidence_atol,
        )
        counts["shared_indices_explicit_cases"] += 1
        if inputs["candidate_indices"].shape[2] != 192:
            failures.append(
                f"{fixture.stem}: shared candidate count is "
                f"{inputs['candidate_indices'].shape[2]}, expected 192"
            )
        else:
            counts["candidate_count_192_cases"] += 1
        if fixture.stem.startswith("long_"):
            counts["long_cases"] += 1
        print(
            f"{fixture.stem}: L={inputs['text_states'].shape[1]} "
            f"Q={inputs['query_states'].shape[1]} C={inputs['candidate_indices'].shape[2]} ok",
            flush=True,
        )
        del composed, wrapped, full, actual, m2, composed_m2, actual_m2, inputs
        gc.collect()

    choice_path = Path(args.golden_dir) / "json_natural_choice.npz"
    with np.load(choice_path, allow_pickle=False) as data:
        explicit_ids = sorted(
            int(name.split("_")[2])
            for name in data.files
            if name.startswith("explicit_scorer_") and name.endswith("_indices")
        )
        if explicit_ids != [0, 1]:
            raise AssertionError(f"unexpected explicit choice captures: {explicit_ids}")
        for explicit_index in explicit_ids:
            marginal_index = explicit_index + 1
            inputs = frozen_inputs(
                data,
                marginal_index=marginal_index,
                candidate_indices=data[
                    f"explicit_scorer_{explicit_index}_indices"
                ],
                candidate_mask=data[
                    f"explicit_scorer_{explicit_index}_valid_mask"
                ],
            )
            composed, wrapped, full = live_outputs(model, wrapper, inputs)
            frozen_capture = {
                "pair_logits": data[
                    f"explicit_scorer_{explicit_index}_pair_logits"
                ].astype(np.float32, copy=True),
                "compatibility": data[
                    f"explicit_scorer_{explicit_index}_compat_logits"
                ].astype(np.float32, copy=True),
                "legal_mask": data[
                    f"explicit_scorer_{explicit_index}_valid_mask"
                ].astype(bool, copy=True),
            }
            compare_output_sets(
                f"choice-{explicit_index}/composed-vs-frozen-capture",
                frozen_capture,
                composed,
                frozen_stats,
                failures,
                atol=args.atol,
                rtol=args.rtol,
                confidence_atol=args.confidence_atol,
            )
            compare_output_sets(
                f"choice-{explicit_index}/wrapper-vs-composed",
                composed,
                wrapped,
                frozen_stats,
                failures,
                atol=args.atol,
                rtol=args.rtol,
                confidence_atol=args.confidence_atol,
            )
            actual = run_explicit(explicit_session, inputs)
            compare_output_sets(
                f"choice-{explicit_index}/onnx-vs-frozen-capture",
                frozen_capture,
                actual,
                onnx_stats,
                failures,
                atol=args.atol,
                rtol=args.rtol,
                confidence_atol=args.confidence_atol,
            )
            compare_output_sets(
                f"choice-{explicit_index}/full-vs-frozen-capture",
                {**frozen_capture, "pair_logits": full},
                frozen_capture,
                full_stats,
                failures,
                atol=args.atol,
                rtol=args.rtol,
                confidence_atol=args.confidence_atol,
            )
            m2 = run_marginals(
                marginal_session,
                inputs["text_states"],
                inputs["text_mask"],
                inputs["query_states"],
                inputs["query_mask"],
            )
            m2_inputs = with_m2_marginals(inputs, m2)
            actual_m2 = run_explicit(explicit_session, m2_inputs)
            compare_output_sets(
                f"choice-{explicit_index}/m2-to-explicit-vs-full",
                {**frozen_capture, "pair_logits": full},
                actual_m2,
                m2_stats,
                failures,
                atol=args.atol,
                rtol=args.rtol,
                confidence_atol=args.confidence_atol,
            )
            counts["frozen_choice_captures"] += 1

    for index, (batch, length, queries, candidates) in enumerate(
        ((1, 1, 1, 1), (1, 9, 3, 7), (2, 17, 4, 7))
    ):
        inputs = synthetic_case(
            model,
            seed=args.seed + 100 + index,
            batch=batch,
            length=length,
            queries=queries,
            candidates=candidates,
        )
        composed, wrapped, full = live_outputs(model, wrapper, inputs)
        actual = run_explicit(explicit_session, inputs)
        compare_output_sets(
            f"synthetic-b{batch}-l{length}-q{queries}-c{candidates}",
            composed,
            actual,
            synthetic_stats,
            failures,
            atol=args.atol,
            rtol=args.rtol,
            confidence_atol=args.confidence_atol,
        )
        compare_output_sets(
            f"synthetic-full-b{batch}-l{length}-q{queries}-c{candidates}",
            {**composed, "pair_logits": full},
            wrapped,
            full_stats,
            failures,
            atol=args.atol,
            rtol=args.rtol,
            confidence_atol=args.confidence_atol,
        )
        counts["synthetic_dynamic_cases"] += 1
        counts[f"candidate_count_{candidates}_synthetic_cases"] += 1

    invalid_inputs = invalid_case(model, args.seed + 200)
    composed, _wrapped, full = live_outputs(model, wrapper, invalid_inputs)
    invalid_actual = run_explicit(explicit_session, invalid_inputs)
    compare_output_sets(
        "synthetic-invalids",
        composed,
        invalid_actual,
        synthetic_stats,
        failures,
        atol=args.atol,
        rtol=args.rtol,
        confidence_atol=args.confidence_atol,
    )
    legal = expected_legal(invalid_inputs)
    if not np.array_equal(invalid_actual["legal_mask"], legal):
        failures.append("synthetic-invalids: graph legal mask differs from exact rule")
    masks.update(assert_mask_contract("synthetic-invalids", invalid_actual, failures))
    if np.any(full[~legal] != MASK_LOGIT):
        failures.append("synthetic-invalids: untouched full method mask differs")
    counts["invalid_negative_reversed_out_of_range_cases"] += 1
    counts["supplied_false_mask_cases"] += 1
    counts["query_masked_cases"] += 1
    counts["batch_two_cases"] += 2

    base = synthetic_case(
        model, seed=args.seed + 300, batch=1, length=3, queries=2, candidates=3
    )
    zero_status = {
        zero: isolated_zero_probe(
            onnx_path,
            zero_inputs(base, zero=zero, hidden=int(model.hidden_size)),
            zero,
        )
        for zero in ("q0", "c0", "l0")
    }
    counts["isolated_zero_dimension_probes"] = 3

    expected_counts = {
        "shared_indices_explicit_cases": 24,
        "candidate_count_192_cases": 24,
        "candidate_count_1_synthetic_cases": 1,
        "candidate_count_7_synthetic_cases": 2,
        "long_cases": 3,
        "frozen_choice_captures": 2,
        "synthetic_dynamic_cases": 3,
        "invalid_negative_reversed_out_of_range_cases": 1,
        "supplied_false_mask_cases": 1,
        "query_masked_cases": 1,
        "batch_two_cases": 2,
        "isolated_zero_dimension_probes": 3,
    }
    for name, expected in expected_counts.items():
        if counts[name] != expected:
            failures.append(f"{name}: {counts[name]} != {expected}")

    report = {
        "onnx": str(onnx_path),
        "marginals_onnx": str(marginals_path),
        "onnxruntime": ort.__version__,
        "atol": args.atol,
        "rtol": args.rtol,
        "final_confidence_atol": args.confidence_atol,
        "graph_audit": graph_audit,
        "fixture_count": len(fixtures),
        "case_counts": dict(sorted(counts.items())),
        "mask_counts": dict(sorted(masks.items())),
        "zero_dimension_contract": (
            "B,L,Q,C must each be >=1; L0/Q0/C0 are rejected by the caller. "
            "Native behavior is probed only in isolated subprocesses."
        ),
        "zero_dimension_status": zero_status,
        "comparison_note": (
            "shared candidate indices are repeated across queries and evaluated "
            "only by the learned explicit path; shared-pool pair logits are never "
            "used as an explicit-scorer reference"
        ),
        "frozen_wrapper_vs_original_modules": frozen_stats,
        "explicit_onnx_vs_frozen_original_modules": onnx_stats,
        "frozen_marginals_vs_full_upstream_method": full_stats,
        "m2_onnx_to_explicit_onnx_vs_full_upstream_method": m2_stats,
        "synthetic_onnx_vs_original_modules": synthetic_stats,
        "gate_failure_count": len(failures),
        "gate_failures": failures,
    }
    report_text = json.dumps(report, indent=2)
    if args.report_json:
        Path(args.report_json).write_text(report_text + "\n")
    print(report_text)
    if failures:
        raise AssertionError(
            f"explicit scorer validation failed {len(failures)} recorded gates"
        )


if __name__ == "__main__":
    main()
