#!/usr/bin/env python3
"""Gate boundary-marginal drift against downstream scorer sensitivity.

This validator uses the pinned, unmodified upstream scorer over every valid
saved pool candidate. It is an acceptance gate, not a statistics-only tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from boundary_marginals import OUTPUT_NAMES
from common import (
    BASE_HF_REVISION,
    GLINER2_COMMIT,
    compare_arrays,
    configure_determinism,
    load_reference_model,
)
from gliner2.models.boundary.pool import PooledCandidates

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--fastino--gliner2.5-base-v1"
    / "snapshots"
    / BASE_HF_REVISION
)
DEFAULT_ONNX_PATH = REPO_ROOT / "onnx/gliner2.5-base-v1/boundary_marginals.onnx"
DEFAULT_GOLDEN_DIR = REPO_ROOT / "fixtures/gliner2.5-base-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--onnx-path", default=str(DEFAULT_ONNX_PATH))
    parser.add_argument("--golden-dir", default=str(DEFAULT_GOLDEN_DIR))
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--confidence-atol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def stats(values: np.ndarray) -> dict[str, float | int]:
    absolute = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    if not np.isfinite(absolute).all():
        raise AssertionError("drift metrics contain non-finite values")
    if absolute.size == 0:
        return {
            "count": 0,
            "mean_abs": 0.0,
            "p95_abs": 0.0,
            "p99_abs": 0.0,
            "max_abs": 0.0,
        }
    return {
        "count": int(absolute.size),
        "mean_abs": float(absolute.mean()),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "max_abs": float(absolute.max()),
    }


def merge_values(parts: list[np.ndarray]) -> dict[str, float | int]:
    values = (
        np.concatenate([np.asarray(value).reshape(-1) for value in parts])
        if parts
        else np.empty(0)
    )
    return stats(values)


def tensor(array: np.ndarray, dtype: torch.dtype | None = None) -> torch.Tensor:
    value = torch.from_numpy(np.array(array, copy=True))
    return value.to(dtype=dtype) if dtype is not None else value


def gather_intervals(
    prefix: np.ndarray,
    mean: np.ndarray,
    indices: np.ndarray,
    query_mask: np.ndarray,
    pool_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the pinned scorer's inside formula in [B,C,Q] order."""
    prefix_tensor = tensor(prefix, torch.float32)
    mean_tensor = tensor(mean, torch.float32)
    index_tensor = tensor(indices, torch.long)
    starts, ends = index_tensor[..., 0], index_tensor[..., 1]
    queries = prefix.shape[1]
    candidates = starts.shape[1]
    start_indices = (
        starts.clamp(0, prefix.shape[2] - 1)
        .unsqueeze(1)
        .expand(-1, queries, candidates)
    )
    end_indices = (
        ends.clamp(0, prefix.shape[2] - 1).unsqueeze(1).expand_as(start_indices)
    )
    interval = prefix_tensor.gather(2, end_indices) - prefix_tensor.gather(
        2, start_indices
    )
    widths = (end_indices - start_indices).to(torch.float32)
    interval = interval + mean_tensor * widths
    normalized = interval / torch.sqrt(widths.clamp_min(1))
    valid = tensor(pool_mask, torch.bool).unsqueeze(1) & tensor(
        query_mask, torch.bool
    ).unsqueeze(-1)
    return (
        interval.transpose(1, 2).numpy(),
        normalized.transpose(1, 2).numpy(),
        valid.transpose(1, 2).numpy(),
    )


def make_pool(data: Any, compat: np.ndarray) -> PooledCandidates:
    return PooledCandidates(
        indices=tensor(data["pool_0_indices"], torch.long),
        mask=tensor(data["pool_0_mask"], torch.bool),
        proposal_logits=tensor(data["pool_0_proposal_logits"], torch.float32),
        gold_mask=None,
        compat_logits=tensor(compat, torch.float32),
    )


def projected_compat(
    start_all: np.ndarray,
    end_all: np.ndarray,
    indices: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    start_tensor = tensor(start_all, torch.float32)
    end_tensor = tensor(end_all, torch.float32)
    index_tensor = tensor(indices, torch.long)
    starts = index_tensor[..., 0].clamp(0, start_tensor.shape[1] - 1)
    ends = index_tensor[..., 1].clamp(0, end_tensor.shape[1] - 1)
    width = start_tensor.shape[-1]
    gathered_start = start_tensor.gather(1, starts.unsqueeze(-1).expand(-1, -1, width))
    gathered_end = end_tensor.gather(1, ends.unsqueeze(-1).expand(-1, -1, width))
    result = (gathered_start * gathered_end).sum(-1) / math.sqrt(width)
    result = torch.where(tensor(mask, torch.bool), result, torch.zeros_like(result))
    return result.numpy()


def run_scorer(
    scorer: torch.nn.Module,
    data: Any,
    outputs: dict[str, np.ndarray],
    compat: np.ndarray,
) -> np.ndarray:
    text_mask = tensor(data["text_mask"], torch.bool)
    with torch.inference_mode():
        score, _ = scorer(
            tensor(outputs["boundary_states"], torch.float32),
            tensor(data["query_states"], torch.float32),
            tensor(data["query_mask"], torch.bool),
            make_pool(data, compat),
            tensor(outputs["start_logits"], torch.float32),
            tensor(outputs["end_logits"], torch.float32),
            tensor(outputs["inside_prefix"], torch.float32),
            text_mask.sum(1).long(),
            tensor(data["text_states"], torch.float32),
            text_mask,
            inside_prefix_mean=tensor(outputs["inside_prefix_mean"], torch.float32),
        )
    result = score.numpy()
    if not np.isfinite(result).all():
        raise AssertionError("upstream scorer produced non-finite logits")
    return result


def long_width_metrics(
    reference_prefix: np.ndarray,
    reference_mean: np.ndarray,
    observed_prefix: np.ndarray,
    observed_mean: np.ndarray,
    valid_length: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    widths = sorted(
        {
            width
            for width in (
                1,
                8,
                32,
                128,
                256,
                512,
                1000,
                1500,
                2000,
                2500,
                3000,
                valid_length,
            )
            if width <= valid_length
        }
    )
    for width in widths:
        starts = np.arange(valid_length - width + 1, dtype=np.int64)
        ends = starts + width
        reference_interval = (
            (reference_prefix[:, :, ends] - reference_prefix[:, :, starts]).astype(
                np.float32
            )
            + (reference_mean * np.float32(width)).astype(np.float32)
        ).astype(np.float32)
        observed_interval = (
            (observed_prefix[:, :, ends] - observed_prefix[:, :, starts]).astype(
                np.float32
            )
            + (observed_mean * np.float32(width)).astype(np.float32)
        ).astype(np.float32)
        delta = observed_interval - reference_interval
        result[str(width)] = {
            "interval": stats(delta),
            "normalized_inside_evidence": stats(delta / np.float32(math.sqrt(width))),
        }
    return result


def top_candidate(score: np.ndarray, valid: np.ndarray, query: int) -> int:
    candidate_ids = np.flatnonzero(valid)
    return int(candidate_ids[np.argmax(score[0, valid, query])])


def main() -> None:
    args = parse_args()
    if not ort.__version__.startswith("1.20."):
        raise RuntimeError(
            f"validation requires ONNX Runtime 1.20.x, found {ort.__version__}"
        )
    configure_determinism(args.seed)
    onnx_path = Path(args.onnx_path)
    golden_dir = Path(args.golden_dir)
    model, config = load_reference_model(args.model_dir)
    if config.get("architecture") != "boundary":
        raise ValueError("expected a boundary checkpoint")
    scorer = model.boundary_head.shared_pool_scorer.eval()
    pair_temperature = float(config["boundary_head"]["pair_temperature"])
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    paths = sorted(golden_dir.glob("*.npz"))
    if len(paths) != 30:
        raise AssertionError(
            f"full-corpus sensitivity requires 30 fixtures, found {len(paths)}"
        )

    all_interval_delta: list[np.ndarray] = []
    all_evidence_delta: list[np.ndarray] = []
    all_isolated_score_delta: list[np.ndarray] = []
    all_full_score_delta: list[np.ndarray] = []
    all_isolated_confidence_delta: list[np.ndarray] = []
    all_full_confidence_delta: list[np.ndarray] = []
    cases: list[dict[str, Any]] = []
    long_intervals: dict[str, Any] = {}
    raw_prefix_failure_count = 0
    raw_prefix_value_count = 0
    valid_pair_count = 0
    head_case_count = 0
    baseline_exact_case_count = 0
    top1_changes_isolated = 0
    top1_changes_full = 0
    threshold_crossings_isolated = 0
    threshold_crossings_full = 0

    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if data["query_states"].shape[1] == 0:
                continue
            head_case_count += 1
            inputs = {
                "text_states": data["text_states"].astype(np.float32, copy=True),
                "text_mask": data["text_mask"].astype(bool, copy=True),
                "query_states": data["query_states"].astype(np.float32, copy=True),
                "query_mask": data["query_mask"].astype(bool, copy=True),
            }
            observed_values = session.run(list(OUTPUT_NAMES), inputs)
            observed = dict(zip(OUTPUT_NAMES, observed_values))
            reference = {
                "boundary_states": data["boundary_encoder_0_states"].copy(),
                "start_logits": data["marginals_0_start_logits"].copy(),
                "end_logits": data["marginals_0_end_logits"].copy(),
                "inside_logits": data["marginals_0_inside_logits"].copy(),
                "inside_prefix": data["marginals_0_inside_prefix"].copy(),
                "inside_prefix_mean": data["marginals_0_inside_prefix_mean"].copy(),
                "start_all": data["pool_start_projection_0_output"].copy(),
                "end_all": data["pool_end_projection_0_output"].copy(),
            }

            prefix_delta = observed["inside_prefix"] - reference["inside_prefix"]
            raw_prefix_value_count += prefix_delta.size
            raw_prefix_failure_count += int(
                np.count_nonzero(
                    ~np.isclose(
                        observed["inside_prefix"],
                        reference["inside_prefix"],
                        atol=args.atol,
                        rtol=args.rtol,
                    )
                )
            )

            reference_interval, reference_evidence, valid = gather_intervals(
                reference["inside_prefix"],
                reference["inside_prefix_mean"],
                data["pool_0_indices"],
                data["query_mask"],
                data["pool_0_mask"],
            )
            observed_interval, observed_evidence, _ = gather_intervals(
                observed["inside_prefix"],
                observed["inside_prefix_mean"],
                data["pool_0_indices"],
                data["query_mask"],
                data["pool_0_mask"],
            )
            reference_evidence_valid = reference_evidence[valid]
            observed_evidence_valid = observed_evidence[valid]
            compare_arrays(
                f"{path.stem}/normalized-inside-evidence",
                reference_evidence_valid,
                observed_evidence_valid,
                atol=args.atol,
                rtol=args.rtol,
            )
            interval_delta = (observed_interval - reference_interval)[valid]
            evidence_delta = observed_evidence_valid - reference_evidence_valid
            all_interval_delta.append(interval_delta)
            all_evidence_delta.append(evidence_delta)
            case_valid_pairs = int(valid.sum())
            valid_pair_count += case_valid_pairs

            saved_compat = data["pool_0_compat_logits"].copy()
            baseline_score = run_scorer(scorer, data, reference, saved_compat)
            saved_score = data["shared_scorer_0_pair_logits_candidate_major"].copy()
            if not np.array_equal(baseline_score, saved_score):
                delta = np.abs(baseline_score - saved_score)
                raise AssertionError(
                    f"{path.stem}: pinned baseline scorer is not bit-exact to golden; "
                    f"max_abs={delta.max(initial=0.0):.9g}"
                )
            baseline_exact_case_count += 1

            isolated = dict(reference)
            isolated["inside_prefix"] = observed["inside_prefix"]
            isolated["inside_prefix_mean"] = observed["inside_prefix_mean"]
            isolated_score = run_scorer(scorer, data, isolated, saved_compat)
            observed_compat = projected_compat(
                observed["start_all"],
                observed["end_all"],
                data["pool_0_indices"],
                data["pool_0_mask"],
            )
            full_score = run_scorer(scorer, data, observed, observed_compat)
            baseline_valid = baseline_score[valid]
            isolated_valid = isolated_score[valid]
            full_valid = full_score[valid]
            compare_arrays(
                f"{path.stem}/isolated-upstream-scorer",
                baseline_valid,
                isolated_valid,
                atol=args.atol,
                rtol=args.rtol,
            )
            compare_arrays(
                f"{path.stem}/full-ort-upstream-scorer",
                baseline_valid,
                full_valid,
                atol=args.atol,
                rtol=args.rtol,
            )
            isolated_score_delta = isolated_valid - baseline_valid
            full_score_delta = full_valid - baseline_valid
            all_isolated_score_delta.append(isolated_score_delta)
            all_full_score_delta.append(full_score_delta)

            baseline_confidence = torch.sigmoid(
                tensor(baseline_score, torch.float32) / pair_temperature
            ).numpy()
            isolated_confidence = torch.sigmoid(
                tensor(isolated_score, torch.float32) / pair_temperature
            ).numpy()
            full_confidence = torch.sigmoid(
                tensor(full_score, torch.float32) / pair_temperature
            ).numpy()
            isolated_confidence_delta = (isolated_confidence - baseline_confidence)[
                valid
            ]
            full_confidence_delta = (full_confidence - baseline_confidence)[valid]
            if (
                np.abs(isolated_confidence_delta).max(initial=0.0)
                > args.confidence_atol
            ):
                raise AssertionError(
                    f"{path.stem}: isolated confidence drift exceeds "
                    f"{args.confidence_atol}"
                )
            if np.abs(full_confidence_delta).max(initial=0.0) > args.confidence_atol:
                raise AssertionError(
                    f"{path.stem}: full ORT confidence drift exceeds "
                    f"{args.confidence_atol}"
                )
            all_isolated_confidence_delta.append(isolated_confidence_delta)
            all_full_confidence_delta.append(full_confidence_delta)
            threshold_crossings_isolated += int(
                np.count_nonzero(
                    ((baseline_confidence >= 0.5) != (isolated_confidence >= 0.5))[
                        valid
                    ]
                )
            )
            threshold_crossings_full += int(
                np.count_nonzero(
                    ((baseline_confidence >= 0.5) != (full_confidence >= 0.5))[valid]
                )
            )
            for query in range(data["query_mask"].shape[1]):
                query_valid = data["pool_0_mask"][0] & bool(
                    data["query_mask"][0, query]
                )
                if not query_valid.any():
                    continue
                baseline_top = top_candidate(baseline_score, query_valid, query)
                isolated_top = top_candidate(isolated_score, query_valid, query)
                full_top = top_candidate(full_score, query_valid, query)
                top1_changes_isolated += int(baseline_top != isolated_top)
                top1_changes_full += int(baseline_top != full_top)

            valid_length = int(data["text_mask"].sum())
            if path.stem.startswith("long_"):
                long_intervals[path.stem] = long_width_metrics(
                    reference["inside_prefix"],
                    reference["inside_prefix_mean"],
                    observed["inside_prefix"],
                    observed["inside_prefix_mean"],
                    valid_length,
                )

            cases.append(
                {
                    "fixture": path.stem,
                    "L": int(data["text_states"].shape[1]),
                    "Q": int(data["query_states"].shape[1]),
                    "valid_candidate_query_pairs": case_valid_pairs,
                    "raw_prefix_delta": stats(prefix_delta),
                    "mean_delta": stats(
                        observed["inside_prefix_mean"] - reference["inside_prefix_mean"]
                    ),
                    "reconstructed_pool_interval_delta": stats(interval_delta),
                    "normalized_pool_inside_evidence_delta": stats(evidence_delta),
                    "isolated_pair_logit_delta": stats(isolated_score_delta),
                    "full_ort_pair_logit_delta": stats(full_score_delta),
                    "isolated_confidence_delta": stats(isolated_confidence_delta),
                    "full_ort_confidence_delta": stats(full_confidence_delta),
                }
            )
        print(f"{path.stem}: {case_valid_pairs} valid pairs ok", flush=True)

    if head_case_count != 24:
        raise AssertionError(f"expected 24 head fixtures, found {head_case_count}")
    if baseline_exact_case_count != 24:
        raise AssertionError(
            "baseline scorer exactness was not established for all 24 head fixtures"
        )
    if top1_changes_isolated or top1_changes_full:
        raise AssertionError(
            "top-1 candidate changed: "
            f"isolated={top1_changes_isolated}, full={top1_changes_full}"
        )
    if threshold_crossings_isolated or threshold_crossings_full:
        raise AssertionError(
            "0.5 confidence crossing observed: "
            f"isolated={threshold_crossings_isolated}, "
            f"full={threshold_crossings_full}"
        )
    if set(long_intervals) != {
        "long_1000_words",
        "long_2000_words",
        "long_3000_words",
    }:
        raise AssertionError("mandatory long-interval metrics are incomplete")

    aggregate = {
        "reconstructed_pool_interval_delta": merge_values(all_interval_delta),
        "normalized_pool_inside_evidence_delta": merge_values(all_evidence_delta),
        "isolated_pair_logit_delta": merge_values(all_isolated_score_delta),
        "full_ort_pair_logit_delta": merge_values(all_full_score_delta),
        "isolated_confidence_delta": merge_values(all_isolated_confidence_delta),
        "full_ort_confidence_delta": merge_values(all_full_confidence_delta),
        "baseline_scorer_vs_saved_golden_bit_exact": True,
        "top1_changes_isolated_prefix": top1_changes_isolated,
        "top1_changes_full_ort": top1_changes_full,
        "confidence_0_5_crossings_isolated_prefix": threshold_crossings_isolated,
        "confidence_0_5_crossings_full_ort": threshold_crossings_full,
    }
    report = {
        "status": "passed",
        "scope": {
            "total_fixtures": len(paths),
            "mandatory_head_fixtures": head_case_count,
            "actual_valid_pool_candidate_query_pairs": valid_pair_count,
            "raw_prefix_values": raw_prefix_value_count,
            "raw_prefix_values_failing_original_isclose": raw_prefix_failure_count,
        },
        "gates": {
            "baseline_upstream_scorer_golden": "bit-exact",
            "normalized_inside_evidence": {
                "atol": args.atol,
                "rtol": args.rtol,
            },
            "upstream_scorer_valid_candidates": {
                "atol": args.atol,
                "rtol": args.rtol,
            },
            "final_confidence_max_abs": args.confidence_atol,
            "top1_changes_allowed": 0,
            "confidence_0_5_crossings_allowed": 0,
        },
        "provenance": {
            "onnx": str(onnx_path.resolve()),
            "onnx_sha256": digest(onnx_path),
            "golden_dir": str(golden_dir.resolve()),
            "onnxruntime": ort.__version__,
            "model_revision": BASE_HF_REVISION,
            "gliner2_commit": GLINER2_COMMIT,
            "python": sys.executable,
        },
        "aggregate": aggregate,
        "long_sliding_intervals": long_intervals,
        "per_fixture": cases,
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["scope"], indent=2))
    print(json.dumps(aggregate, indent=2))
    print(report_path)


if __name__ == "__main__":
    main()
