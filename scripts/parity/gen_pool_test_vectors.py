#!/usr/bin/env python3
"""Generate small shared-pool vectors with the pinned upstream implementation.

This script deliberately calls the unmodified ``DocumentCandidatePool``. Its
linear layers are loaded with deterministic integer matrices, and the actual
projected endpoint tensors are serialized as Rust inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PINNED_COMMIT = "d7c727458bf6929bc9ef5ee04e13c3f717a7c455"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "export"))
from common import assert_pinned_gliner2_installation  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    boundary_mask: list[bool]
    query_mask: list[bool]
    start_logits: list[list[float]]
    end_logits: list[list[float]]
    states: list[list[float]]
    top_k: int
    capacity: int
    quota: int


def logits(rows: int, columns: int, fn) -> list[list[float]]:
    return [[float(fn(q, n)) for n in range(columns)] for q in range(rows)]


def states(n: int) -> list[list[float]]:
    # D=4 keeps compatibility sums bit-exact while still exercising reductions.
    return [
        [float((i % 4) - 1), float((i * 2) % 5 - 2), float(i % 3), 1.0]
        for i in range(n)
    ]


def cases() -> list[Case]:
    return [
        Case(
            "stable_ties",
            [True] * 5,
            [True, True],
            logits(2, 5, lambda _q, _n: 1),
            logits(2, 5, lambda _q, _n: 1),
            states(5),
            5,
            10,
            2,
        ),
        Case(
            "quota_overlap_dedup",
            [True] * 6,
            [True, True, True],
            [
                [9, 8, 2, 1, 0, -1],
                [0, 8, 9, 1, -1, -2],
                [1, 0, -1, 9, 8, 2],
            ],
            [
                [-2, 1, 9, 8, 2, 0],
                [-2, 0, 8, 9, 2, 1],
                [-3, -2, 0, 1, 8, 9],
            ],
            states(6),
            6,
            8,
            2,
        ),
        Case(
            "masked_queries_and_boundaries",
            [True, False, True, True, False, True],
            [True, False, True],
            [
                [4, 100, 3, 2, 100, 1],
                [100, 100, 100, 100, 100, 100],
                [1, 100, 5, 4, 100, 3],
            ],
            [
                [0, 100, 2, 5, 100, 4],
                [100, 100, 100, 100, 100, 100],
                [0, 100, 1, 4, 100, 5],
            ],
            states(6),
            6,
            12,
            3,
        ),
        Case(
            "all_queries_invalid",
            [True, True, True, True],
            [False, False],
            logits(2, 4, lambda q, n: q * 10 + n),
            logits(2, 4, lambda q, n: q * 10 - n),
            states(4),
            4,
            7,
            2,
        ),
        Case(
            "padding",
            [True, True, True],
            [True],
            [[3, 2, 1]],
            [[1, 2, 3]],
            states(3),
            3,
            12,
            1,
        ),
        Case(
            "capacity_truncation",
            [True] * 8,
            [True, True],
            logits(2, 8, lambda q, n: (8 - n) * (q + 1)),
            logits(2, 8, lambda q, n: (n + 1) * (q + 1)),
            states(8),
            8,
            4,
            0,
        ),
        Case(
            "more_than_32_boundaries",
            [True] * 40,
            [True],
            [list(map(float, range(40)))],
            [list(map(float, reversed(range(40))))],
            states(40),
            32,
            20,
            0,
        ),
        Case(
            "quota_zero",
            [True] * 6,
            [True, True],
            logits(2, 6, lambda q, n: n - q),
            logits(2, 6, lambda q, n: 6 - n + q),
            states(6),
            6,
            9,
            0,
        ),
        Case(
            "invalid_placeholder_beats_low_valid_score",
            [True, True, True],
            [True],
            [[-6000, -6000, -6000]],
            [[-6000, -6000, -6000]],
            [[0, 0, 0, 0]] * 3,
            3,
            2,
            0,
        ),
    ]


def as_list(value: torch.Tensor) -> Any:
    return value.detach().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "fixtures" / "pool-test-vectors.json",
    )
    args = parser.parse_args()

    assert_pinned_gliner2_installation()
    from gliner2.models.boundary.pool import DocumentCandidatePool

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    output_cases = []
    for case in cases():
        boundary_states = torch.tensor(case.states, dtype=torch.float32).unsqueeze(0)
        boundary_mask = torch.tensor(case.boundary_mask, dtype=torch.bool).unsqueeze(0)
        query_mask = torch.tensor(case.query_mask, dtype=torch.bool).unsqueeze(0)
        start_logits = torch.tensor(case.start_logits, dtype=torch.float32).unsqueeze(0)
        end_logits = torch.tensor(case.end_logits, dtype=torch.float32).unsqueeze(0)
        d = boundary_states.shape[-1]
        pool = DocumentCandidatePool(
            d,
            pool_boundary_top_k=case.top_k,
            pool_size=case.capacity,
            min_pool_per_query=case.quota,
        ).eval()
        with torch.no_grad():
            pool.start_projection.weight.copy_(torch.eye(d))
            pool.start_projection.bias.zero_()
            # A signed permutation keeps all generated arithmetic simple while
            # making start/end projection inputs observably distinct.
            end_weight = torch.zeros((d, d), dtype=torch.float32)
            end_weight[0, 3] = 1
            end_weight[1, 2] = -1
            end_weight[2, 1] = 1
            end_weight[3, 0] = 1
            pool.end_projection.weight.copy_(end_weight)
            pool.end_projection.bias.zero_()
            start_projection = pool.start_projection(boundary_states)
            end_projection = pool.end_projection(boundary_states)
            expected = pool(
                boundary_states,
                boundary_mask,
                query_mask,
                start_logits,
                end_logits,
            )

        output_cases.append(
            {
                "name": case.name,
                "config": {
                    "boundary_top_k": case.top_k,
                    "capacity": case.capacity,
                    "min_per_query": case.quota,
                },
                "boundary_mask": case.boundary_mask,
                "query_mask": case.query_mask,
                "start_logits": case.start_logits,
                "end_logits": case.end_logits,
                "start_projection": as_list(start_projection[0]),
                "end_projection": as_list(end_projection[0]),
                "expected": {
                    "indices": as_list(expected.indices[0]),
                    "mask": as_list(expected.mask[0]),
                    "compat_logits": as_list(expected.compat_logits[0]),
                    "proposal_logits": as_list(expected.proposal_logits[0]),
                },
            }
        )

    payload = {
        "format_version": 1,
        "oracle": "unmodified pinned GLiNER2 DocumentCandidatePool",
        "upstream_commit": PINNED_COMMIT,
        "torch_version": torch.__version__,
        "cases": output_cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
