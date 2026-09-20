#!/usr/bin/env python3
"""Generate deterministic Hungarian-assignment vectors from pinned GLiNER2."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "export"))
from common import GLINER2_COMMIT, assert_pinned_gliner2_installation  # noqa: E402

SEED = 1729


def cases() -> list[dict[str, Any]]:
    definitions: list[tuple[str, int, int, list[list[float]]]] = [
        (
            "greedy_trap",
            3,
            3,
            [[10.0, 1.0, 1.0], [1.0, 10.0, 1.0], [1.0, 1.0, 100.0]],
        ),
        ("rectangular_wide", 2, 4, [[4.0, 1.0, 3.0, 2.0], [2.0, 0.0, 5.0, 3.0]]),
        (
            "rectangular_tall",
            4,
            2,
            [[4.0, 2.0], [1.0, 0.0], [3.0, 5.0], [2.0, 3.0]],
        ),
        ("negative_values", 3, 3, [[-5.0, -2.0, 0.0], [-4.0, -8.0, -1.0], [-3.0, -6.0, -7.0]]),
        ("all_zero_ties", 4, 4, [[0.0] * 4 for _ in range(4)]),
        (
            "repeated_cost_ties",
            3,
            4,
            [[1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0], [2.0, 2.0, 1.0, 1.0]],
        ),
        ("empty_0x0", 0, 0, []),
        ("empty_0x3", 0, 3, []),
        ("empty_3x0", 3, 0, [[], [], []]),
        (
            "positive_infinite_row",
            3,
            3,
            [[1.0, 2.0, 3.0], [float("inf")] * 3, [3.0, 1.0, 2.0]],
        ),
        (
            "mixed_infinities",
            2,
            3,
            [[float("inf"), -2.0, 4.0], [3.0, float("-inf"), 1.0]],
        ),
        (
            "symmetric_assignment",
            4,
            4,
            [[0.0, 2.0, 2.0, 0.0], [2.0, 0.0, 0.0, 2.0], [2.0, 0.0, 0.0, 2.0], [0.0, 2.0, 2.0, 0.0]],
        ),
    ]

    rng = random.Random(SEED)
    for index in range(16):
        rows = rng.randint(1, 5)
        columns = rng.randint(1, 5)
        # Binary quarters are exact in both JSON/f64 and torch.float64. The
        # narrow range intentionally creates repeated-cost tie cases.
        matrix = [
            [rng.randint(-12, 16) / 4.0 for _ in range(columns)]
            for _ in range(rows)
        ]
        definitions.append((f"random_{index:02d}_{rows}x{columns}", rows, columns, matrix))

    return [
        {"name": name, "rows": rows, "columns": columns, "cost": matrix}
        for name, rows, columns, matrix in definitions
    ]


def json_cost(value: float) -> float | str:
    if value == float("inf"):
        return "+inf"
    if value == float("-inf"):
        return "-inf"
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "fixtures" / "assignment-vectors.json",
    )
    args = parser.parse_args()

    assert_pinned_gliner2_installation()
    if importlib.util.find_spec("scipy") is not None:
        raise RuntimeError(
            "assignment vectors require SciPy to be absent so the pinned internal "
            "shortest-augmenting-path backend is the normative oracle"
        )
    # Initialize the boundary package first; the pinned revision otherwise has
    # a package-level records/matching circular import on a direct submodule import.
    import gliner2.models.boundary  # noqa: F401
    from gliner2.training.matching import linear_sum_assignment

    output_cases = []
    for case in cases():
        matrix = torch.tensor(case["cost"], dtype=torch.float64).reshape(
            case["rows"], case["columns"]
        )
        rows, columns = linear_sum_assignment(matrix)
        output_cases.append(
            {
                **case,
                "cost": [[json_cost(value) for value in row] for row in case["cost"]],
                "expected_rows": rows.tolist(),
                "expected_columns": columns.tolist(),
            }
        )

    payload = {
        "format_version": 1,
        "upstream_commit": GLINER2_COMMIT,
        "backend": "internal_shortest_augmenting_path_scipy_absent",
        "seed": SEED,
        "cases": output_cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} ({len(output_cases)} cases)")


if __name__ == "__main__":
    main()
