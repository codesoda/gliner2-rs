#!/usr/bin/env python3
"""Generate compact decoder oracles with the pinned GLiNER2 implementation.

The expected overlap, grouping, choice-prefix, null-gating, scalar/list, and
character-offset results below are produced by upstream functions. This script
contains case data and JSON adaptation only; it does not implement a second
copy of the resolver or boundary engine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "export"))
from common import GLINER2_COMMIT, assert_pinned_gliner2_installation  # noqa: E402


def f32(value: float) -> float:
    return float(torch.tensor(value, dtype=torch.float32).item())


def overlap_cases(resolve_overlaps) -> list[dict[str, Any]]:
    base = [
        (f32(0.90), 0, 4),
        (f32(0.60), 0, 2),
        (f32(0.60), 2, 4),
    ]
    nested_crossing = [
        (f32(0.95), 0, 6),
        (f32(0.90), 1, 3),
        (f32(0.85), 2, 5),
        (f32(0.80), 6, 8),
    ]
    duplicate = [
        (f32(0.40), 0, 2),
        (f32(0.80), 0, 2),
        (f32(0.70), 2, 4),
        (f32(0.70), 4, 6),
    ]
    cardinality_tie = [
        (f32(1.0), 0, 4),
        (f32(0.5), 0, 2),
        (f32(0.5), 2, 4),
    ]
    lexicographic_tie = [
        (f32(0.5), 0, 2),
        (f32(0.5), 2, 4),
        (f32(0.5), 0, 1),
        (f32(0.5), 1, 4),
    ]
    definitions = [
        ("weighted_not_greedy", "flat", base),
        ("allow_nested_crossing", "allow", nested_crossing),
        ("nested_rejects_crossing", "nested", nested_crossing),
        ("longest_strict_containment", "longest", nested_crossing),
        ("disallow_nested_crossing", "disallow", nested_crossing),
        ("exact_coordinate_duplicate", "all", duplicate),
        ("equal_sum_cardinality", "no-overlap", cardinality_tie),
        ("equal_sum_lexicographic", "non_overlapping", lexicographic_tie),
    ]
    output = []
    for name, policy, spans in definitions:
        expected = resolve_overlaps(
            spans,
            policy,
            score=lambda item: item[0],
            start=lambda item: item[1],
            end=lambda item: item[2],
        )
        output.append(
            {
                "name": name,
                "policy": policy,
                "spans": [list(item) for item in spans],
                "expected": [list(item) for item in expected],
            }
        )
    return output


def grouping_cases(CandidateTensorBatch, group_candidates) -> list[dict[str, Any]]:
    definitions = [
        {
            "name": "threshold_equality_and_masks",
            "indices": [[[0, 1], [1, 2], [2, 3]], [[0, 2], [1, 3], [3, 4]]],
            "valid_mask": [[True, False, True], [True, True, True]],
            "query_mask": [True, False],
            "pair_logits": [[0.0, 9.0, -0.0001], [9.0, 9.0, 9.0]],
            "thresholds": [0.5, 0.1],
            "temperature": 1.0,
        },
        {
            "name": "candidate_order_preserved",
            "indices": [[[4, 5], [0, 1], [2, 4]]],
            "valid_mask": [[True, True, True]],
            "query_mask": [True],
            "pair_logits": [[0.2, 2.0, 1.0]],
            "thresholds": [0.5],
            "temperature": 2.0,
        },
        {
            "name": "zero_candidates",
            "indices": [[]],
            "valid_mask": [[]],
            "query_mask": [True],
            "pair_logits": [[]],
            "thresholds": [0.5],
            "temperature": 1.0,
        },
    ]
    output = []
    for case in definitions:
        q = len(case["query_mask"])
        c = len(case["indices"][0]) if q else 0
        indices = torch.tensor(case["indices"], dtype=torch.long).reshape(1, q, c, 2)
        logits = torch.tensor(case["pair_logits"], dtype=torch.float32).reshape(1, q, c)
        valid_mask = torch.tensor(case["valid_mask"], dtype=torch.bool).reshape(1, q, c)
        query_mask = torch.tensor(case["query_mask"], dtype=torch.bool).reshape(1, q)
        candidates = CandidateTensorBatch(
            indices=indices,
            proposal_logits=None,
            pair_logits=logits,
            valid_mask=valid_mask,
            query_mask=query_mask,
        )
        probabilities = torch.sigmoid(logits / case["temperature"])
        expected = group_candidates(
            candidates,
            threshold=torch.tensor([case["thresholds"]], dtype=torch.float32),
            probabilities=probabilities,
            adaptive_threshold=False,
        )[0]
        output.append(
            {
                **case,
                "expected": [[list(item) for item in query] for query in expected],
            }
        )

    # Q=0 is constructed separately because nested empty Python lists cannot
    # carry the final coordinate dimension without an explicit reshape.
    empty = CandidateTensorBatch(
        indices=torch.empty((1, 0, 0, 2), dtype=torch.long),
        proposal_logits=None,
        pair_logits=torch.empty((1, 0, 0), dtype=torch.float32),
        valid_mask=torch.empty((1, 0, 0), dtype=torch.bool),
        query_mask=torch.empty((1, 0), dtype=torch.bool),
    )
    output.append(
        {
            "name": "zero_queries",
            "indices": [],
            "valid_mask": [],
            "query_mask": [],
            "pair_logits": [],
            "thresholds": [],
            "temperature": 1.0,
            "expected": group_candidates(
                empty,
                threshold=torch.empty((1, 0), dtype=torch.float32),
                probabilities=torch.empty((1, 0, 0), dtype=torch.float32),
                adaptive_threshold=False,
            )[0],
        }
    )
    return output


def codepoint_to_byte(text: str, offset: int) -> int:
    return len(text[:offset].encode("utf-8"))


def entity_cases(BoundaryExtractor, RuntimeMixin) -> list[dict[str, Any]]:
    class Harness:
        boundary_settings = SimpleNamespace(abstention_threshold=0.5)
        _decode_entities = BoundaryExtractor._decode_entities
        _format_spans = RuntimeMixin._format_spans

        @staticmethod
        def _attach_entity_attributes(*_args, **_kwargs) -> None:
            return None

    harness = Harness()
    definitions = [
        {
            "name": "unicode_list_with_choice_prefix",
            "text": " Cafe\u0301 東京 😀 ",
            "start_map": [0, 7, 10],
            "end_map": [6, 9, 12],
            "choice_prefix_words": 2,
            "dtype": "list",
            "null_probability": 0.5,
            "policy": "allow",
            "scored": [
                [f32(0.99), 0, 1],
                [f32(0.90), 2, 3],
                [f32(0.80), 3, 5],
                [f32(0.70), 2, 5],
                [f32(0.60), 5, 6],
            ],
        },
        {
            "name": "scalar_takes_first",
            "text": "alpha beta",
            "start_map": [0, 6],
            "end_map": [5, 10],
            "choice_prefix_words": 0,
            "dtype": "str",
            "null_probability": 0.5,
            "policy": "all",
            "scored": [[f32(0.6), 0, 1], [f32(0.9), 1, 2]],
        },
        {
            "name": "null_strictly_above_abstains",
            "text": "alpha beta",
            "start_map": [0, 6],
            "end_map": [5, 10],
            "choice_prefix_words": 0,
            "dtype": "list",
            "null_probability": f32(0.500001),
            "policy": "flat",
            "scored": [[f32(0.9), 0, 1]],
        },
    ]
    output = []
    for case in definitions:
        result = harness._decode_entities(
            0,
            {},
            [{"field_name": "value", "task_type": "entities"}],
            [case["scored"]],
            {
                "entity_order": ["value"],
                "entity_metadata": {
                    "value": {"dtype": case["dtype"], "validators": ()}
                },
            },
            torch.tensor([case["null_probability"]], dtype=torch.float32),
            case["policy"],
            case["choice_prefix_words"],
            case["start_map"],
            case["end_map"],
            case["text"],
            len(case["start_map"]),
            True,
            True,
        )["value"]
        expected_items = result if isinstance(result, list) else ([] if result is None else [result])
        expected = [
            {
                **item,
                "start": codepoint_to_byte(case["text"], item["start"]),
                "end": codepoint_to_byte(case["text"], item["end"]),
            }
            for item in expected_items
        ]
        offsets = [
            [codepoint_to_byte(case["text"], start), codepoint_to_byte(case["text"], end)]
            for start, end in zip(case["start_map"], case["end_map"])
        ]
        output.append(
            {
                "name": case["name"],
                "text": case["text"],
                "offsets": offsets,
                "choice_prefix_words": case["choice_prefix_words"],
                "dtype": case["dtype"],
                "null_probability": case["null_probability"],
                "policy": case["policy"],
                "scored": case["scored"],
                "expected": expected,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "fixtures" / "boundary-decode-vectors.json",
    )
    args = parser.parse_args()

    assert_pinned_gliner2_installation()
    from gliner2.inference.overlap import resolve_overlaps
    from gliner2.inference.runtime import ExtractorRuntimeMixin
    from gliner2.models.boundary.engine import BoundaryExtractor
    from gliner2.models.boundary.model import _group_scored_candidates
    from gliner2.models.outputs import CandidateTensorBatch

    torch.set_num_threads(1)
    payload = {
        "format_version": 1,
        "oracle": "unmodified pinned GLiNER2 overlap/group/boundary engine",
        "upstream_commit": GLINER2_COMMIT,
        "torch_version": torch.__version__,
        "overlap_cases": overlap_cases(resolve_overlaps),
        "grouping_cases": grouping_cases(
            CandidateTensorBatch, _group_scored_candidates
        ),
        "entity_cases": entity_cases(BoundaryExtractor, ExtractorRuntimeMixin),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
