#!/usr/bin/env python3
"""Generate relation-edge dedup vectors from the pinned unchanged GLiNER2 method.

The oracle is ``BoundaryExtractor._deduplicate_relation_edges`` itself. This
script constructs deterministic synthetic inputs and converts the method's
Python codepoint offsets to the Rust API's UTF-8 byte offsets; it does not
reimplement relation deduplication.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

PINNED_PYTHON = (3, 12, 7)
ENGINE_SHA256 = "80a250e2454ae8b3ae832c98d93d1bedf84113b76b113b42e2eff813313d6595"
METHOD_SHA256 = "26274b4c3fde2efeb0cb5cd50e12a5b1d251ecf79979cd9eef2e2249c12bb018"

if sys.version_info[:3] != PINNED_PYTHON:
    raise SystemExit(f"requires Python {PINNED_PYTHON}, got {sys.version_info[:3]}")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "export"))
from common import GLINER2_COMMIT, assert_pinned_gliner2_installation  # noqa: E402

Mention = tuple[str, int, int]
Edge = dict[str, Any]


def occurrence(text: str, value: str, occurrence_index: int = 0) -> Mention:
    start = -1
    search_from = 0
    for _ in range(occurrence_index + 1):
        start = text.find(value, search_from)
        if start < 0:
            raise ValueError(
                f"missing occurrence {occurrence_index} of {value!r} in {text!r}"
            )
        search_from = start + len(value)
    return (value, start, start + len(value))


def edge(head: Mention, tail: Mention, score: float) -> Edge:
    return {"head": head, "tail": tail, "score": score}


def definitions() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    cases.append({"name": "empty", "text": "", "edges": []})

    text = "Alice--Bob"
    cases.append(
        {
            "name": "single",
            "text": text,
            "edges": [edge(occurrence(text, "Alice"), occurrence(text, "Bob"), 0.5)],
        }
    )

    text = "The New York office hired Ada"
    cases.append(
        {
            "name": "canonical_overlap_exact_higher_and_equal_tie",
            "text": text,
            "edges": [
                edge(occurrence(text, "York"), occurrence(text, "Ada"), 0.8),
                edge(occurrence(text, "New York"), occurrence(text, "Ada"), 0.4),
                edge(occurrence(text, "New York"), occurrence(text, "Ada"), 0.8),
            ],
        }
    )

    text = "abcdef -> T"
    cases.append(
        {
            "name": "canonical_equal_length_tie_earlier_start",
            "text": text,
            "edges": [
                edge(occurrence(text, "cd"), occurrence(text, "T"), 0.9),
                edge(occurrence(text, "abcde"), occurrence(text, "T"), 0.1),
                edge(occurrence(text, "bcdef"), occurrence(text, "T"), 0.2),
            ],
        }
    )

    text = "aaaaX😀😀 -> T"
    cases.append(
        {
            "name": "canonical_codepoint_width_not_byte_width",
            "text": text,
            "edges": [
                edge(occurrence(text, "X"), occurrence(text, "T"), 0.9),
                edge(occurrence(text, "aaaaX"), occurrence(text, "T"), 0.1),
                edge(occurrence(text, "X😀😀"), occurrence(text, "T"), 0.2),
            ],
        }
    )

    text = "Alice met BOB. Later alice met Bob."
    cases.append(
        {
            "name": "repeated_semantic_occurrence_nearest",
            "text": text,
            "edges": [
                edge(occurrence(text, "Alice"), occurrence(text, "BOB"), 0.2),
                edge(occurrence(text, "Alice"), occurrence(text, "Bob"), 0.99),
                edge(occurrence(text, "alice"), occurrence(text, "Bob"), 0.4),
            ],
        }
    )

    text = "H你你T H.....T"
    cases.append(
        {
            "name": "semantic_codepoint_gap_not_byte_gap",
            "text": text,
            "edges": [
                edge(occurrence(text, "H", 0), occurrence(text, "T", 0), 0.2),
                edge(occurrence(text, "H", 1), occurrence(text, "T", 1), 0.9),
            ],
        }
    )

    text = "Straße--X STRASSE--x"
    cases.append(
        {
            "name": "full_casefold_sharp_s",
            "text": text,
            "edges": [
                edge(occurrence(text, "Straße"), occurrence(text, "X"), 0.3),
                edge(occurrence(text, "STRASSE"), occurrence(text, "x"), 0.8),
            ],
        }
    )

    text = "\u001cAlpha\u001f--Beta"
    cases.append(
        {
            "name": "python_control_whitespace_surface_trim",
            "text": text,
            "edges": [
                edge(("Alpha", 0, text.index("--")), occurrence(text, "Beta"), 0.6)
            ],
        }
    )

    text = "ALPHA\u001cBETA--X alpha beta--x"
    cases.append(
        {
            "name": "python_control_whitespace_semantic_split",
            "text": text,
            "edges": [
                edge(
                    occurrence(text, "ALPHA\u001cBETA"),
                    occurrence(text, "X"),
                    0.3,
                ),
                edge(occurrence(text, "alpha beta"), occurrence(text, "x"), 0.7),
            ],
        }
    )

    text = "York--Acme New York--ACME York--Other"
    cases.append(
        {
            "name": "strict_head_token_subset_with_equal_opposite",
            "text": text,
            "edges": [
                edge(occurrence(text, "York", 0), occurrence(text, "Acme"), 0.9),
                edge(occurrence(text, "New York"), occurrence(text, "ACME"), 0.2),
                edge(occurrence(text, "York", 2), occurrence(text, "Other"), 0.1),
            ],
        }
    )

    text = "Acme--York ACME--New York Other--York"
    cases.append(
        {
            "name": "strict_tail_token_subset_with_equal_opposite",
            "text": text,
            "edges": [
                edge(occurrence(text, "Acme"), occurrence(text, "York", 0), 0.9),
                edge(occurrence(text, "ACME"), occurrence(text, "New York"), 0.2),
                edge(occurrence(text, "Other"), occurrence(text, "York", 2), 0.1),
            ],
        }
    )

    text = "A--B ... a--b"
    cases.append(
        {
            "name": "semantic_rank_first_exact_tie",
            "text": text,
            "edges": [
                edge(occurrence(text, "A"), occurrence(text, "B"), 0.5),
                edge(occurrence(text, "a"), occurrence(text, "b"), 0.5),
            ],
        }
    )

    text = "A B C D"
    cases.append(
        {
            "name": "final_stable_coordinate_score_order",
            "text": text,
            "edges": [
                edge(occurrence(text, "C"), occurrence(text, "D"), 0.9),
                edge(occurrence(text, "A"), occurrence(text, "D"), 0.2),
                edge(occurrence(text, "A"), occurrence(text, "B"), 0.1),
            ],
        }
    )
    return cases


def byte_boundaries(text: str) -> list[int]:
    result = [0]
    total = 0
    for character in text:
        total += len(character.encode("utf-8"))
        result.append(total)
    return result


def byte_mention(mention: Mention, boundaries: list[int]) -> dict[str, Any]:
    value, start, end = mention
    return {
        "text": value,
        "start": boundaries[start],
        "end": boundaries[end],
    }


def byte_edge(value: Edge, boundaries: list[int]) -> dict[str, Any]:
    return {
        "head": byte_mention(value["head"], boundaries),
        "tail": byte_mention(value["tail"], boundaries),
        "score": value["score"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "fixtures/gliner2.5-base-v1/relation-aux/relation-decode-vectors.json",
    )
    args = parser.parse_args()

    assert_pinned_gliner2_installation()
    from gliner2.models.boundary.engine import BoundaryExtractor

    source_path = Path(inspect.getsourcefile(BoundaryExtractor) or "")
    engine_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if engine_sha256 != ENGINE_SHA256:
        raise RuntimeError(
            f"boundary engine hash is {engine_sha256}, expected {ENGINE_SHA256}; "
            "the oracle must be unchanged"
        )
    method_source = inspect.getsource(BoundaryExtractor._deduplicate_relation_edges)
    method_sha256 = hashlib.sha256(method_source.encode("utf-8")).hexdigest()
    if method_sha256 != METHOD_SHA256:
        raise RuntimeError(
            f"dedup method hash is {method_sha256}, expected {METHOD_SHA256}; "
            "the oracle must be unchanged"
        )

    output_cases = []
    for case in definitions():
        # The static method is the sole implementation used to derive expected output.
        expected = BoundaryExtractor._deduplicate_relation_edges(case["edges"])
        boundaries = byte_boundaries(case["text"])
        output_cases.append(
            {
                "name": case["name"],
                "text": case["text"],
                "edges": [byte_edge(value, boundaries) for value in case["edges"]],
                "expected": [byte_edge(value, boundaries) for value in expected],
            }
        )

    payload = {
        "format_version": 1,
        "upstream_commit": GLINER2_COMMIT,
        "python_version": ".".join(map(str, PINNED_PYTHON)),
        "oracle": (
            "unchanged gliner2.models.boundary.engine.BoundaryExtractor."
            "_deduplicate_relation_edges"
        ),
        "engine_sha256": engine_sha256,
        "method_sha256": method_sha256,
        "offset_conversion": "Python codepoint offsets to half-open UTF-8 byte offsets",
        "cases": output_cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {args.output} ({len(output_cases)} cases, "
        f"{args.output.stat().st_size} bytes)"
    )


if __name__ == "__main__":
    main()
