#!/usr/bin/env python3
"""Generate two bounded public relation-pipeline oracle cases.

Run this script only with ``scripts/export/env/.venv/bin/python``.  The oracle is
untouched upstream ``model.extract``; no exported graph participates in expected
output generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "scripts" / "export"
sys.path.insert(0, str(EXPORT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    BASE_HF_REVISION,
    BASE_MODEL_ID,
    GLINER2_COMMIT,
    assert_official_base_source,
    assert_pinned_gliner2_installation,
    configure_determinism,
    load_reference_model,
    package_versions,
)
from gen_boundary_goldens import build_schema, utf8_result  # noqa: E402

SEED = 1729
MAX_LEN = 4096
DEFAULT_MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--fastino--gliner2.5-base-v1"
    / "snapshots"
    / BASE_HF_REVISION
)
DEFAULT_OUTPUT = (
    ROOT
    / "fixtures"
    / "gliner2.5-base-v1"
    / "relation-aux"
    / "relation-pipeline-vectors.json"
)

# Keep this tuple small and explicit: its canonical hash is stored in the fixture.
CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "unicode_relations",
        "original_text": "Zoë works for Café Labs in São Paulo. 李雷 works for 東京研究所.",
        "source_schema_call_order": ["relations"],
        "source_schema_arguments": {
            "relations": [{"name": "works for"}],
        },
        "extract_arguments": {
            "threshold": 0.3,
            "format_results": True,
            "include_confidence": True,
            "include_spans": True,
            "max_len": MAX_LEN,
        },
    },
    {
        "case_id": "mixed_choice_relations",
        "original_text": "Alice works for Acme Corp.",
        # Preserve the original metadata-first schema call order explicitly;
        # upstream's processor uses the same task-family routing order.
        "source_schema_call_order": [
            "structures",
            "entities",
            "relations",
            "classifications",
        ],
        "source_schema_arguments": {
            "structures": [
                {
                    "name": "metadata",
                    "fields": [
                        {
                            "name": "kind",
                            "dtype": "str",
                            "choices": ["employment", "other"],
                        }
                    ],
                }
            ],
            "entities": ["person", "organization"],
            "relations": [
                {
                    "name": "works for",
                    "description": "employment relationship",
                }
            ],
            "classifications": [
                {
                    "task": "sentiment",
                    "labels": ["positive", "negative"],
                }
            ],
        },
        "extract_arguments": {
            "threshold": 0.3,
            "format_results": True,
            "include_confidence": True,
            "include_spans": True,
            "max_len": MAX_LEN,
        },
    },
)
EXPECTED_IDS = ("unicode_relations", "mixed_choice_relations")


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def assert_result_contract(case: dict[str, Any], result: Any) -> None:
    case_id = case["case_id"]
    if not isinstance(result, dict):
        raise AssertionError(f"{case_id}: formatted result is not an object")
    relation_result = result.get("relation_extraction")
    if not isinstance(relation_result, dict):
        raise AssertionError(f"{case_id}: relation_extraction is missing")
    works_for = relation_result.get("works for")
    if not isinstance(works_for, list) or not works_for:
        raise AssertionError(f"{case_id}: upstream works for oracle output is empty")
    if case_id == "mixed_choice_relations":
        required = ("metadata", "entities", "sentiment", "relation_extraction")
        missing = [name for name in required if name not in result]
        if missing:
            raise AssertionError(
                f"{case_id}: upstream mixed output is missing task families {missing}"
            )


def main() -> None:
    pinned_python = (ROOT / "scripts/export/env/.venv/bin/python").resolve()
    if Path(sys.executable).resolve() != pinned_python:
        raise RuntimeError(f"run with the pinned interpreter: {pinned_python}")
    args = parse_args()
    ids = tuple(case["case_id"] for case in CASES)
    if ids != EXPECTED_IDS or len(set(ids)) != 2:
        raise AssertionError(f"relation auxiliary case IDs changed: {ids!r}")

    configure_determinism(args.seed, threads=1)
    assert_pinned_gliner2_installation()
    source_hashes = assert_official_base_source(args.model_dir)
    model, config = load_reference_model(args.model_dir)
    if config.get("architecture") != "boundary":
        raise ValueError("relation pipeline vectors require a boundary checkpoint")
    if config.get("architecture_version") != 1:
        raise ValueError("relation pipeline vectors require architecture_version 1")
    if config.get("boundary_head", {}).get("candidate_pool") != "shared":
        raise ValueError("relation pipeline vectors require the shared candidate pool")

    generated: list[dict[str, Any]] = []
    for source_case in CASES:
        schema_arguments = source_case["source_schema_arguments"]
        schema = build_schema(model, schema_arguments)
        extract_arguments = source_case["extract_arguments"]
        with torch.inference_mode():
            result = model.extract(
                source_case["original_text"],
                schema,
                **extract_arguments,
            )
        assert_result_contract(source_case, result)
        generated.append(
            {
                **source_case,
                "upstream_formatted_result_python_offsets": result,
                "upstream_formatted_result_utf8_offsets": utf8_result(
                    result, source_case["original_text"]
                ),
            }
        )

    cases_sha256 = hashlib.sha256(canonical_json(CASES)).hexdigest()
    fixture = {
        "format_version": 1,
        "description": (
            "Two bounded independent public-oracle cases for Unicode relation "
            "endpoints and combined choice-prefix routing."
        ),
        "case_count": 2,
        "cases_sha256": cases_sha256,
        "oracle": "untouched upstream AutoExtractor model.extract formatted output",
        "provenance": {
            "gliner2_repository": "https://github.com/fastino-ai/GLiNER2",
            "gliner2_commit": GLINER2_COMMIT,
            "model_id": BASE_MODEL_ID,
            "hf_revision": BASE_HF_REVISION,
            "source_file_sha256": source_hashes,
            "architecture": "boundary",
            "architecture_version": 1,
            "attention_implementation": "eager",
            "device": "cpu",
            "dtype": "float32",
            "autocast": False,
            "seed": args.seed,
            "torch_threads": torch.get_num_threads(),
            "dependencies": package_versions(),
            "generator": "scripts/parity/gen_relation_pipeline_vectors.py",
        },
        "cases": generated,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(fixture))
    fixture_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(
        f"wrote exactly 2 relation pipeline cases to {args.output}; "
        f"cases_sha256={cases_sha256}; fixture_sha256={fixture_sha256}"
    )


if __name__ == "__main__":
    main()
