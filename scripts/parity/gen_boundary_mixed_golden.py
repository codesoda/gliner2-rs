#!/usr/bin/env python3
"""Generate the small pinned public-oracle mixed-task boundary fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
EXPORT_DIR = SCRIPT_DIR.parent / "export"
sys.path.insert(0, str(EXPORT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    BASE_HF_REVISION,
    BASE_MODEL_ID,
    GLINER2_COMMIT,
    assert_official_base_source,
    configure_determinism,
    load_reference_model,
    package_versions,
)
from gen_boundary_goldens import build_schema, utf8_result  # noqa: E402

SEED = 1729
MAX_LEN = 4096
CASES = (
    {
        "case_id": "mixed_english",
        "text": "Ada Lovelace enjoyed her visit to London.",
        "schema_spec": {
            "entities": ["person", "location"],
            "classifications": [
                {
                    "task": "sentiment",
                    "labels": ["positive", "negative", "neutral"],
                }
            ],
        },
    },
    {
        "case_id": "mixed_unicode_repeated",
        "text": "Zoë met José in São Paulo, and Zoë later thanked José in São Paulo ☕.",
        "schema_spec": {
            "entities": ["person", "location"],
            "classifications": [
                {
                    "task": "sentiment",
                    "labels": ["positive", "negative", "neutral"],
                }
            ],
        },
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=str(
            Path.home()
            / ".cache/huggingface/hub/models--fastino--gliner2.5-base-v1"
            / "snapshots"
            / BASE_HF_REVISION
        ),
    )
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR.parent.parent / "fixtures" / "boundary-mixed.json"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"


def main() -> None:
    args = parse_args()
    configure_determinism(args.seed, threads=1)
    source_hashes = assert_official_base_source(args.model_dir)
    model, config = load_reference_model(args.model_dir)
    if config.get("architecture") != "boundary" or config.get("architecture_version") != 1:
        raise ValueError("mixed fixture requires a boundary architecture_version 1 checkpoint")

    generated = []
    for case in CASES:
        schema = build_schema(model, case["schema_spec"])
        with torch.inference_mode():
            result = model.extract(
                case["text"],
                schema,
                threshold=0.5,
                format_results=True,
                include_confidence=True,
                include_spans=True,
                max_len=MAX_LEN,
            )
        generated.append(
            {
                **case,
                "threshold": 0.5,
                "final_result_utf8_offsets": utf8_result(result, case["text"]),
            }
        )

    fixture = {
        "format_version": 1,
        "description": "Pinned public-oracle mixed entity and classification outputs.",
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
        },
        "cases": generated,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(fixture))
    print(f"wrote {len(generated)} mixed cases to {output}")


if __name__ == "__main__":
    main()
