#!/usr/bin/env python3
"""Generate compact public explicit-span vectors from the pinned upstream model.

This script uses the untouched GLiNER2 preprocessing, encoder/query routing, and
``BoundaryHead.score_explicit_spans`` implementation. It never reimplements the
learned scorer and never uses an ONNX graph as the reference.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = ROOT / "scripts" / "export"
sys.path.insert(0, str(EXPORT_DIR))

from common import (  # noqa: E402
    BASE_HF_REVISION,
    BASE_MODEL_ID,
    GLINER2_COMMIT,
    assert_official_base_source,
    assert_pinned_gliner2_installation,
    configure_determinism,
    load_reference_model,
    package_versions,
    sha256_file,
)
from gliner2.training.trainer import ExtractorCollator  # noqa: E402

SEED = 1729
MAX_LEN = 4096
PINNED_PYTHON = ROOT / "scripts" / "export" / "env" / ".venv" / "bin" / "python"
DEFAULT_MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--fastino--gliner2.5-base-v1"
    / "snapshots"
    / BASE_HF_REVISION
)
DEFAULT_OUTPUT = ROOT / "fixtures" / "public-explicit-vectors.json"
MAX_OUTPUT_BYTES = 15_000

CASES = (
    {
        "case_id": "english_lines_whitespace",
        "splitter": "whitespace",
        "text": "Alice met Bob.\nAcme hired Carol.",
        "labels": ["person", "organization"],
        "surfaces": ["Bob", "Acme", "Acme hired"],
    },
    {
        "case_id": "unicode_multiline_whitespace",
        "splitter": "whitespace",
        "text": "Zoë lives in São Paulo.\n東京 hosts 李雷.",
        "labels": ["person", "place"],
        "surfaces": ["Zoë", "São Paulo", "東京", "李雷"],
    },
    {
        "case_id": "unicode_compact_char",
        "splitter": "char",
        "text": "北京欢迎Zoë\n東京へ行く",
        "labels": ["place", "person"],
        "surfaces": ["北京", "Zoë", "東京", "行く"],
    },
)


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


def utf8_span(text: str, surface: str) -> list[int]:
    if text.count(surface) != 1:
        raise AssertionError(f"surface {surface!r} must occur exactly once in {text!r}")
    start_cp = text.index(surface)
    end_cp = start_cp + len(surface)
    return [
        len(text[:start_cp].encode("utf-8")),
        len(text[:end_cp].encode("utf-8")),
    ]


def byte_to_codepoint(text: str, offset: int) -> int:
    data = text.encode("utf-8")
    if not 0 <= offset <= len(data):
        raise AssertionError(f"UTF-8 offset {offset} out of range")
    return len(data[:offset].decode("utf-8"))


def map_to_token_indices(batch, text: str, requested: list[list[int]]) -> list[list[int]]:
    starts = batch.start_mappings[0]
    ends = batch.end_mappings[0]
    mapped: list[list[int]] = []
    for start_byte, end_byte in requested:
        start_cp = byte_to_codepoint(text, start_byte)
        end_cp = byte_to_codepoint(text, end_byte)
        try:
            start_word = starts.index(start_cp)
            end_word = ends.index(end_cp)
        except ValueError as exc:
            raise AssertionError(
                f"requested span [{start_byte},{end_byte}) is not an upstream word range"
            ) from exc
        if start_word > end_word:
            raise AssertionError("mapped upstream word range is reversed")
        mapped.append([start_word, end_word + 1])
    return mapped


def generate_case(model, specification: dict[str, Any]) -> dict[str, Any]:
    text = specification["text"]
    labels = specification["labels"]
    splitter = specification["splitter"]
    model.set_word_splitter(splitter)

    schema = model.create_schema()
    schema.entities(labels)
    schema_dicts, _ = model._build_schema_dicts_and_metadata([schema])
    collator = ExtractorCollator(
        model.processor,
        is_training=False,
        max_len=MAX_LEN,
        architecture="boundary",
    )
    batch = collator([(text, schema_dicts[0])])

    requested = [utf8_span(text, surface) for surface in specification["surfaces"]]
    token_indices = map_to_token_indices(batch, text, requested)
    query_layout = batch.query_layouts[0].queries
    routed_labels = [query.role_name for query in query_layout]
    if routed_labels != labels:
        raise AssertionError(
            f"{specification['case_id']}: routed labels {routed_labels!r} != {labels!r}"
        )

    with torch.inference_mode():
        core = model._encode_core(batch)
        query_count = len(labels)
        candidate_count = len(token_indices)
        indices = torch.tensor(
            token_indices,
            dtype=torch.long,
            device=core["text_states"].device,
        ).reshape(1, 1, candidate_count, 2).expand(1, query_count, candidate_count, 2).clone()
        valid_mask = torch.ones(
            (1, query_count, candidate_count),
            dtype=torch.bool,
            device=indices.device,
        )
        logits = model.boundary_head.score_explicit_spans(
            core["text_states"],
            core["text_mask"],
            core["query_states"],
            core["query_mask"],
            indices,
            valid_mask,
        )
        temperature = float(model.boundary_settings.pair_temperature)
        probabilities = torch.sigmoid(logits / temperature)

    if logits.shape != (1, len(labels), len(requested)):
        raise AssertionError(f"unexpected explicit score shape {tuple(logits.shape)}")
    if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all():
        raise AssertionError("upstream explicit scorer returned non-finite values")

    span_entries = []
    encoded = text.encode("utf-8")
    for surface, bounds, token_pair in zip(
        specification["surfaces"], requested, token_indices
    ):
        start, end = bounds
        if encoded[start:end].decode("utf-8") != surface:
            raise AssertionError("requested UTF-8 bounds do not reproduce the source surface")
        span_entries.append(
            {
                "requested_utf8": bounds,
                "text": surface,
                "token_indices": token_pair,
            }
        )

    return {
        "case_id": specification["case_id"],
        "labels": labels,
        "logits": logits[0].detach().cpu().tolist(),
        "original_text": text,
        "probabilities": probabilities[0].detach().cpu().tolist(),
        "spans": span_entries,
        "splitter": splitter,
        "text_tokens": batch.text_tokens[0],
    }


def main() -> None:
    expected_python = PINNED_PYTHON.resolve()
    actual_python = Path(sys.executable).resolve()
    if actual_python != expected_python:
        raise RuntimeError(
            f"run with pinned interpreter {expected_python}, got {actual_python}"
        )
    if len({case["case_id"] for case in CASES}) != len(CASES):
        raise AssertionError("public explicit case IDs must be unique")

    configure_determinism(SEED, threads=1)
    assert_pinned_gliner2_installation()
    source_hashes = assert_official_base_source(DEFAULT_MODEL_DIR)
    model, config = load_reference_model(DEFAULT_MODEL_DIR)
    if config.get("architecture") != "boundary" or config.get("architecture_version") != 1:
        raise RuntimeError("public explicit vectors require boundary architecture version 1")
    if config.get("boundary_head", {}).get("candidate_pool") != "shared":
        raise RuntimeError("unexpected pinned boundary candidate-pool configuration")

    cases = [generate_case(model, case) for case in CASES]
    document = {
        "cases": cases,
        "format_version": 1,
        "oracle": "untouched BoundaryHead.score_explicit_spans after upstream preprocessing and encoding",
        "provenance": {
            "attention_implementation": "eager",
            "dependencies": package_versions(),
            "device": "cpu",
            "dtype": "float32",
            "generator_sha256": sha256_file(Path(__file__)),
            "gliner2_commit": GLINER2_COMMIT,
            "gliner2_repository": "https://github.com/fastino-ai/GLiNER2",
            "hf_revision": BASE_HF_REVISION,
            "max_len": MAX_LEN,
            "model_id": BASE_MODEL_ID,
            "onnx_used_as_reference": False,
            "pair_temperature": float(model.boundary_settings.pair_temperature),
            "python_executable": "scripts/export/env/.venv/bin/python",
            "seed": SEED,
            "source_file_sha256": source_hashes,
            "torch_threads": torch.get_num_threads(),
        },
    }
    payload = canonical_json(document)
    if len(payload) > MAX_OUTPUT_BYTES:
        raise RuntimeError(
            f"compact public explicit fixture is {len(payload)} bytes, exceeds {MAX_OUTPUT_BYTES}"
        )
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_bytes(payload)
    print(
        f"wrote {DEFAULT_OUTPUT} ({len(payload)} bytes, sha256={hashlib.sha256(payload).hexdigest()})"
    )


if __name__ == "__main__":
    main()
