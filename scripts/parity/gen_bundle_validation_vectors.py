#!/usr/bin/env python3
"""Generate independent source-oracle vectors for one GLiNER2.5 bundle.

The 30-case corpus is generated independently for each immutable checkpoint.
Additional mixed/Unicode and explicit-span cases live below ``additional/`` so
existing stage validators can continue to require exactly 30 top-level NPZs.
No ONNX graph is used to produce expected values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PARITY_DIR = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "scripts" / "export"
for directory in (EXPORT_DIR, PARITY_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from bundle_profiles import PROFILES, verify_source  # noqa: E402
from common import (  # noqa: E402
    GLINER2_COMMIT,
    assert_pinned_gliner2_installation,
    configure_determinism,
    load_reference_model,
    package_versions,
    sha256_file,
)
from gen_boundary_goldens import (  # noqa: E402
    MAX_LEN,
    Capture,
    add_batch_arrays,
    build_expected_batch,
    build_schema,
    canonical_json,
    expand_case,
    install_hooks,
    load_corpus,
    query_metadata,
    reconstruct_routed_states,
    stages_metadata,
    utf8_result,
    write_deterministic_npz,
    write_json,
)

SEED = 1729
EXPECTED_PROFILES = {
    "small": {
        "model_id": "fastino/gliner2.5-small-v1",
        "revision": "f1e4d8fdd6fe328f45dee6aca3e6a07c9db4296e",
        "bundle_name": "gliner2.5-small-v1",
    },
    "base": {
        "model_id": "fastino/gliner2.5-base-v1",
        "revision": "78cea040597df251eedefa9d7ee2a756af39fe64",
        "bundle_name": "gliner2.5-base-v1",
    },
    "multi": {
        "model_id": "fastino/gliner2.5-multi-v1",
        "revision": "235cf92d6d4318da9bfca0d08975c8fa7250d13b",
        "bundle_name": "gliner2.5-multi-v1",
    },
}

MIXED_CASE = {
    "id": "mixed_unicode_all_tasks",
    "category": "mixed",
    "text": "Zoë joined Café Nova in 東京, sells books, and manages 李雷.",
    "schema": {
        "entities": ["person", "company", "location"],
        "classifications": [
            {"task": "language", "labels": ["English", "multilingual"]}
        ],
        "structures": [
            {
                "name": "vendor",
                "mode": "natural",
                "anchor": "company",
                "fields": [
                    {
                        "name": "company",
                        "dtype": "str",
                        "cardinality": "required_one",
                        "exclusive": True,
                    },
                    {
                        "name": "category",
                        "dtype": "str",
                        "choices": ["books", "hardware"],
                        "cardinality": "optional_one",
                        "exclusive": True,
                    },
                ],
            }
        ],
        "relations": [
            {"name": "manages", "description": "person manages person"}
        ],
    },
}

EXPLICIT_CASE = {
    "case_id": "explicit_unicode_duplicates",
    "splitter": "whitespace",
    "text": "Zoë met 李雷 in São Paulo.",
    "labels": ["person", "place"],
    "surfaces": ["Zoë", "李雷", "São Paulo", "Zoë"],
}


def profile_field(profile: Any, name: str) -> Any:
    if isinstance(profile, dict):
        return profile[name]
    return getattr(profile, name)


def checked_profile(name: str) -> Any:
    if name not in EXPECTED_PROFILES or name not in PROFILES:
        raise KeyError(f"unknown or unavailable bundle profile {name!r}")
    profile = PROFILES[name]
    expected = EXPECTED_PROFILES[name]
    for field, value in expected.items():
        actual = profile_field(profile, field)
        if actual != value:
            raise RuntimeError(
                f"bundle profile {name}.{field}={actual!r}, expected {value!r}"
            )
    return profile


def default_model_dir(profile: Any) -> Path:
    model_id = profile_field(profile, "model_id")
    revision = profile_field(profile, "revision")
    cache_name = "models--" + model_id.replace("/", "--")
    return Path.home() / ".cache" / "huggingface" / "hub" / cache_name / "snapshots" / revision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*EXPECTED_PROFILES, "all"), required=True)
    parser.add_argument(
        "--model-dir",
        help="source checkpoint directory; valid only when --model selects one profile",
    )
    parser.add_argument("--corpus", default=str(Path(__file__).with_name("boundary_corpus.json")))
    parser.add_argument(
        "--out-dir",
        default="/tmp/gliner25-work/m7-validation-fixtures",
        help="non-repository root containing per-bundle fixture directories",
    )
    parser.add_argument("--case", action="append", help="generate only a corpus case ID")
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def provenance(
    profile: Any,
    source_hashes: dict[str, str],
    corpus_hash: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "architecture": "boundary",
        "architecture_version": 1,
        "attention_implementation": "eager",
        "autocast": False,
        "corpus_sha256": corpus_hash,
        "dependencies": package_versions(),
        "device": "cpu",
        "dtype": "float32",
        "generator_sha256": sha256_file(Path(__file__)),
        "gliner2_commit": GLINER2_COMMIT,
        "hf_revision": profile_field(profile, "revision"),
        "model_id": profile_field(profile, "model_id"),
        "onnx_used_as_reference": False,
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "seed": seed,
        "source_file_sha256": dict(sorted(source_hashes.items())),
        "torch_threads": torch.get_num_threads(),
    }


def assert_config(config: dict[str, Any]) -> None:
    if config.get("architecture") != "boundary":
        raise ValueError("bundle vectors require a boundary checkpoint")
    if config.get("architecture_version") != 1:
        raise ValueError("bundle vectors require architecture_version 1")
    if config.get("boundary_head", {}).get("candidate_pool") != "shared":
        raise ValueError("bundle vectors require the shared candidate pool")


def array_metadata(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sorted(arrays.items())
    }


def assert_finite_arrays(case_id: str, arrays: dict[str, np.ndarray]) -> None:
    for name, value in arrays.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise AssertionError(f"{case_id}:{name} contains non-finite values")


def generate_extraction_case(
    model: Any,
    case: dict[str, Any],
    out_dir: Path,
    case_provenance: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Capture one untouched public extraction without base-only helpers."""
    case = expand_case(case)
    case_id = case["id"]
    schema = build_schema(model, case["schema"])
    expected_batch = build_expected_batch(model, case["text"], schema)
    capture = Capture()
    install_hooks(model, capture)
    result = None
    error = None
    try:
        with torch.inference_mode():
            result = model.extract(
                case["text"],
                schema,
                threshold=0.5,
                format_results=True,
                include_confidence=True,
                include_spans=True,
                max_len=MAX_LEN,
                overlap_policy=case.get("overlap_policy"),
            )
    except Exception as exc:  # Preserve actual source behavior for review.
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        capture.close()

    arrays = dict(capture.arrays)
    add_batch_arrays(arrays, expected_batch)
    reconstruct_routed_states(arrays, expected_batch, capture)
    assert_finite_arrays(case_id, arrays)
    if "encoder_0_input_ids" in arrays:
        if not np.array_equal(arrays["encoder_0_input_ids"], arrays["input_ids"]):
            raise AssertionError(f"{case_id}: public extraction input IDs changed")
        if not np.array_equal(
            arrays["encoder_0_attention_mask"], arrays["attention_mask"]
        ):
            raise AssertionError(f"{case_id}: public extraction attention mask changed")

    npz_path = out_dir / f"{case_id}.npz"
    json_path = out_dir / f"{case_id}.json"
    write_deterministic_npz(npz_path, arrays)
    metadata = {
        "format_version": 2,
        "case_id": case_id,
        "category": case["category"],
        "text": case["text"],
        # Public offsets and Rust input refer to the caller's unmodified text.
        # The upstream processor can append punctuation, including for empty input.
        "original_text": case["text"],
        "processor_text": expected_batch.original_texts[0],
        "word_count": case["word_count"],
        "schema_spec": case["schema"],
        "query_metadata": query_metadata(expected_batch),
        "schema_tokens": expected_batch.schema_tokens_list[0],
        "task_types": expected_batch.task_types[0],
        "text_tokens": expected_batch.text_tokens[0],
        "max_len": MAX_LEN,
        "threshold": 0.5,
        "overlap_policy": case.get("overlap_policy"),
        "include_spans": True,
        "include_confidence": True,
        "offset_contract": {
            "python_result": "Unicode code points",
            "rust_result": "UTF-8 bytes",
        },
        "status": "error" if error else "ok",
        "error": error,
        "final_result_python_offsets": result,
        "final_result_utf8_offsets": (
            utf8_result(result, case["text"]) if result is not None else None
        ),
        "stages": stages_metadata(capture, case["category"]),
        "captured_calls": dict(sorted(capture.counts.items())),
        "arrays": array_metadata(arrays),
        "provenance": case_provenance,
    }
    write_json(json_path, metadata)
    entry = {
        "case_id": case_id,
        "category": case["category"],
        "word_count": case["word_count"],
        "status": metadata["status"],
        "npz": npz_path.relative_to(out_dir.parent).as_posix(),
        "npz_sha256": sha256_file(npz_path),
        "npz_bytes": npz_path.stat().st_size,
        "json": json_path.relative_to(out_dir.parent).as_posix(),
        "json_sha256": sha256_file(json_path),
        "json_bytes": json_path.stat().st_size,
    }
    return entry, error is None


def utf8_span(text: str, surface: str, occurrence: int = 0) -> list[int]:
    starts = [index for index in range(len(text)) if text.startswith(surface, index)]
    if occurrence >= len(starts):
        raise AssertionError(f"surface {surface!r} occurrence {occurrence} not found")
    start = starts[occurrence]
    return [
        len(text[:start].encode("utf-8")),
        len(text[: start + len(surface)].encode("utf-8")),
    ]


def byte_to_codepoint(text: str, offset: int) -> int:
    encoded = text.encode("utf-8")
    if not 0 <= offset <= len(encoded):
        raise AssertionError(f"UTF-8 offset {offset} is outside the source text")
    return len(encoded[:offset].decode("utf-8"))


def generate_explicit_case(
    model: Any,
    out_dir: Path,
    case_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Call the untouched sparse explicit head with duplicated labels/spans."""
    specification = EXPLICIT_CASE
    text = specification["text"]
    labels = specification["labels"]
    model.set_word_splitter(specification["splitter"])
    schema = model.create_schema()
    schema.entities(labels)
    batch = build_expected_batch(model, text, schema)
    requested = [utf8_span(text, surface) for surface in specification["surfaces"]]
    token_indices = []
    for start_byte, end_byte in requested:
        start_cp = byte_to_codepoint(text, start_byte)
        end_cp = byte_to_codepoint(text, end_byte)
        try:
            start_word = batch.start_mappings[0].index(start_cp)
            end_word = batch.end_mappings[0].index(end_cp)
        except ValueError as exc:
            raise AssertionError("explicit UTF-8 span is not word-boundary aligned") from exc
        token_indices.append([start_word, end_word + 1])

    capture = Capture()
    install_hooks(model, capture)
    try:
        with torch.inference_mode():
            core = model._encode_core(batch)
            query_count = len(labels)
            candidate_count = len(token_indices)
            indices = (
                torch.tensor(token_indices, dtype=torch.long, device=core["text_states"].device)
                .reshape(1, 1, candidate_count, 2)
                .expand(1, query_count, candidate_count, 2)
                .clone()
            )
            valid_mask = torch.ones(
                (1, query_count, candidate_count), dtype=torch.bool, device=indices.device
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
    finally:
        capture.close()
        model.set_word_splitter("whitespace")

    arrays = dict(capture.arrays)
    add_batch_arrays(arrays, batch)
    reconstruct_routed_states(arrays, batch, capture)
    arrays["explicit_requested_indices"] = indices.detach().cpu().numpy().copy()
    arrays["explicit_requested_mask"] = valid_mask.detach().cpu().numpy().copy()
    arrays["explicit_source_logits"] = logits.detach().cpu().numpy().copy()
    arrays["explicit_source_confidence"] = probabilities.detach().cpu().numpy().copy()
    assert_finite_arrays(specification["case_id"], arrays)

    npz_path = out_dir / f"{specification['case_id']}.npz"
    json_path = out_dir / f"{specification['case_id']}.json"
    write_deterministic_npz(npz_path, arrays)
    span_entries = []
    for surface, bounds, token_pair in zip(
        specification["surfaces"], requested, token_indices
    ):
        start, end = bounds
        if text.encode("utf-8")[start:end].decode("utf-8") != surface:
            raise AssertionError("explicit UTF-8 bounds do not reproduce source text")
        span_entries.append(
            {"requested_utf8": bounds, "text": surface, "token_indices": token_pair}
        )
    metadata = {
        "format_version": 2,
        "case_id": specification["case_id"],
        "kind": "explicit_spans",
        "original_text": text,
        "labels": labels,
        "spans": span_entries,
        "splitter": specification["splitter"],
        "text_tokens": batch.text_tokens[0],
        "logits": logits[0].detach().cpu().tolist(),
        "probabilities": probabilities[0].detach().cpu().tolist(),
        "pair_temperature": temperature,
        "oracle": "untouched BoundaryHead.score_explicit_spans",
        "arrays": array_metadata(arrays),
        "provenance": case_provenance,
    }
    write_json(json_path, metadata)
    return {
        "case_id": specification["case_id"],
        "kind": "explicit_spans",
        "status": "ok",
        "npz": npz_path.relative_to(out_dir.parent).as_posix(),
        "npz_sha256": sha256_file(npz_path),
        "npz_bytes": npz_path.stat().st_size,
        "json": json_path.relative_to(out_dir.parent).as_posix(),
        "json_sha256": sha256_file(json_path),
        "json_bytes": json_path.stat().st_size,
    }


def generate_profile(
    name: str,
    model_dir: Path,
    out_root: Path,
    corpus_path: Path,
    selected_cases: list[str] | None,
    seed: int,
    allow_errors: bool,
) -> None:
    profile = checked_profile(name)
    # This content verification must happen before loading any checkpoint.
    source_hashes = verify_source(name, model_dir)
    expected_source = profile_field(profile, "source_sha256")
    if dict(source_hashes) != dict(expected_source):
        raise RuntimeError(f"verify_source({name!r}) returned unexpected hashes")

    cases, corpus_hash = load_corpus(corpus_path)
    if selected_cases:
        selected = set(selected_cases)
        unknown = selected - {case["id"] for case in cases}
        if unknown:
            raise KeyError(f"unknown case IDs: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in selected]

    model, config = load_reference_model(model_dir)
    assert_config(config)
    case_provenance = provenance(profile, source_hashes, corpus_hash, seed)
    bundle_dir = out_root / profile_field(profile, "bundle_name")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    additional_dir = bundle_dir / "additional"
    additional_dir.mkdir(parents=True, exist_ok=True)
    if not selected_cases:
        for pattern in ("*.json", "*.npz"):
            for stale in bundle_dir.glob(pattern):
                stale.unlink()
            for stale in additional_dir.glob(pattern):
                stale.unlink()

    entries: list[dict[str, Any]] = []
    all_ok = True
    for case in cases:
        print(f"[{name}/{case['id']}] source extraction", flush=True)
        entry, ok = generate_extraction_case(model, case, bundle_dir, case_provenance)
        entry["npz"] = Path(entry["npz"]).name
        entry["json"] = Path(entry["json"]).name
        entries.append(entry)
        all_ok &= ok

    additional_entries: list[dict[str, Any]] = []
    if not selected_cases:
        mixed = dict(MIXED_CASE)
        mixed["word_count"] = len(mixed["text"].split())
        entry, ok = generate_extraction_case(
            model, mixed, additional_dir, case_provenance
        )
        additional_entries.append(entry)
        all_ok &= ok
        additional_entries.append(
            generate_explicit_case(model, additional_dir, case_provenance)
        )

    manifest = {
        "format_version": 2,
        "status": "complete" if len(entries) == 30 and all_ok else "partial-or-error",
        "oracle": "independent pinned upstream source; no ONNX reference",
        "bundle_profile": name,
        "bundle_name": profile_field(profile, "bundle_name"),
        "model_id": profile_field(profile, "model_id"),
        "hf_revision": profile_field(profile, "revision"),
        "gliner2_commit": GLINER2_COMMIT,
        "source_file_sha256": dict(sorted(source_hashes.items())),
        "corpus_sha256": corpus_hash,
        "seed": seed,
        "case_count": len(entries),
        "successful_case_count": sum(item["status"] == "ok" for item in entries),
        "error_case_count": sum(item["status"] != "ok" for item in entries),
        "category_counts": dict(Counter(item["category"] for item in entries)),
        "case_ids": [item["case_id"] for item in entries],
        "additional_case_ids": [item["case_id"] for item in additional_entries],
        "entries": entries,
        "additional_entries": additional_entries,
        "provenance": case_provenance,
    }
    write_json(bundle_dir / "manifest.json", manifest)
    digest = hashlib.sha256(canonical_json(manifest)).hexdigest()
    print(
        f"[{name}] wrote {len(entries)} corpus and {len(additional_entries)} additional "
        f"cases; manifest_payload_sha256={digest}; errors={manifest['error_case_count']}"
    )
    if not all_ok and not allow_errors:
        raise SystemExit(f"{name}: source extraction errors were recorded")


def assert_heavy_output_is_ignored(out_root: Path) -> None:
    resolved = out_root.expanduser().resolve()
    if not resolved.is_relative_to(ROOT):
        return
    probe = resolved / "gliner2.5-small-v1" / "manifest.json"
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", str(probe)],
        cwd=ROOT,
        check=False,
    )
    if ignored.returncode != 0:
        raise RuntimeError(
            f"refusing heavyweight fixture output inside git without an ignore rule: {resolved}"
        )


def main() -> None:
    args = parse_args()
    if args.model == "all" and args.model_dir:
        raise SystemExit("--model-dir is valid only for one selected profile")
    configure_determinism(args.seed, threads=1)
    assert_pinned_gliner2_installation()
    out_root = Path(args.out_dir)
    assert_heavy_output_is_ignored(out_root)
    names = list(EXPECTED_PROFILES) if args.model == "all" else [args.model]
    for name in names:
        profile = checked_profile(name)
        model_dir = (
            Path(args.model_dir).expanduser().resolve()
            if args.model_dir
            else default_model_dir(profile)
        )
        generate_profile(
            name,
            model_dir,
            out_root,
            Path(args.corpus),
            args.case,
            args.seed,
            args.allow_errors,
        )


if __name__ == "__main__":
    main()
