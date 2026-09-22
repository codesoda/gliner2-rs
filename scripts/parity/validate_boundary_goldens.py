#!/usr/bin/env python3
"""Validate structural integrity of GLiNER2.5 boundary golden fixtures.

This verifies corpus coverage, provenance records, metadata, captured tensors,
and deterministic serialization. It does not independently reproduce the
Python oracle computation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
EXPORT_DIR = SCRIPT_DIR.parent / "export"
sys.path.insert(0, str(EXPORT_DIR))

from common import (  # noqa: E402
    BASE_HF_REVISION,
    BASE_MODEL_ID,
    GLINER2_COMMIT,
    OFFICIAL_BASE_SOURCE_SHA256,
    sha256_file,
)
from gen_boundary_goldens import (  # noqa: E402
    EXPECTED_CATEGORIES,
    load_corpus,
    write_deterministic_npz,
)

BASE_ARRAYS = {
    "input_ids",
    "attention_mask",
    "encoder_0_input_ids",
    "encoder_0_attention_mask",
    "encoder_0_last_hidden_state",
    "text_word_indices",
    "text_word_mask",
    "query_marker_indices",
    "query_marker_mask",
    "cls_marker_indices",
    "cls_marker_mask",
    "text_word_counts",
    "start_mappings",
    "end_mappings",
    "text_states",
    "text_mask",
    "query_states",
    "query_mask",
    "classification_states",
    "classification_mask",
}
EXTRACTIVE_ARRAYS = {
    "boundary_encoder_0_text_states",
    "boundary_encoder_0_text_mask",
    "boundary_encoder_0_states",
    "boundary_encoder_0_mask",
    "marginals_0_boundary_states",
    "marginals_0_boundary_mask",
    "marginals_0_text_states",
    "marginals_0_text_mask",
    "marginals_0_query_states",
    "marginals_0_query_mask",
    "marginals_0_start_logits",
    "marginals_0_end_logits",
    "marginals_0_inside_logits",
    "marginals_0_inside_prefix",
    "marginals_0_inside_prefix_mean",
    "pool_start_projection_0_input",
    "pool_start_projection_0_output",
    "pool_end_projection_0_input",
    "pool_end_projection_0_output",
    "pool_0_boundary_states",
    "pool_0_boundary_mask",
    "pool_0_query_mask",
    "pool_0_start_logits",
    "pool_0_end_logits",
    "pool_0_indices",
    "pool_0_mask",
    "pool_0_proposal_logits",
    "pool_0_compat_logits",
    "shared_scorer_0_boundary_states",
    "shared_scorer_0_query_states",
    "shared_scorer_0_query_mask",
    "shared_scorer_0_indices",
    "shared_scorer_0_mask",
    "shared_scorer_0_compat",
    "shared_scorer_0_pair_logits_candidate_major",
    "shared_scorer_0_feature_states",
    "boundary_head_0_indices",
    "boundary_head_0_proposal_logits",
    "boundary_head_0_pair_logits",
    "boundary_head_0_valid_mask",
    "boundary_head_0_query_mask",
    "boundary_head_0_candidate_states",
    "boundary_head_0_null_logits",
    "boundary_head_0_count_log_rates",
}
EXTRACTIVE_CALL_SUFFIXES = {
    "boundary_encoder": {"text_states", "text_mask", "states", "mask"},
    "marginals": {
        "boundary_states",
        "boundary_mask",
        "text_states",
        "text_mask",
        "query_states",
        "query_mask",
        "start_logits",
        "end_logits",
        "inside_logits",
        "inside_prefix",
        "inside_prefix_mean",
    },
    "pool_start_projection": {"input", "output"},
    "pool_end_projection": {"input", "output"},
    "pool": {
        "boundary_states",
        "boundary_mask",
        "query_mask",
        "start_logits",
        "end_logits",
        "indices",
        "mask",
        "proposal_logits",
        "compat_logits",
    },
    "shared_scorer": {
        "boundary_states",
        "query_states",
        "query_mask",
        "indices",
        "mask",
        "compat",
        "pair_logits_candidate_major",
        "feature_states",
    },
    "boundary_head": {
        "indices",
        "proposal_logits",
        "pair_logits",
        "valid_mask",
        "query_mask",
        "candidate_states",
        "null_logits",
        "count_log_rates",
    },
}
STAGE_PREFIXES = {
    "encoder": ("encoder_",),
    "boundary_marginals": ("marginals_",),
    "shared_pool": ("pool_", "pool_start_projection_", "pool_end_projection_"),
    "shared_scorer": ("shared_scorer_",),
    "boundary_head": ("boundary_head_",),
    "classifier": ("classifier_",),
    "explicit_sparse_scorer": ("explicit_scorer_",),
    "record_head": ("record_",),
    "relation_scorer": ("relation_scorer_",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(SCRIPT_DIR / "boundary_corpus.json"))
    parser.add_argument("--fixture-dir", default="fixtures/gliner2.5-base-v1")
    parser.add_argument("--subset-dir", default="fixtures/gliner2.5-base-v1-subset")
    return parser.parse_args()


def validate_reproducible_npz(path: Path) -> None:
    with np.load(path, allow_pickle=False) as fixture:
        arrays = {name: fixture[name] for name in fixture.files}
    for name, value in arrays.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise AssertionError(f"{path.name}:{name} contains non-finite values")
    scratch = path.with_suffix(".repro.npz")
    try:
        write_deterministic_npz(scratch, arrays)
        if sha256_file(scratch) != sha256_file(path):
            raise AssertionError(f"{path.name} is not reproducibly serialized")
    finally:
        scratch.unlink(missing_ok=True)


def validate_provenance(
    provenance: dict[str, Any], *, corpus_hash: str | None, seed: int | None = None
) -> int:
    expected = {
        "model_id": BASE_MODEL_ID,
        "hf_revision": BASE_HF_REVISION,
        "source_file_sha256": OFFICIAL_BASE_SOURCE_SHA256,
        "gliner2_commit": GLINER2_COMMIT,
        "architecture": "boundary",
        "architecture_version": 1,
        "attention_implementation": "eager",
        "device": "cpu",
        "dtype": "float32",
        "autocast": False,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise AssertionError(
                f"provenance {key} is {provenance.get(key)!r}, expected {value!r}"
            )
    if corpus_hash is not None and provenance.get("corpus_sha256") != corpus_hash:
        raise AssertionError("case provenance corpus hash is incorrect")
    actual_seed = provenance.get("seed")
    if not isinstance(actual_seed, int):
        raise AssertionError("provenance seed must be an integer")
    if seed is not None and actual_seed != seed:
        raise AssertionError(f"case provenance seed {actual_seed} != manifest seed {seed}")
    dependencies = provenance.get("dependencies")
    if not isinstance(dependencies, dict) or dependencies.get("gliner2") != "2.0.0":
        raise AssertionError("provenance must record the pinned GLiNER2 2.0.0 package")
    return actual_seed


def _require_equal(name: str, left: np.ndarray, right: np.ndarray) -> None:
    if not np.array_equal(left, right):
        raise AssertionError(f"{name} arrays differ")


def _reconstruct(hidden: np.ndarray, indices: np.ndarray, mask: np.ndarray) -> np.ndarray:
    safe = np.clip(indices.astype(np.int64), 0, hidden.shape[1] - 1)
    gathered = np.take_along_axis(
        hidden, np.repeat(safe[..., None], hidden.shape[-1], axis=-1), axis=1
    )
    return gathered * mask.astype(bool)[..., None].astype(gathered.dtype)


def _validate_complete_calls(case_id: str, names: set[str]) -> None:
    for prefix, required_suffixes in EXTRACTIVE_CALL_SUFFIXES.items():
        calls: dict[int, set[str]] = {}
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)_(.+)$")
        for name in names:
            match = pattern.match(name)
            if match:
                calls.setdefault(int(match.group(1)), set()).add(match.group(2))
        if not calls:
            raise AssertionError(f"{case_id} has no {prefix} captured calls")
        if set(calls) != set(range(max(calls) + 1)):
            raise AssertionError(
                f"{case_id} {prefix} captured call indices are not contiguous"
            )
        for index, suffixes in calls.items():
            missing = required_suffixes - suffixes
            if missing:
                raise AssertionError(
                    f"{case_id} {prefix}_{index} missing arrays {sorted(missing)}"
                )


def validate_case_arrays(
    case_id: str, category: str, metadata: dict[str, Any], data
) -> None:
    names = set(data.files)
    missing = BASE_ARRAYS - names
    if missing:
        raise AssertionError(f"{case_id} missing base arrays {sorted(missing)}")
    if set(metadata.get("arrays", {})) != names:
        raise AssertionError(f"{case_id} metadata array names do not match NPZ")
    for name in names:
        description = metadata["arrays"][name]
        if description.get("shape") != list(data[name].shape):
            raise AssertionError(f"{case_id}:{name} metadata shape is incorrect")
        if description.get("dtype") != str(data[name].dtype):
            raise AssertionError(f"{case_id}:{name} metadata dtype is incorrect")
        if data[name].dtype.kind in "fc" and not np.isfinite(data[name]).all():
            raise AssertionError(f"{case_id}:{name} contains non-finite values")

    _require_equal(
        f"{case_id} encoder input_ids",
        data["input_ids"],
        data["encoder_0_input_ids"],
    )
    _require_equal(
        f"{case_id} encoder attention_mask",
        data["attention_mask"],
        data["encoder_0_attention_mask"],
    )
    hidden = data["encoder_0_last_hidden_state"]
    routed = {}
    for prefix, index_name, mask_name in (
        ("text", "text_word_indices", "text_word_mask"),
        ("query", "query_marker_indices", "query_marker_mask"),
        ("classification", "cls_marker_indices", "cls_marker_mask"),
    ):
        routed[prefix] = _reconstruct(hidden, data[index_name], data[mask_name])
        _require_equal(
            f"{case_id} reconstructed {prefix}_states",
            routed[prefix],
            data[f"{prefix}_states"],
        )
        _require_equal(
            f"{case_id} reconstructed {prefix}_mask",
            data[mask_name].astype(bool),
            data[f"{prefix}_mask"],
        )

    q = int(data["query_states"].shape[1])
    if data["query_mask"].shape != data["query_marker_mask"].shape:
        raise AssertionError(f"{case_id} query mask shape is inconsistent")
    if len(metadata.get("query_metadata", [])) != q:
        raise AssertionError(f"{case_id} query metadata count does not match Q={q}")
    stages = metadata.get("stages", {})
    if set(stages) != set(STAGE_PREFIXES):
        raise AssertionError(f"{case_id} stage metadata is incomplete")

    for stage, prefixes in STAGE_PREFIXES.items():
        actual = any(
            any(name.startswith(prefix) for prefix in prefixes) for name in names
        )
        recorded = stages[stage].get("invoked")
        if recorded is not actual:
            raise AssertionError(f"{case_id} stage {stage} invocation metadata is incorrect")
        if actual and stages[stage].get("reason_if_not") is not None:
            raise AssertionError(
                f"{case_id} stage {stage} has a reason_if_not despite invocation"
            )

    if q > 0:
        missing = EXTRACTIVE_ARRAYS - names
        if missing:
            raise AssertionError(f"{case_id} missing extractive arrays {sorted(missing)}")
        _validate_complete_calls(case_id, names)
        for stage in (
            "boundary_marginals",
            "shared_pool",
            "shared_scorer",
            "boundary_head",
        ):
            if not stages[stage]["invoked"]:
                raise AssertionError(f"{case_id} extractive Q={q} requires {stage}")
        for captured_name, routed_name in (
            ("boundary_encoder_0_text_states", "text_states"),
            ("boundary_encoder_0_text_mask", "text_mask"),
            ("marginals_0_text_states", "text_states"),
            ("marginals_0_text_mask", "text_mask"),
            ("marginals_0_query_states", "query_states"),
            ("marginals_0_query_mask", "query_mask"),
            ("shared_scorer_0_query_states", "query_states"),
            ("shared_scorer_0_query_mask", "query_mask"),
        ):
            _require_equal(
                f"{case_id} {captured_name}", data[captured_name], data[routed_name]
            )
    elif any(name in names for name in EXTRACTIVE_ARRAYS):
        raise AssertionError(f"{case_id} has extractive tensors with Q=0")

    if category == "classification":
        if q != 0:
            raise AssertionError(f"{case_id} classification should have Q=0")
        if not stages["classifier"]["invoked"]:
            raise AssertionError(f"{case_id} lacks classifier tensors")


def validate_fixture_corpus(
    corpus_path: Path, fixture_dir: Path, subset_dir: Path
) -> dict[str, Any]:
    cases, corpus_hash = load_corpus(corpus_path)
    manifest = json.loads((fixture_dir / "manifest.json").read_text())
    expected_ids = [case["id"] for case in cases]
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise AssertionError("full fixture entries must be a list")
    entry_ids = [entry.get("case_id") for entry in entries]
    duplicates = sorted(
        case_id for case_id, count in Counter(entry_ids).items() if count > 1
    )
    if duplicates:
        raise AssertionError(f"duplicate manifest case IDs: {duplicates}")
    if set(entry_ids) != set(expected_ids) or len(entry_ids) != len(expected_ids):
        missing = sorted(set(expected_ids) - set(entry_ids))
        extra = sorted(set(entry_ids) - set(expected_ids))
        raise AssertionError(
            f"manifest case ID set mismatch; missing={missing}, extra={extra}"
        )
    if manifest.get("status") != "complete" or manifest.get("case_count") != len(cases):
        raise AssertionError("full fixture manifest is not a complete corpus")
    if (
        manifest.get("successful_case_count") != len(cases)
        or manifest.get("error_case_count") != 0
    ):
        raise AssertionError("full fixture manifest success/error counts are incorrect")
    if manifest.get("corpus_sha256") != corpus_hash:
        raise AssertionError("fixture corpus hash does not match boundary_corpus.json")
    if Counter(manifest.get("category_counts", {})) != Counter(EXPECTED_CATEGORIES):
        raise AssertionError("fixture category counts are incorrect")
    if sorted(manifest.get("long_word_counts", [])) != [1000, 2000, 3000]:
        raise AssertionError("fixture long word counts are incorrect")
    manifest_seed = validate_provenance(manifest.get("provenance", {}), corpus_hash=None)

    by_id = {case["id"]: case for case in cases}
    full_entries = {}
    long_sequences = {}
    for entry in entries:
        case_id = entry["case_id"]
        case = by_id[case_id]
        expected_entry = {
            "category": case["category"],
            "word_count": case["word_count"],
            "status": "ok",
            "npz": f"{case_id}.npz",
            "json": f"{case_id}.json",
        }
        for key, value in expected_entry.items():
            if entry.get(key) != value:
                raise AssertionError(f"{case_id} manifest {key} is incorrect")
        npz_path = fixture_dir / entry["npz"]
        json_path = fixture_dir / entry["json"]
        for path, hash_key, bytes_key in (
            (npz_path, "npz_sha256", "npz_bytes"),
            (json_path, "json_sha256", "json_bytes"),
        ):
            if path.stat().st_size != entry.get(bytes_key):
                raise AssertionError(f"byte count mismatch for {path.name}")
            if sha256_file(path) != entry.get(hash_key):
                raise AssertionError(f"hash mismatch for {path.name}")
        validate_reproducible_npz(npz_path)
        metadata = json.loads(json_path.read_text())
        expected_metadata = {
            "format_version": 1,
            "case_id": case_id,
            "category": case["category"],
            "text": case["text"],
            "word_count": case["word_count"],
            "schema_spec": case["schema"],
            "status": "ok",
            "error": None,
            "max_len": 4096,
        }
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                raise AssertionError(f"{case_id} metadata {key} is incorrect")
        validate_provenance(
            metadata.get("provenance", {}), corpus_hash=corpus_hash, seed=manifest_seed
        )
        with np.load(npz_path, allow_pickle=False) as data:
            validate_case_arrays(case_id, case["category"], metadata, data)
            if case["category"] == "long":
                long_sequences[case["word_count"]] = int(data["input_ids"].shape[1])
        full_entries[case_id] = entry

    derived_categories = Counter(entry["category"] for entry in entries)
    if dict(derived_categories) != manifest["category_counts"]:
        raise AssertionError("manifest category counts do not match entries")

    subset_manifest = json.loads((subset_dir / "manifest.json").read_text())
    if subset_manifest.get("corpus_sha256") != corpus_hash:
        raise AssertionError("subset corpus hash mismatch")
    if subset_manifest.get("hf_revision") != BASE_HF_REVISION:
        raise AssertionError("subset revision mismatch")
    if subset_manifest.get("source_file_sha256") != OFFICIAL_BASE_SOURCE_SHA256:
        raise AssertionError("subset source hashes mismatch")
    subset_entries = subset_manifest.get("entries", [])
    subset_ids = [entry.get("case_id") for entry in subset_entries]
    if len(subset_ids) != len(set(subset_ids)):
        raise AssertionError("duplicate subset case IDs")
    for entry in subset_entries:
        case_id = entry["case_id"]
        if entry != full_entries.get(case_id):
            raise AssertionError(f"subset entry for {case_id} differs from full manifest")
        for key, hash_key, bytes_key in (
            ("npz", "npz_sha256", "npz_bytes"),
            ("json", "json_sha256", "json_bytes"),
        ):
            path = subset_dir / entry[key]
            full_path = fixture_dir / entry[key]
            if (
                path.stat().st_size != entry[bytes_key]
                or path.read_bytes() != full_path.read_bytes()
            ):
                raise AssertionError(f"subset file differs from full fixture: {entry[key]}")
            if sha256_file(path) != entry[hash_key]:
                raise AssertionError(f"subset hash mismatch for {entry[key]}")
    expected_subset_files = {"manifest.json"} | {
        entry[key] for entry in subset_entries for key in ("npz", "json")
    }
    actual_subset_files = {path.name for path in subset_dir.iterdir() if path.is_file()}
    if actual_subset_files != expected_subset_files:
        raise AssertionError("subset directory files do not exactly match its manifest")
    subset_total = sum(
        path.stat().st_size for path in subset_dir.iterdir() if path.is_file()
    )
    if subset_total > 2 * 1024 * 1024:
        raise AssertionError(f"subset is {subset_total} bytes including manifest, over 2 MiB")

    return {
        "case_count": len(entries),
        "corpus_sha256": corpus_hash,
        "category_counts": manifest["category_counts"],
        "long_word_to_subword_lengths": long_sequences,
        "subset_bytes_including_manifest": subset_total,
        "deterministic_npz_hashes": "verified",
        "finite_float_arrays": "verified",
        "validation_scope": (
            "structural fixture integrity; independent oracle reproduction not performed"
        ),
    }


def main() -> None:
    args = parse_args()
    report = validate_fixture_corpus(
        Path(args.corpus), Path(args.fixture_dir), Path(args.subset_dir)
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
