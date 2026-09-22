#!/usr/bin/env python3
"""Generate deterministic GLiNER2.5 public-oracle stage goldens.

The extraction itself always goes through ``AutoExtractor``'s public ``extract``
method. Forward hooks observe tensors without replacing candidate selection,
scoring, or decoding. Full output belongs in the ignored fixture directory; a
small real-tensor subset is copied to the separately committed subset directory.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import traceback
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
EXPORT_DIR = SCRIPT_DIR.parent / "export"
sys.path.insert(0, str(EXPORT_DIR))

from common import (  # noqa: E402
    BASE_HF_REVISION,
    BASE_MODEL_ID,
    GLINER2_COMMIT,
    OFFICIAL_BASE_SOURCE_SHA256,
    assert_official_base_source,
    assert_pinned_gliner2_installation,
    configure_determinism,
    load_reference_model,
    package_versions,
    sha256_file,
)

SEED = 1729
MAX_LEN = 4096
MODEL_ID = BASE_MODEL_ID
EXPECTED_CATEGORIES = {
    "ner": 8,
    "classification": 5,
    "json": 5,
    "relation": 4,
    "unicode": 3,
    "long": 3,
    "edge": 2,
}
DEFAULT_SUBSET = (
    "unicode_combining_emoji",
    "classification_multi_task",
    "relation_employment",
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
    parser.add_argument("--corpus", default=str(SCRIPT_DIR / "boundary_corpus.json"))
    parser.add_argument("--out-dir", default="fixtures/gliner2.5-base-v1")
    parser.add_argument(
        "--subset-dir", default="fixtures/gliner2.5-base-v1-subset"
    )
    parser.add_argument("--subset-ids", default=",".join(DEFAULT_SUBSET))
    parser.add_argument("--case", action="append", help="Generate only this case ID")
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--no-subset", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write NPZ with fixed ZIP metadata so fixture hashes are reproducible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.ascontiguousarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, buffer.getvalue(), compresslevel=9)


def expand_case(raw: dict[str, Any]) -> dict[str, Any]:
    case = dict(raw)
    if "text" not in case:
        repeat_count = int(case.pop("repeat_count"))
        repeat_word = case.pop("repeat_word")
        tail = case.pop("tail")
        case["text"] = " ".join([repeat_word] * repeat_count) + tail
    case["word_count"] = len(case["text"].split())
    expected = case.get("expected_word_count")
    if expected is not None and case["word_count"] != expected:
        raise AssertionError(
            f"{case['id']}: word count {case['word_count']} != {expected}"
        )
    return case


def load_corpus(path: Path) -> tuple[list[dict[str, Any]], str]:
    document = json.loads(path.read_text())
    cases = [expand_case(case) for case in document["cases"]]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise AssertionError("corpus case IDs must be unique")
    counts = Counter(case["category"] for case in cases)
    if counts != Counter(EXPECTED_CATEGORIES):
        raise AssertionError(f"corpus categories {dict(counts)} != {EXPECTED_CATEGORIES}")
    if len(cases) != 30:
        raise AssertionError(f"expected 30 corpus cases, found {len(cases)}")
    long_counts = sorted(
        case["word_count"] for case in cases if case["category"] == "long"
    )
    if long_counts != [1000, 2000, 3000]:
        raise AssertionError(f"long corpus word counts are {long_counts}")
    digest = hashlib.sha256(canonical_json(cases)).hexdigest()
    return cases, digest


def build_schema(model, specification: dict[str, Any]):
    schema = model.create_schema()
    if "entities" in specification:
        schema.entities(specification["entities"])
    if "entities_config" in specification:
        schema.entities(specification["entities_config"])
    for task in specification.get("classifications", []):
        config = dict(task)
        name = config.pop("task")
        labels = config.pop("labels")
        schema.classification(name, labels, **config)
    for structure in specification.get("structures", []):
        builder = schema.structure(
            structure["name"],
            mode=structure.get("mode"),
            anchor=structure.get("anchor"),
            occurrence_policy=structure.get("occurrence_policy"),
        )
        for field in structure["fields"]:
            config = dict(field)
            name = config.pop("name")
            builder.field(name, **config)
        builder._auto_finish()
        schema._active_builder = None
    relation_input: dict[str, Any] = {}
    for relation in specification.get("relations", []):
        if isinstance(relation, str):
            relation_input[relation] = {}
        else:
            config = dict(relation)
            name = config.pop("name")
            relation_input[name] = config
    if relation_input:
        schema.relations(relation_input)
    return schema


def array(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def add_array(
    arrays: dict[str, np.ndarray], name: str, value: torch.Tensor | np.ndarray
) -> None:
    result = array(value)
    if result.dtype == object:
        raise TypeError(f"refusing object array for {name}")
    arrays[name] = result


def codepoint_to_utf8_offset(text: str, offset: int) -> int:
    return len(text[:offset].encode("utf-8"))


def utf8_result(value: Any, text: str) -> Any:
    if isinstance(value, list):
        return [utf8_result(item, text) for item in value]
    if isinstance(value, tuple):
        return [utf8_result(item, text) for item in value]
    if isinstance(value, dict):
        converted = {key: utf8_result(item, text) for key, item in value.items()}
        if (
            isinstance(value.get("start"), int)
            and isinstance(value.get("end"), int)
            and 0 <= value["start"] <= value["end"] <= len(text)
        ):
            converted["start"] = codepoint_to_utf8_offset(text, value["start"])
            converted["end"] = codepoint_to_utf8_offset(text, value["end"])
            converted["offset_unit"] = "utf8_byte"
        return converted
    return value


def query_metadata(batch) -> list[dict[str, Any]]:
    result = []
    for spec in batch.query_layouts[0].queries:
        result.append(
            {
                "query_id": int(spec.query_id),
                "task_index": int(spec.task_index),
                "task_type": spec.task_type,
                "task_name": spec.task_name,
                "role_index": int(spec.role_index),
                "role_name": spec.role_name,
                "field_path": list(spec.field_path),
            }
        )
    return result


class Capture:
    """Forward-hook collector; callbacks only detach/copy observed tensors."""

    def __init__(self):
        self.arrays: dict[str, np.ndarray] = {}
        self.counts: defaultdict[str, int] = defaultdict(int)
        self.handles = []

    def indexed(self, stage: str, name: str, value) -> None:
        index = self.counts[stage]
        add_array(self.arrays, f"{stage}_{index}_{name}", value)

    def finish(self, stage: str) -> None:
        self.counts[stage] += 1

    def hook(self, module, callback: Callable, *, prepend: bool = False) -> None:
        self.handles.append(
            module.register_forward_hook(
                callback, with_kwargs=True, prepend=prepend
            )
        )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def install_hooks(model, capture: Capture) -> None:
    def encoder_hook(_module, args, kwargs, output):
        ids = kwargs.get("input_ids", args[0] if args else None)
        mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
        capture.indexed("encoder", "input_ids", ids)
        capture.indexed("encoder", "attention_mask", mask)
        capture.indexed("encoder", "last_hidden_state", output.last_hidden_state)
        capture.finish("encoder")

    def boundary_encoder_hook(_module, args, _kwargs, output):
        capture.indexed("boundary_encoder", "text_states", args[0])
        capture.indexed("boundary_encoder", "text_mask", args[1])
        capture.indexed("boundary_encoder", "states", output.states)
        capture.indexed("boundary_encoder", "mask", output.mask)
        capture.finish("boundary_encoder")

    def marginal_hook(_module, args, _kwargs, output):
        capture.indexed("marginals", "boundary_states", args[0])
        capture.indexed("marginals", "boundary_mask", args[1])
        capture.indexed("marginals", "text_states", args[2])
        capture.indexed("marginals", "text_mask", args[3])
        capture.indexed("marginals", "query_states", args[4])
        capture.indexed("marginals", "query_mask", args[5])
        for name in (
            "start_logits",
            "end_logits",
            "inside_logits",
            "inside_prefix",
            "inside_prefix_mean",
        ):
            capture.indexed("marginals", name, getattr(output, name))
        capture.finish("marginals")

    def projection_hook(stage):
        def callback(_module, args, _kwargs, output):
            capture.indexed(stage, "input", args[0])
            capture.indexed(stage, "output", output)
            capture.finish(stage)
        return callback

    def pool_hook(_module, args, _kwargs, output):
        capture.indexed("pool", "boundary_states", args[0])
        capture.indexed("pool", "boundary_mask", args[1])
        capture.indexed("pool", "query_mask", args[2])
        capture.indexed("pool", "start_logits", args[3])
        capture.indexed("pool", "end_logits", args[4])
        for name in ("indices", "mask", "proposal_logits", "compat_logits"):
            value = getattr(output, name)
            if value is not None:
                capture.indexed("pool", name, value)
        capture.finish("pool")

    def shared_scorer_hook(_module, args, _kwargs, output):
        names = (
            "boundary_states",
            "query_states",
            "query_mask",
        )
        for name, value in zip(names, args[:3]):
            capture.indexed("shared_scorer", name, value)
        pooled = args[3]
        capture.indexed("shared_scorer", "indices", pooled.indices)
        capture.indexed("shared_scorer", "mask", pooled.mask)
        if pooled.compat_logits is not None:
            capture.indexed("shared_scorer", "compat", pooled.compat_logits)
        capture.indexed("shared_scorer", "pair_logits_candidate_major", output[0])
        capture.indexed("shared_scorer", "feature_states", output[1])
        capture.finish("shared_scorer")

    def head_hook(_module, _args, _kwargs, output):
        if output.candidates is not None:
            candidates = output.candidates
            for name in (
                "indices",
                "proposal_logits",
                "pair_logits",
                "valid_mask",
                "query_mask",
                "candidate_states",
            ):
                value = getattr(candidates, name)
                if value is not None:
                    capture.indexed("boundary_head", name, value)
        if output.null_logits is not None:
            capture.indexed("boundary_head", "null_logits", output.null_logits)
        if output.count_log_rates is not None:
            capture.indexed(
                "boundary_head", "count_log_rates", output.count_log_rates
            )
        capture.finish("boundary_head")

    def classifier_hook(_module, args, _kwargs, output):
        capture.indexed("classifier", "input", args[0])
        capture.indexed("classifier", "raw_logits", output.squeeze(-1))
        capture.finish("classifier")

    def explicit_scorer_hook(_module, args, _kwargs, output):
        proposals = args[2]
        capture.indexed("explicit_scorer", "boundary_states", args[0])
        capture.indexed("explicit_scorer", "query_states", args[1])
        capture.indexed("explicit_scorer", "indices", proposals.indices)
        capture.indexed("explicit_scorer", "valid_mask", proposals.valid_mask)
        if proposals.compat_logits is not None:
            capture.indexed(
                "explicit_scorer", "compat_logits", proposals.compat_logits
            )
        capture.indexed("explicit_scorer", "pair_logits", output)
        capture.finish("explicit_scorer")

    def linear_record_hook(stage):
        def callback(_module, args, _kwargs, output):
            capture.indexed(stage, "input", args[0])
            capture.indexed(stage, "output", output)
            capture.finish(stage)
        return callback

    def relation_hook(_module, args, _kwargs, output):
        boundary_states, relation_states, _candidates, pairs = args
        capture.indexed("relation_scorer", "text_states", boundary_states)
        capture.indexed("relation_scorer", "relation_states", relation_states)
        for name in (
            "batch_index",
            "relation_index",
            "head_start",
            "head_end",
            "tail_start",
            "tail_end",
            "head_prob",
            "tail_prob",
            "pair_mask",
        ):
            value = getattr(pairs, name)
            if value is not None:
                capture.indexed("relation_scorer", name, value)
        capture.indexed("relation_scorer", "logits", output)
        capture.finish("relation_scorer")

    head = model.boundary_head
    capture.hook(model.encoder, encoder_hook)
    capture.hook(head.boundary_encoder, boundary_encoder_hook)
    capture.hook(head.boundary_query_head, marginal_hook)
    capture.hook(
        head.shared_pool_builder.start_projection,
        projection_hook("pool_start_projection"),
    )
    capture.hook(
        head.shared_pool_builder.end_projection,
        projection_hook("pool_end_projection"),
    )
    capture.hook(head.shared_pool_builder, pool_hook)
    capture.hook(head.shared_pool_scorer, shared_scorer_hook)
    capture.hook(head, head_hook)
    capture.hook(model.classifier, classifier_hook)
    capture.hook(head.pair_scorer, explicit_scorer_hook)
    if head.null_projection is not None:
        capture.hook(head.null_projection, projection_hook("null_projection"))
    if head.count_head is not None:
        capture.hook(head.count_head, projection_hook("count_head"))
    if getattr(model, "enable_records", False):
        record = model.record_decoder
        for name in (
            "inst_proj",
            "field_proj",
            "cand_proj",
            "object_head",
            "latent_seed_head",
            "q_proj",
            "k_proj",
            "v_proj",
        ):
            capture.hook(
                getattr(record, name), linear_record_hook(f"record_{name}")
            )
    if getattr(model, "enable_relations", False):
        capture.hook(model.relation_scorer, relation_hook)


def build_expected_batch(model, text: str, schema):
    from gliner2.training.trainer import ExtractorCollator

    schema_dicts, _ = model._build_schema_dicts_and_metadata([schema])
    collator = ExtractorCollator(
        model.processor,
        is_training=False,
        max_len=MAX_LEN,
        architecture="boundary",
    )
    return collator([(text, schema_dicts[0])])


def add_batch_arrays(arrays: dict[str, np.ndarray], batch) -> None:
    for name in (
        "input_ids",
        "attention_mask",
        "text_word_indices",
        "text_word_mask",
        "query_marker_indices",
        "query_marker_mask",
        "cls_marker_indices",
        "cls_marker_mask",
        "text_word_counts",
    ):
        value = getattr(batch, name, None)
        if value is not None:
            add_array(arrays, name, value)
    if batch.start_mappings:
        add_array(arrays, "start_mappings", np.asarray(batch.start_mappings[0]))
        add_array(arrays, "end_mappings", np.asarray(batch.end_mappings[0]))


def reconstruct_routed_states(
    arrays: dict[str, np.ndarray], batch, capture: Capture
) -> None:
    if "encoder_0_last_hidden_state" not in capture.arrays:
        return
    hidden = capture.arrays["encoder_0_last_hidden_state"]
    for prefix, index_name, mask_name in (
        ("text", "text_word_indices", "text_word_mask"),
        ("query", "query_marker_indices", "query_marker_mask"),
        ("classification", "cls_marker_indices", "cls_marker_mask"),
    ):
        indices = getattr(batch, index_name, None)
        masks = getattr(batch, mask_name, None)
        if indices is None or masks is None:
            continue
        index_array = array(indices).astype(np.int64)
        mask_array = array(masks).astype(bool)
        safe = np.clip(index_array, 0, hidden.shape[1] - 1)
        gathered = np.take_along_axis(
            hidden, np.repeat(safe[..., None], hidden.shape[-1], axis=-1), axis=1
        )
        gathered *= mask_array[..., None].astype(gathered.dtype)
        arrays[f"{prefix}_states"] = gathered
        arrays[f"{prefix}_mask"] = mask_array


def stages_metadata(capture: Capture, category: str) -> dict[str, Any]:
    boundary = capture.counts["boundary_head"] > 0
    return {
        "encoder": {"invoked": capture.counts["encoder"] > 0},
        "boundary_marginals": {
            "invoked": capture.counts["marginals"] > 0,
            "reason_if_not": (
                "classification has Q=0 or schema has no extractive queries"
                if not capture.counts["marginals"]
                else None
            ),
        },
        "shared_pool": {"invoked": capture.counts["pool"] > 0},
        "shared_scorer": {"invoked": capture.counts["shared_scorer"] > 0},
        "boundary_head": {"invoked": boundary},
        "classifier": {
            "invoked": capture.counts["classifier"] > 0,
            "reason_if_not": "not a classification task"
            if category != "classification"
            else None,
        },
        "explicit_sparse_scorer": {
            "invoked": capture.counts["explicit_scorer"] > 0,
            "reason_if_not": "no explicit choice/attribute span scoring in this case"
            if not capture.counts["explicit_scorer"]
            else None,
        },
        "record_head": {
            "invoked": any(
                count for name, count in capture.counts.items() if name.startswith("record_")
            ),
            "reason_if_not": (
                None
                if any(
                    count
                    for name, count in capture.counts.items()
                    if name.startswith("record_")
                )
                else (
                    "record metadata not applicable"
                    if category != "json"
                    else "legacy structure path or no record candidates"
                )
            ),
        },
        "relation_scorer": {
            "invoked": capture.counts["relation_scorer"] > 0,
            "reason_if_not": "relation task not applicable or produced no relation pairs"
            if not capture.counts["relation_scorer"]
            else None,
        },
    }


def golden_provenance(
    *,
    seed: int,
    source_hashes: dict[str, str],
    corpus_hash: str | None = None,
) -> dict[str, Any]:
    if source_hashes != OFFICIAL_BASE_SOURCE_SHA256:
        raise ValueError("official base provenance requires verified source file hashes")
    provenance = {
        "model_id": MODEL_ID,
        "hf_revision": BASE_HF_REVISION,
        "source_file_sha256": dict(source_hashes),
        "gliner2_commit": GLINER2_COMMIT,
        "architecture": "boundary",
        "architecture_version": 1,
        "attention_implementation": "eager",
        "device": "cpu",
        "dtype": "float32",
        "autocast": False,
        "seed": seed,
        "torch_threads": torch.get_num_threads(),
        "dependencies": package_versions(),
    }
    if corpus_hash is not None:
        provenance["corpus_sha256"] = corpus_hash
    return provenance


def assert_captured_routing(case_id: str, arrays: dict[str, np.ndarray]) -> None:
    """Ensure hooks observed the same text/query states reconstructed from routing."""
    comparisons = (
        ("boundary_encoder_0_text_states", "text_states"),
        ("boundary_encoder_0_text_mask", "text_mask"),
        ("marginals_0_text_states", "text_states"),
        ("marginals_0_text_mask", "text_mask"),
        ("marginals_0_query_states", "query_states"),
        ("marginals_0_query_mask", "query_mask"),
        ("shared_scorer_0_query_states", "query_states"),
        ("shared_scorer_0_query_mask", "query_mask"),
    )
    for captured_name, routed_name in comparisons:
        if captured_name in arrays and not np.array_equal(
            arrays[captured_name], arrays[routed_name]
        ):
            raise AssertionError(
                f"{case_id}: {captured_name} differs from reconstructed {routed_name}"
            )


def generate_case(
    model,
    case: dict[str, Any],
    out_dir: Path,
    corpus_hash: str,
    *,
    seed: int,
    source_hashes: dict[str, str],
) -> tuple[dict[str, Any], bool]:
    case_id = case["id"]
    print(f"[{case_id}] words={case['word_count']} category={case['category']}", flush=True)
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
    except Exception as exc:  # Stored verbatim as the observed public behavior.
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
    assert_captured_routing(case_id, arrays)
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
        "format_version": 1,
        "case_id": case_id,
        "category": case["category"],
        "text": case["text"],
        "word_count": case["word_count"],
        "schema_spec": case["schema"],
        "query_metadata": query_metadata(expected_batch),
        "schema_tokens": expected_batch.schema_tokens_list[0],
        "task_types": expected_batch.task_types[0],
        "text_tokens": expected_batch.text_tokens[0],
        "original_text": expected_batch.original_texts[0],
        "max_len": MAX_LEN,
        "threshold": 0.5,
        "include_spans": True,
        "include_confidence": True,
        "offset_contract": {
            "python_result": "Unicode code points",
            "rust_result": "UTF-8 bytes",
        },
        "status": "error" if error else "ok",
        "error": error,
        "final_result_python_offsets": result,
        "final_result_utf8_offsets": utf8_result(result, case["text"])
        if result is not None
        else None,
        "stages": stages_metadata(capture, case["category"]),
        "captured_calls": dict(sorted(capture.counts.items())),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in sorted(arrays.items())
        },
        "provenance": golden_provenance(
            seed=seed, source_hashes=source_hashes, corpus_hash=corpus_hash
        ),
    }
    write_json(json_path, metadata)
    entry = {
        "case_id": case_id,
        "category": case["category"],
        "word_count": case["word_count"],
        "status": metadata["status"],
        "npz": npz_path.name,
        "npz_sha256": sha256_file(npz_path),
        "json": json_path.name,
        "json_sha256": sha256_file(json_path),
        "npz_bytes": npz_path.stat().st_size,
        "json_bytes": json_path.stat().st_size,
    }
    return entry, error is None


def copy_subset(
    out_dir: Path,
    subset_dir: Path,
    requested: list[str],
    entries: list[dict[str, Any]],
    corpus_hash: str,
) -> None:
    subset_dir.mkdir(parents=True, exist_ok=True)
    retained = {f"{case_id}{suffix}" for case_id in requested for suffix in (".json", ".npz")}
    retained.add("manifest.json")
    for stale in subset_dir.iterdir():
        if stale.is_file() and stale.name not in retained:
            stale.unlink()
    selected = {entry["case_id"]: entry for entry in entries}
    missing = [case_id for case_id in requested if case_id not in selected]
    if missing:
        raise AssertionError(f"subset cases were not generated: {missing}")
    subset_entries = []
    for case_id in requested:
        entry = selected[case_id]
        if entry["status"] != "ok":
            raise AssertionError(f"cannot commit errored subset case {case_id}")
        for suffix in (".json", ".npz"):
            shutil.copy2(out_dir / f"{case_id}{suffix}", subset_dir / f"{case_id}{suffix}")
        copied = dict(entry)
        if sha256_file(subset_dir / copied["npz"]) != entry["npz_sha256"]:
            raise AssertionError(f"subset NPZ differs from full fixture for {case_id}")
        if sha256_file(subset_dir / copied["json"]) != entry["json_sha256"]:
            raise AssertionError(f"subset JSON differs from full fixture for {case_id}")
        if (subset_dir / copied["npz"]).stat().st_size != entry["npz_bytes"]:
            raise AssertionError(f"subset NPZ size differs for {case_id}")
        if (subset_dir / copied["json"]).stat().st_size != entry["json_bytes"]:
            raise AssertionError(f"subset JSON size differs for {case_id}")
        subset_entries.append(copied)
    total_before_manifest = sum(
        path.stat().st_size for path in subset_dir.iterdir() if path.name != "manifest.json"
    )
    write_json(
        subset_dir / "manifest.json",
        {
            "format_version": 1,
            "description": "Committed real-oracle M1 subset; full corpus is gitignored.",
            "corpus_sha256": corpus_hash,
            "gliner2_commit": GLINER2_COMMIT,
            "hf_revision": BASE_HF_REVISION,
            "source_file_sha256": OFFICIAL_BASE_SOURCE_SHA256,
            "entries": subset_entries,
            "total_bytes_before_manifest": total_before_manifest,
        },
    )
    total = sum(path.stat().st_size for path in subset_dir.iterdir())
    if total > 2 * 1024 * 1024:
        raise AssertionError(
            f"committed subset including manifest is {total} bytes, exceeds 2 MiB"
        )


def main() -> None:
    args = parse_args()
    configure_determinism(args.seed, threads=1)
    assert_pinned_gliner2_installation()
    source_hashes = assert_official_base_source(args.model_dir)
    cases, corpus_hash = load_corpus(Path(args.corpus))
    if args.case:
        selected = set(args.case)
        unknown = selected - {case["id"] for case in cases}
        if unknown:
            raise KeyError(f"unknown case IDs: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in selected]

    model, config = load_reference_model(args.model_dir)
    if config.get("architecture") != "boundary":
        raise ValueError("goldens require a boundary checkpoint")
    if config.get("architecture_version") != 1:
        raise ValueError("goldens require architecture_version 1")
    if config.get("boundary_head", {}).get("candidate_pool") != "shared":
        raise ValueError("goldens require the shared candidate pool")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    all_ok = True
    for case in cases:
        entry, ok = generate_case(
            model,
            case,
            out_dir,
            corpus_hash,
            seed=args.seed,
            source_hashes=source_hashes,
        )
        entries.append(entry)
        all_ok &= ok
        print(
            f"[{case['id']}] {entry['status']} npz_sha256={entry['npz_sha256']}",
            flush=True,
        )

    manifest = {
        "format_version": 1,
        "status": "complete" if len(cases) == 30 else "partial-case-selection",
        "case_count": len(entries),
        "successful_case_count": sum(entry["status"] == "ok" for entry in entries),
        "error_case_count": sum(entry["status"] == "error" for entry in entries),
        "corpus_sha256": corpus_hash,
        "category_counts": dict(Counter(entry["category"] for entry in entries)),
        "long_word_counts": [
            entry["word_count"] for entry in entries if entry["category"] == "long"
        ],
        "provenance": {
            **golden_provenance(seed=args.seed, source_hashes=source_hashes),
            "source_file_sha256": source_hashes,
            "gliner2_repository": "https://github.com/fastino-ai/GLiNER2",
            "max_len": MAX_LEN,
        },
        "entries": entries,
    }
    write_json(out_dir / "manifest.json", manifest)

    if not args.no_subset:
        subset_ids = [value for value in args.subset_ids.split(",") if value]
        generated_ids = {entry["case_id"] for entry in entries}
        if all(case_id in generated_ids for case_id in subset_ids):
            copy_subset(
                out_dir, Path(args.subset_dir), subset_ids, entries, corpus_hash
            )
        elif not args.case:
            raise AssertionError("default subset IDs missing from complete generation")

    print(
        f"generated {len(entries)} cases; corpus_sha256={corpus_hash}; "
        f"errors={manifest['error_case_count']}"
    )
    if not all_ok and not args.allow_errors:
        raise SystemExit(
            "one or more public oracle calls failed; behavior was recorded. "
            "Re-run with --allow-errors only after reviewing the case JSON."
        )


if __name__ == "__main__":
    main()
