"""Model-free unit tests for boundary golden generation and validation."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from gen_boundary_goldens import (
    golden_provenance,
    load_corpus,
    write_deterministic_npz,
    write_json,
)
from validate_boundary_goldens import (
    EXTRACTIVE_ARRAYS,
    STAGE_PREFIXES,
    validate_fixture_corpus,
)
from common import (
    BASE_HF_REVISION,
    OFFICIAL_BASE_SOURCE_SHA256,
    assert_official_base_source,
    update_partial_manifest,
)

CORPUS = Path(__file__).with_name("boundary_corpus.json")


def _arrays(q: int, classification: bool) -> dict[str, np.ndarray]:
    hidden = np.arange(6, dtype=np.float32).reshape(1, 3, 2)
    query_indices = np.arange(q, dtype=np.int64).reshape(1, q)
    query_mask = np.ones((1, q), dtype=bool)
    cls_count = 1 if classification else 0
    cls_indices = np.zeros((1, cls_count), dtype=np.int64)
    cls_mask = np.ones((1, cls_count), dtype=bool)
    arrays = {
        "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
        "attention_mask": np.ones((1, 3), dtype=np.int64),
        "encoder_0_input_ids": np.array([[1, 2, 3]], dtype=np.int64),
        "encoder_0_attention_mask": np.ones((1, 3), dtype=np.int64),
        "encoder_0_last_hidden_state": hidden,
        "text_word_indices": np.array([[0]], dtype=np.int64),
        "text_word_mask": np.ones((1, 1), dtype=bool),
        "query_marker_indices": query_indices,
        "query_marker_mask": query_mask,
        "cls_marker_indices": cls_indices,
        "cls_marker_mask": cls_mask,
        "text_word_counts": np.array([1], dtype=np.int64),
        "start_mappings": np.array([0], dtype=np.int64),
        "end_mappings": np.array([0], dtype=np.int64),
        "text_states": hidden[:, :1],
        "text_mask": np.ones((1, 1), dtype=bool),
        "query_states": hidden[:, :q],
        "query_mask": query_mask,
        "classification_states": hidden[:, :cls_count],
        "classification_mask": cls_mask,
    }
    if classification:
        arrays["classifier_0_input"] = hidden[:, :1]
        arrays["classifier_0_raw_logits"] = np.zeros((1, 1), dtype=np.float32)
    if q:
        # Shapes need only be internally recorded for these structural tests.
        for name in EXTRACTIVE_ARRAYS:
            if name.endswith(("mask", "valid_mask", "query_mask", "text_mask", "boundary_mask")):
                arrays[name] = np.ones((1, 1), dtype=bool)
            elif name.endswith("indices"):
                arrays[name] = np.zeros((1, 1, 2), dtype=np.int64)
            else:
                arrays[name] = np.zeros((1, 1, 2), dtype=np.float32)
        arrays["boundary_encoder_0_text_states"] = arrays["text_states"].copy()
        arrays["boundary_encoder_0_text_mask"] = arrays["text_mask"].copy()
        arrays["marginals_0_text_states"] = arrays["text_states"].copy()
        arrays["marginals_0_text_mask"] = arrays["text_mask"].copy()
        arrays["marginals_0_query_states"] = arrays["query_states"].copy()
        arrays["marginals_0_query_mask"] = arrays["query_mask"].copy()
        arrays["shared_scorer_0_query_states"] = arrays["query_states"].copy()
        arrays["shared_scorer_0_query_mask"] = arrays["query_mask"].copy()
    return arrays


class BoundaryGoldenValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.full = root / "full"
        self.subset = root / "subset"
        self.full.mkdir()
        self.subset.mkdir()
        self.cases, self.corpus_hash = load_corpus(CORPUS)
        self._build(seed=991)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build(self, seed: int) -> None:
        entries = []
        for case in self.cases:
            case_id = case["id"]
            classification = case["category"] == "classification"
            q = 0 if classification or not case["schema"] else 1
            arrays = _arrays(q, classification)
            npz = self.full / f"{case_id}.npz"
            metadata_path = self.full / f"{case_id}.json"
            write_deterministic_npz(npz, arrays)
            stages = {}
            names = set(arrays)
            for stage, prefixes in STAGE_PREFIXES.items():
                invoked = any(
                    any(name.startswith(prefix) for prefix in prefixes) for name in names
                )
                stages[stage] = {
                    "invoked": invoked,
                    "reason_if_not": None if invoked else "not invoked",
                }
            metadata = {
                "format_version": 1,
                "case_id": case_id,
                "category": case["category"],
                "text": case["text"],
                "original_text": case["text"],
                "word_count": case["word_count"],
                "schema_spec": case["schema"],
                "status": "ok",
                "error": None,
                "max_len": 4096,
                "query_metadata": [{} for _ in range(q)],
                "stages": stages,
                "arrays": {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in sorted(arrays.items())
                },
                "provenance": golden_provenance(
                    seed=seed,
                    source_hashes=OFFICIAL_BASE_SOURCE_SHA256,
                    corpus_hash=self.corpus_hash,
                ),
            }
            write_json(metadata_path, metadata)
            entries.append(
                {
                    "case_id": case_id,
                    "category": case["category"],
                    "word_count": case["word_count"],
                    "status": "ok",
                    "npz": npz.name,
                    "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
                    "json": metadata_path.name,
                    "json_sha256": hashlib.sha256(
                        metadata_path.read_bytes()
                    ).hexdigest(),
                    "npz_bytes": npz.stat().st_size,
                    "json_bytes": metadata_path.stat().st_size,
                }
            )
        manifest = {
            "format_version": 1,
            "status": "complete",
            "case_count": 30,
            "successful_case_count": 30,
            "error_case_count": 0,
            "corpus_sha256": self.corpus_hash,
            "category_counts": dict(Counter(e["category"] for e in entries)),
            "long_word_counts": [1000, 2000, 3000],
            "provenance": {
                **golden_provenance(
                    seed=seed, source_hashes=OFFICIAL_BASE_SOURCE_SHA256
                ),
                "gliner2_repository": "https://github.com/fastino-ai/GLiNER2",
                "max_len": 4096,
            },
            "entries": entries,
        }
        write_json(self.full / "manifest.json", manifest)
        selected = [entries[0]]
        for key in ("npz", "json"):
            (self.subset / selected[0][key]).write_bytes(
                (self.full / selected[0][key]).read_bytes()
            )
        write_json(
            self.subset / "manifest.json",
            {
                "format_version": 1,
                "corpus_sha256": self.corpus_hash,
                "hf_revision": manifest["provenance"]["hf_revision"],
                "source_file_sha256": manifest["provenance"]["source_file_sha256"],
                "entries": selected,
            },
        )

    def _manifest(self) -> dict:
        return json.loads((self.full / "manifest.json").read_text())

    def _write_manifest(self, manifest: dict) -> None:
        write_json(self.full / "manifest.json", manifest)

    def _mutate_metadata(self, case_id: str, mutation) -> None:
        path = self.full / f"{case_id}.json"
        metadata = json.loads(path.read_text())
        mutation(metadata)
        write_json(path, metadata)
        self._refresh_entry(case_id, "json")

    def _mutate_npz(self, case_id: str, mutation) -> None:
        path = self.full / f"{case_id}.npz"
        with np.load(path, allow_pickle=False) as fixture:
            arrays = {name: fixture[name] for name in fixture.files}
        mutation(arrays)
        write_deterministic_npz(path, arrays)
        self._refresh_entry(case_id, "npz")

    def _refresh_entry(self, case_id: str, kind: str) -> None:
        path = self.full / f"{case_id}.{kind}"
        manifest = self._manifest()
        entry = next(item for item in manifest["entries"] if item["case_id"] == case_id)
        entry[f"{kind}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        entry[f"{kind}_bytes"] = path.stat().st_size
        self._write_manifest(manifest)

    def test_accepts_consistent_nondefault_seed(self) -> None:
        report = validate_fixture_corpus(CORPUS, self.full, self.subset)
        self.assertEqual(report["case_count"], 30)
        self.assertEqual(
            golden_provenance(
                seed=123, source_hashes=OFFICIAL_BASE_SOURCE_SHA256
            )["seed"],
            123,
        )

    def test_rejects_duplicate_and_missing_manifest_rows(self) -> None:
        manifest = self._manifest()
        manifest["entries"][-1] = copy.deepcopy(manifest["entries"][0])
        self._write_manifest(manifest)
        with self.assertRaisesRegex(AssertionError, "duplicate manifest"):
            validate_fixture_corpus(CORPUS, self.full, self.subset)

    def test_rejects_wrong_metadata(self) -> None:
        case_id = self.cases[0]["id"]
        self._mutate_metadata(case_id, lambda value: value.__setitem__("text", "wrong"))
        with self.assertRaisesRegex(AssertionError, "metadata text"):
            validate_fixture_corpus(CORPUS, self.full, self.subset)

    def test_rejects_wrong_array_metadata(self) -> None:
        case_id = self.cases[0]["id"]
        self._mutate_metadata(
            case_id,
            lambda value: value["arrays"]["input_ids"].__setitem__("shape", [99]),
        )
        with self.assertRaisesRegex(AssertionError, "metadata shape"):
            validate_fixture_corpus(CORPUS, self.full, self.subset)

    def test_rejects_wrong_routed_array(self) -> None:
        case_id = self.cases[0]["id"]
        self._mutate_npz(
            case_id,
            lambda arrays: arrays.__setitem__(
                "text_states", arrays["text_states"] + np.float32(1.0)
            ),
        )
        with self.assertRaisesRegex(AssertionError, "reconstructed text_states"):
            validate_fixture_corpus(CORPUS, self.full, self.subset)

    def test_rejects_missing_stage_even_if_metadata_is_edited(self) -> None:
        case_id = self.cases[0]["id"]
        self._mutate_npz(
            case_id,
            lambda arrays: [
                arrays.pop(name)
                for name in list(arrays)
                if name.startswith("boundary_head_")
            ],
        )
        self._mutate_metadata(
            case_id,
            lambda value: (
                [
                    value["arrays"].pop(name)
                    for name in list(value["arrays"])
                    if name.startswith("boundary_head_")
                ],
                value["stages"]["boundary_head"].update(
                    {"invoked": False, "reason_if_not": "edited out"}
                ),
            ),
        )
        with self.assertRaisesRegex(AssertionError, "missing extractive arrays"):
            validate_fixture_corpus(CORPUS, self.full, self.subset)

    def test_rejects_bad_provenance(self) -> None:
        manifest = self._manifest()
        manifest["provenance"]["hf_revision"] = "unverified"
        self._write_manifest(manifest)
        with self.assertRaisesRegex(AssertionError, "provenance hf_revision"):
            validate_fixture_corpus(CORPUS, self.full, self.subset)

    def test_unverified_path_does_not_claim_official_revision(self) -> None:
        root = Path(self.temporary.name) / BASE_HF_REVISION
        model = root / "model"
        output = root / "output"
        model.mkdir(parents=True)
        output.mkdir(parents=True)
        (model / "config.json").write_text('{"architecture":"boundary"}')
        (output / "encoder.onnx").write_bytes(b"not-an-onnx-model")
        with self.assertRaisesRegex(RuntimeError, "exact pinned official base"):
            assert_official_base_source(model)
        manifest_path = update_partial_manifest(
            output,
            model,
            opset=17,
            graph_name="encoder.onnx",
            graph_signature={"inputs": {}, "outputs": {}},
        )
        manifest = json.loads(manifest_path.read_text())
        self.assertIsNone(manifest["hf_model"])
        self.assertIsNone(manifest["hf_revision"])
        self.assertEqual(manifest["ort_crate_version"], "2.0.0-rc.9")
        self.assertIn("config.json", manifest["source_file_sha256"])


if __name__ == "__main__":
    unittest.main()
