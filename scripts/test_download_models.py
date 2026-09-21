#!/usr/bin/env python3
"""Model-free tests for scripts/download_models.py."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from typing import Any

MODULE_PATH = Path(__file__).with_name("download_models.py")
SPEC = importlib.util.spec_from_file_location("download_models", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
DOWNLOAD_MODELS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOAD_MODELS)

BundleError = DOWNLOAD_MODELS.BundleError


def file_spec(data: bytes) -> dict[str, Any]:
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def boundary_manifest(bundle: str = "gliner2.5-base-v1") -> tuple[dict[str, Any], dict[str, bytes]]:
    profile = next(
        item
        for item in DOWNLOAD_MODELS.BOUNDARY_MODELS.values()
        if item["bundle"] == bundle
    )
    contents = {
        name: f"contents:{name}".encode()
        for name in DOWNLOAD_MODELS.BOUNDARY_REQUIRED_FILES
    }
    contents["config.json"] = json.dumps({
        "architecture": "boundary", "architecture_version": 1,
        "model_name": profile["encoder_model"],
    }).encode()
    manifest = {
        "manifest_version": 1,
        "architecture": "boundary",
        "architecture_version": 1,
        "status": "validated",
        "release_ready": True,
        "hf_model": profile["hf_model"],
        "hf_revision": profile["hf_revision"],
        "gliner2_commit": DOWNLOAD_MODELS.GLINER2_COMMIT,
        "opset": 17,
        "precision": "fp32",
        "ort_crate_version": "2.0.0-rc.13",
        "native_onnx_runtime": "1.28.0",
        "validation_onnxruntime": "1.20.1",
        "dependencies": dict(DOWNLOAD_MODELS.EXPECTED_EXPORT_DEPENDENCIES),
        "source_file_sha256": {"gliner2/model.py": "a" * 64},
        "files": {name: file_spec(data) for name, data in contents.items()},
        "graphs": {
            name: {
                "inputs": {
                    "input": {"dtype": "FLOAT", "shape": [1, "L"]}
                },
                "outputs": {
                    "output": {"dtype": "FLOAT", "shape": [1, "L", 1]}
                },
            }
            for name in DOWNLOAD_MODELS.BOUNDARY_GRAPHS
        },
        "validation": {"report": "validation.json", "sha256": "b" * 64},
    }
    return manifest, contents


def tiny_v2_pins() -> tuple[dict[str, Any], dict[tuple[str, str, str], bytes]]:
    files: dict[tuple[str, str, str], bytes] = {}
    models = {}
    hosted_revision = "c" * 40
    source_revisions = {"base": "d" * 40, "large": "e" * 40}
    for selector in ("base", "large"):
        bundle = f"gliner2-{selector}-v1"
        source_repo = f"fastino/{bundle}"
        source_revision = source_revisions[selector]
        metadata_contents = {
            "config.json": f"{selector}-config".encode(),
            "tokenizer.json": f"{selector}-tokenizer".encode(),
            "tokenizer_config.json": f"{selector}-tokenizer-config".encode(),
        }
        onnx_contents = {
            "encoder.onnx": f"{selector}-encoder".encode(),
            "extractor_padded.onnx": f"{selector}-extractor".encode(),
            "classifier.onnx": f"{selector}-classifier".encode(),
        }
        for name, data in metadata_contents.items():
            files[(source_repo, source_revision, name)] = data
        for name, data in onnx_contents.items():
            files[(DOWNLOAD_MODELS.REPO_ID, hosted_revision, f"{bundle}/{name}")] = data
        models[bundle] = {
            "selector": selector,
            "source_repository": source_repo,
            "source_revision": source_revision,
            "source_metadata": {
                name: file_spec(data) for name, data in metadata_contents.items()
            },
            "hosted_onnx_files": {
                name: file_spec(data) for name, data in onnx_contents.items()
            },
            "observed_identity": {"hidden_size": 1},
        }
    return (
        {
            "schema_version": 1,
            "description": "test pins",
            "hash_algorithm": "sha256",
            "hosted_onnx": {
                "repository": DOWNLOAD_MODELS.REPO_ID,
                "verified_revision": hosted_revision,
            },
            "models": models,
        },
        files,
    )


class PathAndSelectorTests(unittest.TestCase):
    def test_selectors_and_full_bundle_aliases(self) -> None:
        self.assertEqual(DOWNLOAD_MODELS.resolve_selection("base"), ["base"])
        self.assertEqual(
            DOWNLOAD_MODELS.resolve_selection("gliner2.5-multi-v1"), ["2.5-multi"]
        )
        self.assertEqual(
            DOWNLOAD_MODELS.resolve_selection("all"),
            ["base", "large", "2.5-small", "2.5-base", "2.5-multi"],
        )
        with self.assertRaisesRegex(BundleError, "unknown model"):
            DOWNLOAD_MODELS.resolve_selection("small")

    def test_unsafe_paths_are_rejected(self) -> None:
        for value in (
            "/absolute",
            "../escape",
            "nested/../../escape",
            r"nested\windows",
            "C:/windows",
            "nested//not-normalized",
            "nested/./not-normalized",
            "",
        ):
            with self.subTest(value=value), self.assertRaises(BundleError):
                DOWNLOAD_MODELS.safe_relative_path(value)

    def test_resolved_path_cannot_escape_through_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "bundle"
            outside = parent / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "link").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(BundleError, "escapes destination"):
                DOWNLOAD_MODELS._target(root, "link/artifact")


class ManifestTests(unittest.TestCase):
    def test_booleans_and_numbers_are_not_interchangeable(self) -> None:
        for key, value in (("manifest_version", True), ("architecture_version", 1.0), ("release_ready", 1)):
            with self.subTest(key=key):
                manifest, _ = boundary_manifest()
                manifest[key] = value
                with self.assertRaises(BundleError):
                    DOWNLOAD_MODELS.validate_boundary_manifest(manifest, "gliner2.5-base-v1")

    def test_dependency_pins_are_required(self) -> None:
        manifest, _ = boundary_manifest()
        manifest["dependencies"]["torch"] = "different"
        with self.assertRaisesRegex(BundleError, "dependency pins"):
            DOWNLOAD_MODELS.validate_boundary_manifest(manifest, "gliner2.5-base-v1")

    def test_exporter_shaped_boundary_manifest_is_accepted(self) -> None:
        manifest, _ = boundary_manifest()
        files = DOWNLOAD_MODELS.validate_boundary_manifest(
            manifest, "gliner2.5-base-v1"
        )
        self.assertEqual(set(files), set(DOWNLOAD_MODELS.BOUNDARY_REQUIRED_FILES))
        encoder = manifest["graphs"]["encoder.onnx"]
        self.assertEqual(
            encoder["inputs"],
            {"input": {"dtype": "FLOAT", "shape": [1, "L"]}},
        )

    def test_legacy_and_malformed_graph_signatures_are_rejected(self) -> None:
        cases = (
            (
                [{"name": "input", "dtype": "FLOAT", "shape": [1]}],
                "nonempty inputs object",
            ),
            ({"": {"dtype": "FLOAT", "shape": [1]}}, "invalid inputs tensor name"),
            (
                {"input": {"dtype": "float32", "shape": [1]}},
                "invalid ONNX dtype",
            ),
            (
                {"input": {"dtype": "FLOAT", "shape": [-1]}},
                "invalid shape dimension",
            ),
            (
                {"input": {"dtype": "FLOAT", "shape": [True]}},
                "invalid shape dimension",
            ),
            (
                {"input": {"dtype": "FLOAT", "shape": [" "]}},
                "invalid shape dimension",
            ),
        )
        for inputs, message in cases:
            with self.subTest(inputs=inputs):
                manifest, _ = boundary_manifest()
                manifest["graphs"]["encoder.onnx"]["inputs"] = inputs
                with self.assertRaisesRegex(BundleError, message):
                    DOWNLOAD_MODELS.validate_boundary_manifest(
                        manifest, "gliner2.5-base-v1"
                    )

    def test_duplicate_json_keys_are_rejected_before_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text('{"graphs": {}, "graphs": {}}', encoding="utf-8")
            with self.assertRaisesRegex(BundleError, "duplicate JSON object key 'graphs'"):
                DOWNLOAD_MODELS._read_json_document(
                    path, DOWNLOAD_MODELS.MAX_MANIFEST_BYTES, "test manifest"
                )

    def test_partial_and_unreleased_manifests_are_rejected(self) -> None:
        manifest, _ = boundary_manifest()
        del manifest["files"]["boundary_explicit_scorer.onnx"]
        with self.assertRaisesRegex(BundleError, "partial"):
            DOWNLOAD_MODELS.validate_boundary_manifest(
                manifest, "gliner2.5-base-v1"
            )

        manifest, _ = boundary_manifest()
        manifest["release_ready"] = False
        with self.assertRaisesRegex(BundleError, "release_ready"):
            DOWNLOAD_MODELS.validate_boundary_manifest(
                manifest, "gliner2.5-base-v1"
            )

    def test_unsafe_manifest_path_is_rejected(self) -> None:
        manifest, _ = boundary_manifest()
        manifest["files"]["../escape"] = file_spec(b"bad")
        with self.assertRaisesRegex(BundleError, "unsafe bundle path"):
            DOWNLOAD_MODELS.validate_boundary_manifest(
                manifest, "gliner2.5-base-v1"
            )

    def test_wrong_architecture_and_graph_set_are_rejected(self) -> None:
        manifest, _ = boundary_manifest()
        manifest["architecture"] = "span"
        with self.assertRaisesRegex(BundleError, "architecture"):
            DOWNLOAD_MODELS.validate_boundary_manifest(
                manifest, "gliner2.5-base-v1"
            )

        manifest, _ = boundary_manifest()
        del manifest["graphs"]["boundary_records.onnx"]
        with self.assertRaisesRegex(BundleError, "exactly the seven"):
            DOWNLOAD_MODELS.validate_boundary_manifest(
                manifest, "gliner2.5-base-v1"
            )


class HashTests(unittest.TestCase):
    def test_stream_hash_and_size_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact"
            path.write_bytes(b"correct")
            DOWNLOAD_MODELS.verify_file(path, file_spec(b"correct"))
            with self.assertRaisesRegex(BundleError, "size mismatch"):
                DOWNLOAD_MODELS.verify_file(path, file_spec(b"too long"))
            bad_hash = file_spec(b"correct")
            bad_hash["sha256"] = "0" * 64
            with self.assertRaisesRegex(BundleError, "SHA-256 mismatch"):
                DOWNLOAD_MODELS.verify_file(path, bad_hash)

    def test_hash_failure_does_not_replace_existing_bundle(self) -> None:
        manifest, contents = boundary_manifest()
        revision = "f" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            remote = root / "remote"
            remote.mkdir()
            mapping: dict[tuple[str, str, str], Path] = {}
            manifest_path = remote / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            bundle = "gliner2.5-base-v1"
            mapping[(DOWNLOAD_MODELS.REPO_ID, revision, f"{bundle}/export_manifest.json")] = manifest_path
            for index, (name, data) in enumerate(contents.items()):
                path = remote / f"file-{index}"
                path.write_bytes(b"corrupt" if name == "classifier.onnx" else data)
                mapping[(DOWNLOAD_MODELS.REPO_ID, revision, f"{bundle}/{name}")] = path

            destination = root / "onnx"
            old = destination / bundle
            old.mkdir(parents=True)
            (old / "marker").write_text("old", encoding="utf-8")

            def downloader(repo: str, rev: str, filename: str) -> Path:
                return mapping[(repo, rev, filename)]

            with self.assertRaisesRegex(BundleError, "mismatch"):
                DOWNLOAD_MODELS._download_boundary(
                    DOWNLOAD_MODELS.BOUNDARY_MODELS["2.5-base"],
                    destination,
                    revision,
                    downloader,
                )
            self.assertEqual((old / "marker").read_text(encoding="utf-8"), "old")


class InstallationTests(unittest.TestCase):
    def test_boundary_install_checks_authenticated_config_identity(self) -> None:
        for override in (None, {"architecture": "span"}, {"architecture_version": True}, {"model_name": "wrong"}):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                remote = root / "remote"
                remote.mkdir()
                manifest, contents = boundary_manifest()
                if override is not None:
                    config = json.loads(contents["config.json"])
                    config.update(override)
                    contents["config.json"] = json.dumps(config).encode()
                    manifest["files"]["config.json"] = file_spec(contents["config.json"])
                for name, data in contents.items():
                    path = remote / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                (remote / "export_manifest.json").write_text(json.dumps(manifest))
                dest = root / "onnx"
                bundle = dest / "gliner2.5-base-v1"
                bundle.mkdir(parents=True)
                (bundle / "marker").write_text("old")
                def download(_repo, _revision, filename):
                    return remote / filename.split("/", 1)[1]
                def install():
                    DOWNLOAD_MODELS._download_boundary(
                        DOWNLOAD_MODELS.BOUNDARY_MODELS["2.5-base"], dest, "f" * 40, download
                    )
                if override is None:
                    install()
                    self.assertFalse((bundle / "marker").exists())
                    self.assertEqual((bundle / "config.json").read_bytes(), contents["config.json"])
                else:
                    with self.assertRaises(BundleError):
                        install()
                    self.assertEqual((bundle / "marker").read_text(), "old")

    def test_failed_install_restores_existing_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "bundle"
            destination.mkdir()
            (destination / "marker").write_text("old", encoding="utf-8")
            stage = root / "stage"
            stage.mkdir()
            (stage / "marker").write_text("new", encoding="utf-8")
            original_rename = Path.rename

            def fail_stage_rename(path: Path, target: Path) -> Path:
                if path == stage:
                    raise OSError("injected install failure")
                return original_rename(path, target)

            with mock.patch.object(Path, "rename", new=fail_stage_rename):
                with self.assertRaisesRegex(OSError, "injected install failure"):
                    DOWNLOAD_MODELS._replace_bundle(stage, destination)
            self.assertEqual(
                (destination / "marker").read_text(encoding="utf-8"), "old"
            )

    def test_symlink_destination_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            actual = root / "actual"
            actual.mkdir()
            destination = root / "bundle"
            destination.symlink_to(actual, target_is_directory=True)
            stage = root / "stage"
            stage.mkdir()
            with self.assertRaisesRegex(BundleError, "symlink destination"):
                DOWNLOAD_MODELS._replace_bundle(stage, destination)


class MetadataResolutionTests(unittest.TestCase):
    def test_committed_pins_resolve_verified_original_sources(self) -> None:
        pins = DOWNLOAD_MODELS.load_v2_metadata_pins()
        self.assertEqual(
            pins["models"]["gliner2-base-v1"]["source_revision"],
            "79c3a777abc572b4767922f3916cf63fb5754df2",
        )
        self.assertEqual(
            pins["models"]["gliner2-large-v1"]["source_revision"],
            "f32ea6ef6e26d8264fdc72431b4b3b041eadc537",
        )
        for model in pins["models"].values():
            self.assertIn("config.json", model["source_metadata"])
            self.assertIn("tokenizer.json", model["source_metadata"])

    def test_v2_download_uses_pinned_source_revision_and_collocates_metadata(self) -> None:
        pins, contents = tiny_v2_pins()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pins_path = root / "pins.json"
            pins_path.write_text(json.dumps(pins), encoding="utf-8")
            remote = root / "remote"
            remote.mkdir()
            paths: dict[tuple[str, str, str], Path] = {}
            for index, (key, data) in enumerate(contents.items()):
                path = remote / str(index)
                path.write_bytes(data)
                paths[key] = path
            calls: list[tuple[str, str, str]] = []

            def downloader(repo: str, revision: str, filename: str) -> Path:
                calls.append((repo, revision, filename))
                return paths[(repo, revision, filename)]

            installed = DOWNLOAD_MODELS.download_selected(
                "base",
                root / "onnx",
                pins_path=pins_path,
                downloader=downloader,
            )
            bundle = installed[0]
            self.assertTrue((bundle / "encoder.onnx").is_file())
            self.assertTrue((bundle / "config.json").is_file())
            self.assertTrue((bundle / "tokenizer.json").is_file())
            self.assertIn(
                (
                    "fastino/gliner2-base-v1",
                    "d" * 40,
                    "tokenizer.json",
                ),
                calls,
            )
            self.assertNotIn("main", {revision for _, revision, _ in calls})

    def test_unpublished_boundary_revision_must_be_explicit(self) -> None:
        pins, _ = tiny_v2_pins()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "pins.json"
            path.write_text(json.dumps(pins), encoding="utf-8")
            with self.assertRaisesRegex(BundleError, "boundary downloads require --revision"):
                DOWNLOAD_MODELS.download_selected(
                    "2.5-base",
                    root / "onnx",
                    pins_path=path,
                    downloader=lambda *_: self.fail("network callback was invoked"),
                )

    def test_moving_revisions_are_rejected_without_network(self) -> None:
        pins, _ = tiny_v2_pins()
        pins["models"]["gliner2-base-v1"]["source_revision"] = "main"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "pins.json"
            path.write_text(json.dumps(pins), encoding="utf-8")
            with self.assertRaisesRegex(BundleError, "not immutable"):
                DOWNLOAD_MODELS.load_v2_metadata_pins(path)

            pins, _ = tiny_v2_pins()
            path.write_text(json.dumps(pins), encoding="utf-8")
            with self.assertRaisesRegex(BundleError, "immutable commit SHA"):
                DOWNLOAD_MODELS.download_selected(
                    "base",
                    root / "onnx",
                    revision="main",
                    pins_path=path,
                    downloader=lambda *_: self.fail("network callback was invoked"),
                )


if __name__ == "__main__":
    unittest.main()
