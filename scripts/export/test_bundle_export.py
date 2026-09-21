#!/usr/bin/env python3
"""Lightweight synthetic tests for GLiNER2.5 bundle assembly."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import onnx
from onnx import TensorProto, helper

import bundle_profiles
import export_bundle_2_5 as bundle_export
from common import OFFICIAL_BASE_SOURCE_SHA256, sha256_file


class BundleProfileTests(unittest.TestCase):
    def test_pinned_profiles(self) -> None:
        expected = {
            "small": (
                "fastino/gliner2.5-small-v1",
                "f1e4d8fdd6fe328f45dee6aca3e6a07c9db4296e",
                "gliner2.5-small-v1",
            ),
            "base": (
                "fastino/gliner2.5-base-v1",
                "78cea040597df251eedefa9d7ee2a756af39fe64",
                "gliner2.5-base-v1",
            ),
            "multi": (
                "fastino/gliner2.5-multi-v1",
                "235cf92d6d4318da9bfca0d08975c8fa7250d13b",
                "gliner2.5-multi-v1",
            ),
        }
        self.assertEqual(set(bundle_profiles.PROFILES), set(expected))
        for name, identity in expected.items():
            profile = bundle_profiles.PROFILES[name]
            self.assertEqual(
                (profile.model_id, profile.revision, profile.bundle_name), identity
            )
            for digest in profile.source_sha256.values():
                self.assertRegex(digest, r"^[0-9a-f]{64}$")
        base_hashes = bundle_profiles.PROFILES["base"].source_sha256
        self.assertEqual(
            {name: base_hashes[name] for name in OFFICIAL_BASE_SOURCE_SHA256},
            OFFICIAL_BASE_SOURCE_SHA256,
        )
        self.assertEqual(
            set(base_hashes) - set(OFFICIAL_BASE_SOURCE_SHA256),
            {"tokenizer_config.json", "README.md"},
        )

    def test_verify_source_rejects_missing_and_mismatched_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            expected = "good".encode()
            expected_hash = __import__("hashlib").sha256(expected).hexdigest()
            profile = bundle_profiles.BundleProfile(
                "test/model", "a" * 40, "test-bundle", {"source.bin": expected_hash}
            )
            with mock.patch.dict(
                bundle_profiles.PROFILES, {"synthetic": profile}, clear=True
            ):
                with self.assertRaisesRegex(RuntimeError, "missing regular file"):
                    bundle_profiles.verify_source("synthetic", source)
                (source / "source.bin").write_bytes(b"wrong")
                with self.assertRaisesRegex(RuntimeError, "expected sha256"):
                    bundle_profiles.verify_source("synthetic", source)
                (source / "source.bin").write_bytes(expected)
                self.assertEqual(
                    bundle_profiles.verify_source("synthetic", source),
                    {"source.bin": expected_hash},
                )

    def test_unknown_profile_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown GLiNER2.5 profile"):
            bundle_profiles.verify_source("not-a-model", ".")


class BundleAssemblyTests(unittest.TestCase):
    def _write_source(self, root: Path, *, card: bool = True) -> dict[str, str]:
        files = {
            "config.json": json.dumps(
                {
                    "architecture": "boundary",
                    "architecture_version": 1,
                    "model_name": "microsoft/deberta-v3-xsmall",
                },
                sort_keys=True,
            ).encode(),
            "encoder_config/config.json": json.dumps(
                {"hidden_size": 384}, sort_keys=True
            ).encode(),
            "model.safetensors": b"synthetic weights, never a fixture",
            "tokenizer.json": b'{"version":"1.0"}',
            "tokenizer_config.json": b"{}",
        }
        if card:
            files["README.md"] = b"---\nlicense: apache-2.0\n---\n# Test card\n"
        for relative_name, content in files.items():
            path = root / relative_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        return {name: sha256_file(root / name) for name in files}

    def _write_graphs(self, root: Path) -> None:
        for index, graph_name in enumerate(bundle_export.GRAPH_NAMES):
            input_info = helper.make_tensor_value_info(
                f"input_{index}", TensorProto.FLOAT, ["batch", index + 1]
            )
            output_info = helper.make_tensor_value_info(
                f"output_{index}", TensorProto.FLOAT, ["batch", index + 1]
            )
            node = helper.make_node(
                "Identity", [input_info.name], [output_info.name]
            )
            graph = helper.make_graph(
                [node], f"synthetic_{index}", [input_info], [output_info]
            )
            model = helper.make_model(
                graph,
                opset_imports=[helper.make_opsetid("", 17)],
                producer_name="test_bundle_export",
            )
            onnx.save(model, root / graph_name)

    def test_complete_manifest_uses_actual_signatures_and_replaces_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            bundle = root / "bundle"
            upstream = root / "upstream"
            source.mkdir()
            bundle.mkdir()
            upstream.mkdir()
            hashes = self._write_source(source)
            self._write_graphs(bundle)
            (bundle / "export_manifest.json").write_text(
                '{"status":"m1-partial-incomplete","ort_crate_version":"2.0.0-rc.9"}'
            )
            license_path = upstream / "LICENSE"
            license_path.write_text("Apache License\nVersion 2.0, January 2004\n")
            profile = bundle_profiles.BundleProfile(
                "test/small",
                "b" * 40,
                "gliner2.5-small-test",
                hashes,
            )
            dependencies = dict(bundle_export.EXPECTED_DEPENDENCIES)
            with (
                mock.patch.dict(
                    bundle_profiles.PROFILES, {"small": profile}, clear=True
                ),
                mock.patch.object(
                    bundle_export,
                    "verify_upstream_source",
                    return_value=license_path,
                ),
            ):
                manifest = bundle_export.finalize_bundle(
                    "small",
                    source,
                    bundle,
                    upstream,
                    hashes,
                    dependencies,
                )

            written = json.loads((bundle / "export_manifest.json").read_text())
            self.assertEqual(written, manifest)
            self.assertEqual(written["status"], "exported-unvalidated")
            self.assertFalse(written["release_ready"])
            self.assertIsNone(written["validation"])
            self.assertEqual(written["ort_crate_version"], "2.0.0-rc.13")
            self.assertNotIn("rc.9", (bundle / "export_manifest.json").read_text())
            self.assertEqual(set(written["graphs"]), set(bundle_export.GRAPH_NAMES))
            signature = written["graphs"]["encoder.onnx"]
            self.assertEqual(
                signature["inputs"],
                {"input_0": {"dtype": "FLOAT", "shape": ["batch", 1]}},
            )
            self.assertEqual(
                set(written["files"]), set(bundle_export.REQUIRED_BUNDLE_FILES)
            )
            for metadata in written["files"].values():
                self.assertGreater(metadata["bytes"], 0)
                self.assertRegex(metadata["sha256"], r"^[0-9a-f]{64}$")
            self.assertIn("fp32, opset-17 conversions", (bundle / "NOTICE").read_text())
            self.assertIn("test/small", (bundle / "NOTICE").read_text())

    def test_absent_model_card_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            upstream = root / "upstream"
            source.mkdir()
            upstream.mkdir()
            hashes = self._write_source(source, card=False)
            profile = bundle_profiles.BundleProfile(
                "test/small", "c" * 40, "test", hashes
            )
            license_path = upstream / "LICENSE"
            license_path.write_text("Apache License")
            with (
                mock.patch.dict(
                    bundle_profiles.PROFILES, {"small": profile}, clear=True
                ),
                mock.patch.object(
                    bundle_export,
                    "verify_upstream_source",
                    return_value=license_path,
                ),
                self.assertRaisesRegex(
                    RuntimeError, "original README.md from the exact immutable HF revision"
                ),
            ):
                bundle_export.preflight_source("small", source, upstream)

    def test_export_commands_are_opset17_subprocesses(self) -> None:
        commands = bundle_export.export_commands(Path("model"), Path("bundle"))
        self.assertEqual(len(commands), 7)
        self.assertTrue(all(command[0] == __import__("sys").executable for command in commands))
        self.assertTrue(all(command[-2:] == ["--opset", "17"] for command in commands))


if __name__ == "__main__":
    unittest.main()
