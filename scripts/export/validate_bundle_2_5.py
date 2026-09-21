#!/usr/bin/env python3
"""Validate one or all GLiNER2.5 bundles against checkpoint-specific oracles.

This orchestrator verifies source identity and fixture provenance before running
all seven source-vs-ONNX stage validators. By default it then runs the native
Rust full-extraction parity test. It never promotes a bundle manifest; parent
release review owns that decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
PARITY_DIR = ROOT / "scripts" / "parity"
for directory in (SCRIPT_DIR, PARITY_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from bundle_profiles import PROFILES, verify_source  # noqa: E402
from common import (  # noqa: E402
    GLINER2_COMMIT,
    assert_pinned_gliner2_installation,
    package_versions,
    sha256_file,
)
from gen_boundary_goldens import load_corpus  # noqa: E402

ATOL = 1e-4
RTOL = 1e-3
CONFIDENCE_ATOL = 1e-3
PREFIX_COORDINATE_ATOL = 1.1e-6
GRAPH_FILES = (
    "encoder.onnx",
    "classifier.onnx",
    "boundary_marginals.onnx",
    "boundary_scorer.onnx",
    "boundary_explicit_scorer.onnx",
    "boundary_records.onnx",
    "boundary_relations.onnx",
)
REQUIRED_BUNDLE_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "encoder_config/config.json",
    "SOURCE_MODEL_CARD.md",
    "LICENSE",
    "NOTICE",
    *GRAPH_FILES,
)
EXPECTED_PROFILES = {
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
EXPECTED_ADDITIONAL_IDS = ["mixed_unicode_all_tasks", "explicit_unicode_duplicates"]
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def profile_field(profile: Any, name: str) -> Any:
    if isinstance(profile, dict):
        return profile[name]
    return getattr(profile, name)


def checked_profile(name: str) -> Any:
    if name not in EXPECTED_PROFILES or name not in PROFILES:
        raise KeyError(f"unknown or unavailable profile {name!r}")
    profile = PROFILES[name]
    model_id, revision, bundle_name = EXPECTED_PROFILES[name]
    for field, expected in (
        ("model_id", model_id),
        ("revision", revision),
        ("bundle_name", bundle_name),
    ):
        actual = profile_field(profile, field)
        if actual != expected:
            raise RuntimeError(f"profile {name}.{field}={actual!r}, expected {expected!r}")
    return profile


def default_model_dir(profile: Any) -> Path:
    cache_name = "models--" + profile_field(profile, "model_id").replace("/", "--")
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / cache_name
        / "snapshots"
        / profile_field(profile, "revision")
    )


def resolve_profile_dir(root: Path, profile: Any, marker: str) -> Path:
    direct = root.expanduser().resolve()
    if (direct / marker).is_file():
        return direct
    return direct / profile_field(profile, "bundle_name")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*EXPECTED_PROFILES, "all"), required=True)
    parser.add_argument(
        "--model-dir",
        help="source checkpoint directory; valid only for one selected profile",
    )
    parser.add_argument(
        "--bundle-dir",
        required=True,
        help="one bundle directory or root containing named bundle directories",
    )
    parser.add_argument(
        "--fixture-dir",
        required=True,
        help="one fixture directory or root containing named fixture directories",
    )
    parser.add_argument(
        "--corpus", default=str(PARITY_DIR / "boundary_corpus.json")
    )
    parser.add_argument("--report-json", help="aggregate machine-readable report path")
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="run all source/ONNX stages but deliberately omit native parity",
    )
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def safe_relative(name: str) -> Path:
    if not name or "\\" in name:
        raise AssertionError(f"unsafe manifest path {name!r}")
    raw_parts = name.split("/")
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in ("", ".", "..") for part in raw_parts)
    ):
        raise AssertionError(f"unsafe manifest path {name!r}")
    return Path(*posix.parts)


def checked_manifest_file(root: Path, name: str, entry: dict[str, Any]) -> Path:
    relative = safe_relative(name)
    path = root / relative
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise AssertionError(f"manifest path escapes root: {name!r}") from exc
    if not path.is_file():
        raise AssertionError(f"missing manifest file {path}")
    size = entry.get("bytes")
    digest = entry.get("sha256")
    if not isinstance(size, int) or size <= 0 or path.stat().st_size != size:
        raise AssertionError(f"size mismatch for {path}")
    if not isinstance(digest, str) or not HEX_SHA256.fullmatch(digest):
        raise AssertionError(f"invalid SHA-256 metadata for {path}")
    actual = sha256_file(path)
    if actual != digest:
        raise AssertionError(f"SHA-256 mismatch for {path}: {actual} != {digest}")
    return path


def dim_signature(value: onnx.ValueInfoProto) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append(None)
    return result


def graph_audit(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    imports = {item.domain: item.version for item in model.opset_import}
    if imports.get("") != 17:
        raise AssertionError(f"{path.name}: default opset {imports.get('')} != 17")
    counts = {"float_initializers": 0, "float_constants": 0}

    def inspect_tensor(tensor: onnx.TensorProto, label: str, initializer: bool) -> None:
        value = numpy_helper.to_array(tensor)
        if value.dtype.kind not in "fc":
            return
        if initializer and value.dtype != np.float32:
            raise AssertionError(f"{path.name}:{label} is {value.dtype}, expected fp32")
        if not np.isfinite(value).all():
            raise AssertionError(f"{path.name}:{label} contains non-finite values")
        key = "float_initializers" if initializer else "float_constants"
        counts[key] += 1

    def visit(graph: onnx.GraphProto) -> None:
        for initializer in graph.initializer:
            inspect_tensor(initializer, f"initializer {initializer.name!r}", True)
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    inspect_tensor(attribute.t, f"node {node.name or node.op_type!r}", False)
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    for value in attribute.tensors:
                        inspect_tensor(value, f"node {node.name or node.op_type!r}", False)
                elif attribute.type == onnx.AttributeProto.GRAPH:
                    visit(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for child in attribute.graphs:
                        visit(child)

    visit(model.graph)
    return {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "opset": imports[""],
        "inputs": {
            value.name: {
                "dtype": onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type),
                "shape": dim_signature(value),
            }
            for value in model.graph.input
        },
        "outputs": {
            value.name: {
                "dtype": onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type),
                "shape": dim_signature(value),
            }
            for value in model.graph.output
        },
        **counts,
    }


def validate_bundle_manifest(
    bundle_dir: Path,
    profile: Any,
    source_hashes: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = bundle_dir / "export_manifest.json"
    document = json.loads(path.read_text())
    expected = {
        "manifest_version": 1,
        "architecture": "boundary",
        "architecture_version": 1,
        "hf_model": profile_field(profile, "model_id"),
        "hf_revision": profile_field(profile, "revision"),
        "gliner2_commit": GLINER2_COMMIT,
        "opset": 17,
        "precision": "fp32",
        "ort_crate_version": "2.0.0-rc.13",
        "native_onnx_runtime": "1.28.0",
        "validation_onnxruntime": "1.20.1",
        "release_ready": False,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise AssertionError(
                f"{path}: {key}={document.get(key)!r}, expected {value!r}"
            )
    if document.get("status") != "exported-unvalidated":
        raise AssertionError("development validation requires exported-unvalidated status")
    if document.get("source_file_sha256") != source_hashes:
        raise AssertionError("bundle source hashes differ from verified checkpoint")
    dependencies = document.get("dependencies")
    if not isinstance(dependencies, dict) or not dependencies:
        raise AssertionError("bundle manifest lacks actual export dependencies")

    files = document.get("files")
    if not isinstance(files, dict):
        raise AssertionError("bundle manifest files must be an object")
    checked_files = {
        name: checked_manifest_file(bundle_dir, name, entry)
        for name, entry in files.items()
    }
    missing_files = set(REQUIRED_BUNDLE_FILES) - set(checked_files)
    if missing_files:
        raise AssertionError(
            f"bundle manifest omits required files {sorted(missing_files)}"
        )

    graphs = document.get("graphs")
    if not isinstance(graphs, dict) or set(graphs) != set(GRAPH_FILES):
        raise AssertionError(
            f"bundle graph set {sorted(graphs or {})} != {sorted(GRAPH_FILES)}"
        )
    audits = {name: graph_audit(bundle_dir / name) for name in GRAPH_FILES}
    for name, audit in audits.items():
        declared = graphs[name]
        if declared.get("inputs") != audit["inputs"]:
            raise AssertionError(f"{name}: manifest input signature differs from graph")
        if declared.get("outputs") != audit["outputs"]:
            raise AssertionError(f"{name}: manifest output signature differs from graph")
        file_entry = files[name]
        if (file_entry["sha256"], file_entry["bytes"]) != (
            audit["sha256"],
            audit["bytes"],
        ):
            raise AssertionError(f"{name}: graph audit differs from files metadata")
    return document, audits


def validate_fixture_manifest(
    fixture_dir: Path,
    profile_name: str,
    profile: Any,
    source_hashes: dict[str, str],
    corpus_path: Path,
) -> dict[str, Any]:
    cases, corpus_hash = load_corpus(corpus_path)
    expected_ids = [case["id"] for case in cases]
    path = fixture_dir / "manifest.json"
    document = json.loads(path.read_text())
    expected = {
        "format_version": 2,
        "status": "complete",
        "bundle_profile": profile_name,
        "bundle_name": profile_field(profile, "bundle_name"),
        "model_id": profile_field(profile, "model_id"),
        "hf_revision": profile_field(profile, "revision"),
        "gliner2_commit": GLINER2_COMMIT,
        "source_file_sha256": source_hashes,
        "corpus_sha256": corpus_hash,
        "case_count": 30,
        "successful_case_count": 30,
        "error_case_count": 0,
        "case_ids": expected_ids,
        "additional_case_ids": EXPECTED_ADDITIONAL_IDS,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise AssertionError(
                f"{path}: {key}={document.get(key)!r}, expected {value!r}"
            )
    provenance = document.get("provenance", {})
    for key, value in (
        ("model_id", profile_field(profile, "model_id")),
        ("hf_revision", profile_field(profile, "revision")),
        ("gliner2_commit", GLINER2_COMMIT),
        ("source_file_sha256", source_hashes),
        ("onnx_used_as_reference", False),
        ("dependencies", package_versions()),
    ):
        if provenance.get(key) != value:
            raise AssertionError(f"fixture provenance {key} is not current and pinned")

    entries = document.get("entries")
    additional = document.get("additional_entries")
    if not isinstance(entries, list) or len(entries) != 30:
        raise AssertionError("fixture manifest must contain exactly 30 corpus entries")
    if not isinstance(additional, list) or len(additional) != 2:
        raise AssertionError("fixture manifest must contain exactly two additional entries")
    if [item.get("case_id") for item in entries] != expected_ids:
        raise AssertionError("fixture entries are not the closed corpus order")
    if [item.get("case_id") for item in additional] != EXPECTED_ADDITIONAL_IDS:
        raise AssertionError("fixture additional entries are not the closed case set")

    stage_invocations: set[str] = set()
    checked: list[dict[str, Any]] = []
    for entry in [*entries, *additional]:
        if entry.get("status") != "ok":
            raise AssertionError(f"fixture entry {entry.get('case_id')} is not successful")
        paths = {}
        for kind in ("npz", "json"):
            relative = entry.get(kind)
            if not isinstance(relative, str):
                raise AssertionError(f"fixture entry lacks {kind} path")
            paths[kind] = checked_manifest_file(
                fixture_dir,
                relative,
                {
                    "bytes": entry.get(f"{kind}_bytes"),
                    "sha256": entry.get(f"{kind}_sha256"),
                },
            )
        metadata = json.loads(paths["json"].read_text())
        if metadata.get("case_id") != entry["case_id"]:
            raise AssertionError(f"fixture ID mismatch in {paths['json']}")
        if metadata.get("provenance") != provenance:
            raise AssertionError(f"fixture provenance differs in {paths['json']}")
        if metadata.get("kind") != "explicit_spans":
            expected_result = metadata.get("final_result_utf8_offsets")
            if not isinstance(expected_result, dict):
                raise AssertionError(f"{paths['json']} lacks a typed final result")
            if metadata.get("original_text") != metadata.get("text"):
                raise AssertionError(f"{paths['json']} did not preserve original UTF-8 text")
            stage_invocations.update(
                name
                for name, state in metadata.get("stages", {}).items()
                if state.get("invoked") is True
            )
        else:
            stage_invocations.add("explicit_sparse_scorer")
            if len(metadata.get("labels", [])) != 2 or len(metadata.get("spans", [])) != 4:
                raise AssertionError("explicit Unicode/duplicate-span coverage changed")
        with np.load(paths["npz"], allow_pickle=False) as arrays:
            for array_name in arrays.files:
                value = arrays[array_name]
                if value.dtype.kind in "fc" and not np.isfinite(value).all():
                    raise AssertionError(
                        f"{paths['npz']}:{array_name} contains non-finite values"
                    )
        checked.append(
            {
                "case_id": entry["case_id"],
                "json_sha256": entry["json_sha256"],
                "npz_sha256": entry["npz_sha256"],
            }
        )

    required_stages = {
        "encoder",
        "boundary_marginals",
        "shared_pool",
        "shared_scorer",
        "boundary_head",
        "classifier",
        "explicit_sparse_scorer",
        "record_head",
        "relation_scorer",
    }
    missing = required_stages - stage_invocations
    if missing:
        raise AssertionError(f"fixture suite does not invoke stages {sorted(missing)}")
    return {
        "manifest_sha256": sha256_file(path),
        "corpus_sha256": corpus_hash,
        "case_count": len(entries),
        "additional_case_count": len(additional),
        "stage_invocations": sorted(stage_invocations),
        "entries": checked,
    }


def run_stage(label: str, command: list[str], report_path: Path) -> dict[str, Any]:
    print(f"[{label}] {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")
    if not report_path.is_file():
        raise RuntimeError(f"{label} did not create {report_path}")
    report = json.loads(report_path.read_text())
    return {
        "command": command,
        "report": report,
        "report_sha256": sha256_file(report_path),
    }


def stage_commands(
    model_dir: Path,
    bundle_dir: Path,
    fixture_dir: Path,
    report_dir: Path,
    seed: int,
) -> list[tuple[str, list[str], Path]]:
    # Resolving the venv executable symlink loses its site-packages in child processes.
    python = str(Path(sys.executable).absolute())
    common = ["--model-dir", str(model_dir), "--golden-dir", str(fixture_dir), "--seed", str(seed)]
    definitions = [
        (
            "encoder",
            "validate_encoder.py",
            ["--onnx-path", str(bundle_dir / "encoder.onnx"), *common],
        ),
        (
            "classifier",
            "validate_classifier.py",
            ["--onnx-path", str(bundle_dir / "classifier.onnx"), *common],
        ),
        (
            "boundary_marginals",
            "validate_boundary_marginals.py",
            ["--onnx-path", str(bundle_dir / "boundary_marginals.onnx"), *common],
        ),
        (
            "boundary_scorer",
            "validate_boundary_scorer.py",
            ["--onnx-path", str(bundle_dir / "boundary_scorer.onnx"), *common],
        ),
        (
            "boundary_explicit_scorer",
            "validate_boundary_explicit_scorer.py",
            [
                "--onnx-path",
                str(bundle_dir / "boundary_explicit_scorer.onnx"),
                "--marginals-onnx",
                str(bundle_dir / "boundary_marginals.onnx"),
                *common,
            ],
        ),
        (
            "boundary_records",
            "validate_boundary_records.py",
            ["--onnx-path", str(bundle_dir / "boundary_records.onnx"), *common],
        ),
        (
            "boundary_relations",
            "validate_boundary_relations.py",
            ["--onnx-path", str(bundle_dir / "boundary_relations.onnx"), *common],
        ),
    ]
    result = []
    for label, script, arguments in definitions:
        report = report_dir / f"{label}.json"
        result.append(
            (
                label,
                [python, str(SCRIPT_DIR / script), *arguments, "--report-json", str(report)],
                report,
            )
        )
    return result


def run_native(
    profile_name: str,
    bundle_dir: Path,
    fixture_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "GLINER2_BUNDLE_MODEL": profile_name,
            "GLINER2_BUNDLE_GRAPH_DIR": str(bundle_dir),
            "GLINER2_BUNDLE_FIXTURE_DIR": str(fixture_dir),
            "GLINER2_STRICT_BUNDLE_VALIDATION": "1",
            "GLINER2_BUNDLE_REPORT_JSON": str(report_path),
        }
    )
    command = ["cargo", "test", "--locked", "--test", "bundle_parity", "--", "--nocapture"]
    print(f"[native/{profile_name}] {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"native/{profile_name} failed with exit code {completed.returncode}")
    if not report_path.is_file():
        raise RuntimeError(f"native test did not create {report_path}")
    return {
        "command": command,
        "report": json.loads(report_path.read_text()),
        "report_sha256": sha256_file(report_path),
    }


def validate_profile(
    name: str,
    model_dir: Path,
    bundle_dir: Path,
    fixture_dir: Path,
    corpus_path: Path,
    seed: int,
    python_only: bool,
) -> dict[str, Any]:
    profile = checked_profile(name)
    source_hashes = verify_source(name, model_dir)
    expected_source = profile_field(profile, "source_sha256")
    if dict(source_hashes) != dict(expected_source):
        raise AssertionError(f"verify_source({name!r}) returned unexpected hashes")
    manifest, graph_audits = validate_bundle_manifest(bundle_dir, profile, source_hashes)
    fixtures = validate_fixture_manifest(
        fixture_dir, name, profile, source_hashes, corpus_path
    )

    with tempfile.TemporaryDirectory(prefix=f"gliner25-{name}-validation-") as temporary:
        report_dir = Path(temporary)
        stages = {
            label: run_stage(label, command, report_path)
            for label, command, report_path in stage_commands(
                model_dir, bundle_dir, fixture_dir, report_dir, seed
            )
        }
        native = None
        if not python_only:
            native = run_native(
                name, bundle_dir, fixture_dir, report_dir / "native.json"
            )

    return {
        "profile": name,
        "bundle_name": profile_field(profile, "bundle_name"),
        "hf_model": profile_field(profile, "model_id"),
        "hf_revision": profile_field(profile, "revision"),
        "gliner2_commit": GLINER2_COMMIT,
        "source_file_sha256": dict(sorted(source_hashes.items())),
        "bundle_manifest_sha256": sha256_file(bundle_dir / "export_manifest.json"),
        "bundle_manifest_status_before_validation": manifest["status"],
        "graph_audits": graph_audits,
        "fixtures": fixtures,
        "tolerances": {
            "atol": ATOL,
            "rtol": RTOL,
            "inside_prefix_coordinate_atol": PREFIX_COORDINATE_ATOL,
            "final_confidence_atol": CONFIDENCE_ATOL,
            "discrete_outputs": "exact",
        },
        "python_stages": stages,
        "native": native,
        "status": "passed" if native is not None else "python-passed-native-not-run",
        "release_ready": False,
    }


def main() -> None:
    args = parse_args()
    if ort.__version__ != "1.20.1":
        raise RuntimeError(
            f"bundle validation requires Python ONNX Runtime 1.20.1, found {ort.__version__}"
        )
    assert_pinned_gliner2_installation()
    if args.model == "all" and args.model_dir:
        raise SystemExit("--model-dir is valid only for one selected profile")
    names = list(EXPECTED_PROFILES) if args.model == "all" else [args.model]
    reports = []
    for name in names:
        profile = checked_profile(name)
        model_dir = (
            Path(args.model_dir).expanduser().resolve()
            if args.model_dir
            else default_model_dir(profile)
        )
        bundle_dir = resolve_profile_dir(
            Path(args.bundle_dir), profile, "export_manifest.json"
        )
        fixture_dir = resolve_profile_dir(Path(args.fixture_dir), profile, "manifest.json")
        reports.append(
            validate_profile(
                name,
                model_dir,
                bundle_dir,
                fixture_dir,
                Path(args.corpus),
                args.seed,
                args.python_only,
            )
        )

    document = {
        "format_version": 1,
        "kind": "gliner2.5-bundle-validation",
        "status": (
            "passed" if all(item["status"] == "passed" for item in reports)
            else "partial-native-not-run"
        ),
        "release_ready": False,
        "runtime": {
            "python": platform.python_version(),
            "python_executable": str(Path(sys.executable).resolve()),
            "onnxruntime": ort.__version__,
            "dependencies": package_versions(),
        },
        "profiles": reports,
    }
    payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    report_path = (
        Path(args.report_json)
        if args.report_json
        else Path(args.bundle_dir) / "bundle-validation-report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(payload)
    print(
        json.dumps(
            {
                "status": document["status"],
                "report": str(report_path),
                "report_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "release_ready": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
