#!/usr/bin/env python3
"""Build complete, explicitly unvalidated GLiNER2.5 ONNX bundles.

The seven existing graph exporters are run as isolated subprocesses so each
model is released before the next graph is loaded.  This command exports only;
it never promotes, validates, uploads, or marks a bundle release-ready.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import onnx

import bundle_profiles
from common import (
    GLINER2_COMMIT,
    assert_pinned_gliner2_installation,
    package_versions,
    sha256_file,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_SOURCE_ROOT = Path("/tmp/gliner25-work/GLiNER2")
DEFAULT_OUTPUT_ROOT = Path("/tmp/gliner25-work/m7-bundles")
MODEL_ORDER = ("small", "multi", "base")

GRAPH_EXPORTS = (
    ("encoder.onnx", "export_encoder.py"),
    ("classifier.onnx", "export_classifier.py"),
    ("boundary_marginals.onnx", "export_boundary_marginals.py"),
    ("boundary_scorer.onnx", "export_boundary_scorer.py"),
    ("boundary_explicit_scorer.onnx", "export_boundary_explicit_scorer.py"),
    ("boundary_records.onnx", "export_boundary_records.py"),
    ("boundary_relations.onnx", "export_boundary_relations.py"),
)
GRAPH_NAMES = tuple(item[0] for item in GRAPH_EXPORTS)
RUNTIME_METADATA = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "encoder_config/config.json",
)
BUNDLE_DOCUMENTS = ("SOURCE_MODEL_CARD.md", "LICENSE", "NOTICE")
REQUIRED_BUNDLE_FILES = (*RUNTIME_METADATA, *BUNDLE_DOCUMENTS, *GRAPH_NAMES)

EXPECTED_ENCODERS = {
    "small": ("microsoft/deberta-v3-xsmall", 384),
    "base": ("microsoft/deberta-v3-base", 768),
    "multi": ("microsoft/mdeberta-v3-base", 768),
}
EXPECTED_DEPENDENCIES = {
    "gliner2": "2.0.0",
    "torch": "2.8.0",
    "transformers": "4.57.6",
    "onnx": "1.17.0",
    "onnxruntime": "1.20.1",
    "numpy": "2.2.6",
    "peft": "0.17.1",
    "sentencepiece": "0.2.1",
}
FLOAT_DTYPES = {
    onnx.TensorProto.FLOAT16,
    onnx.TensorProto.FLOAT,
    onnx.TensorProto.DOUBLE,
    onnx.TensorProto.BFLOAT16,
}
DTYPE_NAMES = {
    onnx.TensorProto.FLOAT: "float32",
    onnx.TensorProto.FLOAT16: "float16",
    onnx.TensorProto.DOUBLE: "float64",
    onnx.TensorProto.BFLOAT16: "bfloat16",
    onnx.TensorProto.BOOL: "bool",
    onnx.TensorProto.INT8: "int8",
    onnx.TensorProto.INT16: "int16",
    onnx.TensorProto.INT32: "int32",
    onnx.TensorProto.INT64: "int64",
    onnx.TensorProto.UINT8: "uint8",
    onnx.TensorProto.UINT16: "uint16",
    onnx.TensorProto.UINT32: "uint32",
    onnx.TensorProto.UINT64: "uint64",
    onnx.TensorProto.STRING: "string",
}


def default_model_dir(profile_name: str) -> Path:
    profile = bundle_profiles.PROFILES[profile_name]
    cache_name = profile.model_id.replace("/", "--")
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{cache_name}"
        / "snapshots"
        / profile.revision
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON metadata {path} must contain an object")
    return value


def _require_apache_model_card(path: Path, expected_sha256: str | None) -> None:
    if expected_sha256 is None or not path.is_file():
        raise RuntimeError(
            f"pinned source model card is absent: {path}. A complete bundle requires "
            "the original README.md from the exact immutable HF revision. Retrieve it "
            "from the public model repository (without credentials), pin its SHA-256 in "
            "bundle_profiles.py, and retry; do not substitute another model's card or "
            "use an unpinned local README.md."
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"source model card {path} has sha256 {actual_sha256}, expected "
            f"{expected_sha256}"
        )
    text = path.read_text(encoding="utf-8")
    if re.search(r"(?mi)^license:\s*apache-2\.0\s*$", text) is None:
        raise RuntimeError(
            f"source model card {path} does not declare license: apache-2.0"
        )


def verify_upstream_source(source_dir: Path) -> Path:
    source = source_dir.expanduser().resolve()
    if not (source / ".git").exists():
        raise RuntimeError(f"GLiNER2 source is not a git checkout: {source}")
    try:
        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot verify GLiNER2 checkout {source}: {exc}") from exc
    if commit != GLINER2_COMMIT:
        raise RuntimeError(
            f"GLiNER2 checkout is {commit}, expected pinned commit {GLINER2_COMMIT}"
        )
    license_path = source / "LICENSE"
    if not license_path.is_file() or "Apache License" not in license_path.read_text(
        encoding="utf-8"
    ):
        raise RuntimeError(f"pinned GLiNER2 Apache-2.0 LICENSE is missing: {license_path}")
    return license_path


def verify_export_environment() -> dict[str, str]:
    assert_pinned_gliner2_installation()
    actual = package_versions()
    mismatches = [
        f"{name}={actual.get(name)!r}, expected {expected!r}"
        for name, expected in EXPECTED_DEPENDENCIES.items()
        if actual.get(name) != expected
    ]
    if mismatches:
        raise RuntimeError(
            "export must use scripts/export/env/.venv pinned dependencies: "
            + "; ".join(mismatches)
        )
    return actual


def preflight_source(
    profile_name: str, model_dir: Path, gliner2_source_dir: Path
) -> dict[str, Any]:
    profile = bundle_profiles.PROFILES[profile_name]
    source_hashes = bundle_profiles.verify_source(profile_name, model_dir)
    config = _load_json(model_dir / "config.json")
    encoder_config = _load_json(model_dir / "encoder_config/config.json")
    expected_encoder, expected_hidden = EXPECTED_ENCODERS[profile_name]
    errors: list[str] = []
    if config.get("architecture") != "boundary":
        errors.append(f"architecture={config.get('architecture')!r}, expected 'boundary'")
    if config.get("architecture_version") != 1:
        errors.append(
            f"architecture_version={config.get('architecture_version')!r}, expected 1"
        )
    if config.get("model_name") != expected_encoder:
        errors.append(
            f"model_name={config.get('model_name')!r}, expected {expected_encoder!r}"
        )
    if encoder_config.get("hidden_size") != expected_hidden:
        errors.append(
            f"hidden_size={encoder_config.get('hidden_size')!r}, expected {expected_hidden}"
        )
    if errors:
        raise RuntimeError(
            f"unsupported metadata for {profile.model_id}@{profile.revision}: "
            + "; ".join(errors)
        )
    _require_apache_model_card(
        model_dir / "README.md", profile.source_sha256.get("README.md")
    )
    license_path = verify_upstream_source(gliner2_source_dir)
    return {
        "model": profile_name,
        "model_dir": str(model_dir.resolve()),
        "hf_model": profile.model_id,
        "hf_revision": profile.revision,
        "bundle_name": profile.bundle_name,
        "encoder": expected_encoder,
        "hidden_size": expected_hidden,
        "source_file_sha256": source_hashes,
        "license": str(license_path),
    }


def export_commands(model_dir: Path, bundle_dir: Path) -> list[list[str]]:
    return [
        [
            sys.executable,
            str(SCRIPT_DIR / script_name),
            "--model-dir",
            str(model_dir),
            "--out-dir",
            str(bundle_dir),
            "--opset",
            "17",
        ]
        for _, script_name in GRAPH_EXPORTS
    ]


def _shape(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
    dimensions: list[int | str | None] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param") and dimension.dim_param:
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(None)
    return dimensions


def _signature(value_info: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    return {
        "dtype": onnx.TensorProto.DataType.Name(tensor_type.elem_type),
        "shape": _shape(value_info),
    }


def _walk_graphs(graph: onnx.GraphProto) -> Iterable[onnx.GraphProto]:
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _walk_graphs(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for nested in attribute.graphs:
                    yield from _walk_graphs(nested)


def inspect_graph(path: Path) -> dict[str, Any]:
    """Return the actual ONNX ABI while rejecting non-fp32 floating weights."""

    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model)
    default_opsets = [item.version for item in model.opset_import if item.domain == ""]
    if default_opsets != [17]:
        raise RuntimeError(f"{path.name} has default ONNX opsets {default_opsets}, expected [17]")

    for graph in _walk_graphs(model.graph):
        for value in (*graph.input, *graph.output, *graph.value_info):
            element_type = value.type.tensor_type.elem_type
            if element_type in FLOAT_DTYPES and element_type != onnx.TensorProto.FLOAT:
                raise RuntimeError(
                    f"{path.name} tensor {value.name!r} is "
                    f"{DTYPE_NAMES.get(element_type, element_type)}, expected float32"
                )
        for initializer in graph.initializer:
            if (
                initializer.data_type in FLOAT_DTYPES
                and initializer.data_type != onnx.TensorProto.FLOAT
            ):
                raise RuntimeError(
                    f"{path.name} initializer {initializer.name!r} is not fp32"
                )
        for node in graph.node:
            for attribute in node.attribute:
                tensors = []
                if attribute.type == onnx.AttributeProto.TENSOR:
                    tensors = [attribute.t]
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    tensors = list(attribute.tensors)
                for tensor in tensors:
                    if tensor.data_type in FLOAT_DTYPES and tensor.data_type != onnx.TensorProto.FLOAT:
                        raise RuntimeError(
                            f"{path.name} constant in {node.name or node.op_type!r} is not fp32"
                        )

    return {
        "inputs": {value.name: _signature(value) for value in model.graph.input},
        "outputs": {value.name: _signature(value) for value in model.graph.output},
    }


def _notice(profile_name: str) -> str:
    profile = bundle_profiles.PROFILES[profile_name]
    return f"""GLiNER2.5 Rust conversion notice

Source software: https://github.com/fastino-ai/GLiNER2
Source software commit: {GLINER2_COMMIT}
Source model: https://huggingface.co/{profile.model_id}
Source model revision: {profile.revision}
Source software and source model license: Apache License 2.0

The seven ONNX graphs in this bundle are fp32, opset-17 conversions produced
from the source model above for gliner2-rs. They are not the publisher's original
safetensors files. Export success is not validation: this bundle is explicitly
exported-unvalidated and is not release-ready until independent per-checkpoint
source/ONNX/native validation is reviewed and the manifest is promoted.
"""


def _copy_metadata(
    profile_name: str, model_dir: Path, bundle_dir: Path, license_path: Path
) -> None:
    for relative_name in RUNTIME_METADATA:
        source = model_dir / relative_name
        if not source.is_file():
            raise RuntimeError(f"required runtime metadata disappeared: {source}")
        destination = bundle_dir / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination, follow_symlinks=True)
    model_card = model_dir / "README.md"
    profile = bundle_profiles.PROFILES[profile_name]
    _require_apache_model_card(model_card, profile.source_sha256.get("README.md"))
    shutil.copy2(model_card, bundle_dir / "SOURCE_MODEL_CARD.md", follow_symlinks=True)
    shutil.copy2(license_path, bundle_dir / "LICENSE", follow_symlinks=True)
    (bundle_dir / "NOTICE").write_text(_notice(profile_name), encoding="utf-8")


def build_manifest(
    profile_name: str,
    bundle_dir: Path,
    source_hashes: dict[str, str],
    dependencies: dict[str, str],
) -> dict[str, Any]:
    profile = bundle_profiles.PROFILES[profile_name]
    graphs = {name: inspect_graph(bundle_dir / name) for name in GRAPH_NAMES}
    files: dict[str, dict[str, Any]] = {}
    for relative_name in REQUIRED_BUNDLE_FILES:
        path = bundle_dir / relative_name
        if not path.is_file():
            raise RuntimeError(f"complete bundle is missing {relative_name}")
        size = path.stat().st_size
        if size <= 0:
            raise RuntimeError(f"complete bundle file is empty: {relative_name}")
        files[relative_name] = {"bytes": size, "sha256": sha256_file(path)}

    return {
        "manifest_version": 1,
        "architecture": "boundary",
        "architecture_version": 1,
        "status": "exported-unvalidated",
        "release_ready": False,
        "hf_model": profile.model_id,
        "hf_revision": profile.revision,
        "gliner2_commit": GLINER2_COMMIT,
        "opset": 17,
        "precision": "fp32",
        "ort_crate_version": "2.0.0-rc.13",
        "native_onnx_runtime": "1.28.0",
        "validation_onnxruntime": "1.20.1",
        "dependencies": dependencies,
        "source_file_sha256": source_hashes,
        "files": files,
        "graphs": graphs,
        "validation": None,
    }


def finalize_bundle(
    profile_name: str,
    model_dir: Path,
    bundle_dir: Path,
    gliner2_source_dir: Path,
    source_hashes_before: dict[str, str],
    dependencies: dict[str, str],
) -> dict[str, Any]:
    source_hashes_after = bundle_profiles.verify_source(profile_name, model_dir)
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("verified checkpoint source changed during export")
    license_path = verify_upstream_source(gliner2_source_dir)
    _copy_metadata(profile_name, model_dir, bundle_dir, license_path)

    # encoder/marginals deliberately write the historical M1 partial manifest.
    # It is deleted rather than merged so rc.9/partial state cannot leak.
    partial_manifest = bundle_dir / "export_manifest.json"
    partial_manifest.unlink(missing_ok=True)

    actual_files = {
        path.relative_to(bundle_dir).as_posix()
        for path in bundle_dir.rglob("*")
        if path.is_file()
    }
    expected_files = set(REQUIRED_BUNDLE_FILES)
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise RuntimeError(f"bundle file set is not exact; missing={missing}, extra={extra}")

    manifest = build_manifest(
        profile_name, bundle_dir, source_hashes_after, dependencies
    )
    partial_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def export_one(
    profile_name: str,
    model_dir: Path,
    output_root: Path,
    gliner2_source_dir: Path,
    dependencies: dict[str, str],
) -> Path:
    profile = bundle_profiles.PROFILES[profile_name]
    source_hashes = bundle_profiles.verify_source(profile_name, model_dir)
    destination = output_root / profile.bundle_name
    if destination.exists():
        raise RuntimeError(
            f"refusing to overwrite existing bundle {destination}; use a fresh staging root"
        )
    work = output_root / f".{profile.bundle_name}.exporting-{os.getpid()}"
    if work.exists():
        raise RuntimeError(f"temporary export directory already exists: {work}")
    output_root.mkdir(parents=True, exist_ok=True)
    work.mkdir()
    try:
        for command in export_commands(model_dir, work):
            print(f"+ {shlex.join(command)}", flush=True)
            subprocess.run(command, cwd=SCRIPT_DIR, check=True)
        finalize_bundle(
            profile_name,
            model_dir,
            work,
            gliner2_source_dir,
            source_hashes,
            dependencies,
        )
        work.rename(destination)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    print(f"wrote unvalidated bundle {destination}")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=(*MODEL_ORDER, "all"))
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="source checkpoint directory (only valid for one selected model)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="fresh staging root; each selected bundle is created below it",
    )
    parser.add_argument(
        "--gliner2-source-dir", type=Path, default=DEFAULT_SOURCE_ROOT
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preflight", action="store_true", help="verify inputs/environment only"
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="run preflight and print all exporter commands without writing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = MODEL_ORDER if args.model == "all" else (args.model,)
    if args.model_dir is not None and len(selected) != 1:
        raise ValueError("--model-dir requires one of --model small|base|multi")

    dependencies = verify_export_environment()
    plans: list[tuple[str, Path, dict[str, Any]]] = []
    for profile_name in selected:
        model_dir = (
            args.model_dir.expanduser().resolve()
            if args.model_dir is not None
            else default_model_dir(profile_name)
        )
        report = preflight_source(profile_name, model_dir, args.gliner2_source_dir)
        plans.append((profile_name, model_dir, report))
        print(json.dumps({"preflight": report}, sort_keys=True))

    if args.preflight:
        return
    if args.dry_run:
        for profile_name, model_dir, _ in plans:
            destination = args.out_dir / bundle_profiles.PROFILES[profile_name].bundle_name
            for command in export_commands(model_dir, destination):
                print(shlex.join(command))
        return

    for profile_name, model_dir, _ in plans:
        export_one(
            profile_name,
            model_dir,
            args.out_dir.expanduser().resolve(),
            args.gliner2_source_dir,
            dependencies,
        )


if __name__ == "__main__":
    main()
