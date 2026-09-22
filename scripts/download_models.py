#!/usr/bin/env python3
"""Explicitly download verified GLiNER2 ONNX bundles from Hugging Face.

Examples:
    python scripts/download_models.py --model base
    python scripts/download_models.py --model 2.5-base --revision <immutable-HF-SHA>
    python scripts/download_models.py --model all --dest ./onnx --revision <immutable-HF-SHA>

``all`` downloads both legacy v2 bundles and all three (substantially larger)
GLiNER2.5 bundles. Downloads are staged, SHA-256 checked, and installed only
when the selected bundle is complete. Nothing invokes this script at build or
inference time.

Requires: ``pip install huggingface_hub``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ID = "codesoda/gliner2-onnx"
ROOT = Path(__file__).resolve().parent.parent
V2_PINS_PATH = ROOT / "docs" / "checkpoints" / "v2-metadata-pins.json"
GLINER2_COMMIT = "d7c727458bf6929bc9ef5ee04e13c3f717a7c455"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_PINS_BYTES = 1024 * 1024
EXPECTED_EXPORT_DEPENDENCIES = {
    "gliner2": "2.0.0", "torch": "2.8.0", "transformers": "4.57.6",
    "onnx": "1.17.0", "onnxruntime": "1.20.1", "numpy": "2.2.6",
    "peft": "0.17.1", "sentencepiece": "0.2.1",
}
ONNX_TENSOR_DTYPES = frozenset(
    {
        "FLOAT",
        "UINT8",
        "INT8",
        "UINT16",
        "INT16",
        "INT32",
        "INT64",
        "STRING",
        "BOOL",
        "FLOAT16",
        "DOUBLE",
        "UINT32",
        "UINT64",
        "COMPLEX64",
        "COMPLEX128",
        "BFLOAT16",
        "FLOAT8E4M3FN",
        "FLOAT8E4M3FNUZ",
        "FLOAT8E5M2",
        "FLOAT8E5M2FNUZ",
        "UINT4",
        "INT4",
        "FLOAT4E2M1",
    }
)

BOUNDARY_MODELS = {
    "2.5-small": {
        "bundle": "gliner2.5-small-v1",
        "encoder_model": "microsoft/deberta-v3-xsmall",
        "hf_model": "fastino/gliner2.5-small-v1",
        "hf_revision": "f1e4d8fdd6fe328f45dee6aca3e6a07c9db4296e",
    },
    "2.5-base": {
        "bundle": "gliner2.5-base-v1",
        "encoder_model": "microsoft/deberta-v3-base",
        "hf_model": "fastino/gliner2.5-base-v1",
        "hf_revision": "78cea040597df251eedefa9d7ee2a756af39fe64",
    },
    "2.5-multi": {
        "bundle": "gliner2.5-multi-v1",
        "encoder_model": "microsoft/mdeberta-v3-base",
        "hf_model": "fastino/gliner2.5-multi-v1",
        "hf_revision": "235cf92d6d4318da9bfca0d08975c8fa7250d13b",
    },
}
SELECTOR_ALIASES = {
    "gliner2-base-v1": "base",
    "gliner2-large-v1": "large",
    **{profile["bundle"]: selector for selector, profile in BOUNDARY_MODELS.items()},
}
SELECTORS = ("base", "large", *BOUNDARY_MODELS)

BOUNDARY_REQUIRED_FILES = frozenset(
    {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "encoder_config/config.json",
        "SOURCE_MODEL_CARD.md",
        "LICENSE",
        "NOTICE",
        "encoder.onnx",
        "classifier.onnx",
        "boundary_marginals.onnx",
        "boundary_scorer.onnx",
        "boundary_explicit_scorer.onnx",
        "boundary_records.onnx",
        "boundary_relations.onnx",
    }
)
BOUNDARY_GRAPHS = frozenset(
    {
        "encoder.onnx",
        "classifier.onnx",
        "boundary_marginals.onnx",
        "boundary_scorer.onnx",
        "boundary_explicit_scorer.onnx",
        "boundary_records.onnx",
        "boundary_relations.onnx",
    }
)

Downloader = Callable[[str, str, str], Path]


class BundleError(ValueError):
    """A remote bundle or local pin document violates the download contract."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_abi_name(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value.strip() == value
        and not any(ord(character) <= 0x1F or ord(character) == 0x7F for character in value)
    )


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _read_json_document(path: Path, limit: int, context: str) -> tuple[object, bytes]:
    try:
        with path.open("rb") as handle:
            content = handle.read(limit + 1)
    except OSError as error:
        raise BundleError(f"cannot read {context} {path}: {error}") from error
    if not content:
        raise BundleError(f"{context} is empty: {path}")
    if len(content) > limit:
        raise BundleError(f"{context} exceeds the {limit} byte limit: {path}")
    try:
        value = json.loads(content, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError(f"invalid JSON in {context} {path}: {error}") from error
    return value, content


def safe_relative_path(value: object) -> str:
    """Return a normalized bundle-relative POSIX path or reject it."""
    if not isinstance(value, str) or not value:
        raise BundleError("bundle path must be a nonempty string")
    if "\\" in value or ":" in value:
        raise BundleError(f"unsafe bundle path (Windows separator/prefix): {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise BundleError(f"unsafe bundle path: {value!r}")
    normalized = path.as_posix()
    if normalized != value:
        raise BundleError(f"non-normalized bundle path: {value!r}")
    return normalized


def _file_spec(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise BundleError(f"{context} must be an object")
    size = value.get("bytes")
    digest = value.get("sha256")
    if not _is_int(size) or size <= 0:
        raise BundleError(f"{context}.bytes must be a positive integer")
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        raise BundleError(f"{context}.sha256 must be 64 lowercase hex characters")
    return {"bytes": size, "sha256": digest}


def _file_table(value: object, context: str) -> dict[str, dict[str, object]]:
    if not isinstance(value, Mapping):
        raise BundleError(f"{context} must be an object")
    result: dict[str, dict[str, object]] = {}
    for raw_path, raw_spec in value.items():
        path = safe_relative_path(raw_path)
        result[path] = _file_spec(raw_spec, f"{context}[{path!r}]")
    return result


def _require(manifest: Mapping[str, Any], key: str, expected: object) -> None:
    actual = manifest.get(key)
    if type(actual) is not type(expected) or actual != expected:
        raise BundleError(
            f"manifest {key!r} is {manifest.get(key)!r}; expected {expected!r}"
        )


def validate_boundary_manifest(
    value: object, bundle_name: str
) -> dict[str, dict[str, object]]:
    """Validate a release boundary manifest and return its checked file table."""
    if not isinstance(value, Mapping):
        raise BundleError("manifest must be a JSON object")
    manifest = value
    profile = next(
        (item for item in BOUNDARY_MODELS.values() if item["bundle"] == bundle_name),
        None,
    )
    if profile is None:
        raise BundleError(f"unsupported boundary bundle: {bundle_name}")

    _require(manifest, "manifest_version", 1)
    _require(manifest, "architecture", "boundary")
    _require(manifest, "architecture_version", 1)
    _require(manifest, "status", "validated")
    _require(manifest, "release_ready", True)
    _require(manifest, "hf_model", profile["hf_model"])
    _require(manifest, "hf_revision", profile["hf_revision"])
    _require(manifest, "gliner2_commit", GLINER2_COMMIT)
    _require(manifest, "opset", 17)
    _require(manifest, "precision", "fp32")
    _require(manifest, "ort_crate_version", "2.0.0-rc.13")
    _require(manifest, "native_onnx_runtime", "1.28.0")
    _require(manifest, "validation_onnxruntime", "1.20.1")

    dependencies = manifest.get("dependencies")
    if dependencies != EXPECTED_EXPORT_DEPENDENCIES:
        raise BundleError("manifest export dependency pins do not match the supported environment")
    source_hashes = manifest.get("source_file_sha256")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise BundleError("manifest source_file_sha256 must be a nonempty object")
    for source_path, digest in source_hashes.items():
        safe_relative_path(source_path)
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise BundleError(f"invalid source hash for {source_path!r}")

    files = _file_table(manifest.get("files"), "manifest files")
    missing = sorted(BOUNDARY_REQUIRED_FILES - files.keys())
    if missing:
        raise BundleError(f"manifest is partial; missing required files: {missing}")
    if "export_manifest.json" in files:
        raise BundleError("manifest must not list itself in files")

    graphs = manifest.get("graphs")
    if not isinstance(graphs, Mapping) or set(graphs) != BOUNDARY_GRAPHS:
        raise BundleError(
            "manifest graphs must contain exactly the seven required graph files"
        )
    for name, signature in graphs.items():
        if not isinstance(signature, Mapping):
            raise BundleError(f"graph signature {name!r} must be an object")
        for axis in ("inputs", "outputs"):
            tensors = signature.get(axis)
            if not isinstance(tensors, Mapping) or not tensors:
                raise BundleError(
                    f"graph signature {name!r} needs a nonempty {axis} object"
                )
            for tensor_name, tensor in tensors.items():
                if not _valid_abi_name(tensor_name):
                    raise BundleError(
                        f"graph signature {name!r} has an invalid {axis} tensor name"
                    )
                if not isinstance(tensor, Mapping):
                    raise BundleError(
                        f"graph signature {name!r} tensor {tensor_name!r} must be an object"
                    )
                dtype = tensor.get("dtype")
                if not isinstance(dtype, str) or dtype not in ONNX_TENSOR_DTYPES:
                    raise BundleError(
                        f"graph signature {name!r} tensor {tensor_name!r} "
                        f"has invalid ONNX dtype {dtype!r}"
                    )
                shape = tensor.get("shape")
                if not isinstance(shape, list):
                    raise BundleError(
                        f"graph signature {name!r} tensor {tensor_name!r} shape must be a list"
                    )
                if any(
                    dimension is not None
                    and not (
                        (_is_int(dimension) and dimension >= 0)
                        or _valid_abi_name(dimension)
                    )
                    for dimension in shape
                ):
                    raise BundleError(
                        f"graph signature {name!r} tensor {tensor_name!r} "
                        "has an invalid shape dimension"
                    )

    validation = manifest.get("validation")
    if not isinstance(validation, Mapping) or not validation:
        raise BundleError("validated manifest must contain validation evidence")
    return files


def load_v2_metadata_pins(path: Path = V2_PINS_PATH) -> dict[str, Any]:
    """Load and fail-closed validate the committed legacy metadata pins."""
    value, _ = _read_json_document(path, MAX_PINS_BYTES, "v2 metadata pins")
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise BundleError("unsupported v2 metadata pin schema")
    if value.get("hash_algorithm") != "sha256":
        raise BundleError("v2 metadata pins must use sha256")
    hosted = value.get("hosted_onnx")
    if not isinstance(hosted, dict) or hosted.get("repository") != REPO_ID:
        raise BundleError("v2 pins name an unexpected hosted ONNX repository")
    revision = hosted.get("verified_revision")
    if not isinstance(revision, str) or REVISION_RE.fullmatch(revision) is None:
        raise BundleError("v2 hosted ONNX revision must be an immutable commit SHA")

    models = value.get("models")
    if not isinstance(models, dict) or set(models) != {
        "gliner2-base-v1",
        "gliner2-large-v1",
    }:
        raise BundleError("v2 pins must define exactly the base and large bundles")
    expected_selectors = {"gliner2-base-v1": "base", "gliner2-large-v1": "large"}
    for bundle_name, model in models.items():
        if not isinstance(model, dict):
            raise BundleError(f"v2 pin {bundle_name!r} must be an object")
        if model.get("selector") != expected_selectors[bundle_name]:
            raise BundleError(f"v2 pin {bundle_name!r} has an invalid selector")
        source_repo = model.get("source_repository")
        source_revision = model.get("source_revision")
        if source_repo != f"fastino/{bundle_name}":
            raise BundleError(f"v2 pin {bundle_name!r} has an invalid source repository")
        if not isinstance(source_revision, str) or REVISION_RE.fullmatch(source_revision) is None:
            raise BundleError(f"v2 pin {bundle_name!r} source revision is not immutable")
        metadata = _file_table(model.get("source_metadata"), "source_metadata")
        if not {"config.json", "tokenizer.json"}.issubset(metadata):
            raise BundleError(f"v2 pin {bundle_name!r} lacks runtime config/tokenizer")
        onnx_files = _file_table(model.get("hosted_onnx_files"), "hosted_onnx_files")
        if not {"encoder.onnx", "extractor_padded.onnx", "classifier.onnx"}.issubset(
            onnx_files
        ):
            raise BundleError(f"v2 pin {bundle_name!r} lacks required ONNX graphs")
    return value


def resolve_selection(selector: str) -> list[str]:
    """Resolve one CLI selector to canonical selectors in download order."""
    canonical = SELECTOR_ALIASES.get(selector, selector)
    if canonical == "all":
        return list(SELECTORS)
    if canonical not in SELECTORS:
        choices = "|".join((*SELECTORS, "all"))
        raise BundleError(f"unknown model {selector!r} (expected {choices})")
    return [canonical]


def stream_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def verify_file(path: Path, spec: Mapping[str, object]) -> None:
    if not path.is_file():
        raise BundleError(f"downloaded file is missing: {path}")
    size, digest = stream_sha256(path)
    if size != spec["bytes"]:
        raise BundleError(
            f"size mismatch for {path.name}: got {size}, expected {spec['bytes']}"
        )
    if digest != spec["sha256"]:
        raise BundleError(
            f"SHA-256 mismatch for {path.name}: got {digest}, expected {spec['sha256']}"
        )


def _target(root: Path, relative: str) -> Path:
    relative = safe_relative_path(relative)
    target = root.joinpath(*PurePosixPath(relative).parts)
    resolved_root = root.resolve()
    resolved_target = target.resolve(strict=False)
    try:
        common = Path(os.path.commonpath((resolved_root, resolved_target)))
    except ValueError as error:
        raise BundleError(f"bundle path escapes destination: {relative!r}") from error
    if common != resolved_root:
        raise BundleError(f"bundle path escapes destination: {relative!r}")
    return target


def _copy_verified(source: Path, target: Path, spec: Mapping[str, object]) -> None:
    verify_file(source, spec)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
    verify_file(target, spec)


def _default_downloader(repo_id: str, revision: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(repo_id=repo_id, revision=revision, filename=filename)
    )


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _replace_bundle(stage: Path, destination: Path) -> None:
    if destination.is_symlink():
        raise BundleError(f"refusing to replace symlink destination: {destination}")
    if destination.exists() and not destination.is_dir():
        raise BundleError(f"bundle destination is not a directory: {destination}")
    if not stage.is_dir() or stage.is_symlink():
        raise BundleError(f"staged bundle is not a directory: {stage}")

    backup_root: Path | None = None
    backup: Path | None = None
    if destination.exists():
        backup_root = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.previous.", dir=destination.parent
            )
        )
        backup = backup_root / destination.name
        try:
            destination.rename(backup)
        except BaseException:
            shutil.rmtree(backup_root, ignore_errors=True)
            raise
    try:
        stage.rename(destination)
    except BaseException as install_error:
        if backup is not None and backup.exists() and not destination.exists():
            try:
                backup.rename(destination)
            except BaseException as restore_error:
                raise BundleError(
                    f"failed to install {destination} and restore its previous bundle; "
                    f"backup remains at {backup}: {restore_error}"
                ) from install_error
        if backup_root is not None:
            shutil.rmtree(backup_root, ignore_errors=True)
        raise
    if backup_root is not None:
        shutil.rmtree(backup_root)


def _download_boundary(
    profile: Mapping[str, str],
    dest: Path,
    revision: str,
    downloader: Downloader,
) -> None:
    bundle = profile["bundle"]
    manifest_name = f"{bundle}/export_manifest.json"
    manifest_source = downloader(REPO_ID, revision, manifest_name)
    manifest, manifest_bytes = _read_json_document(
        manifest_source, MAX_MANIFEST_BYTES, f"manifest for {bundle}"
    )
    files = validate_boundary_manifest(manifest, bundle)

    temporary = Path(tempfile.mkdtemp(prefix=f".{bundle}.", dir=dest))
    stage = temporary / bundle
    stage.mkdir()
    try:
        (stage / "export_manifest.json").write_bytes(manifest_bytes)
        for relative, spec in files.items():
            source = downloader(REPO_ID, revision, f"{bundle}/{relative}")
            _copy_verified(source, _target(stage, relative), spec)
        config, _ = _read_json_document(stage / "config.json", MAX_MANIFEST_BYTES, "authenticated config")
        if not isinstance(config, Mapping):
            raise BundleError("authenticated config must be an object")
        _require(config, "architecture", "boundary")
        _require(config, "architecture_version", 1)
        _require(config, "model_name", profile["encoder_model"])
        _replace_bundle(stage, dest / bundle)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _download_v2(
    selector: str,
    pins: Mapping[str, Any],
    dest: Path,
    revision: str,
    downloader: Downloader,
) -> None:
    bundle = f"gliner2-{selector}-v1"
    model = pins["models"][bundle]
    temporary = Path(tempfile.mkdtemp(prefix=f".{bundle}.", dir=dest))
    stage = temporary / bundle
    stage.mkdir()
    try:
        onnx_files = _file_table(model["hosted_onnx_files"], "hosted_onnx_files")
        for relative, spec in onnx_files.items():
            source = downloader(REPO_ID, revision, f"{bundle}/{relative}")
            _copy_verified(source, _target(stage, relative), spec)

        metadata = _file_table(model["source_metadata"], "source_metadata")
        for relative, spec in metadata.items():
            source = downloader(
                model["source_repository"], model["source_revision"], relative
            )
            _copy_verified(source, _target(stage, relative), spec)
        _replace_bundle(stage, dest / bundle)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def download_selected(
    selector: str,
    dest: Path,
    revision: str | None = None,
    *,
    pins_path: Path = V2_PINS_PATH,
    downloader: Downloader = _default_downloader,
) -> list[Path]:
    """Download selected complete bundles and return their installed paths."""
    pins = load_v2_metadata_pins(pins_path)
    selected = resolve_selection(selector)
    if revision is None:
        if any(item not in ("base", "large") for item in selected):
            raise BundleError(
                "boundary downloads require --revision with the immutable published "
                "commit containing validated bundles"
            )
        revision = pins["hosted_onnx"]["verified_revision"]
    if not isinstance(revision, str) or REVISION_RE.fullmatch(revision) is None:
        raise BundleError("hosted ONNX revision must be an immutable commit SHA")

    dest = Path(dest)
    if dest.is_symlink():
        raise BundleError(f"destination must not be a symlink: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    installed: list[Path] = []
    for item in selected:
        if item in ("base", "large"):
            print(f"Downloading gliner2-{item}-v1 from {REPO_ID}@{revision} ...")
            _download_v2(item, pins, dest, revision, downloader)
            bundle = f"gliner2-{item}-v1"
        else:
            profile = BOUNDARY_MODELS[item]
            print(f"Downloading {profile['bundle']} from {REPO_ID}@{revision} ...")
            _download_boundary(profile, dest, revision, downloader)
            bundle = profile["bundle"]
        installed.append(dest / bundle)
    return installed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="all",
        help=(
            "base, large, 2.5-small, 2.5-base, 2.5-multi, a full bundle name, "
            "or all (all five; default: all)"
        ),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=ROOT / "onnx",
        help="destination directory (default: ./onnx)",
    )
    parser.add_argument(
        "--revision",
        help=(
            "immutable codesoda/gliner2-onnx revision; defaults to the verified "
            "v2-only revision for base/large, and is required for boundary/all"
        ),
    )
    args = parser.parse_args()
    paths = download_selected(args.model, args.dest, args.revision)
    print("Done. Verified bundles:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
