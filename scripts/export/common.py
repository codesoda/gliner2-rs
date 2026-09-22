"""Shared deterministic helpers for GLiNER2 exports and validation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

GLINER2_COMMIT = "d7c727458bf6929bc9ef5ee04e13c3f717a7c455"
BASE_HF_REVISION = "78cea040597df251eedefa9d7ee2a756af39fe64"
BASE_MODEL_ID = "fastino/gliner2.5-base-v1"
OFFICIAL_BASE_SOURCE_SHA256 = {
    "model.safetensors": "7274094de2e0c2a37a386f55fc4e23061a954da5bd7a335e7dfe56f2743c277a",
    "config.json": "0eb92d00584d613aab32b2178f84a85176b62c87ae3689ce9084e83f6eba64d1",
    "tokenizer.json": "cbc8ae6037812709c9c26f2a160f8dc48b0440bcb79c8141804259ae2d6adac3",
    "encoder_config/config.json": (
        "d36a845b9f25dcaf1ec45a1c4bdf65ea4ac20596537e14530ec9f660a63aeca4"
    ),
}
BOUNDARY_REVISIONS = {
    "f1e4d8fdd6fe328f45dee6aca3e6a07c9db4296e": "fastino/gliner2.5-small-v1",
    BASE_HF_REVISION: "fastino/gliner2.5-base-v1",
    "235cf92d6d4318da9bfca0d08975c8fa7250d13b": "fastino/gliner2.5-multi-v1",
}
MASK_LOGIT = -1.0e4
DEPENDENCIES = (
    "gliner2",
    "torch",
    "transformers",
    "onnx",
    "onnxruntime",
    "numpy",
    "peft",
    "sentencepiece",
)


def configure_determinism(seed: int = 1729, threads: int = 1) -> None:
    """Configure the CPU reference path; autocast is deliberately not used."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only permits setting this before inter-op work starts.
        pass
    torch.use_deterministic_algorithms(True, warn_only=False)


def load_reference_model(model_dir: str | Path, *, device: str = "cpu"):
    """Load span checkpoints compatibly and boundary checkpoints via AutoExtractor."""
    assert_pinned_gliner2_installation()
    from gliner2 import AutoExtractor, ExtractorConfig, GLiNER2

    path = Path(model_dir).expanduser().resolve()
    config_data = json.loads((path / "config.json").read_text())
    architecture = config_data.get("architecture", "span")
    if architecture == "boundary":
        config = ExtractorConfig.from_pretrained(str(path / "config.json"))
        # transformers 4.57.6 cannot use DeBERTa-v2 SDPA here. Make the measured
        # eager fallback explicit and reproducible before the encoder is built.
        config.attn_implementation = "eager"
        model = AutoExtractor.from_pretrained(str(path), config=config)
    elif architecture == "span":
        model = GLiNER2.from_pretrained(str(path))
    else:
        raise ValueError(f"unsupported architecture {architecture!r}")
    model = model.to(device=device, dtype=torch.float32).eval()
    for name, parameter in model.named_parameters():
        if parameter.dtype != torch.float32:
            raise TypeError(f"parameter {name} is {parameter.dtype}, expected fp32")
        if not torch.isfinite(parameter).all():
            raise ValueError(f"parameter {name} contains non-finite values")
    return model, config_data


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in DEPENDENCIES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_file_hashes(model_dir: str | Path) -> dict[str, str]:
    """Hash the source files that identify a checkpoint without trusting its path."""
    source = Path(model_dir).expanduser().resolve()
    hashes: dict[str, str] = {}
    for relative_name in OFFICIAL_BASE_SOURCE_SHA256:
        path = source / relative_name
        if path.is_file():
            hashes[relative_name] = sha256_file(path)
    return hashes


def assert_official_base_source(model_dir: str | Path) -> dict[str, str]:
    """Require the exact pinned official base checkpoint, wherever it is located."""
    actual = source_file_hashes(model_dir)
    if actual != OFFICIAL_BASE_SOURCE_SHA256:
        details = [
            f"{name}: expected {expected}, got {actual.get(name, 'missing')}"
            for name, expected in OFFICIAL_BASE_SOURCE_SHA256.items()
            if actual.get(name) != expected
        ]
        raise RuntimeError(
            "goldens require the exact pinned official base checkpoint source files; "
            + "; ".join(details)
        )
    return actual


def assert_pinned_gliner2_installation() -> None:
    """Reject an sdist/wheel or checkout that cannot prove the pinned VCS commit."""
    try:
        distribution = importlib.metadata.distribution("gliner2")
        direct_url_text = distribution.read_text("direct_url.json")
        if direct_url_text is None:
            raise OSError("distribution has no direct_url.json")
        direct_url = json.loads(direct_url_text)
        vcs_info = direct_url.get("vcs_info", {})
        commit = vcs_info.get("commit_id")
        vcs = vcs_info.get("vcs")
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError, TypeError) as exc:
        raise RuntimeError(
            "cannot verify the installed GLiNER2 VCS commit; use the locked "
            "scripts/export/env environment"
        ) from exc
    if vcs != "git" or commit != GLINER2_COMMIT:
        raise RuntimeError(
            f"installed GLiNER2 VCS/commit is {vcs!r}/{commit!r}, expected "
            f"'git'/{GLINER2_COMMIT}; "
            "use the locked scripts/export/env environment"
        )


def copy_runtime_metadata(model_dir: str | Path, out_dir: str | Path) -> None:
    source = Path(model_dir).expanduser().resolve()
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        candidate = source / name
        if candidate.exists():
            shutil.copy2(candidate, destination / name, follow_symlinks=True)
    encoder_config = source / "encoder_config"
    if encoder_config.is_dir():
        shutil.copytree(
            encoder_config,
            destination / "encoder_config",
            dirs_exist_ok=True,
            symlinks=False,
        )


def normalize_onnx_mask_constants(
    path: str | Path, *, mask_logit: float = MASK_LOGIT
) -> dict[str, Any]:
    """Replace exporter-emitted fp32-min attention masks and reject infinities.

    transformers' eager DeBERTa attention traces ``torch.finfo(float32).min``.
    It is finite, but bundles standardize mask sentinels to -1e4. Replacing only
    scalar constants below -1e30 preserves softmax masking while avoiding
    backend-specific extreme-value behavior; learned initializers are untouched.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    replaced: list[str] = []
    checked = 0
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            if attribute.type != onnx.AttributeProto.TENSOR:
                continue
            value = numpy_helper.to_array(attribute.t)
            if value.dtype.kind not in "fc":
                continue
            checked += 1
            if not np.isfinite(value).all():
                raise ValueError(f"ONNX constant {node.name!r} is non-finite")
            if value.ndim == 0 and float(value) < -1e30:
                replacement = np.asarray(mask_logit, dtype=value.dtype)
                attribute.t.CopyFrom(numpy_helper.from_array(replacement))
                replaced.append(node.name)
    for initializer in model.graph.initializer:
        value = numpy_helper.to_array(initializer)
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"ONNX initializer {initializer.name!r} is non-finite")
    onnx.save(model, str(path))
    return {
        "finite_constants_checked": checked,
        "mask_constants_rewritten": len(replaced),
        "mask_logit": mask_logit,
        "rewritten_nodes": replaced,
    }


def update_partial_manifest(
    out_dir: str | Path,
    model_dir: str | Path,
    *,
    opset: int,
    graph_name: str,
    graph_signature: dict[str, Any],
) -> Path:
    """Update an explicitly incomplete M1 manifest; this is not release-ready."""
    out = Path(out_dir)
    manifest_path = out / "export_manifest.json"
    model_config = json.loads((Path(model_dir).expanduser() / "config.json").read_text())
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    source_hashes = source_file_hashes(model_dir)
    is_official_base = source_hashes == OFFICIAL_BASE_SOURCE_SHA256
    manifest.update(
        {
            "status": "m1-partial-incomplete",
            "release_ready": False,
            "architecture": model_config.get("architecture", "span"),
            "architecture_version": model_config.get("architecture_version"),
            "gliner2_repository": "https://github.com/fastino-ai/GLiNER2",
            "gliner2_commit": GLINER2_COMMIT,
            # A directory name is not provenance. M1 only recognizes the exact
            # official base files; unverified/custom checkpoints remain explicit.
            "hf_model": BASE_MODEL_ID if is_official_base else None,
            "hf_revision": BASE_HF_REVISION if is_official_base else None,
            "source_file_sha256": source_hashes,
            "opset": opset,
            "onnx_runtime": "1.20.1",
            "ort_crate_version": "2.0.0-rc.9",
            "precision": "fp32",
            "attention_implementation": "eager",
            "mask_logit": MASK_LOGIT,
            "dependencies": package_versions(),
            "graphs": manifest.get("graphs", {}),
        }
    )
    graph_path = out / graph_name
    manifest["graphs"][graph_name] = {
        **graph_signature,
        "sha256": sha256_file(graph_path),
        "bytes": graph_path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def assert_finite(name: str, value: np.ndarray | torch.Tensor) -> None:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
    if not np.isfinite(array).all():
        raise AssertionError(f"{name} contains non-finite values")


def compare_arrays(
    name: str,
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> dict[str, float | str | list[int]]:
    if expected.shape != actual.shape:
        raise AssertionError(f"{name}: shape {actual.shape} != {expected.shape}")
    assert_finite(f"{name} expected", expected)
    assert_finite(f"{name} actual", actual)
    absolute = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    relative = absolute / np.maximum(np.abs(expected.astype(np.float64)), 1e-12)
    report: dict[str, float | str | list[int]] = {
        "name": name,
        "shape": list(expected.shape),
        "max_abs": float(absolute.max(initial=0.0)),
        "max_rel": float(relative.max(initial=0.0)),
    }
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        flat = int(np.argmax(absolute))
        index = np.unravel_index(flat, absolute.shape)
        raise AssertionError(
            f"{name}: numerical mismatch at {index}: expected={expected[index]!r} "
            f"actual={actual[index]!r}; max_abs={report['max_abs']:.9g}, "
            f"max_rel={report['max_rel']:.9g}, atol={atol}, rtol={rtol}"
        )
    return report
