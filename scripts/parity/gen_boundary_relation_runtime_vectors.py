#!/usr/bin/env python3
"""Generate compact relation-runtime vectors from the untouched pinned scorer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = ROOT / "scripts" / "export"
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from boundary_relations import INPUT_NAMES, source_logits, validate_abi_inputs  # noqa: E402
from common import (  # noqa: E402
    BASE_HF_REVISION,
    BASE_MODEL_ID,
    GLINER2_COMMIT,
    assert_official_base_source,
    configure_determinism,
    load_reference_model,
    package_versions,
    sha256_file,
)
from gen_boundary_goldens import write_deterministic_npz  # noqa: E402
from validate_boundary_relations import (  # noqa: E402
    REAL_CASES,
    compare,
    fixture_inputs,
    numpy_inputs,
    run_onnx,
)

DEFAULT_MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--fastino--gliner2.5-base-v1"
    / "snapshots"
    / BASE_HF_REVISION
)
DEFAULT_OUT_DIR = ROOT / "fixtures/gliner2.5-base-v1/relation-runtime-vectors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--golden-dir", type=Path, default=ROOT / "fixtures/gliner2.5-base-v1"
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=ROOT / "onnx/gliner2.5-base-v1/boundary_relations.onnx",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if ort.__version__ != "1.20.1":
        raise RuntimeError(f"vector generation requires ORT 1.20.1, found {ort.__version__}")
    configure_determinism(args.seed)
    source_hashes = assert_official_base_source(args.model_dir)
    model, config = load_reference_model(args.model_dir)
    head = config.get("boundary_head", {})
    if not (
        config.get("architecture") == "boundary"
        and config.get("architecture_version") == 1
        and head.get("directional_relation_states") is True
        and head.get("relation_biaffine_content") is True
    ):
        raise RuntimeError("relation vectors require the supported boundary relation flags")
    if not args.onnx_path.is_file():
        raise FileNotFoundError(args.onnx_path)
    session = ort.InferenceSession(
        str(args.onnx_path), providers=["CPUExecutionProvider"]
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    onnx_stats: dict[str, float | int] = {"failure_count": 0}
    frozen_stats: dict[str, float | int] = {"failure_count": 0}
    failures: list[str] = []
    for case_id in REAL_CASES:
        source_path = args.golden_dir / f"{case_id}.npz"
        metadata_path = args.golden_dir / f"{case_id}.json"
        with np.load(source_path, allow_pickle=False) as data:
            inputs = fixture_inputs(data)
            frozen = data["relation_scorer_0_logits"].copy()
        validate_abi_inputs(inputs)
        with torch.inference_mode():
            expected_tensor = source_logits(model.relation_scorer, inputs)
        expected = expected_tensor.detach().cpu().numpy()
        compare(
            f"{case_id}/untouched_vs_frozen",
            frozen,
            expected,
            frozen_stats,
            failures,
            atol=args.atol,
            rtol=args.rtol,
        )
        actual = run_onnx(session, inputs)
        compare(
            f"{case_id}/onnx_vs_untouched",
            expected,
            actual,
            onnx_stats,
            failures,
            atol=args.atol,
            rtol=args.rtol,
            confidence_atol=1e-3,
        )
        if failures:
            raise AssertionError("; ".join(failures))

        arrays = {**numpy_inputs(inputs), "relation_logits": expected}
        output_path = args.out_dir / f"{case_id}.npz"
        write_deterministic_npz(output_path, arrays)
        entries.append(
            {
                "case_id": case_id,
                "source_json_sha256": sha256_file(metadata_path),
                "source_npz_sha256": sha256_file(source_path),
                "vector": output_path.name,
                "vector_bytes": output_path.stat().st_size,
                "vector_sha256": sha256_file(output_path),
                "B": int(inputs[0].shape[0]),
                "L": int(inputs[0].shape[1]),
                "H": int(inputs[0].shape[2]),
                "R": int(inputs[1].shape[1]),
                "P": int(inputs[2].shape[0]),
            }
        )
        print(f"wrote {output_path}", flush=True)

    manifest = {
        "format_version": 1,
        "oracle": "untouched pinned GLiNER2 SparseRelationScorer.forward",
        "case_count": len(entries),
        "atol": args.atol,
        "rtol": args.rtol,
        "provenance": {
            "gliner2_commit": GLINER2_COMMIT,
            "model_id": BASE_MODEL_ID,
            "hf_revision": BASE_HF_REVISION,
            "source_file_sha256": source_hashes,
            "boundary_relations_onnx_sha256": sha256_file(args.onnx_path),
            "seed": args.seed,
            "dtype": "float32",
            "device": "cpu",
            "python_onnxruntime": ort.__version__,
            "dependencies": package_versions(),
        },
        "untouched_vs_frozen": frozen_stats,
        "onnx_vs_untouched": onnx_stats,
        "entries": entries,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
