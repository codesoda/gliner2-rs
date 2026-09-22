#!/usr/bin/env python3
"""Generate optional Rust record-runtime vectors from untouched forward_group.

The four vectors contain only the already-routed ONNX ABI inputs and the three
outputs returned by the pinned checkpoint's original RecordHead.forward_group.
They intentionally omit encoder tensors, model weights, and high-level record
metadata, and are written to an ignored directory by default.
"""

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
from gen_boundary_goldens import (  # noqa: E402
    build_expected_batch,
    build_schema,
    write_deterministic_npz,
)
from validate_boundary_records import (  # noqa: E402
    REAL_CASES,
    assert_masks_and_shapes,
    compare_outputs,
    expected_outputs,
    fixture_candidates,
    numpy_inputs,
    prepare_group_inputs,
    run_onnx,
    source_group_and_states,
)

DEFAULT_MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--fastino--gliner2.5-base-v1"
    / "snapshots"
    / BASE_HF_REVISION
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--golden-dir", type=Path, default=ROOT / "fixtures/gliner2.5-base-v1"
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=ROOT / "onnx/gliner2.5-base-v1/boundary_records.onnx",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "fixtures/gliner2.5-records"
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_determinism(args.seed)
    source_hashes = assert_official_base_source(args.model_dir)
    model, config = load_reference_model(args.model_dir)
    if config.get("architecture") != "boundary":
        raise RuntimeError("record vectors require a boundary checkpoint")
    if not args.onnx_path.is_file():
        raise FileNotFoundError(args.onnx_path)
    session = ort.InferenceSession(
        str(args.onnx_path), providers=["CPUExecutionProvider"]
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    stats: dict[str, dict[str, float | int]] = {}
    for case_id in REAL_CASES:
        metadata_path = args.golden_dir / f"{case_id}.json"
        source_npz = args.golden_dir / f"{case_id}.npz"
        metadata = json.loads(metadata_path.read_text())
        schema = build_schema(model, metadata["schema_spec"])
        batch = build_expected_batch(model, metadata["text"], schema)
        if len(batch.record_specs[0]) != 1:
            raise AssertionError(f"{case_id}: expected exactly one RecordSpec")
        spec = next(iter(batch.record_specs[0].values()))

        with np.load(source_npz, allow_pickle=False) as data:
            query_states = torch.from_numpy(data["query_states"][0].copy()).float()
            candidates = fixture_candidates(data)
        group, instance_states = source_group_and_states(
            model.record_decoder, spec, query_states, candidates
        )
        inputs = prepare_group_inputs(spec, query_states, candidates)
        expected = expected_outputs(group, instance_states, inputs[1].shape[1])

        # The vectors remain an original-source oracle, but generation also
        # proves the selected graph matches that oracle at the unchanged gate.
        actual = run_onnx(session, inputs)
        compare_outputs(
            case_id,
            expected,
            actual,
            stats,
            atol=args.atol,
            rtol=args.rtol,
        )
        shape_report = assert_masks_and_shapes(
            case_id,
            inputs,
            actual,
            instance_queries=int(model.record_decoder.instance_queries),
        )

        arrays = {**numpy_inputs(inputs), **expected}
        output_path = args.out_dir / f"{case_id}.npz"
        write_deterministic_npz(output_path, arrays)
        entries.append(
            {
                "case_id": case_id,
                "mode": spec.mode,
                "source_json_sha256": sha256_file(metadata_path),
                "source_npz_sha256": sha256_file(source_npz),
                "vector": output_path.name,
                "vector_bytes": output_path.stat().st_size,
                "vector_sha256": sha256_file(output_path),
                "field_count": int(inputs[0].shape[0]),
                "candidate_count": int(inputs[1].shape[1]),
                "context_count": int(inputs[3].shape[0]),
                "seed_count": int(inputs[5].shape[0]),
                "instance_count": int(expected["instance_states"].shape[0]),
                **shape_report,
            }
        )
        print(f"wrote {output_path}", flush=True)

    manifest = {
        "format_version": 1,
        "oracle": "untouched pinned GLiNER2 RecordHead.forward_group",
        "case_count": len(entries),
        "atol": args.atol,
        "rtol": args.rtol,
        "provenance": {
            "gliner2_commit": GLINER2_COMMIT,
            "model_id": BASE_MODEL_ID,
            "hf_revision": BASE_HF_REVISION,
            "source_file_sha256": source_hashes,
            "boundary_records_onnx_sha256": sha256_file(args.onnx_path),
            "seed": args.seed,
            "dtype": "float32",
            "device": "cpu",
            "dependencies": package_versions(),
        },
        "onnx_vs_source": stats,
        "entries": entries,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
