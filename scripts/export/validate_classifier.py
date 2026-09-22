#!/usr/bin/env python3
"""Numerically validate classifier.onnx against raw and calibrated PyTorch logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from common import compare_arrays, configure_determinism, load_reference_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/gliner2-base-v1")
    parser.add_argument("--onnx-path", default="onnx/gliner2-base-v1/classifier.onnx")
    parser.add_argument("--rows", default="1,3,7,13")
    parser.add_argument("--golden-dir", help="Optional directory of golden .npz inputs")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--report-json")
    return parser.parse_args()


def golden_inputs(path: Path):
    for fixture in sorted(path.glob("*.npz")):
        with np.load(fixture, allow_pickle=False) as data:
            for key in sorted(data.files):
                if (
                    key.startswith("classifier_")
                    and key.endswith("_input")
                    and data[key].ndim == 2
                ):
                    yield f"golden-{fixture.stem}-{key}", data[key].astype(np.float32)


def main() -> None:
    args = parse_args()
    configure_determinism(args.seed)
    model, config = load_reference_model(args.model_dir)
    model.classifier.float().eval()
    hidden = int(model.hidden_size)
    temperature = float(
        config.get("boundary_head", {}).get("classification_temperature", 1.0)
    )
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"invalid classification temperature {temperature}")

    rng = np.random.default_rng(args.seed)
    cases = [
        (f"generated-n{rows}", rng.standard_normal((rows, hidden), dtype=np.float32))
        for rows in (int(value) for value in args.rows.split(",") if value)
    ]
    if args.golden_dir:
        cases.extend(golden_inputs(Path(args.golden_dir)))
    if len({values.shape[0] for _, values in cases}) < 2:
        raise AssertionError("validation requires multiple dynamic row counts")

    session = ort.InferenceSession(
        str(Path(args.onnx_path)), providers=["CPUExecutionProvider"]
    )
    reports = []
    with torch.inference_mode():
        for name, values in cases:
            expected = (
                model.classifier(torch.from_numpy(values))
                .squeeze(-1)
                .detach()
                .cpu()
                .numpy()
            )
            (actual,) = session.run(["logits"], {"cls_embeds": values})
            raw = compare_arrays(
                f"{name}-raw", expected, actual, atol=args.atol, rtol=args.rtol
            )
            calibrated = compare_arrays(
                f"{name}-temperature",
                expected / temperature,
                actual / temperature,
                atol=args.atol,
                rtol=args.rtol,
            )
            reports.extend((raw, calibrated))
            print(
                f"{name}: rows={values.shape[0]} max_abs={raw['max_abs']:.9g} "
                f"max_rel={raw['max_rel']:.9g} temperature={temperature:g}"
            )
    summary = {
        "onnx": str(Path(args.onnx_path)),
        "atol": args.atol,
        "rtol": args.rtol,
        "classification_temperature": temperature,
        "cases": reports,
        "maximum_abs_error": max(float(item["max_abs"]) for item in reports),
        "maximum_rel_error": max(float(item["max_rel"]) for item in reports),
    }
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
