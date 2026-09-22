#!/usr/bin/env python3
"""Numerically validate encoder.onnx against the pinned PyTorch oracle."""

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
    parser.add_argument("--onnx-path", default="onnx/gliner2-base-v1/encoder.onnx")
    parser.add_argument("--lengths", default="8,31,73")
    parser.add_argument("--golden-dir", help="Optional directory of golden .npz inputs")
    parser.add_argument("--golden-pattern", default="*.npz")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--report-json")
    return parser.parse_args()


def generated_cases(model, lengths: list[int], seed: int):
    rng = np.random.default_rng(seed)
    vocab_size = int(model.encoder.config.vocab_size)
    for index, length in enumerate(lengths):
        batch = 1 + (index % 2)
        ids = rng.integers(5, vocab_size, size=(batch, length), dtype=np.int64)
        mask = np.ones((batch, length), dtype=np.int64)
        if batch > 1 and length > 3:
            mask[1, -max(1, length // 5) :] = 0
        yield f"generated-b{batch}-l{length}", ids, mask


def golden_cases(path: Path, pattern: str):
    for fixture in sorted(path.glob(pattern)):
        with np.load(fixture, allow_pickle=False) as data:
            if "input_ids" in data and "attention_mask" in data:
                yield (
                    f"golden-{fixture.stem}",
                    data["input_ids"].astype(np.int64, copy=True),
                    data["attention_mask"].astype(np.int64, copy=True),
                )


def main() -> None:
    args = parse_args()
    configure_determinism(args.seed)
    model, _ = load_reference_model(args.model_dir)
    model.encoder.float().eval()
    session = ort.InferenceSession(
        str(Path(args.onnx_path)), providers=["CPUExecutionProvider"]
    )

    lengths = [int(value) for value in args.lengths.split(",") if value]
    cases = list(generated_cases(model, lengths, args.seed))
    if args.golden_dir:
        cases.extend(golden_cases(Path(args.golden_dir), args.golden_pattern))
    if len({ids.shape[1] for _, ids, _ in cases}) < 2:
        raise AssertionError("validation requires multiple dynamic sequence lengths")

    reports = []
    with torch.inference_mode():
        for name, input_ids, attention_mask in cases:
            expected = model.encoder(
                input_ids=torch.from_numpy(input_ids),
                attention_mask=torch.from_numpy(attention_mask),
            ).last_hidden_state.detach().cpu().numpy()
            (actual,) = session.run(
                ["last_hidden_state"],
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )
            report = compare_arrays(
                name, expected, actual, atol=args.atol, rtol=args.rtol
            )
            reports.append(report)
            print(
                f"{name}: shape={tuple(actual.shape)} "
                f"max_abs={report['max_abs']:.9g} max_rel={report['max_rel']:.9g}"
            )
    summary = {
        "onnx": str(Path(args.onnx_path)),
        "atol": args.atol,
        "rtol": args.rtol,
        "cases": reports,
        "maximum_abs_error": max(float(item["max_abs"]) for item in reports),
        "maximum_rel_error": max(float(item["max_rel"]) for item in reports),
    }
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
