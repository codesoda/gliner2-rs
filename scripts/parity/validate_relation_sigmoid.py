#!/usr/bin/env python3
"""Validate relation-proposal sigmoid arithmetic against pinned PyTorch 2.8.

This is a development-only numerical probe. It extracts the relevant functions
from ``src/boundary/relation_pairs.rs``, compiles them with the installed
``rustc`` in a temporary directory, and compares their f32 output bits with the
pinned macOS arm64 CPU reference environment. It does not load model weights or
participate in inference/builds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPORT_SCRIPTS = ROOT / "scripts" / "export"
sys.path.insert(0, str(EXPORT_SCRIPTS))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import configure_determinism  # noqa: E402

SEED = 1729
EXPECTED_TORCH_VERSION = "2.8.0"
EXPECTED_TORCH_GIT_VERSION = "a1cb3cc05d46d198467bebbb6e8fba50a325d4e7"
SLEEF_COMMIT = "5a1d179df9cf652951b59010a2d2075372d67f68"
RUST_FUNCTIONS = (
    "sleef_expf_u10",
    "scalar_source_sigmoid",
    "vector_source_sigmoid",
)
SOURCE_PATH = ROOT / "src" / "boundary" / "relation_pairs.rs"
PUBLIC_API_TEST_PATH = ROOT / "tests" / "boundary_relation_pairs.rs"
PINNED_PYTHON = ROOT / "scripts" / "export" / "env" / ".venv" / "bin" / "python"
VECTOR_SAMPLE_COUNT = 12_033_560
SCALAR_SAMPLE_COUNT = 2_000_001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-json",
        type=Path,
        help="optionally write the detailed validation report to this path",
    )
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def extract_rust_function(source: str, name: str) -> str:
    """Extract one non-nested Rust function, including its complete body."""
    matches = list(re.finditer(rf"(?m)^fn\s+{re.escape(name)}\s*\(", source))
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one Rust function {name!r}, found {len(matches)}")
    start = matches[0].start()
    opening = source.find("{", matches[0].end())
    if opening < 0:
        raise AssertionError(f"Rust function {name!r} has no body")
    depth = 0
    for offset in range(opening, len(source)):
        character = source[offset]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[start : offset + 1]
    raise AssertionError(f"Rust function {name!r} has an unterminated body")


def assert_reference_environment() -> None:
    # Do not resolve the venv executable symlink: its lexical path is the proof
    # that this script was launched through the repository's locked environment.
    actual_python = Path(sys.executable).absolute()
    expected_python = PINNED_PYTHON.absolute()
    if actual_python != expected_python:
        raise RuntimeError(
            f"run this probe with {PINNED_PYTHON.relative_to(ROOT)}; "
            f"current interpreter is {actual_python}"
        )
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError(
            "relation sigmoid bit parity is normative only for the pinned macOS arm64 "
            f"SLEEF path, found {platform.system()} {platform.machine()}"
        )
    if sys.byteorder != "little":
        raise RuntimeError(f"the arithmetic probe requires little-endian f32 bytes, found {sys.byteorder}")
    if torch.__version__ != EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            f"expected torch {EXPECTED_TORCH_VERSION}, found {torch.__version__}"
        )
    if torch.version.git_version != EXPECTED_TORCH_GIT_VERSION:
        raise RuntimeError(
            "unexpected Torch source revision: expected "
            f"{EXPECTED_TORCH_GIT_VERSION}, found {torch.version.git_version}"
        )


def build_probe(directory: Path, functions: list[str]) -> tuple[Path, str, str]:
    main = r'''
use std::io::{Read, Write};

fn main() {
    let mode = std::env::args().nth(1).expect("mode must be scalar or vector");
    let sigmoid: fn(f32) -> f32 = match mode.as_str() {
        "scalar" => scalar_source_sigmoid,
        "vector" => vector_source_sigmoid,
        _ => panic!("mode must be scalar or vector"),
    };
    let mut input = Vec::new();
    std::io::stdin().read_to_end(&mut input).unwrap();
    assert_eq!(input.len() % 4, 0, "input must contain complete little-endian f32 values");
    let mut output = Vec::with_capacity(input.len());
    for bytes in input.chunks_exact(4) {
        let value = f32::from_le_bytes(bytes.try_into().unwrap());
        output.extend_from_slice(&sigmoid(value).to_bits().to_le_bytes());
    }
    std::io::stdout().write_all(&output).unwrap();
}
'''
    probe_source = "\n\n".join(functions) + "\n" + main
    source_path = directory / "relation_sigmoid_probe.rs"
    binary_path = directory / "relation_sigmoid_probe"
    source_path.write_text(probe_source)
    completed = subprocess.run(
        ["rustc", "--edition", "2024", "-O", str(source_path), "-o", str(binary_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stderr:
        raise RuntimeError(f"rustc emitted diagnostics:\n{completed.stderr}")
    rustc_version = subprocess.check_output(["rustc", "--version"], text=True).strip()
    return binary_path, sha256_bytes(probe_source.encode()), rustc_version


def rust_sigmoid(binary: Path, values: np.ndarray, mode: str) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    completed = subprocess.run(
        [str(binary), mode],
        input=contiguous.tobytes(),
        check=True,
        capture_output=True,
    )
    expected_bytes = contiguous.size * np.dtype(np.float32).itemsize
    if len(completed.stdout) != expected_bytes:
        raise AssertionError(
            f"Rust probe returned {len(completed.stdout)} bytes for {expected_bytes} input bytes"
        )
    return np.frombuffer(completed.stdout, dtype="<f4")


def bits(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def compare_bits(name: str, expected: np.ndarray, actual: np.ndarray) -> int:
    if expected.shape != actual.shape:
        raise AssertionError(f"{name}: shape {actual.shape} != expected {expected.shape}")
    failures = int(np.count_nonzero(bits(actual) != bits(expected)))
    if failures:
        raise AssertionError(f"{name}: {failures} exact-bit failures")
    return failures


def vector_cases() -> dict[str, np.ndarray]:
    lower = np.float32(-1.388).view(np.uint32).item()
    upper = np.float32(-1.384).view(np.uint32).item()
    threshold_bits = np.arange(min(lower, upper), max(lower, upper) + 1, dtype=np.uint32)
    cases = {
        "all_f32_threshold_window": np.sort(threshold_bits.view(np.float32)),
        "dense_minus110_to110": np.linspace(-110, 110, 8_000_004, dtype=np.float32),
        "seeded_uniform_minus110_to110": np.random.default_rng(SEED)
        .uniform(-110, 110, 4_000_000)
        .astype(np.float32),
    }
    count = sum(values.size for values in cases.values())
    if count != VECTOR_SAMPLE_COUNT:
        raise AssertionError(f"vector corpus changed: expected {VECTOR_SAMPLE_COUNT}, found {count}")
    return cases


def layout_inputs(rounding_probe: np.float32) -> dict[str, torch.Tensor]:
    value = float(rounding_probe)
    return {
        "row_major_2x5": torch.full((2, 5), value, dtype=torch.float32),
        "row_major_2x192": torch.full((2, 192), value, dtype=torch.float32),
        "row_major_4x192": torch.full((4, 192), value, dtype=torch.float32),
        "dense_transpose_2x5": torch.full((5, 2), value, dtype=torch.float32).t(),
        "dense_transpose_2x192": torch.full((192, 2), value, dtype=torch.float32).t(),
        "dense_transpose_4x192": torch.full((192, 4), value, dtype=torch.float32).t(),
        "row_gap_2x9": torch.full((4, 9), value, dtype=torch.float32)[::2],
        "column_gap_9x2": torch.full((2, 18), value, dtype=torch.float32)[:, :9].t(),
        "broadcast_scalar_2x5": torch.tensor(value, dtype=torch.float32)
        .reshape(1, 1)
        .expand(2, 5),
        "broadcast_row_2x5": torch.full((1, 5), value, dtype=torch.float32).expand(2, 5),
        "broadcast_column_2x5": torch.full((2, 1), value, dtype=torch.float32).expand(2, 5),
        "stride2_columns_2x5": torch.full((2, 10), value, dtype=torch.float32)[:, ::2],
        "stride2_columns_2x192": torch.full((2, 384), value, dtype=torch.float32)[:, ::2],
        "stride2_columns_4x192": torch.full((4, 384), value, dtype=torch.float32)[:, ::2],
    }


def mirrored_layout_result(
    binary: Path, tensor: torch.Tensor
) -> tuple[np.ndarray, str]:
    """Mirror proposal_probabilities traversal; this does not execute that Rust function."""
    query_count, candidate_count = tensor.shape
    query_stride, candidate_stride = tensor.stride()
    source_count = tensor.numel()
    if tensor.is_contiguous() or (query_stride == 0 and candidate_stride == 0):
        traversal = "global_row"
    elif query_stride == 1 and candidate_stride == query_count:
        traversal = "global_column"
    elif candidate_stride in (0, 1):
        traversal = "rows"
    elif query_stride == 1:
        traversal = "columns"
    else:
        traversal = "scalar"

    logical_values = tensor.numpy().reshape(-1).astype(np.float32, copy=False)
    vector_values = rust_sigmoid(binary, logical_values, "vector")
    expected = rust_sigmoid(binary, logical_values, "scalar").copy()
    global_cut = source_count // 8 * 8
    row_cut = candidate_count // 8 * 8
    column_cut = query_count // 8 * 8
    for query in range(query_count):
        for candidate in range(candidate_count):
            logical = query * candidate_count + candidate
            uses_vector = (
                (traversal == "global_row" and logical < global_cut)
                or (
                    traversal == "global_column"
                    and candidate * query_count + query < global_cut
                )
                or (traversal == "rows" and candidate < row_cut)
                or (traversal == "columns" and query < column_cut)
            )
            if uses_vector:
                expected[logical] = vector_values[logical]
    return expected, traversal


def run_validation(binary: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    vector_results: list[dict[str, Any]] = []
    for name, values in vector_cases().items():
        expected = torch.sigmoid(torch.from_numpy(values.copy())).numpy()
        actual = rust_sigmoid(binary, values, "vector")
        failures = compare_bits(name, expected, actual)
        vector_results.append(
            {"name": name, "sample_count": int(values.size), "bit_failures": failures}
        )

    scalar_values = np.linspace(-20, 20, SCALAR_SAMPLE_COUNT, dtype=np.float32)
    scalar_storage = np.empty(scalar_values.size * 2, dtype=np.float32)
    scalar_storage[::2] = scalar_values
    scalar_storage[1::2] = 123.0
    expected_scalar = torch.sigmoid(torch.from_numpy(scalar_storage)[::2]).numpy()
    actual_scalar = rust_sigmoid(binary, scalar_values, "scalar")
    scalar_failures = compare_bits("strided_scalar", expected_scalar, actual_scalar)
    scalar_result = {
        "name": "strided_scalar",
        "sample_count": int(scalar_values.size),
        "input_stride": 2,
        "bit_failures": scalar_failures,
    }

    rounding_probe = np.array([0xBFB1A843], dtype=np.uint32).view(np.float32)[0]
    layout_results: list[dict[str, Any]] = []
    for name, tensor in layout_inputs(rounding_probe).items():
        output = torch.sigmoid(tensor)
        mirrored, traversal = mirrored_layout_result(binary, tensor)
        actual_logical = output.numpy().reshape(-1).astype(np.float32, copy=False)
        failures = compare_bits(name, mirrored, actual_logical)
        output_bits, counts = np.unique(bits(actual_logical), return_counts=True)
        layout_results.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "sample_count": tensor.numel(),
                "input_strides": list(tensor.stride()),
                "output_strides": list(output.stride()),
                "mirrored_traversal": traversal,
                "bit_failures": failures,
                "output_bit_counts": {
                    f"0x{int(key):08x}": int(count)
                    for key, count in zip(output_bits, counts)
                },
            }
        )
    return vector_results, scalar_result, layout_results


def main() -> None:
    args = parse_args()
    assert_reference_environment()
    configure_determinism(SEED, threads=1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError(
            "Torch thread pinning failed: "
            f"intra={torch.get_num_threads()}, inter={torch.get_num_interop_threads()}"
        )

    if not SOURCE_PATH.is_file() or not PUBLIC_API_TEST_PATH.is_file():
        raise FileNotFoundError("relation source or public API regression test is missing")
    rust_source = SOURCE_PATH.read_text()
    if SLEEF_COMMIT not in rust_source:
        raise AssertionError(f"relation source no longer identifies pinned SLEEF commit {SLEEF_COMMIT}")
    extracted = [extract_rust_function(rust_source, name) for name in RUST_FUNCTIONS]
    if "vector_source_sigmoid(logit)" not in rust_source or "scalar_source_sigmoid(logit)" not in rust_source:
        raise AssertionError("proposal traversal no longer calls both extracted sigmoid paths")

    with tempfile.TemporaryDirectory(prefix="relation-sigmoid-") as temporary:
        binary, probe_sha256, rustc_version = build_probe(Path(temporary), extracted)
        vector_results, scalar_result, layout_results = run_validation(binary)

    vector_count = sum(case["sample_count"] for case in vector_results)
    scalar_count = scalar_result["sample_count"]
    layout_count = sum(case["sample_count"] for case in layout_results)
    test_case_count = len(vector_results) + 1 + len(layout_results)
    exact_bit_comparisons = vector_count + scalar_count + layout_count
    bit_failures = sum(case["bit_failures"] for case in vector_results)
    bit_failures += scalar_result["bit_failures"]
    bit_failures += sum(case["bit_failures"] for case in layout_results)
    if bit_failures:
        raise AssertionError(f"validation completed with {bit_failures} exact-bit failures")

    report: dict[str, Any] = {
        "scope": (
            "Pinned macOS arm64 Torch 2.8 CPU/SLEEF relation-proposal sigmoid "
            "development validation; not a universal platform parity claim"
        ),
        "status": "passed",
        "seed": SEED,
        "source": {
            "path": str(SOURCE_PATH.relative_to(ROOT)),
            "sha256": sha256_file(SOURCE_PATH),
            "extracted_functions": {
                name: sha256_bytes(function.encode())
                for name, function in zip(RUST_FUNCTIONS, extracted)
            },
            "sleef_commit": SLEEF_COMMIT,
        },
        "script": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "compiled_probe_sha256": probe_sha256,
        "environment": {
            "python": str(Path(sys.executable).absolute().relative_to(ROOT)),
            "torch_version": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "rustc_version": rustc_version,
        },
        "counts": {
            "test_case_count": test_case_count,
            "vector_sample_count": vector_count,
            "scalar_strided_sample_count": scalar_count,
            "layout_sample_count": layout_count,
            "exact_bit_comparison_count": exact_bit_comparisons,
            "bit_failures": bit_failures,
            "errors": 0,
        },
        "vector_cases": vector_results,
        "scalar_strided_case": scalar_result,
        "layout_cases": layout_results,
        "layout_validation": {
            "kind": "Python mirror of Rust traversal classification, not actual Rust traversal",
            "limitation": (
                "The layout comparison executes extracted Rust scalar/vector arithmetic but "
                "mirrors TensorIterator traversal classification in Python. Actual Rust public "
                "API layout regressions are covered by tests/boundary_relation_pairs.rs."
            ),
            "public_api_regression_tests": str(PUBLIC_API_TEST_PATH.relative_to(ROOT)),
        },
    }

    if args.report_json is not None:
        report_path = args.report_json.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Detailed report: {report_path}")

    print("relation sigmoid parity: PASS")
    print(
        f"cases={test_case_count} exact_bit_comparisons={exact_bit_comparisons:,} "
        "bit_failures=0 errors=0"
    )
    print(
        f"vector={vector_count:,} scalar_strided={scalar_count:,} "
        f"layout_mirror={layout_count:,} threads={torch.get_num_threads()}"
    )
    print(f"source_sha256={report['source']['sha256']}")
    print(f"script_sha256={report['script']['sha256']}")
    print(
        "layout limitation: Python mirrors traversal classification; actual Rust public API "
        "regressions are in tests/boundary_relation_pairs.rs"
    )


if __name__ == "__main__":
    main()
