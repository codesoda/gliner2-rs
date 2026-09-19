#!/usr/bin/env python3
"""Regression tests for the narrowly scoped inside-prefix ONNX tolerance."""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from boundary_marginals import OUTPUT_NAMES
from validate_boundary_marginals import (
    PREFIX_COORDINATE_ATOL,
    compare_onnx_array,
    compare_onnx_output_sets,
    compare_output_sets,
)


class PrefixToleranceTests(unittest.TestCase):
    def test_coordinate_zero_has_no_extra_allowance(self) -> None:
        expected = np.zeros((1, 1, 3), dtype=np.float32)
        actual = expected.copy()
        actual[0, 0, 0] = np.float32(1.005e-4)
        with self.assertRaisesRegex(AssertionError, "numerical mismatch"):
            compare_onnx_array(
                "inside_prefix",
                expected,
                actual,
                atol=1e-4,
                rtol=1e-3,
                prefix_coordinate_atol=PREFIX_COORDINATE_ATOL,
            )

    def test_allowance_grows_by_coordinate_not_global_length(self) -> None:
        expected = np.zeros((1, 1, 8), dtype=np.float32)
        actual = expected.copy()
        actual[0, 0, 1] = np.float32(1e-4 + PREFIX_COORDINATE_ATOL * 0.95)
        report = compare_onnx_array(
            "inside_prefix",
            expected,
            actual,
            atol=1e-4,
            rtol=1e-3,
            prefix_coordinate_atol=PREFIX_COORDINATE_ATOL,
        )
        self.assertEqual(report["raw_original_failure_count"], 1)

        actual[0, 0, 1] = np.float32(1e-4 + PREFIX_COORDINATE_ATOL * 2.0)
        with self.assertRaisesRegex(AssertionError, r"\(0, 0, 1\)"):
            compare_onnx_array(
                "inside_prefix",
                expected,
                actual,
                atol=1e-4,
                rtol=1e-3,
                prefix_coordinate_atol=PREFIX_COORDINATE_ATOL,
            )

    def test_nonprefix_output_keeps_original_tolerance(self) -> None:
        expected = {}
        actual = {}
        for name in OUTPUT_NAMES:
            if name == "boundary_mask":
                expected[name] = np.ones((1, 3), dtype=bool)
            else:
                expected[name] = np.zeros((1, 1, 3), dtype=np.float32)
            actual[name] = expected[name].copy()
        actual["start_logits"][0, 0, 2] = np.float32(1e-4 + PREFIX_COORDINATE_ATOL)
        failures: list[str] = []
        compare_onnx_output_sets(
            "test",
            expected,
            actual,
            {},
            Counter(),
            atol=1e-4,
            rtol=1e-3,
            strict_prefix=False,
            failures=failures,
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("start_logits", failures[0])

    def test_non_onnx_comparison_never_gets_prefix_allowance(self) -> None:
        expected = {}
        actual = {}
        for name in OUTPUT_NAMES:
            if name == "boundary_mask":
                expected[name] = np.ones((1, 3), dtype=bool)
            else:
                expected[name] = np.zeros((1, 1, 3), dtype=np.float32)
            actual[name] = expected[name].copy()
        actual["inside_prefix"][0, 0, 2] = np.float32(1e-4 + PREFIX_COORDINATE_ATOL)
        with self.assertRaisesRegex(AssertionError, "inside_prefix"):
            compare_output_sets(
                "oracle-golden", expected, actual, {}, atol=1e-4, rtol=1e-3
            )

    def test_shape_and_nonfinite_values_are_rejected(self) -> None:
        expected = np.zeros((1, 1, 3), dtype=np.float32)
        with self.assertRaisesRegex(AssertionError, "shape"):
            compare_onnx_array(
                "inside_prefix",
                expected,
                np.zeros((1, 1, 4), dtype=np.float32),
                atol=1e-4,
                rtol=1e-3,
                prefix_coordinate_atol=PREFIX_COORDINATE_ATOL,
            )
        actual = expected.copy()
        actual[0, 0, 1] = np.nan
        with self.assertRaisesRegex(AssertionError, "non-finite"):
            compare_onnx_array(
                "inside_prefix",
                expected,
                actual,
                atol=1e-4,
                rtol=1e-3,
                prefix_coordinate_atol=PREFIX_COORDINATE_ATOL,
            )

    def test_excessive_prefix_error_still_fails(self) -> None:
        expected = np.zeros((1, 1, 100), dtype=np.float32)
        actual = expected.copy()
        coordinate = 50
        actual[0, 0, coordinate] = np.float32(
            1e-4 + PREFIX_COORDINATE_ATOL * coordinate + 1e-5
        )
        with self.assertRaisesRegex(AssertionError, "numerical mismatch"):
            compare_onnx_array(
                "inside_prefix",
                expected,
                actual,
                atol=1e-4,
                rtol=1e-3,
                prefix_coordinate_atol=PREFIX_COORDINATE_ATOL,
            )


if __name__ == "__main__":
    unittest.main()
