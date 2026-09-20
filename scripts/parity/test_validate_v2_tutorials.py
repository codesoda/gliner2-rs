"""Model-free tests for the executable v2 tutorial regression gate."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from validate_v2_tutorials import (  # noqa: E402
    ABSOLUTE_TOLERANCE,
    DEFAULT_REFERENCE_DIR,
    EXPECTED_CASE_IDS,
    GateError,
    compare_case,
    discover_confidence_slots,
    run_tutorials,
    validate_tutorials,
)


TUPLE_REFERENCE = b'''"topics": MultiWithConfidence(
    [
        (
            "technology",
            0.5,
        ),
        (
            "health",
            0.75,
        ),
    ],
),
'''
SPAN_REFERENCE = b'''TextWithConfidenceAndSpans {
    text: "item 123",
    confidence: 0.5,
    start: 10,
    end: 18,
},
'''


def _replace_once(data: bytes, old: bytes, new: bytes) -> bytes:
    if data.count(old) != 1:
        raise AssertionError("test replacement is not unique")
    return data.replace(old, new, 1)


class V2TutorialValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _materialize_actuals(self, prefix: str = "test") -> Path:
        actual = self.root / "actual"
        actual.mkdir(exist_ok=True)
        for case_id in EXPECTED_CASE_IDS:
            name = "tutorial_{}.txt".format(case_id)
            (actual / "{}-{}".format(prefix, name)).write_bytes(
                (DEFAULT_REFERENCE_DIR / name).read_bytes()
            )
        return actual

    def test_exact_round_trip_all_six_cases(self) -> None:
        report = validate_tutorials(
            DEFAULT_REFERENCE_DIR, self._materialize_actuals(), "test"
        )
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["case_count"], 6)
        self.assertEqual(report["confidence_difference_count"], 0)

    def test_known_ort_128_output_delta_passes(self) -> None:
        actual = self._materialize_actuals("rc13")
        path = actual / "rc13-tutorial_1_classification.txt"
        output = _replace_once(path.read_bytes(), b"0.86740035,", b"0.8674009,")
        output = (
            b"model load took: 17.85s (onnx=/provided/model)\n"
            + output
            + b"inference took: 867.79ms\n"
        )
        path.write_bytes(output)
        report = validate_tutorials(DEFAULT_REFERENCE_DIR, actual, "rc13")
        self.assertEqual(report["confidence_difference_count"], 1)
        self.assertEqual(
            report["max_absolute_confidence_error"],
            5.364418029785156e-07,
        )

    def test_within_at_and_above_tolerance(self) -> None:
        slots = discover_confidence_slots(TUPLE_REFERENCE)
        within = _replace_once(TUPLE_REFERENCE, b"0.5,", b"0.5000009,")
        self.assertEqual(
            compare_case("synthetic", TUPLE_REFERENCE, within, slots)["status"],
            "pass",
        )

        # The decimal delta is exactly the approved bound; its finite f32
        # representation is fractionally below it and must pass as-is.
        zero_reference = _replace_once(TUPLE_REFERENCE, b"0.5,", b"0.0,")
        zero_slots = discover_confidence_slots(zero_reference)
        at = _replace_once(zero_reference, b"0.0,", b"0.000001,")
        self.assertLessEqual(
            compare_case(
                "synthetic", zero_reference, at, zero_slots, ABSOLUTE_TOLERANCE
            )["max_absolute_confidence_error"],
            ABSOLUTE_TOLERANCE,
        )

        above = _replace_once(TUPLE_REFERENCE, b"0.5,", b"0.5000011,")
        with self.assertRaisesRegex(GateError, "exceeds"):
            compare_case(
                "synthetic", TUPLE_REFERENCE, above, slots, ABSOLUTE_TOLERANCE
            )

    def test_coordinate_integers_and_digits_in_strings_are_exact(self) -> None:
        slots = discover_confidence_slots(SPAN_REFERENCE)
        coordinate = _replace_once(SPAN_REFERENCE, b"start: 10,", b"start: 11,")
        string_digits = _replace_once(SPAN_REFERENCE, b"item 123", b"item 124")
        for changed in (coordinate, string_digits):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(GateError, "outside a declared"):
                    compare_case("synthetic", SPAN_REFERENCE, changed, slots)

    def test_labels_missing_extra_and_reordered_results_are_exact(self) -> None:
        slots = discover_confidence_slots(TUPLE_REFERENCE)
        label = _replace_once(TUPLE_REFERENCE, b"technology", b"technologies")
        missing = TUPLE_REFERENCE.replace(b"            \"health\",\n", b"")
        extra = TUPLE_REFERENCE + b"payload\n"
        first = b'''        (
            "technology",
            0.5,
        ),
'''
        second = b'''        (
            "health",
            0.75,
        ),
'''
        reordered = TUPLE_REFERENCE.replace(first + second, second + first)
        for changed in (label, missing, extra, reordered):
            with self.subTest(changed=changed):
                with self.assertRaises(GateError):
                    compare_case("synthetic", TUPLE_REFERENCE, changed, slots)

    def test_nan_and_infinity_are_rejected(self) -> None:
        slots = discover_confidence_slots(TUPLE_REFERENCE)
        for token in (b"NaN", b"nan", b"inf", b"Infinity", b"1e999"):
            changed = _replace_once(TUPLE_REFERENCE, b"0.5,", token + b",")
            with self.subTest(token=token):
                with self.assertRaisesRegex(GateError, "decimal|finite"):
                    compare_case("synthetic", TUPLE_REFERENCE, changed, slots)

    def test_invalid_or_relaxed_tolerance_is_rejected_before_comparison(self) -> None:
        slots = discover_confidence_slots(TUPLE_REFERENCE)
        for tolerance in (float("nan"), float("inf"), -1.0, 1.000001e-6):
            with self.subTest(tolerance=tolerance):
                with self.assertRaisesRegex(GateError, "tolerance must be finite"):
                    compare_case(
                        "synthetic", TUPLE_REFERENCE, TUPLE_REFERENCE, slots, tolerance
                    )

    def test_malformed_confidence_context_cannot_be_blessed(self) -> None:
        malformed = b'''UnknownOutput {
    label: "technology",
    confidence: 0.5,
},
'''
        self.assertEqual(discover_confidence_slots(malformed), [])
        fake_slot = {
            "line": 3,
            "kind": "named",
            "variant": "UnknownOutput",
            "prefix": "    confidence: ",
            "suffix": ",",
            "reference": "0.5",
        }
        changed = malformed.replace(b"0.5,", b"0.5000001,")
        with self.assertRaisesRegex(GateError, "not the declared typed"):
            compare_case("synthetic", malformed, changed, [fake_slot])

    def test_reference_hash_tamper_is_rejected(self) -> None:
        reference = self.root / "reference"
        shutil.copytree(DEFAULT_REFERENCE_DIR, reference)
        actual = self._materialize_actuals()
        path = reference / "tutorial_5_validator.txt"
        path.write_bytes(path.read_bytes() + b"tamper\n")
        with self.assertRaisesRegex(GateError, "reference hash mismatch"):
            validate_tutorials(reference, actual, "test")

    def test_confidence_line_ending_bytes_are_exact(self) -> None:
        slots = discover_confidence_slots(TUPLE_REFERENCE)
        changed = _replace_once(
            TUPLE_REFERENCE, b"            0.5,\n", b"            0.5,\r\n"
        )
        with self.assertRaisesRegex(GateError, "line-ending bytes"):
            compare_case("synthetic", TUPLE_REFERENCE, changed, slots)

    def test_only_complete_timing_lines_are_ignored(self) -> None:
        slots = discover_confidence_slots(TUPLE_REFERENCE)
        actual = (
            b"model load took: 1.25s (onnx=/tmp/model)\n"
            + TUPLE_REFERENCE
            + b"batch inference took: 2.00ms\n"
            + b"batch took: 3us\n"
            + b"inference took: 4ns\n"
        )
        report = compare_case("synthetic", TUPLE_REFERENCE, actual, slots)
        self.assertEqual(report["ignored_timing_line_count"], 4)

        payload = TUPLE_REFERENCE + b"inference took: 4ns payload must remain\n"
        with self.assertRaisesRegex(GateError, "line count changed"):
            compare_case("synthetic", TUPLE_REFERENCE, payload, slots)

    def test_missing_and_extra_actual_cases_fail(self) -> None:
        actual = self._materialize_actuals()
        (actual / "test-tutorial_6_relation_extraction.txt").unlink()
        with self.assertRaisesRegex(GateError, "missing/unreadable actual"):
            validate_tutorials(DEFAULT_REFERENCE_DIR, actual, "test")

        actual = self._materialize_actuals("extra")
        (actual / "extra-tutorial_7_unexpected.txt").write_text("unexpected\n")
        with self.assertRaisesRegex(GateError, "actual case set mismatch"):
            validate_tutorials(DEFAULT_REFERENCE_DIR, actual, "extra")

    def test_run_rejects_prefix_before_model_or_output_work(self) -> None:
        output = self.root / "must-not-exist"
        with self.assertRaisesRegex(GateError, "actual prefix"):
            run_tutorials(self.root / "missing-model", output, "../unsafe")
        self.assertFalse(output.exists())

    def test_failure_report_contains_hashes_and_drift_counts(self) -> None:
        slots = discover_confidence_slots(TUPLE_REFERENCE)
        changed = _replace_once(TUPLE_REFERENCE, b"0.5,", b"0.6,")
        try:
            compare_case("synthetic", TUPLE_REFERENCE, changed, slots)
        except GateError as exc:
            self.assertIsNotNone(exc.report)
            report = exc.report
        else:
            self.fail("comparison unexpectedly passed")
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["confidence_difference_count"], 1)
        self.assertGreater(report["max_absolute_confidence_error"], 0)
        self.assertEqual(len(report["reference_sha256"]), 64)
        self.assertEqual(len(report["actual_sha256"]), 64)

    def test_manifest_is_json_and_records_fixed_tolerance(self) -> None:
        manifest = json.loads((DEFAULT_REFERENCE_DIR / "manifest.json").read_text())
        self.assertEqual(manifest["absolute_confidence_tolerance"], 1e-6)
        self.assertEqual(manifest["case_ids"], list(EXPECTED_CASE_IDS))


if __name__ == "__main__":
    unittest.main()
