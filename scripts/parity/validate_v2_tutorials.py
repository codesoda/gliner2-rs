#!/usr/bin/env python3
"""Fail-closed regression gate for the six GLiNER v2 tutorial outputs.

The checked-in references are captured ORT 1.20.0 output. Runtime timing lines
are removed only when the complete line has a known shape. Every other byte
must match except manifest-declared, typed confidence slots, which are compared
as finite f32 values with an absolute tolerance of 1e-6.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_REFERENCE_DIR = REPO_ROOT / "fixtures" / "v2-tutorials"
MANIFEST_NAME = "manifest.json"
ABSOLUTE_TOLERANCE = 1e-6
EXPECTED_CASE_IDS = (
    "1_classification",
    "2_ner",
    "3_json_extraction",
    "4_combined",
    "5_validator",
    "6_relation_extraction",
)
EXPECTED_REFERENCE_SHA256 = {
    "1_classification": "6d419dd4833d8f3ad02c55b387496396569c126106b76f9910d80271bbe20242",
    "2_ner": "51982dcfb71709747e8ff2c26f9d9a0a5ab79f9a9e46b69a01c6ca42a67ac2d4",
    "3_json_extraction": "2c1d38de6c8e0f6295a619ee2dfdbd75acb4e7295d026a2961ad559861f9502f",
    "4_combined": "b77ee5cb41bb642ba72ba4363b07e33a67a96991e2f42a29c43eb8b49c929902",
    "5_validator": "1aa77586e815f68884ccbc77808343efdea4fafff2e3794270a91d09202bfe69",
    "6_relation_extraction": "29ef8e1db814938cc69ddf1b32c3d10338f20f68140e3d95998e15b99e000e44",
}

_NUMBER = r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
_NUMBER_RE = re.compile(r"^" + _NUMBER + r"$")
_NAMED_CONFIDENCE_RE = re.compile(
    r"^(?P<indent> *)confidence: (?P<number>" + _NUMBER + r"),$"
)
_BARE_NUMBER_RE = re.compile(
    r"^(?P<indent> *)(?P<number>" + _NUMBER + r"),$"
)
_RUST_STRING_RE = r'"(?:[^"\\]|\\.)*"'
_NAMED_VARIANTS = {
    "Single": "label",
    "SingleWithConfidence": "label",
    "TextWithConfidence": "text",
    "TextWithConfidenceAndSpans": "text",
}
_DURATION = r"[0-9]+(?:\.[0-9]+)?(?:ns|us|µs|ms|s)"
_TIMING_RES = (
    re.compile(r"^model load took: " + _DURATION + r" \(onnx=[^\r\n]*\)$"),
    re.compile(r"^inference took: " + _DURATION + r"$"),
    re.compile(r"^batch inference took: " + _DURATION + r"$"),
    re.compile(r"^batch took: " + _DURATION + r"$"),
)
_PREFIX_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class GateError(AssertionError):
    """Validation failure with an optional machine-readable partial report."""

    def __init__(self, message: str, report: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.report = report


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _line_count(data: bytes) -> int:
    if not data:
        return 0
    return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)


def _body(line: str) -> str:
    if line.endswith("\r\n"):
        return line[:-2]
    if line.endswith("\n") or line.endswith("\r"):
        return line[:-1]
    return line


def _line_ending(line: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    if line.endswith("\r"):
        return "\r"
    return ""


def _split_utf8(data: bytes, description: str) -> List[str]:
    try:
        return data.decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError as exc:
        raise GateError("{} is not valid UTF-8: {}".format(description, exc))


def strip_known_timing_lines(data: bytes) -> Tuple[bytes, int]:
    """Remove only complete, anchored timing lines emitted by tutorials 1-6."""

    lines = _split_utf8(data, "actual output")
    kept = []
    removed = 0
    for line in lines:
        text = _body(line)
        if any(pattern.fullmatch(text) for pattern in _TIMING_RES):
            removed += 1
        else:
            kept.append(line)
    return "".join(kept).encode("utf-8"), removed


def _named_context(lines: Sequence[str], index: int) -> Optional[str]:
    match = _NAMED_CONFIDENCE_RE.fullmatch(_body(lines[index]))
    if match is None or index < 2:
        return None
    indent = match.group("indent")
    if len(indent) < 4:
        return None
    opener_indent = indent[:-4]
    field_match = re.fullmatch(
        re.escape(indent) + r"(?P<field>label|text): " + _RUST_STRING_RE + r",",
        _body(lines[index - 1]),
    )
    opener_match = re.fullmatch(
        re.escape(opener_indent)
        + r"(?:.+: )?(?P<variant>Single|SingleWithConfidence|TextWithConfidence|TextWithConfidenceAndSpans) \{",
        _body(lines[index - 2]),
    )
    if field_match is None or opener_match is None:
        return None
    variant = opener_match.group("variant")
    if field_match.group("field") != _NAMED_VARIANTS[variant]:
        return None
    if variant == "TextWithConfidenceAndSpans":
        if index + 2 >= len(lines):
            return None
        if re.fullmatch(re.escape(indent) + r"start: [0-9]+,", _body(lines[index + 1])) is None:
            return None
        if re.fullmatch(re.escape(indent) + r"end: [0-9]+,", _body(lines[index + 2])) is None:
            return None
    else:
        if index + 1 >= len(lines) or _body(lines[index + 1]) != opener_indent + "},":
            return None
    return variant


def _tuple_context(lines: Sequence[str], index: int) -> Optional[str]:
    match = _BARE_NUMBER_RE.fullmatch(_body(lines[index]))
    if match is None or index < 2 or index + 1 >= len(lines):
        return None
    indent = match.group("indent")
    if len(indent) < 12:
        return None
    tuple_indent = indent[:-4]
    variant_indent = indent[:-12]
    if re.fullmatch(re.escape(indent) + _RUST_STRING_RE + r",", _body(lines[index - 1])) is None:
        return None
    if _body(lines[index - 2]) != tuple_indent + "(":
        return None
    if _body(lines[index + 1]) != tuple_indent + "),":
        return None

    opener = re.compile(
        re.escape(variant_indent) + _RUST_STRING_RE + r": MultiWithConfidence\($"
    )
    opener_index = None
    for candidate in range(index - 3, -1, -1):
        text = _body(lines[candidate])
        if text == "-----------------":
            break
        if opener.fullmatch(text):
            opener_index = candidate
            break
    if opener_index is None:
        return None
    closing = variant_indent + "),"
    if not any(_body(lines[candidate]) == closing for candidate in range(index + 2, len(lines))):
        return None
    return "MultiWithConfidence"


def discover_confidence_slots(reference: bytes) -> List[Dict[str, Any]]:
    """Discover only typed Debug confidence positions for manifest creation/tests."""

    lines = _split_utf8(reference, "reference")
    slots = []
    for index, line in enumerate(lines):
        text = _body(line)
        match = _NAMED_CONFIDENCE_RE.fullmatch(text)
        kind = "named"
        variant = _named_context(lines, index) if match is not None else None
        if variant is None:
            match = _BARE_NUMBER_RE.fullmatch(text)
            kind = "tuple"
            variant = _tuple_context(lines, index) if match is not None else None
        if match is None or variant is None:
            continue
        start, end = match.span("number")
        slots.append(
            {
                "line": index + 1,
                "kind": kind,
                "variant": variant,
                "prefix": text[:start],
                "suffix": text[end:],
                "reference": match.group("number"),
            }
        )
    return slots


def _validate_slots(reference: bytes, slots: Any, case_id: str) -> Dict[int, Dict[str, Any]]:
    if not isinstance(slots, list):
        raise GateError("{} confidence_slots must be a list".format(case_id))
    lines = _split_utf8(reference, "{} reference".format(case_id))
    discovered = {slot["line"]: slot for slot in discover_confidence_slots(reference)}
    by_line = {}
    required_keys = {"line", "kind", "variant", "prefix", "suffix", "reference"}
    for slot in slots:
        if not isinstance(slot, dict) or set(slot) != required_keys:
            raise GateError("{} has malformed confidence slot metadata".format(case_id))
        line_number = slot.get("line")
        if not isinstance(line_number, int) or isinstance(line_number, bool):
            raise GateError("{} confidence slot line must be an integer".format(case_id))
        if line_number in by_line:
            raise GateError("{} duplicates confidence slot line {}".format(case_id, line_number))
        if line_number not in discovered or slot != discovered[line_number]:
            raise GateError(
                "{} line {} is not the declared typed confidence context".format(
                    case_id, line_number
                )
            )
        if line_number < 1 or line_number > len(lines):
            raise GateError("{} confidence slot line is out of range".format(case_id))
        by_line[line_number] = slot
    return by_line


def _f32(token: str, case_id: str, line_number: int) -> float:
    if _NUMBER_RE.fullmatch(token) is None:
        raise GateError(
            "{} line {} confidence is not a decimal number".format(case_id, line_number)
        )
    try:
        value = struct.unpack("!f", struct.pack("!f", float(token)))[0]
    except (OverflowError, ValueError, struct.error) as exc:
        raise GateError(
            "{} line {} confidence is not a finite f32: {}".format(
                case_id, line_number, exc
            )
        )
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise GateError(
            "{} line {} confidence must be finite and in [0, 1]".format(
                case_id, line_number
            )
        )
    return value


def _slot_token(line: str, slot: Dict[str, Any], case_id: str) -> str:
    text = _body(line)
    prefix = slot["prefix"]
    suffix = slot["suffix"]
    if not text.startswith(prefix) or not text.endswith(suffix):
        raise GateError(
            "{} line {} changed bytes outside its confidence number".format(
                case_id, slot["line"]
            )
        )
    end = len(text) - len(suffix) if suffix else len(text)
    token = text[len(prefix) : end]
    if prefix + token + suffix != text:
        raise GateError(
            "{} line {} has malformed confidence boundaries".format(
                case_id, slot["line"]
            )
        )
    return token


def compare_case(
    case_id: str,
    reference: bytes,
    actual: bytes,
    confidence_slots: List[Dict[str, Any]],
    tolerance: float = ABSOLUTE_TOLERANCE,
) -> Dict[str, Any]:
    """Compare one output without normalizing or rewriting runtime values."""

    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance < 0.0
        or tolerance > ABSOLUTE_TOLERANCE
    ):
        raise GateError("confidence tolerance must be finite and in [0, 1e-6]")
    normalized_actual, ignored_timing_lines = strip_known_timing_lines(actual)
    report: Dict[str, Any] = {
        "case_id": case_id,
        "reference_sha256": _sha256(reference),
        "actual_sha256": _sha256(actual),
        "normalized_actual_sha256": _sha256(normalized_actual),
        "reference_bytes": len(reference),
        "actual_bytes": len(actual),
        "reference_line_count": _line_count(reference),
        "normalized_actual_line_count": _line_count(normalized_actual),
        "ignored_timing_line_count": ignored_timing_lines,
        "confidence_slot_count": len(confidence_slots),
        "confidence_difference_count": 0,
        "max_absolute_confidence_error": 0.0,
        "confidence_differences": [],
    }
    try:
        slots = _validate_slots(reference, confidence_slots, case_id)
        reference_lines = _split_utf8(reference, "{} reference".format(case_id))
        actual_lines = _split_utf8(normalized_actual, "{} actual".format(case_id))
        if len(reference_lines) != len(actual_lines):
            raise GateError(
                "{} line count changed: reference={}, actual={}".format(
                    case_id, len(reference_lines), len(actual_lines)
                )
            )

        for index, (reference_line, actual_line) in enumerate(
            zip(reference_lines, actual_lines), 1
        ):
            if reference_line == actual_line:
                continue
            slot = slots.get(index)
            if slot is None:
                raise GateError(
                    "{} line {} changed outside a declared confidence slot".format(
                        case_id, index
                    )
                )
            if _line_ending(reference_line) != _line_ending(actual_line):
                raise GateError(
                    "{} line {} changed line-ending bytes outside its confidence number".format(
                        case_id, index
                    )
                )
            reference_token = _slot_token(reference_line, slot, case_id)
            actual_token = _slot_token(actual_line, slot, case_id)
            reference_value = _f32(reference_token, case_id, index)
            actual_value = _f32(actual_token, case_id, index)
            error = abs(actual_value - reference_value)
            difference = {
                "line": index,
                "variant": slot["variant"],
                "reference": reference_token,
                "actual": actual_token,
                "absolute_f32_error": error,
            }
            report["confidence_differences"].append(difference)
            report["confidence_difference_count"] += 1
            report["max_absolute_confidence_error"] = max(
                report["max_absolute_confidence_error"], error
            )
            if error > tolerance:
                raise GateError(
                    "{} line {} confidence error {} exceeds {}".format(
                        case_id, index, error, tolerance
                    )
                )
        report["status"] = "pass"
        return report
    except GateError as exc:
        report["status"] = "fail"
        report["error"] = str(exc)
        raise GateError(str(exc), report) from exc


def _load_manifest(reference_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    manifest_path = reference_dir / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError("cannot read reference manifest {}: {}".format(manifest_path, exc))
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
        raise GateError("reference manifest format_version must be 1")
    if manifest.get("case_ids") != list(EXPECTED_CASE_IDS):
        raise GateError("reference manifest must contain the exact six canonical case IDs in order")
    if manifest.get("absolute_confidence_tolerance") != ABSOLUTE_TOLERANCE:
        raise GateError("reference manifest confidence tolerance must remain exactly 1e-6")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != len(EXPECTED_CASE_IDS):
        raise GateError("reference manifest must contain exactly six entries")
    by_id = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("case_id"), str):
            raise GateError("reference manifest contains a malformed entry")
        case_id = entry["case_id"]
        if case_id in by_id:
            raise GateError("duplicate reference manifest case ID {}".format(case_id))
        by_id[case_id] = entry
    if list(by_id) != list(EXPECTED_CASE_IDS):
        raise GateError("reference manifest entry IDs/order do not match canonical cases")
    return manifest, by_id


def _validate_prefix(actual_prefix: str) -> None:
    if not isinstance(actual_prefix, str) or not _PREFIX_RE.fullmatch(actual_prefix):
        raise GateError("actual prefix must contain only letters, digits, '.', '_' or '-'")


def validate_tutorials(
    reference_dir: Path, actual_dir: Path, actual_prefix: str
) -> Dict[str, Any]:
    """Validate all six cases and return a JSON-serializable report."""

    reference_dir = Path(reference_dir)
    actual_dir = Path(actual_dir)
    _validate_prefix(actual_prefix)
    manifest, entries = _load_manifest(reference_dir)
    expected_reference_files = {MANIFEST_NAME}
    expected_actual_files = set()
    aggregate: Dict[str, Any] = {
        "status": "fail",
        "absolute_confidence_tolerance": ABSOLUTE_TOLERANCE,
        "case_count": len(EXPECTED_CASE_IDS),
        "confidence_difference_count": 0,
        "max_absolute_confidence_error": 0.0,
        "reference_manifest_sha256": _sha256(
            (reference_dir / MANIFEST_NAME).read_bytes()
        ),
        "cases": [],
    }
    try:
        for case_id in EXPECTED_CASE_IDS:
            entry = entries[case_id]
            reference_name = "tutorial_{}.txt".format(case_id)
            if entry.get("reference") != reference_name:
                raise GateError("{} reference filename is not canonical".format(case_id))
            expected_reference_files.add(reference_name)
            actual_name = "{}-{}".format(actual_prefix, reference_name)
            expected_actual_files.add(actual_name)
            reference_path = reference_dir / reference_name
            actual_path = actual_dir / actual_name
            try:
                reference = reference_path.read_bytes()
            except OSError as exc:
                raise GateError("missing/unreadable reference {}: {}".format(reference_path, exc))
            actual_hash = _sha256(reference)
            if entry.get("sha256") != EXPECTED_REFERENCE_SHA256[case_id]:
                raise GateError("{} manifest reference hash is not canonical".format(case_id))
            if actual_hash != entry.get("sha256"):
                raise GateError("{} reference hash mismatch".format(case_id))
            if entry.get("bytes") != len(reference):
                raise GateError("{} reference byte count mismatch".format(case_id))
            if entry.get("line_count") != _line_count(reference):
                raise GateError("{} reference line count mismatch".format(case_id))
            try:
                actual = actual_path.read_bytes()
            except OSError as exc:
                raise GateError("missing/unreadable actual {}: {}".format(actual_path, exc))
            case_report = compare_case(
                case_id,
                reference,
                actual,
                entry.get("confidence_slots"),
                ABSOLUTE_TOLERANCE,
            )
            aggregate["cases"].append(case_report)
            aggregate["confidence_difference_count"] += case_report[
                "confidence_difference_count"
            ]
            aggregate["max_absolute_confidence_error"] = max(
                aggregate["max_absolute_confidence_error"],
                case_report["max_absolute_confidence_error"],
            )

        actual_matches = {
            path.name
            for path in actual_dir.glob("{}-tutorial_*.txt".format(actual_prefix))
            if path.is_file()
        }
        if actual_matches != expected_actual_files:
            missing = sorted(expected_actual_files - actual_matches)
            extra = sorted(actual_matches - expected_actual_files)
            raise GateError(
                "actual case set mismatch; missing={}, extra={}".format(missing, extra)
            )
        reference_files = {
            path.name for path in reference_dir.iterdir() if path.is_file()
        }
        if reference_files != expected_reference_files:
            raise GateError("reference directory contains a missing or extra file")
        aggregate["status"] = "pass"
        aggregate["reference_total_bytes_including_manifest"] = sum(
            path.stat().st_size for path in reference_dir.iterdir() if path.is_file()
        )
        return aggregate
    except GateError as exc:
        if exc.report is not None:
            aggregate["cases"].append(exc.report)
            aggregate["confidence_difference_count"] += exc.report.get(
                "confidence_difference_count", 0
            )
            aggregate["max_absolute_confidence_error"] = max(
                aggregate["max_absolute_confidence_error"],
                exc.report.get("max_absolute_confidence_error", 0.0),
            )
        aggregate["error"] = str(exc)
        raise GateError(str(exc), aggregate) from exc


def run_tutorials(model: Path, actual_dir: Path, actual_prefix: str) -> None:
    """Build and execute every canonical tutorial, publishing only a complete run."""

    _validate_prefix(actual_prefix)
    model = Path(model).resolve()
    actual_dir = Path(actual_dir)
    if not model.is_dir():
        raise GateError("--model must be an existing ONNX bundle directory")
    required = ("encoder.onnx", "extractor_padded.onnx", "classifier.onnx")
    missing = [name for name in required if not (model / name).is_file()]
    metadata_dir = REPO_ROOT / "models" / model.name
    metadata_required = ("config.json", "tokenizer.json")
    missing.extend(
        "models/{}/{}".format(model.name, name)
        for name in metadata_required
        if not (metadata_dir / name).is_file()
    )
    if missing:
        raise GateError("model bundle is missing required artifacts: {}".format(missing))
    examples = ["tutorial_{}".format(case_id) for case_id in EXPECTED_CASE_IDS]

    with tempfile.TemporaryDirectory(prefix="v2-tutorial-run-") as temporary:
        staged = Path(temporary)
        for case_id, example in zip(EXPECTED_CASE_IDS, examples):
            command = [
                "cargo",
                "run",
                "--locked",
                "--quiet",
                "--example",
                example,
                "--",
                "--model",
                str(model),
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(REPO_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
            except OSError as exc:
                raise GateError("failed to build/run {}: {}".format(example, exc))
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode("utf-8", errors="replace")[-4000:]
                raise GateError(
                    "{} failed with exit {}: {}".format(
                        example, exc.returncode, stderr.strip()
                    )
                )
            if not completed.stdout:
                raise GateError("{} produced no stdout".format(example))
            (staged / "{}-tutorial_{}.txt".format(actual_prefix, case_id)).write_bytes(
                completed.stdout
            )
        actual_dir.mkdir(parents=True, exist_ok=True)
        for path in staged.iterdir():
            shutil.copyfile(str(path), str(actual_dir / path.name))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--actual-dir", type=Path, required=True)
    parser.add_argument("--actual-prefix", required=True)
    parser.add_argument(
        "--run", action="store_true", help="build and execute all six examples before comparison"
    )
    parser.add_argument("--model", type=Path, help="ONNX model bundle passed to every example")
    args = parser.parse_args(argv)
    if args.run and args.model is None:
        parser.error("--run requires --model")
    if not args.run and args.model is not None:
        parser.error("--model is only valid with --run")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.run:
            run_tutorials(args.model, args.actual_dir, args.actual_prefix)
        report = validate_tutorials(
            args.reference_dir, args.actual_dir, args.actual_prefix
        )
    except GateError as exc:
        report = exc.report or {"status": "fail", "error": str(exc)}
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
