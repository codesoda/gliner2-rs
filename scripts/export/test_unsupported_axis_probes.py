#!/usr/bin/env python3
"""Model-free regressions for unsupported-axis diagnostic opt-in policy.

Execute the real parser and main's diagnostic statements via AST extraction,
without importing model/runtime dependencies. All inference and subprocess
entry points are mocks; these tests never launch a native crash probe.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent
VALIDATORS = (
    "validate_boundary_marginals.py",
    "validate_boundary_scorer.py",
    "validate_boundary_explicit_scorer.py",
)


def source_tree(filename: str) -> ast.Module:
    return ast.parse((ROOT / filename).read_text(), filename=filename)


def execute(nodes: list[ast.stmt], namespace: dict) -> None:
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "<validator-policy>", "exec"), namespace)


def policy_block(filename: str, start: str) -> list[ast.stmt]:
    main = next(
        node for node in source_tree(filename).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    def assigns(node: ast.stmt, name: str) -> bool:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        return any(isinstance(target, ast.Name) and target.id == name for target in targets)

    first = next(i for i, node in enumerate(main.body) if assigns(node, start))
    last = next(i for i, node in enumerate(main.body) if assigns(node, "expected_counts"))
    return main.body[first:last]


def namespace(enabled: bool = False) -> dict:
    return {
        "args": SimpleNamespace(
            probe_unsupported_axes=enabled, seed=1729, atol=1e-4, rtol=1e-3,
            strict_prefix=False,
        ),
        "model": SimpleNamespace(hidden_size=8),
        "wrapper": object(),
        "session": object(),
        "onnx_path": Path("unused.onnx"),
        "counts": Counter(),
        "synthetic_stats": {},
        "raw_original_failures": Counter(),
        "tolerance_failures": [],
        "synthetic_inputs": Mock(return_value={}),
        "synthetic_case": Mock(return_value={}),
        "zero_inputs": Mock(return_value={}),
        "live_outputs": Mock(return_value=({}, {})),
        "run_onnx": Mock(return_value={}),
        "compare_onnx_output_sets": Mock(),
        "isolated_zero_probe": Mock(return_value="mock-probed"),
        "subprocess": SimpleNamespace(run=Mock(return_value=SimpleNamespace(returncode=-11))),
        "tempfile": SimpleNamespace(
            TemporaryDirectory=Mock(side_effect=lambda: contextlib.nullcontext("/unused"))
        ),
        "np": SimpleNamespace(savez=Mock()),
        "Path": Path,
        "sys": sys,
    }


class UnsupportedAxisProbeTests(unittest.TestCase):
    def test_cli_defaults_opt_in_and_warning(self) -> None:
        for filename in VALIDATORS:
            with self.subTest(validator=filename):
                parser = next(
                    node for node in source_tree(filename).body
                    if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
                )
                env = {"argparse": argparse, "DEFAULT_MODEL_DIR": Path("unused")}
                execute([parser], env)
                with patch.object(sys, "argv", [filename]):
                    self.assertIs(env["parse_args"]().probe_unsupported_axes, False)
                with patch.object(sys, "argv", [filename, "--probe-unsupported-axes"]):
                    self.assertIs(env["parse_args"]().probe_unsupported_axes, True)
                output = io.StringIO()
                with patch.object(sys, "argv", [filename, "--help"]):
                    with contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
                        env["parse_args"]()
                self.assertIn("OS crash dialog", " ".join(output.getvalue().split()))

    def test_marginals_default_skips_l0_but_keeps_supported_q0(self) -> None:
        env = namespace()
        execute(policy_block(VALIDATORS[0], "q0_inputs"), env)
        env["subprocess"].run.assert_not_called()
        env["tempfile"].TemporaryDirectory.assert_not_called()
        env["run_onnx"].assert_called_once()
        env["compare_onnx_output_sets"].assert_called_once()
        self.assertEqual(env["compare_onnx_output_sets"].call_args.args[0], "synthetic-l2-q0")
        self.assertEqual(env["counts"]["synthetic_q0_cases"], 1)
        self.assertEqual(env["l0_status"]["isolated_probe_status"], "not-run-unsupported-contract")
        self.assertIsNone(env["l0_status"]["isolated_probe_returncode"])

    def test_supported_q0_failure_is_not_suppressed(self) -> None:
        env = namespace()
        env["run_onnx"].side_effect = RuntimeError("valid-input failure")
        with self.assertRaisesRegex(RuntimeError, "valid-input failure"):
            execute(policy_block(VALIDATORS[0], "q0_inputs"), env)

    def test_marginals_opt_in_retains_historical_status(self) -> None:
        env = namespace(True)
        execute(policy_block(VALIDATORS[0], "q0_inputs"), env)
        env["subprocess"].run.assert_called_once()
        self.assertEqual(env["l0_status"]["isolated_probe_status"], "known-ort-1.20-sigsegv-observed")
        self.assertEqual(env["l0_status"]["isolated_probe_returncode"], -11)

    def test_marginals_unexpected_diagnostic_failure_still_raises(self) -> None:
        env = namespace(True)
        env["subprocess"].run.return_value.returncode = 1
        with self.assertRaisesRegex(AssertionError, "unexpected L=0"):
            execute(policy_block(VALIDATORS[0], "q0_inputs"), env)

    def test_shared_default_skips_both_probes_and_reports_zero_counts(self) -> None:
        env = namespace()
        execute(policy_block(VALIDATORS[1], "zero_status"), env)
        env["isolated_zero_probe"].assert_not_called()
        env["subprocess"].run.assert_not_called()
        env["synthetic_inputs"].assert_not_called()
        for zero in ("q0", "c0"):
            self.assertEqual(env["zero_status"][zero], "not-run-caller-bypass")
            self.assertEqual(dict(env["counts"])[f"synthetic_{zero}_probes"], 0)

    def test_shared_opt_in_counts_actual_probes_only(self) -> None:
        for upstream_rejects in (False, True):
            with self.subTest(upstream_rejects=upstream_rejects):
                env = namespace(True)
                if upstream_rejects:
                    env["live_outputs"].side_effect = ValueError("unsupported")
                execute(policy_block(VALIDATORS[1], "zero_status"), env)
                self.assertEqual(
                    [call.args[0] for call in env["synthetic_inputs"].call_args_list],
                    [2029, 2030],
                )
                self.assertEqual(env["isolated_zero_probe"].call_count, 0 if upstream_rejects else 2)
                for zero in ("q0", "c0"):
                    self.assertEqual(env["counts"][f"synthetic_{zero}_probes"], 0 if upstream_rejects else 1)

    def test_explicit_default_skips_all_probes(self) -> None:
        env = namespace()
        execute(policy_block(VALIDATORS[2], "zero_status"), env)
        env["isolated_zero_probe"].assert_not_called()
        env["subprocess"].run.assert_not_called()
        env["synthetic_case"].assert_not_called()
        env["zero_inputs"].assert_not_called()
        self.assertEqual(env["counts"]["isolated_zero_dimension_probes"], 0)
        for zero in ("q0", "c0", "l0"):
            status = env["zero_status"][zero]
            self.assertEqual(status["contract"], "rejected-before-ORT")
            self.assertEqual(status["isolated_status"], "not-run-unsupported-contract")
            self.assertIsNone(status["isolated_returncode"])

    def test_explicit_opt_in_runs_three_mock_probes(self) -> None:
        env = namespace(True)
        execute(policy_block(VALIDATORS[2], "zero_status"), env)
        self.assertEqual(env["isolated_zero_probe"].call_count, 3)
        self.assertEqual(env["counts"]["isolated_zero_dimension_probes"], 3)
        self.assertEqual(list(env["zero_status"]), ["q0", "c0", "l0"])


if __name__ == "__main__":
    unittest.main()
