"""Tiny-only experiment: python -m unittest discover -s scripts/export -p test_boundary_reductions.py -v.

Uses the existing export venv. No checkpoints are loaded. Optional saved logits
are read from BOUNDARY_REDUCTION_FIXTURES (default /tmp/gliner25-work/m7-validation-fixtures).
Tiny graphs and a detailed report go to BOUNDARY_REDUCTION_EXPERIMENT_DIR
(default /tmp/gliner25-work/m7-prefix-experiment). Never runs ORT with a zero axis.
The source .sum parity assertion is AArch64-specific; the independent literal
fp32 reference and ONNX parity tests are portable.
"""

import json
import math
import os
from pathlib import Path
import platform
import re
import unittest

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch

from boundary_reductions import AArch64SumLastAxis, _level_step, sum_last_axis


LENGTHS = (1, 3, 4, 15, 16, 255, 256, 257, 1001, 2001, 3001, 4095, 4096, 4097,
           65535, 65536, 65537)
IS_AARCH64 = platform.machine().lower() in ("arm64", "aarch64")
EXPERIMENT = Path(os.environ.get(
    "BOUNDARY_REDUCTION_EXPERIMENT_DIR", "/tmp/gliner25-work/m7-prefix-experiment"))
FIXTURES = Path(os.environ.get(
    "BOUNDARY_REDUCTION_FIXTURES", "/tmp/gliner25-work/m7-validation-fixtures"))


def literal_numpy_sum(values):
    """Independent literal translation of pool.rs, vectorized only over B/Q."""
    batch, queries, length = values.shape
    vectors = length // 4
    size = vectors // 4
    power = max(4, max(0, size - 1).bit_length() // 4)
    step = 1 << power
    accum = np.zeros((4, 4, batch, queries, 4), dtype=np.float32)
    index = 0
    while index + step <= size:
        for _ in range(step):
            for partial in range(4):
                offset = (index * 4 + partial) * 4
                accum[0, partial] += values[:, :, offset:offset + 4]
            index += 1
        for level in range(1, 4):
            for partial in range(4):
                accum[level, partial] += accum[level - 1, partial]
                accum[level - 1, partial].fill(0)
            if index & ((step - 1) << (level * power)):
                break
    while index < size:
        for partial in range(4):
            offset = (index * 4 + partial) * 4
            accum[0, partial] += values[:, :, offset:offset + 4]
        index += 1
    for level in range(1, 4):
        for partial in range(4):
            accum[0, partial] += accum[level, partial]
    for vector in range(size * 4, vectors):
        accum[0, 0] += values[:, :, vector * 4:vector * 4 + 4]
    for partial in range(1, 4):
        accum[0, 0] += accum[0, partial]
    total = np.zeros((batch, queries, 1), dtype=np.float32)
    for element in range(vectors * 4, length):
        total += values[:, :, element:element + 1]
    for lane in range(4):
        total += accum[0, 0, :, :, lane:lane + 1]
    return total


class PrefixSuffix(torch.nn.Module):
    """Test-only original suffix; the experimental variant replaces ONLY sum."""

    def __init__(self, source_order):
        super().__init__()
        self.source_order = source_order

    def forward(self, values, keep):
        total = sum_last_axis(values) if self.source_order else values.sum(-1, keepdim=True)
        count = keep.sum(-1, keepdim=True).clamp_min(1)
        mean = (total / count).detach()
        centered = (values - mean) * keep.to(values.dtype)
        zeros = torch.zeros((values.shape[0], values.shape[1], 1),
                            dtype=torch.float32, device=values.device)
        prefix = torch.cat((zeros, centered.cumsum(dim=-1)), dim=-1)
        return total, mean, prefix


def assert_bits(actual, expected):
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def stats(actual, expected):
    # Diagnostic arithmetic only; these arrays never enter the exported graph.
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {"elements": int(actual.size),
            "different_bits": int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32))),
            "max_absolute_error": float(difference.max(initial=0))}


def audit_graph(graph):
    """Recursively reject non-fp32 floats/nonfinite constants, including Loop/If."""
    nodes = []
    floats = {onnx.TensorProto.FLOAT16, onnx.TensorProto.DOUBLE,
              onnx.TensorProto.BFLOAT16, onnx.TensorProto.FLOAT}

    def tensor(value):
        if value.data_type in floats:
            assert value.data_type == onnx.TensorProto.FLOAT
            assert np.isfinite(numpy_helper.to_array(value)).all()

    for initializer in graph.initializer:
        tensor(initializer)
    for info in list(graph.input) + list(graph.output) + list(graph.value_info):
        kind = info.type.tensor_type.elem_type
        assert kind not in floats or kind == onnx.TensorProto.FLOAT
    for node in graph.node:
        nodes.append(node.op_type)
        for attr in node.attribute:
            if node.op_type == "Cast" and attr.name == "to":
                assert attr.i not in floats or attr.i == onnx.TensorProto.FLOAT
            if attr.type == onnx.AttributeProto.TENSOR:
                tensor(attr.t)
            elif attr.type == onnx.AttributeProto.TENSORS:
                for value in attr.tensors:
                    tensor(value)
            elif attr.type == onnx.AttributeProto.FLOAT:
                assert math.isfinite(attr.f)
            elif attr.type == onnx.AttributeProto.FLOATS:
                assert all(math.isfinite(value) for value in attr.floats)
            elif attr.type == onnx.AttributeProto.GRAPH:
                nodes.extend(audit_graph(attr.g))
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for nested in attr.graphs:
                    nodes.extend(audit_graph(nested))
    return nodes


def export_tiny(module, name, suffix=False):
    path = EXPERIMENT / name
    sample = torch.zeros((2, 3, 17), dtype=torch.float32)
    args = (sample, torch.ones_like(sample, dtype=torch.bool)) if suffix else (sample,)
    axes = {"values": {0: "B", 1: "Q", 2: "L"}, "sum": {0: "B", 1: "Q"}}
    inputs, outputs = ["values"], ["sum"]
    if suffix:
        inputs.append("keep")
        outputs += ["mean", "prefix"]
        axes.update({"keep": {0: "B", 1: "Q", 2: "L"},
                     "mean": {0: "B", 1: "Q"},
                     "prefix": {0: "B", 1: "Q", 2: "L_plus_one"}})
    torch.onnx.export(module.eval(), args, str(path), input_names=inputs,
                      output_names=outputs, dynamic_axes=axes,
                      opset_version=17, dynamo=False)
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    ops = audit_graph(onnx.shape_inference.infer_shapes(model).graph)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    return session, {"bytes": path.stat().st_size,
                     "recursive_nodes": len(ops), "loops": ops.count("Loop"),
                     "ifs": ops.count("If"), "reducesums": ops.count("ReduceSum"),
                     "cumsums": ops.count("CumSum"), "finite_fp32_audit": True}


def positive_run(session, values, keep=None):
    # This is a safety precondition, not a graph workaround. No zero-axis ORT probes.
    assert values.ndim == 3 and all(axis > 0 for axis in values.shape)
    feeds = {"values": values}
    if keep is not None:
        feeds["keep"] = keep
    result = session.run(None, feeds)
    for value in result:
        assert value.dtype == np.float32 and np.isfinite(value).all()
    return result


def saved_cases():
    for model in ("small", "multi", "base"):
        for path in sorted((FIXTURES / f"gliner2.5-{model}-v1").rglob("*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                for key in sorted(archive.files):
                    match = re.fullmatch(r"(marginals_\d+)_inside_logits", key)
                    if match:
                        stem = match[1]
                        logits = archive[key]
                        keep = (archive[stem + "_text_mask"][:, None, :]
                                & archive[stem + "_query_mask"][:, :, None])
                        values = np.where(keep, logits, np.float32(0))
                        yield (model, str(path.relative_to(FIXTURES)), stem, values, keep,
                               archive[stem + "_inside_prefix_mean"],
                               archive[stem + "_inside_prefix"])


class BoundaryReductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        EXPERIMENT.mkdir(parents=True, exist_ok=True)
        cls.sum_session, sum_audit = export_tiny(AArch64SumLastAxis(), "sum.onnx")
        cls.old_session, old_audit = export_tiny(PrefixSuffix(False), "original_suffix.onnx", True)
        cls.new_session, new_audit = export_tiny(PrefixSuffix(True), "source_order_suffix.onnx", True)
        cls.report = {"scope": "Saved logits and tiny reduction/suffix ONLY; no full graph validation",
                      "platform": platform.machine(), "torch": torch.__version__,
                      "onnxruntime": ort.__version__, "export_sample_shape": [2, 3, 17],
                      "graphs": {"sum": sum_audit, "original_suffix": old_audit,
                                 "source_order_suffix": new_audit}}

    @classmethod
    def tearDownClass(cls):
        (EXPERIMENT / "report.json").write_text(json.dumps(cls.report, indent=2) + "\n")

    def test_graph_contract(self):
        audit = self.report["graphs"]["sum"]
        self.assertGreater(audit["loops"], 0)
        self.assertGreaterEqual(audit["ifs"], 2)
        self.assertEqual(audit["reducesums"], 0)
        self.assertLess(audit["bytes"], 100_000)
        self.assertEqual(self.report["graphs"]["original_suffix"]["cumsums"], 1)
        self.assertEqual(self.report["graphs"]["source_order_suffix"]["cumsums"], 1)

    def test_dynamic_level_power(self):
        # The first change from step 16 to 32 needs >8M tokens. Check integer
        # shape logic without allocating/running that many ORT loop iterations.
        for size in (0, 1, 15, 16, 255, 256, 4096, 2**19, 2**19 + 1,
                     2**20, 2**23, 2**23 + 1, 2**31 + 1):
            self.assertEqual(_level_step(size), 1 << max(4, max(0, size - 1).bit_length() // 4))
        self.assertEqual(_level_step(2**19), 16)
        self.assertEqual(_level_step(2**19 + 1), 32)

    def test_dynamic_positive_axes(self):
        rng = np.random.default_rng(731)
        rows = []
        for length in LENGTHS:
            for batch, queries in ((1, 1), (2, 3)):
                for distribution in ("normal", "mixed_exponents"):
                    with self.subTest(length=length, batch=batch, queries=queries,
                                      distribution=distribution):
                        values = rng.standard_normal((batch, queries, length), dtype=np.float32)
                        if distribution == "mixed_exponents":
                            values = np.ldexp(values, rng.integers(-15, 16, values.shape))
                        expected = literal_numpy_sum(values)
                        scripted = sum_last_axis(torch.from_numpy(values)).numpy()
                        actual = positive_run(self.sum_session, values)[0]
                        assert_bits(scripted, expected)
                        assert_bits(actual, expected)
                        if IS_AARCH64:
                            assert_bits(actual, torch.from_numpy(values).sum(-1, keepdim=True).numpy())
                        rows.append({"shape": list(values.shape), "distribution": distribution,
                                     "numpy_script_ort_bit_exact": True,
                                     "torch_sum_bit_exact": True if IS_AARCH64 else None})
        self.report["synthetic"] = rows

    def test_masked_suffix_positive_axes(self):
        rng = np.random.default_rng(182)
        for length in (1, 16, 257, 3001, 4097):
            logits = rng.standard_normal((2, 3, length), dtype=np.float32)
            keep = rng.random(logits.shape) > 0.3
            keep[0, 1, :] = False  # Invalid query, but all tensor axes positive.
            logits[~keep] = np.float32(-1e4)
            values = np.where(keep, logits, np.float32(0))
            total, mean, prefix = positive_run(self.new_session, values, keep)
            expected_sum = literal_numpy_sum(values)
            expected_mean = (torch.from_numpy(expected_sum)
                             / torch.from_numpy(keep).sum(-1, keepdim=True).clamp_min(1)).numpy()
            assert_bits(total, expected_sum)
            assert_bits(mean, expected_mean)
            assert_bits(prefix[0, 1], np.zeros(length + 1, dtype=np.float32))
            if IS_AARCH64:
                assert_bits(total, torch.from_numpy(values).sum(-1, keepdim=True).numpy())
        self.report["synthetic_masks"] = {
            "lengths": [1, 16, 257, 3001, 4097],
            "shape_prefix": [2, 3], "includes_all_masked_query": True,
            "sum_and_mean_bit_exact": True}

    def test_zero_axes_source_only(self):
        for shape in ((0, 2, 17), (2, 0, 257), (2, 3, 0), (0, 0, 0)):
            values = np.zeros(shape, dtype=np.float32)
            actual = sum_last_axis(torch.from_numpy(values)).numpy()
            self.assertEqual(actual.shape, (*shape[:2], 1))
            assert_bits(actual, literal_numpy_sum(values))
            assert_bits(actual, torch.from_numpy(values).sum(-1, keepdim=True).numpy())

    def test_source_input_contract(self):
        for values in (torch.zeros(2, 3), torch.zeros(1, 2, 3, dtype=torch.float64)):
            with self.assertRaises(torch.jit.Error):
                sum_last_axis(values)

    def test_saved_masked_logits(self):
        cases = list(saved_cases())
        if not cases:
            self.skipTest(f"optional saved fixtures absent: {FIXTURES}")
        self.assertEqual({case[0] for case in cases}, {"small", "multi", "base"})
        rows = []
        for model, path, stem, values, keep, saved_mean, saved_prefix in cases:
            with self.subTest(path=path, stem=stem):
                expected_sum = literal_numpy_sum(values)
                assert_bits(sum_last_axis(torch.from_numpy(values)).numpy(), expected_sum)
                source_sum, source_mean, source_prefix = (
                    value.numpy() for value in PrefixSuffix(False)(
                        torch.from_numpy(values), torch.from_numpy(keep)))
                if IS_AARCH64:
                    assert_bits(expected_sum, source_sum)
                    assert_bits(source_mean, saved_mean)
                    assert_bits(source_prefix, saved_prefix)
                row = {"model": model, "path": path, "stage": stem,
                       "shape": list(values.shape), "source_sum": stats(expected_sum, source_sum)}
                if not all(axis > 0 for axis in values.shape):
                    row["ort"] = "skipped: zero axis (source tested only)"
                    rows.append(row)
                    continue
                tiny_sum = positive_run(self.sum_session, values)[0]
                old_sum, old_mean, old_prefix = positive_run(self.old_session, values, keep)
                new_sum, new_mean, new_prefix = positive_run(self.new_session, values, keep)
                assert_bits(tiny_sum, expected_sum)
                assert_bits(new_sum, tiny_sum)
                # Isolate the sum correction with the *unchanged source cumsum*.
                count = torch.from_numpy(keep).sum(-1, keepdim=True).clamp_min(1)
                isolated_mean = torch.from_numpy(tiny_sum) / count
                centered = ((torch.from_numpy(values) - isolated_mean)
                            * torch.from_numpy(keep).to(torch.float32))
                isolated_prefix = torch.cat((
                    torch.zeros((*values.shape[:2], 1), dtype=torch.float32),
                    centered.cumsum(-1)), -1).numpy()
                if IS_AARCH64:
                    assert_bits(new_mean, saved_mean)
                    assert_bits(isolated_prefix, saved_prefix)
                row.update({"ort": "positive axes only", "sum": stats(tiny_sum, expected_sum),
                            "old_sum": stats(old_sum, source_sum),
                            "old_mean": stats(old_mean, saved_mean),
                            "new_mean": stats(new_mean, saved_mean),
                            "old_prefix": stats(old_prefix, saved_prefix),
                            "new_prefix": stats(new_prefix, saved_prefix),
                            "old_endpoint": stats(old_prefix[:, :, -1], saved_prefix[:, :, -1]),
                            "new_endpoint": stats(new_prefix[:, :, -1], saved_prefix[:, :, -1]),
                            "sum_plus_source_cumsum": stats(isolated_prefix, saved_prefix)})
                rows.append(row)
        self.report["saved_cases"] = rows
        self.report["summary"] = {}
        for model in ("small", "multi", "base"):
            eligible = [row for row in rows if row["model"] == model and "sum" in row]
            aggregate = {"stages": len(eligible),
                         "query_sums": sum(row["sum"]["elements"] for row in eligible)}
            for field in ("sum", "old_sum", "old_mean", "new_mean", "old_prefix", "new_prefix",
                          "old_endpoint", "new_endpoint", "sum_plus_source_cumsum"):
                aggregate[field] = {
                    "different_bits": sum(row[field]["different_bits"] for row in eligible),
                    "max_absolute_error": max((row[field]["max_absolute_error"] for row in eligible), default=0)}
            self.report["summary"][model] = aggregate


if __name__ == "__main__":
    unittest.main()
