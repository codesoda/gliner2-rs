#!/usr/bin/env python3
"""Model-free readback policy tests; mocked transport, real local SHA-256 checks."""

import hashlib
import json
import os
import re
import sys
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("verify_external_consumer.sh")


@unittest.skipUnless(shutil.which("bash") and shutil.which("jq"), "requires bash and jq")
class ConsumerReadbackTests(unittest.TestCase):
    def run_readback(self, filename, etag, remote=b"synthetic bytes"):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = root / "local"
            local.mkdir()
            payload = b"synthetic bytes"
            (local / filename).write_bytes(payload)
            remote_file = root / "remote"
            remote_file.write_bytes(remote)
            source = SCRIPT.read_text()
            self.assertIn("\nsource_url=", source)
            helpers = source.split("\nsource_url=", 1)[0]
            harness = helpers + r'''
report_dir="$TEST_ROOT/report"
work_dir="$TEST_ROOT/work"
mkdir -p "$report_dir/remote-identity" "$work_dir"
remote_head() {
  printf 'HTTP/2 307\r\nx-linked-etag: "%s"\r\n\r\n' "$TEST_ETAG" > "$3"
}
small_get() {
  touch "$TEST_ROOT/get-called"
  cp "$TEST_ROOT/remote" "$2"
  : > "$2.headers"
}
verify_remote_file "$TEST_ROOT/local" "$TEST_FILENAME" owner/repo \
  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa "$TEST_FILENAME" \
  "$TEST_BYTES" "$TEST_SHA" test
'''
            environment = dict(os.environ, TEST_ROOT=str(root), TEST_ETAG=etag,
                               TEST_FILENAME=filename, TEST_BYTES=str(len(payload)),
                               TEST_SHA=hashlib.sha256(payload).hexdigest())
            process = subprocess.run(["bash", "-c", harness], env=environment,
                                     capture_output=True, text=True)
            receipt_path = root / "report" / "remote-identities.jsonl"
            receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
            return process, receipt, (root / "get-called").exists()

    def test_git_metadata_sha1_etag_uses_actual_get_sha256(self):
        process, receipt, fetched = self.run_readback("config.json", "b" * 40)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(fetched)
        self.assertEqual(receipt["method"], "get-sha256")
        self.assertEqual(receipt["sha256"], hashlib.sha256(b"synthetic bytes").hexdigest())

    def test_metadata_body_mismatch_fails_despite_valid_header(self):
        process, receipt, fetched = self.run_readback("config.json", "b" * 40, b"different bytes")
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("remote metadata SHA-256 mismatch", process.stderr)
        self.assertTrue(fetched)
        self.assertIsNone(receipt)

    def test_graph_sha1_etag_is_rejected_without_get_fallback(self):
        process, receipt, fetched = self.run_readback("encoder.onnx", "b" * 40)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("lacks LFS SHA-256", process.stderr)
        self.assertFalse(fetched)
        self.assertIsNone(receipt)

    def test_graph_lfs_sha256_is_verified_without_get(self):
        digest = hashlib.sha256(b"synthetic bytes").hexdigest()
        process, receipt, fetched = self.run_readback("encoder.onnx", digest)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertFalse(fetched)
        self.assertEqual(receipt["method"], "head-lfs-sha256")
        self.assertEqual(receipt["sha256"], digest)

    def test_graph_lfs_sha256_mismatch_is_rejected(self):
        process, receipt, fetched = self.run_readback("encoder.onnx", "b" * 64)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("mismatched LFS SHA-256", process.stderr)
        self.assertFalse(fetched)
        self.assertIsNone(receipt)


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("sandbox-exec"), "macOS sandbox test")
class PythonExecutionPolicyTests(unittest.TestCase):
    def test_absolute_lowercase_and_framework_python_names_are_denied(self):
        match = re.search(r"^python_policy='(.+)'$", SCRIPT.read_text(), re.MULTILINE)
        self.assertIsNotNone(match)
        policy = match.group(1)
        with tempfile.TemporaryDirectory() as temporary:
            for name in ("python3.12", "Python", "PYTHON3"):
                with self.subTest(name=name):
                    executable = Path(temporary) / name
                    shutil.copyfile("/bin/echo", executable)
                    executable.chmod(0o755)
                    process = subprocess.run(
                        ["sandbox-exec", "-p", policy, "/bin/sh", "-c", '"$1" forbidden', "sh", str(executable)],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(process.returncode, 126, process.stderr)
                    self.assertNotIn("forbidden", process.stdout)
            process = subprocess.run(
                ["sandbox-exec", "-p", policy, "/bin/echo", "allowed"],
                capture_output=True, text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(process.stdout.strip(), "allowed")


if __name__ == "__main__":
    unittest.main()
