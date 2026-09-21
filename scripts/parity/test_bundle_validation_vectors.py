#!/usr/bin/env python3
"""Model-free regression checks for checkpoint-specific oracle metadata."""

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gen_bundle_validation_vectors as generator


class OriginalTextTests(unittest.TestCase):
    def test_processor_punctuation_never_replaces_caller_text(self):
        for text, processed in (("", "."), ("東京。", "東京。."), ("Zoë", "Zoë.")):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                batch = SimpleNamespace(
                    original_texts=[processed], schema_tokens_list=[[]],
                    task_types=[[]], text_tokens=[[]],
                )
                case = {
                    "id": "raw_text", "category": "edge", "text": text,
                    "word_count": len(text.split()), "schema": {},
                }
                model = mock.Mock()
                model.extract.return_value = {}
                with ExitStack() as stack:
                    for name in ("install_hooks", "add_batch_arrays", "reconstruct_routed_states"):
                        stack.enter_context(mock.patch.object(generator, name))
                    stack.enter_context(mock.patch.object(generator, "build_schema", return_value={}))
                    stack.enter_context(mock.patch.object(generator, "build_expected_batch", return_value=batch))
                    stack.enter_context(mock.patch.object(generator, "query_metadata", return_value={}))
                    stack.enter_context(mock.patch.object(generator, "stages_metadata", return_value={}))
                    entry, ok = generator.generate_extraction_case(model, case, root, {})
                self.assertTrue(ok)
                self.assertEqual(entry["status"], "ok")
                metadata = json.loads((root / "raw_text.json").read_text())
                self.assertEqual(metadata["text"], text)
                self.assertEqual(metadata["original_text"], text)
                self.assertEqual(metadata["processor_text"], processed)
                self.assertEqual(model.extract.call_args.args[0], text)


if __name__ == "__main__":
    unittest.main()
