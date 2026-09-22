#!/usr/bin/env python3
"""Generate compact boundary preprocessing oracles from pinned GLiNER2.

The upstream splitter implementations are invoked directly. This file only
supplies cases, reproduces `_collate_batch`'s terminal-punctuation boundary,
and converts Python code-point offsets to explicit UTF-8 byte offsets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "export"))
from common import GLINER2_COMMIT, assert_pinned_gliner2_installation  # noqa: E402


def normalize(text: str) -> tuple[str, bool]:
    # Pinned gliner2/processor.py `_collate_batch`, lines 523-527.
    if text and not text.endswith((".", "!", "?")):
        return text + ".", True
    if not text:
        return ".", True
    return text, False


def codepoint_to_byte(text: str, offset: int) -> int:
    return len(text[:offset].encode("utf-8"))


def cases() -> list[dict[str, Any]]:
    return [
        {"name": "empty", "text": ""},
        {"name": "punctuationless", "text": "Alpha βeta"},
        {"name": "already_punctuated", "text": "Done!"},
        {"name": "combining_mark", "text": "Cafe\u0301"},
        {"name": "cjk", "text": "東京"},
        {"name": "emoji", "text": "wave 😀"},
        {
            "name": "unicode_ignorecase_email_and_handle",
            "text": "İıſK x@İı.ſK @İıſK",
        },
        {
            "name": "url_absorbs_synthetic_period",
            "text": "visit https://example.com/path",
        },
        {
            "name": "url_email_handle",
            "text": "WWW.Example.org mail A+B@example.COM @Handle?",
        },
        {
            "name": "python_control_whitespace",
            "text": "a\u001cb\u001dc\u001ed\u001fe",
        },
        {
            "name": "char_mode",
            "text": "ABC東京 Cafe\u0301 @Tag",
            "splitter": "char",
        },
        {
            "name": "cap_before_choice_prefix",
            "text": "one two three",
            "max_len": 2,
            "choice_prefix": ["(", "kind:", "Red", ")"],
        },
        {"name": "trailing_space_before_synthetic_period", "text": "tail "},
    ]


def build_case(definition: dict[str, Any], splitters: dict[str, Any]) -> dict[str, Any]:
    text = definition["text"]
    normalized, suffix_added = normalize(text)
    splitter_name = definition.get("splitter", "whitespace")
    max_len = definition.get("max_len", 4096)
    prefix = definition.get("choice_prefix", [])
    split = list(splitters[splitter_name](normalized, lower=True))[:max_len]

    offsets = []
    original_codepoints = len(text)
    for _token, start, end in split:
        safe_start = min(start, original_codepoints)
        safe_end = min(end, original_codepoints)
        if start >= original_codepoints:
            coverage = "synthetic_only"
        elif end > original_codepoints:
            coverage = "original_with_synthetic_suffix"
        else:
            coverage = "original"
        offsets.append(
            {
                "start": codepoint_to_byte(text, safe_start),
                "end": codepoint_to_byte(text, safe_end),
                "coverage": coverage,
            }
        )

    return {
        "name": definition["name"],
        "text": text,
        "splitter": splitter_name,
        "max_len": max_len,
        "choice_prefix": prefix,
        "choice_prefix_words": len(prefix),
        "normalized_text": normalized,
        "synthetic_suffix_added": suffix_added,
        "text_tokens": prefix + [token for token, _start, _end in split],
        "original_offsets_utf8": offsets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "fixtures" / "boundary-token-vectors.json",
    )
    args = parser.parse_args()

    assert_pinned_gliner2_installation()
    from gliner2.processing.word_splitter import (
        CharLevelSplitter,
        WhitespaceTokenSplitter,
    )

    splitters = {
        "whitespace": WhitespaceTokenSplitter(),
        "char": CharLevelSplitter(),
    }
    payload = {
        "format_version": 1,
        "oracle": "unmodified pinned GLiNER2 word splitters after _collate_batch punctuation normalization",
        "upstream_commit": GLINER2_COMMIT,
        "offset_unit": "utf8_byte",
        "cases": [build_case(case, splitters) for case in cases()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
