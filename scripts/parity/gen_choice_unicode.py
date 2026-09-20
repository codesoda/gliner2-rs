#!/usr/bin/env python3
"""Generate the pinned Unicode tables used by boundary choice helpers.

Run with scripts/export/env/.venv/bin/python.  The output deliberately contains
only compact arithmetic runs, exceptional scalar mappings/expansions, and
property ranges; runtime inference never invokes Python or host Rust Unicode
algorithms.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import re
import sys
import unicodedata
from pathlib import Path

import _sre
from re._casefix import _EXTRA_CASES

PINNED_PYTHON = (3, 12, 7)
PINNED_UNICODE = "15.0.0"
MAX_SCALAR = 0x10FFFF
SURROGATE_START = 0xD800
SURROGATE_END = 0xDFFF


def scalars():
    for cp in range(MAX_SCALAR + 1):
        if not SURROGATE_START <= cp <= SURROGATE_END:
            yield cp


def mapping(method: str) -> dict[int, tuple[int, ...]]:
    result = {}
    for cp in scalars():
        mapped = tuple(map(ord, getattr(chr(cp), method)()))
        if mapped != (cp,):
            result[cp] = mapped
    return result


def ranges_for(predicate) -> list[tuple[int, int]]:
    ranges = []
    start = end = None
    for cp in scalars():
        if predicate(cp):
            if start is None:
                start = end = cp
            elif cp == end + 1:
                end = cp
            else:
                ranges.append((start, end))
                start = end = cp
        elif start is not None:
            ranges.append((start, end))
            start = end = None
    if start is not None:
        ranges.append((start, end))
    return ranges


class UnionFind:
    def __init__(self):
        self.parent = list(range(MAX_SCALAR + 1))

    def find(self, item: int) -> int:
        parent = self.parent[item]
        while parent != item:
            self.parent[item] = self.parent[parent]
            item = self.parent[item]
            parent = self.parent[item]
        return item

    def union(self, left: int, right: int):
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return
        if left < right:
            self.parent[right] = left
        else:
            self.parent[left] = right


def ignorecase_mapping() -> dict[int, tuple[int, ...]]:
    """Build equivalence classes used by CPython's Unicode literal matcher."""
    union = UnionFind()
    for cp in scalars():
        union.union(cp, _sre.unicode_tolower(cp))
    for lowered, extras in _EXTRA_CASES.items():
        for extra in extras:
            union.union(lowered, extra)
    result = {}
    for cp in scalars():
        representative = union.find(cp)
        if representative != cp:
            result[cp] = (representative,)
    return result


def compress_mapping(values: dict[int, tuple[int, ...]]):
    one = {cp: mapped[0] for cp, mapped in values.items() if len(mapped) == 1}
    expansions = {cp: mapped for cp, mapped in values.items() if len(mapped) != 1}
    keys = sorted(one)
    runs = []
    singles = []
    index = 0
    while index < len(keys):
        start = keys[index]
        delta = one[start] - start
        best_end = index
        best_step = 0
        for step in (1, 2):
            end = index
            while (
                end + 1 < len(keys)
                and keys[end + 1] == keys[end] + step
                and one[keys[end + 1]] - keys[end + 1] == delta
            ):
                end += 1
            if end - index > best_end - index:
                best_end, best_step = end, step
        if best_end - index + 1 >= 3:
            runs.append((start, keys[best_end], best_step, delta))
            index = best_end + 1
        else:
            singles.append((start, one[start]))
            index += 1
    return runs, singles, sorted(expansions.items())


def decode_compressed(cp, runs, singles, expansions):
    for start, end, step, delta in runs:
        if start <= cp <= end and (cp - start) % step == 0:
            return (cp + delta,)
    for source, target in singles:
        if source == cp:
            return (target,)
    for source, target in expansions:
        if source == cp:
            return target
    return (cp,)


def rust_array(name: str, rust_type: str, rows: list[str]) -> str:
    body = "\n".join(f"    {row}," for row in rows)
    return f"const {name}: &[{rust_type}] = &[\n{body}\n];\n"


def emit_mapping(prefix: str, values: dict[int, tuple[int, ...]], *, include_expansions=True):
    runs, singles, expansions = compress_mapping(values)
    flat = []
    expansion_rows = []
    for source, mapped in expansions:
        offset = len(flat)
        flat.extend(mapped)
        expansion_rows.append(f"Expansion {{ source: 0x{source:X}, offset: {offset}, len: {len(mapped)} }}")
    sections = [
        rust_array(
            f"{prefix}_RUNS",
            "DeltaRun",
            [
                f"DeltaRun {{ start: 0x{start:X}, end: 0x{end:X}, step: {step}, delta: {delta} }}"
                for start, end, step, delta in runs
            ],
        ),
        rust_array(
            f"{prefix}_SINGLES",
            "ScalarMap",
            [f"ScalarMap {{ source: 0x{source:X}, target: 0x{target:X} }}" for source, target in singles],
        ),
    ]
    if include_expansions:
        sections.extend([
            rust_array(f"{prefix}_EXPANSIONS", "Expansion", expansion_rows),
            rust_array(f"{prefix}_EXPANSION_DATA", "u32", [f"0x{cp:X}" for cp in flat]),
        ])
    else:
        assert not expansions
    return "\n".join(sections), (runs, singles, expansions)


def emit_ranges(name: str, ranges: list[tuple[int, int]]) -> str:
    return rust_array(name, "ScalarRange", [f"ScalarRange {{ start: 0x{start:X}, end: 0x{end:X} }}" for start, end in ranges])


def byte_spans(text: str, choice: str):
    starts = [0]
    for char in text:
        starts.append(starts[-1] + len(char.encode()))
    return [(starts[m.start()], starts[m.end()]) for m in re.finditer(rf"(?<!\w){re.escape(choice)}(?!\w)", text, re.IGNORECASE)]


LOWER_ORACLES = [
    ("ΟΣ", "ος"),
    ("ΟΣΑ", "οσα"),
    ("AΣ", "aς"),
    ("A\u0301Σ", "a\u0301ς"),
    ("Σ", "σ"),
    ("İΣ", "i\u0307ς"),
    ("𐐀Σ", "𐐨ς"),
    ("ᲐΣ", "აς"),
]

LITERAL_ORACLES = [
    ("İ I ı i", "i", [(0, 1), (2, 3), (4, 5), (6, 7)]),
    ("ſ S s", "s", [(0, 1), (2, 3), (4, 5)]),
    ("K K k", "k", [(0, 1), (2, 3), (4, 5)]),
    ("Σ σ ς", "Σ", [(0, 1), (2, 3), (4, 5)]),
    ("ß ss ẞ SS", "ß", [(0, 1), (5, 6)]),
    ("a\u0301a a\u0301 a", "a", [(0, 1), (2, 3), (4, 5), (7, 8)]),
    ("猫,猫咪,猫", "猫", [(0, 1), (5, 6)]),
    ("🙂 ok 🙂", "🙂", [(0, 1), (5, 6)]),
    (".a..", "", [(0, 0), (3, 3), (4, 4)]),
    ("a+b aab a+b", "a+b", [(0, 3), (8, 11)]),
]


def in_compressed_ranges(cp: int, ranges: list[tuple[int, int]]) -> bool:
    low = 0
    high = len(ranges)
    while low < high:
        middle = (low + high) // 2
        if ranges[middle][0] <= cp:
            low = middle + 1
        else:
            high = middle
    return low > 0 and cp <= ranges[low - 1][1]


def simulated_lower(text, compressed, cased_ranges, ignorable_ranges):
    chars = list(map(ord, text))
    output = []
    for index, cp in enumerate(chars):
        if cp == 0x03A3:
            before = index - 1
            while before >= 0 and in_compressed_ranges(chars[before], ignorable_ranges):
                before -= 1
            preceded = before >= 0 and in_compressed_ranges(chars[before], cased_ranges)
            after = index + 1
            while after < len(chars) and in_compressed_ranges(chars[after], ignorable_ranges):
                after += 1
            followed = after < len(chars) and in_compressed_ranges(chars[after], cased_ranges)
            if preceded and not followed:
                output.append(0x03C2)
                continue
        output.extend(decode_compressed(cp, *compressed))
    return "".join(map(chr, output))


def simulated_literal_spans(text, choice, ignorecase, word_ranges):
    text_chars = list(map(ord, text))
    choice_chars = list(map(ord, choice))
    result = []
    cursor = 0
    while cursor <= len(text_chars):
        end = cursor + len(choice_chars)
        if end > len(text_chars):
            break
        left = cursor == 0 or not in_compressed_ranges(text_chars[cursor - 1], word_ranges)
        right = end == len(text_chars) or not in_compressed_ranges(text_chars[end], word_ranges)
        equal = all(
            decode_compressed(actual, *ignorecase)[0]
            == decode_compressed(expected, *ignorecase)[0]
            for actual, expected in zip(text_chars[cursor:end], choice_chars)
        )
        if left and right and equal:
            result.append((cursor, end))
            cursor += max(len(choice_chars), 1)
        else:
            cursor += 1
    return result


def independent_string_self_check(lower, ignorecase, word_ranges, cased_ranges, ignorable_ranges):
    digest = hashlib.sha256()
    for text, expected in LOWER_ORACLES:
        assert text.lower() == expected, (text, text.lower(), expected)
        actual = simulated_lower(text, lower, cased_ranges, ignorable_ranges)
        assert actual == expected, (text, actual, expected)
        digest.update(repr((text, expected)).encode())
    for text, choice, expected in LITERAL_ORACLES:
        python_spans = [
            (match.start(), match.end())
            for match in re.finditer(rf"(?<!\w){re.escape(choice)}(?!\w)", text, re.IGNORECASE)
        ]
        assert python_spans == expected, (text, choice, python_spans, expected)
        actual = simulated_literal_spans(text, choice, ignorecase, word_ranges)
        assert actual == expected, (text, choice, actual, expected)
        digest.update(repr((text, choice, expected, byte_spans(text, choice))).encode())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    default = Path(__file__).resolve().parents[2] / "src" / "boundary" / "choice_unicode.rs"
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()

    actual_python = sys.version_info[:3]
    if actual_python != PINNED_PYTHON:
        raise SystemExit(f"requires Python {PINNED_PYTHON}, got {actual_python}")
    if unicodedata.unidata_version != PINNED_UNICODE:
        raise SystemExit(f"requires Unicode {PINNED_UNICODE}, got {unicodedata.unidata_version}")

    is_cased = ctypes.pythonapi._PyUnicode_IsCased
    is_cased.argtypes = [ctypes.c_uint32]
    is_cased.restype = ctypes.c_int
    is_case_ignorable = ctypes.pythonapi._PyUnicode_IsCaseIgnorable
    is_case_ignorable.argtypes = [ctypes.c_uint32]
    is_case_ignorable.restype = ctypes.c_int

    lower = mapping("lower")
    casefold = mapping("casefold")
    ignorecase = ignorecase_mapping()
    word_ranges = ranges_for(lambda cp: chr(cp).isalnum() or cp == ord("_"))
    cased_ranges = ranges_for(lambda cp: bool(is_cased(cp)))
    ignorable_ranges = ranges_for(lambda cp: bool(is_case_ignorable(cp)))

    lower_text, lower_compressed = emit_mapping("LOWER", lower)
    casefold_text, casefold_compressed = emit_mapping("CASEFOLD", casefold)
    ignorecase_text, ignorecase_compressed = emit_mapping(
        "IGNORECASE", ignorecase, include_expansions=False
    )

    # Exhaustive generator-side verification catches compressor defects.  The
    # source values themselves come directly from the pinned interpreter.
    for cp in scalars():
        assert decode_compressed(cp, *lower_compressed) == lower.get(cp, (cp,))
        assert decode_compressed(cp, *casefold_compressed) == casefold.get(cp, (cp,))
        assert decode_compressed(cp, *ignorecase_compressed) == ignorecase.get(cp, (cp,))
        assert in_compressed_ranges(cp, word_ranges) == (chr(cp).isalnum() or cp == ord("_"))
        assert in_compressed_ranges(cp, cased_ranges) == bool(is_cased(cp))
        assert in_compressed_ranges(cp, ignorable_ranges) == bool(is_case_ignorable(cp))

    oracle_digest = independent_string_self_check(
        lower_compressed,
        ignorecase_compressed,
        word_ranges,
        cased_ranges,
        ignorable_ranges,
    )
    source = f'''//! Generated Unicode 15.0.0 data for boundary choice matching.
//!
//! Generated by `scripts/parity/gen_choice_unicode.py` with CPython 3.12.7
//! (`unicodedata` 15.0.0). Do not hand edit. The generator exhaustively checks
//! every non-surrogate scalar after compression and fixed string/literal
//! oracles. See `THIRD-PARTY-NOTICES.md` for source-data notices and licenses.
//! Oracle digest: `{oracle_digest}`.

#[derive(Clone, Copy)]
struct DeltaRun {{
    start: u32,
    end: u32,
    step: u8,
    delta: i32,
}}

#[derive(Clone, Copy)]
struct ScalarMap {{
    source: u32,
    target: u32,
}}

#[derive(Clone, Copy)]
struct Expansion {{
    source: u32,
    offset: u32,
    len: u8,
}}

#[derive(Clone, Copy)]
struct ScalarRange {{
    start: u32,
    end: u32,
}}

pub(super) fn lower(value: &str) -> String {{
    let chars: Vec<char> = value.chars().collect();
    let mut result = String::with_capacity(value.len());
    for (index, &character) in chars.iter().enumerate() {{
        // Final_Sigma is the only locale-independent contextual lower rule.
        if character == '\\u{{03A3}}' && is_final_sigma(&chars, index) {{
            result.push('\\u{{03C2}}');
        }} else {{
            append_mapping(character as u32, LOWER_RUNS, LOWER_SINGLES, LOWER_EXPANSIONS, LOWER_EXPANSION_DATA, &mut result);
        }}
    }}
    result
}}

pub(super) fn casefold(value: &str) -> String {{
    let mut result = String::with_capacity(value.len());
    for character in value.chars() {{
        append_mapping(character as u32, CASEFOLD_RUNS, CASEFOLD_SINGLES, CASEFOLD_EXPANSIONS, CASEFOLD_EXPANSION_DATA, &mut result);
    }}
    result
}}

pub(super) fn ignorecase_equal(left: char, right: char) -> bool {{
    canonical(left as u32) == canonical(right as u32)
}}

pub(super) fn is_word(character: char) -> bool {{
    in_ranges(character as u32, WORD_RANGES)
}}

fn is_final_sigma(chars: &[char], index: usize) -> bool {{
    let preceded_by_cased = chars[..index]
        .iter()
        .rev()
        .find(|character| !in_ranges(**character as u32, CASE_IGNORABLE_RANGES))
        .is_some_and(|character| in_ranges(*character as u32, CASED_RANGES));
    let followed_by_cased = chars[index + 1..]
        .iter()
        .find(|character| !in_ranges(**character as u32, CASE_IGNORABLE_RANGES))
        .is_some_and(|character| in_ranges(*character as u32, CASED_RANGES));
    preceded_by_cased && !followed_by_cased
}}

fn canonical(scalar: u32) -> u32 {{
    mapped_scalar(scalar, IGNORECASE_RUNS, IGNORECASE_SINGLES).unwrap_or(scalar)
}}

fn append_mapping(
    scalar: u32,
    runs: &[DeltaRun],
    singles: &[ScalarMap],
    expansions: &[Expansion],
    expansion_data: &[u32],
    output: &mut String,
) {{
    if let Ok(index) = expansions.binary_search_by_key(&scalar, |entry| entry.source) {{
        let entry = expansions[index];
        let start = entry.offset as usize;
        let end = start + entry.len as usize;
        for &mapped in &expansion_data[start..end] {{
            output.push(char::from_u32(mapped).expect("generated mapping is a Unicode scalar"));
        }}
    }} else if let Some(mapped) = mapped_scalar(scalar, runs, singles) {{
        output.push(char::from_u32(mapped).expect("generated mapping is a Unicode scalar"));
    }} else {{
        output.push(char::from_u32(scalar).expect("input char is a Unicode scalar"));
    }}
}}

fn mapped_scalar(scalar: u32, runs: &[DeltaRun], singles: &[ScalarMap]) -> Option<u32> {{
    let insertion = runs.partition_point(|run| run.start <= scalar);
    if insertion > 0 {{
        let run = runs[insertion - 1];
        if scalar <= run.end && (scalar - run.start).is_multiple_of(u32::from(run.step)) {{
            return Some((i64::from(scalar) + i64::from(run.delta)) as u32);
        }}
    }}
    singles
        .binary_search_by_key(&scalar, |entry| entry.source)
        .ok()
        .map(|index| singles[index].target)
}}

fn in_ranges(scalar: u32, ranges: &[ScalarRange]) -> bool {{
    let insertion = ranges.partition_point(|range| range.start <= scalar);
    insertion > 0 && scalar <= ranges[insertion - 1].end
}}

{lower_text}
{casefold_text}
{ignorecase_text}
{emit_ranges("WORD_RANGES", word_ranges)}
{emit_ranges("CASED_RANGES", cased_ranges)}
{emit_ranges("CASE_IGNORABLE_RANGES", ignorable_ranges)}
'''
    source = source.rstrip() + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source)
    print(
        f"wrote {args.output}: {len(source.encode())} bytes; "
        f"lower={len(lower)}, casefold={len(casefold)}, ignorecase={len(ignorecase)}, "
        f"word_ranges={len(word_ranges)}, cased_ranges={len(cased_ranges)}, "
        f"case_ignorable_ranges={len(ignorable_ranges)}"
    )


if __name__ == "__main__":
    main()
