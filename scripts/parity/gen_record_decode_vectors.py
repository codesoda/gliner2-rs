#!/usr/bin/env python3
"""Generate compact record-decoder vectors from pinned, unchanged GLiNER2."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "export"))
from common import GLINER2_COMMIT, assert_pinned_gliner2_installation  # noqa: E402


def f32(value: float) -> float:
    return float(torch.tensor(value, dtype=torch.float32).item())


def field(query_id: int, cardinality: str, *, exclusive: bool = False) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "cardinality": cardinality,
        "exclusive": exclusive,
    }


def case(
    name: str,
    mode: str,
    fields: list[dict[str, Any]],
    object_logits: list[float],
    assign_logits: list[list[list[float]]],
    field_spans: list[list[list[int]]],
    *,
    anchor_query_id: int | None = None,
    instance_seed: list[list[int] | None] | None = None,
    instance_spans: list[list[int] | None] | None = None,
    anchor_threshold: float = 0.5,
    field_threshold: float = 0.5,
    object_threshold: float = 0.5,
    temperature: float = 1.0,
) -> dict[str, Any]:
    instances = len(object_logits)
    return {
        "name": name,
        "mode": mode,
        "fields": fields,
        "anchor_query_id": anchor_query_id,
        "object_logits": object_logits,
        "assign_logits": assign_logits,
        "field_spans": field_spans,
        "instance_seed": instance_seed if instance_seed is not None else [None] * instances,
        "instance_spans": instance_spans if instance_spans is not None else [None] * instances,
        "anchor_threshold": anchor_threshold,
        "field_threshold": field_threshold,
        "object_threshold": object_threshold,
        "temperature": temperature,
    }


def definitions() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    cases.append(
        case(
            "exclusive_scalar_greedy_trap",
            "latent",
            [field(10, "required_one", exclusive=True)],
            [3.0, 2.0, 1.0],
            [[[-10.0, 5.0, 4.0, 0.0], [-10.0, 5.0, 0.0, 0.0], [-10.0, 0.0, 0.0, 5.0]]],
            [[ [10, 11], [20, 21], [30, 31] ]],
        )
    )
    cases.append(
        case(
            "exclusive_optional_absent",
            "latent",
            [field(11, "optional_one", exclusive=True)],
            [2.0, 1.0],
            [[[0.0, 5.0], [5.0, 4.0]]],
            [[[40, 41]]],
        )
    )
    cases.append(
        case(
            "exclusive_required_under_capacity",
            "latent",
            [field(12, "required_one", exclusive=True)],
            [3.0, 2.0, 1.0],
            [[[-10.0, 3.0], [-10.0, 2.0], [-10.0, 1.0]]],
            [[[50, 51]]],
        )
    )
    cases.append(
        case(
            "exclusive_list_first_selected_owns_ties",
            "anchorless",
            [field(13, "zero_or_more", exclusive=True)],
            [1.0, 0.5, 2.0],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            [[[60, 61], [62, 63]]],
        )
    )

    natural_count = 33
    natural_spans = [[index * 2, index * 2 + 1] for index in reversed(range(natural_count))]
    cases.append(
        case(
            "natural_33_instances_anchor_span_order",
            "natural",
            [field(20, "required_one")],
            [0.0] * natural_count,
            [[[0.0] * (natural_count + 1) for _ in range(natural_count)]],
            [natural_spans],
            anchor_query_id=20,
            instance_seed=[[0, index] for index in range(natural_count)],
            instance_spans=natural_spans,
        )
    )
    cases.append(
        case(
            "natural_nonanchor_fields_and_missing_seed",
            "natural",
            [
                field(30, "required_one"),
                field(5, "optional_one"),
                field(40, "one_or_more", exclusive=True),
            ],
            [2.0, 1.0],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 2.0, 1.0], [3.0, 2.0, 1.0]],
                [[0.0, 2.0, -1.0], [0.0, 1.0, 3.0]],
            ],
            [
                [[10, 11], [20, 21]],
                [[30, 31], [32, 33]],
                [[40, 41], [42, 43]],
            ],
            anchor_query_id=30,
            instance_seed=[[0, 0], None],
            instance_spans=[[10, 11], [20, 21]],
        )
    )
    cases.append(
        case(
            "selection_threshold_rejects_all_instances",
            "latent",
            [field(31, "required_one")],
            [-2.0, -3.0],
            [[[0.0, 1.0], [0.0, 2.0]]],
            [[[44, 45]]],
            anchor_threshold=0.9,
        )
    )
    cases.append(
        case(
            "latent_duplicate_dedup_and_first_key_order",
            "latent",
            [field(21, "required_one")],
            [1.0, 3.0, 2.0],
            [[[0.0, 5.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 5.0]]],
            [[[70, 71], [72, 73]]],
        )
    )
    cases.append(
        case(
            "anchorless_uses_object_threshold_and_candidate_order",
            "anchorless",
            [field(22, "zero_or_more")],
            [0.0],
            [[[8.0, 0.0, 1.0]]],
            [[[80, 81], [82, 83]]],
            anchor_threshold=0.99,
            object_threshold=0.5,
            field_threshold=0.5,
        )
    )
    cases.append(
        case(
            "no_instances",
            "latent",
            [field(23, "optional_one")],
            [],
            [[]],
            [[[90, 91]]],
        )
    )
    cases.append(
        case(
            "no_candidates_scalar_and_lists",
            "latent",
            [
                field(24, "required_one"),
                field(25, "zero_or_more"),
                field(26, "one_or_more"),
            ],
            [2.0],
            [[[0.0]], [[0.0]], [[0.0]]],
            [[], [], []],
        )
    )
    cases.append(
        case(
            "empty_fields_drop_selected_instances",
            "anchorless",
            [],
            [2.0, 1.0],
            [],
            [],
        )
    )
    equal_object = f32(torch.sigmoid(torch.tensor(0.5, dtype=torch.float32)).item())
    cases.append(
        case(
            "temperature_and_threshold_equality",
            "latent",
            [field(27, "zero_or_more")],
            [1.0],
            [[[9.0, 0.0]]],
            [[[100, 101]]],
            anchor_threshold=equal_object,
            field_threshold=0.5,
            temperature=2.0,
        )
    )
    cases.append(
        case(
            "one_or_more_does_not_invent_fallback",
            "anchorless",
            [field(28, "one_or_more")],
            [2.0],
            [[[0.0, -10.0]]],
            [[[110, 111]]],
        )
    )

    for width in (17, 33, 193):
        candidate_count = width - 1
        spans = [[1000 + candidate * 2, 1001 + candidate * 2] for candidate in range(candidate_count)]
        cases.append(
            case(
                f"nonexclusive_scalar_tie_width_{width}",
                "latent",
                [field(100 + width, "required_one")],
                [2.0],
                [[[0.0] * width]],
                [spans],
            )
        )
    return cases


def argsort_definitions(random_cases_per_width: int) -> list[dict[str, Any]]:
    """Capture the pinned CPU torch.argsort permutation, including unstable ties."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0x6A09E667)
    cases: list[dict[str, Any]] = []
    for width in (17, 33, 193):
        equal = torch.zeros(width, dtype=torch.float32)
        cases.append(
            {
                "name": f"all_equal_width_{width}",
                "values": equal.tolist(),
                "expected": torch.argsort(equal, descending=True).tolist(),
            }
        )
        for index in range(random_cases_per_width):
            if index % 2 == 0:
                # Heavy exact ties with positive, negative, and zero buckets.
                values = torch.randint(
                    -4,
                    5,
                    (width,),
                    generator=generator,
                    dtype=torch.int64,
                ).to(torch.float32)
                values /= 4.0
                kind = "quantized"
            else:
                # Mostly distinct values with deliberately repeated pivot-like
                # positions, exercising both comparison and equality paths.
                values = torch.randn(width, generator=generator, dtype=torch.float32)
                repeated = values[width // 3].clone()
                values[0] = repeated
                values[width // 2] = repeated
                values[-1] = repeated
                kind = "mixed"
            cases.append(
                {
                    "name": f"{kind}_seeded_{index}_width_{width}",
                    "values": values.tolist(),
                    "expected": torch.argsort(values, descending=True).tolist(),
                }
            )
    return cases


def expected_record(record) -> dict[str, Any]:
    return {
        "fields": [
            {"query_id": query_id, "spans": [list(span) for span in spans]}
            for query_id, spans in sorted(record.fields.items())
        ],
        "field_scores": [
            {"query_id": query_id, "scores": scores}
            for query_id, scores in sorted(record.field_scores.items())
        ],
        "anchor_span": None if record.anchor_span is None else list(record.anchor_span),
        "score": record.score,
    }


def run_oracle(raw: dict[str, Any], RecordFieldSpec, RecordGroupOutput, RecordSpec, FieldCardinality, decode_group):
    specs = [
        RecordFieldSpec(
            query_id=entry["query_id"],
            name=f"field_{entry['query_id']}",
            role_index=index,
            cardinality=FieldCardinality(entry["cardinality"]),
            is_anchor=entry["query_id"] == raw["anchor_query_id"],
            exclusive=entry["exclusive"],
        )
        for index, entry in enumerate(raw["fields"])
    ]
    spec = RecordSpec(
        task_index=0,
        task_name=raw["name"],
        task_type="json_structures",
        mode=raw["mode"],
        fields=tuple(specs),
        anchor_query_id=raw["anchor_query_id"],
    )
    instances = len(raw["object_logits"])
    spans = [torch.tensor(item, dtype=torch.long).reshape(-1, 2) for item in raw["field_spans"]]
    assignments = []
    for field_index, values in enumerate(raw["assign_logits"]):
        assignments.append(
            torch.tensor(values, dtype=torch.float32).reshape(
                instances, len(raw["field_spans"][field_index]) + 1
            )
        )
    group = RecordGroupOutput(
        spec=spec,
        object_logits=torch.tensor(raw["object_logits"], dtype=torch.float32),
        assign_logits=assignments,
        field_query_ids=[entry["query_id"] for entry in raw["fields"]],
        field_specs=specs,
        field_spans=spans,
        field_cand_mask=[torch.ones(len(item), dtype=torch.bool) for item in spans],
        field_cand_logits=[torch.zeros(len(item), dtype=torch.float32) for item in spans],
        instance_seed=[None if item is None else tuple(item) for item in raw["instance_seed"]],
        instance_spans=[None if item is None else tuple(item) for item in raw["instance_spans"]],
    )
    decoded = decode_group(
        group,
        anchor_threshold=raw["anchor_threshold"],
        field_threshold=raw["field_threshold"],
        object_threshold=raw["object_threshold"],
        temperature=raw["temperature"],
    )
    return [expected_record(record) for record in decoded]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "fixtures" / "record-decode-vectors.json",
    )
    parser.add_argument(
        "--argsort-random-cases",
        type=int,
        default=3,
        help="seeded random/tied argsort probes per width (17, 33, 193)",
    )
    args = parser.parse_args()
    if args.argsort_random_cases < 0:
        parser.error("--argsort-random-cases must be non-negative")

    assert_pinned_gliner2_installation()
    if importlib.util.find_spec("scipy") is not None:
        raise RuntimeError("record vectors require the pinned internal assignment backend (SciPy absent)")
    # Initialize package exports before the records submodule; the pinned tree
    # otherwise exposes a records/matching circular import on direct import.
    import gliner2.models.boundary  # noqa: F401
    from gliner2.models.boundary.records import RecordGroupOutput, decode_group
    from gliner2.processing.records import FieldCardinality, RecordFieldSpec, RecordSpec

    torch.set_num_threads(1)
    output_cases = []
    for raw in definitions():
        output_cases.append(
            {
                **raw,
                "expected": run_oracle(
                    raw,
                    RecordFieldSpec,
                    RecordGroupOutput,
                    RecordSpec,
                    FieldCardinality,
                    decode_group,
                ),
            }
        )

    payload = {
        "format_version": 2,
        "upstream_commit": GLINER2_COMMIT,
        "oracle": "unchanged pinned GLiNER2 RecordGroupOutput/RecordSpec decode_group",
        "torch_version": torch.__version__,
        "assignment_backend": "internal_shortest_augmenting_path_scipy_absent",
        "oracle_platform": f"{platform.system().lower()}-{platform.machine().lower()}",
        "argsort_contract": "torch.argsort(descending=True, stable=False) exact permutation",
        "argsort_cases": argsort_definitions(args.argsort_random_cases),
        "cases": output_cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} ({len(output_cases)} cases, {args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
