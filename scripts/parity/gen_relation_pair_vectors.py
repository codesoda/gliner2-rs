#!/usr/bin/env python3
"""Export deterministic typed relation-pair vectors from pinned GLiNER2.

The oracle is the unmodified upstream ``TypedRelationPairGenerator``. This
script only constructs inputs and serializes its compact output; it does not
reimplement proposal selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "export"))
from common import GLINER2_COMMIT, assert_pinned_gliner2_installation  # noqa: E402

SEED = 1729
PINNED_RELATIONS_SHA256 = (
    "59e0b20c040e95a0a0cd0c59fe5d5563f6c85d9d5f060138bda583a7f4a90cbf"
)


def assert_pinned_relation_source() -> None:
    spec = importlib.util.find_spec("gliner2.models.boundary.relations")
    if spec is None or spec.origin is None:
        raise RuntimeError("cannot locate installed pinned relation source")
    source = Path(spec.origin).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != PINNED_RELATIONS_SHA256:
        raise RuntimeError(
            f"installed relation source {source} has sha256 {digest}, expected "
            f"{PINNED_RELATIONS_SHA256}"
        )


@dataclass(frozen=True)
class Case:
    name: str
    indices: list[list[list[int]]]
    valid_mask: list[list[bool]]
    query_mask: list[bool]
    pair_logits: list[list[float]]
    specs: list[dict[str, Any]]
    config: dict[str, Any]


def checkpoint_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "heads_per_relation": 32,
        "tails_per_relation": 32,
        "pair_cap": 64,
        "argument_threshold": 0.2,
    }
    config.update(overrides)
    return config


def cases() -> list[Case]:
    rng = random.Random(SEED)
    random_indices: list[list[list[int]]] = []
    random_logits: list[list[float]] = []
    random_valid: list[list[bool]] = []
    for query in range(5):
        spans = []
        logits = []
        valid = []
        for candidate in range(8):
            start = rng.randrange(0, 14)
            width = rng.randrange(1, 4)
            spans.append([start, start + width])
            # A narrow integer range deliberately creates sigmoid/product ties.
            logits.append(float(rng.randrange(-3, 5)))
            valid.append(rng.randrange(5) != 0)
        random_indices.append(spans)
        random_logits.append(logits)
        random_valid.append(valid)

    return [
        Case(
            name="tie_heavy_coordinates_and_flat_order",
            indices=[
                [[5, 6], [1, 2], [1, 2], [3, 4]],
                [[3, 4], [1, 3], [8, 9], [0, 1]],
                [[10, 11], [7, 8], [7, 8], [9, 10]],
            ],
            valid_mask=[[True] * 4 for _ in range(3)],
            query_mask=[True, True, True],
            pair_logits=[[0.0] * 4 for _ in range(3)],
            specs=[{
                "relation_type": "tie",
                "head_query_ids": [0, 1],
                "tail_query_ids": [2],
                "allow_self": False,
            }],
            config=checkpoint_config(
                heads_per_relation=6, tails_per_relation=3, pair_cap=12
            ),
        ),
        Case(
            name="source_sigmoid_rounding_endpoint_cap",
            indices=[[[0, 1], [10, 11]], [[20, 21], [-1, -1]]],
            valid_mask=[[True, True], [True, False]],
            query_mask=[True, True],
            pair_logits=[[-1.3859999, -1.3859998], [0.0, 0.0]],
            specs=[{
                "relation_type": "rounding_endpoint",
                "head_query_ids": [0],
                "tail_query_ids": [1],
                "allow_self": False,
            }],
            config=checkpoint_config(
                heads_per_relation=1,
                tails_per_relation=1,
                pair_cap=1,
                argument_threshold=0.2,
            ),
        ),
        Case(
            name="source_sigmoid_rounding_product_cap",
            indices=[[[0, 1], [10, 11]], [[20, 21], [-1, -1]]],
            valid_mask=[[True, True], [True, False]],
            query_mask=[True, True],
            pair_logits=[[-1.3859999, -1.3859998], [0.0, 0.0]],
            specs=[{
                "relation_type": "rounding_product",
                "head_query_ids": [0],
                "tail_query_ids": [1],
                "allow_self": False,
            }],
            config=checkpoint_config(
                heads_per_relation=2,
                tails_per_relation=1,
                pair_cap=1,
                argument_threshold=0.2,
            ),
        ),
        Case(
            name="threshold_masks_invalid_ids_and_self",
            indices=[
                [[0, 1], [-99, -7], [4, 5]],
                [[0, 1], [6, 7], [8, 9]],
                [[11, 12], [13, 14], [15, 16]],
            ],
            valid_mask=[
                [True, False, True],
                [True, True, True],
                [True, True, True],
            ],
            query_mask=[True, True, False],
            pair_logits=[
                [0.0, 100.0, -100.0],
                [0.0, -0.0001, 100.0],
                [100.0, 100.0, 100.0],
            ],
            specs=[
                {
                    "relation_type": "no_self",
                    "head_query_ids": [-1, 0, 999],
                    "tail_query_ids": [1, 500],
                    "allow_self": False,
                },
                {
                    "relation_type": "self_allowed",
                    "head_query_ids": [0],
                    "tail_query_ids": [1],
                    "allow_self": True,
                },
            ],
            config=checkpoint_config(argument_threshold=0.5),
        ),
        Case(
            name="sigmoid_saturation_not_raw_logit_order",
            indices=[[[9, 10], [1, 2], [5, 6]], [[12, 13], [11, 12], [10, 11]]],
            valid_mask=[[True] * 3, [True] * 3],
            query_mask=[True, True],
            pair_logits=[[100.0, 90.0, 80.0], [70.0, 60.0, 50.0]],
            specs=[{
                "relation_type": "saturated",
                "head_query_ids": [0],
                "tail_query_ids": [1],
                "allow_self": False,
            }],
            config=checkpoint_config(
                heads_per_relation=3, tails_per_relation=3, pair_cap=5
            ),
        ),
        Case(
            name="multiple_query_sets_and_caps",
            indices=[
                [[0, 1], [4, 5], [8, 9]],
                [[1, 2], [5, 6], [9, 10]],
                [[2, 3], [6, 7], [10, 11]],
                [[3, 4], [7, 8], [11, 12]],
            ],
            valid_mask=[[True] * 3 for _ in range(4)],
            query_mask=[True] * 4,
            pair_logits=[
                [4.0, 3.0, 2.0],
                [3.0, 2.0, 1.0],
                [2.0, 1.0, 0.0],
                [1.0, 0.0, -1.0],
            ],
            specs=[
                {
                    "relation_type": "many_to_many",
                    "head_query_ids": [0, 1],
                    "tail_query_ids": [2, 3],
                    "allow_self": False,
                },
                {
                    "relation_type": "reversed",
                    "head_query_ids": [3, 2],
                    "tail_query_ids": [1, 0],
                    "allow_self": False,
                },
            ],
            config=checkpoint_config(
                heads_per_relation=3, tails_per_relation=2, pair_cap=4
            ),
        ),
        Case(
            name="seeded_random",
            indices=random_indices,
            valid_mask=random_valid,
            query_mask=[True, False, True, True, True],
            pair_logits=random_logits,
            specs=[
                {
                    "relation_type": "random_a",
                    "head_query_ids": [0, 2, 4],
                    "tail_query_ids": [1, 3],
                    "allow_self": False,
                },
                {
                    "relation_type": "random_b",
                    "head_query_ids": [3, 4],
                    "tail_query_ids": [0, 2],
                    "allow_self": True,
                },
            ],
            config=checkpoint_config(
                heads_per_relation=5,
                tails_per_relation=4,
                pair_cap=11,
                argument_threshold=0.25,
            ),
        ),
    ]


def as_list(value: torch.Tensor) -> Any:
    return value.detach().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "fixtures" / "gliner2.5-base-v1" / "relation-aux" / "relation_pair_vectors.json",
    )
    args = parser.parse_args()

    assert_pinned_gliner2_installation()
    assert_pinned_relation_source()
    from gliner2.models.base import QueryLayout, QuerySpec
    from gliner2.models.boundary.relations import (
        RelationProposalSettings,
        RelationTypeSpec,
        TypedRelationPairGenerator,
    )
    from gliner2.models.outputs import CandidateTensorBatch

    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    output_cases = []
    for case in cases():
        indices = torch.tensor(case.indices, dtype=torch.long)
        valid_mask = torch.tensor(case.valid_mask, dtype=torch.bool)
        query_mask = torch.tensor(case.query_mask, dtype=torch.bool)
        pair_logits = torch.tensor(case.pair_logits, dtype=torch.float32)
        query_count, candidate_count, _ = indices.shape
        candidates = CandidateTensorBatch(
            indices=indices.unsqueeze(0),
            proposal_logits=None,
            pair_logits=pair_logits.unsqueeze(0),
            valid_mask=valid_mask.unsqueeze(0),
            query_mask=query_mask.unsqueeze(0),
        )
        layout = QueryLayout(
            queries=tuple(
                QuerySpec(
                    query_id=query_id,
                    task_index=0,
                    task_type="relations",
                    task_name="vectors",
                    role_index=query_id,
                    role_name=f"q{query_id}",
                    field_path=(f"q{query_id}",),
                    extractive=True,
                )
                for query_id in range(query_count)
            )
        )
        specs = [
            RelationTypeSpec(
                relation_type=spec["relation_type"],
                head_query_ids=tuple(spec["head_query_ids"]),
                tail_query_ids=tuple(spec["tail_query_ids"]),
                allow_self=spec["allow_self"],
            )
            for spec in case.specs
        ]
        settings = RelationProposalSettings(**case.config)
        with torch.no_grad():
            result = TypedRelationPairGenerator(settings).generate(
                candidates, [layout], specs, compact=True
            )

        head_query_ids = [int(key[0].removeprefix("q")) for key in result.head_keys]
        tail_query_ids = [int(key[0].removeprefix("q")) for key in result.tail_keys]
        expected = [
            {
                "relation_index": int(result.relation_index[index]),
                "head_query_id": head_query_ids[index],
                "tail_query_id": tail_query_ids[index],
                "head": [
                    int(result.head_start[index]),
                    int(result.head_end[index]),
                ],
                "tail": [
                    int(result.tail_start[index]),
                    int(result.tail_end[index]),
                ],
                "head_probability": float(result.head_prob[index]),
                "tail_probability": float(result.tail_prob[index]),
            }
            for index in range(len(result))
        ]
        output_cases.append(
            {
                "name": case.name,
                "source_pool_count": query_count * candidate_count,
                "indices": case.indices,
                "valid_mask": case.valid_mask,
                "query_mask": case.query_mask,
                "pair_logits": case.pair_logits,
                "specs": case.specs,
                "config": case.config,
                "expected": expected,
                "oracle_pair_mask": as_list(result.pair_mask),
            }
        )

    payload = {
        "format_version": 1,
        "oracle": "unmodified pinned GLiNER2 TypedRelationPairGenerator compact output",
        "upstream_commit": GLINER2_COMMIT,
        "relation_source_sha256": PINNED_RELATIONS_SHA256,
        "torch_version": torch.__version__,
        "seed": SEED,
        "cases": output_cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output} ({len(output_cases)} cases, {args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
