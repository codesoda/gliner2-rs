#!/usr/bin/env python3
"""Generate optional explicit-scorer vectors from the untouched pinned model.

The vectors contain the low-level ONNX ABI inputs plus outputs computed by the
original BoundaryHead modules. Pair logits are also required to equal the
original full ``score_explicit_spans`` method; the ONNX graph is never used as
a reference. The default output directory is intentionally gitignored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EXPORT_DIR = ROOT / "scripts" / "export"
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    BASE_HF_REVISION,
    BASE_MODEL_ID,
    GLINER2_COMMIT,
    MASK_LOGIT,
    assert_official_base_source,
    configure_determinism,
    load_reference_model,
    package_versions,
    sha256_file,
)
from gen_boundary_goldens import write_deterministic_npz  # noqa: E402
from gliner2.models.boundary.proposal import BoundaryProposals  # noqa: E402

DEFAULT_MODEL_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--fastino--gliner2.5-base-v1"
    / "snapshots"
    / BASE_HF_REVISION
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "fixtures/gliner2.5-explicit"
    )
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def make_inputs(
    model: torch.nn.Module,
    *,
    seed: int,
    batch: int,
    length: int,
    queries: int,
    candidates: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = int(model.hidden_size)
    text_states = torch.randn(
        batch, length, hidden, generator=generator, dtype=torch.float32
    )
    query_states = torch.randn(
        batch, queries, hidden, generator=generator, dtype=torch.float32
    )
    text_mask = torch.ones(batch, length, dtype=torch.bool)
    query_mask = torch.ones(batch, queries, dtype=torch.bool)
    if batch > 1:
        text_mask[1, max(1, length // 2) :] = False
        if queries > 1:
            query_mask[1, 1] = False
    if queries > 2:
        query_mask[0, -1] = False

    indices = torch.zeros(batch, queries, candidates, 2, dtype=torch.long)
    candidate_mask = torch.ones(batch, queries, candidates, dtype=torch.bool)
    for batch_index in range(batch):
        live_length = int(text_mask[batch_index].sum())
        for query in range(queries):
            for candidate in range(candidates):
                start = candidate % live_length
                end = min(live_length, start + 1 + candidate % 3)
                indices[batch_index, query, candidate] = torch.tensor((start, end))

    return {
        "text_states": text_states,
        "text_mask": text_mask,
        "query_states": query_states,
        "query_mask": query_mask,
        "candidate_indices": indices,
        "candidate_mask": candidate_mask,
    }


def add_standard_invalids(inputs: dict[str, torch.Tensor], *, extremes: bool) -> None:
    indices = inputs["candidate_indices"]
    mask = inputs["candidate_mask"]
    candidates = indices.shape[2]
    patterns = [(-1, 1), (3, 2), (2, 2), (0, 99)]
    # Keep at least one source-scored legal candidate in every vector.
    invalid_count = min(len(patterns), candidates - 1)
    for candidate, pair in enumerate(patterns[:invalid_count]):
        indices[:, :, candidate] = torch.tensor(pair)
    if candidates > 4:
        mask[:, :, 4] = False
    if extremes and candidates > 6:
        indices[:, :, 5] = torch.tensor((torch.iinfo(torch.int64).min, torch.iinfo(torch.int64).max))
        indices[:, :, 6] = torch.tensor((torch.iinfo(torch.int64).max, torch.iinfo(torch.int64).min))


def source_outputs(
    model: torch.nn.Module, inputs: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    head = model.boundary_head
    text_states = inputs["text_states"]
    text_mask = inputs["text_mask"]
    query_states = inputs["query_states"]
    query_mask = inputs["query_mask"]
    indices = inputs["candidate_indices"]
    candidate_mask = inputs["candidate_mask"]

    with torch.inference_mode():
        encoding = head.boundary_encoder(text_states, text_mask)
        marginals = head.boundary_query_head(
            encoding.states,
            encoding.mask,
            text_states,
            text_mask,
            query_states,
            query_mask,
        )
        text_lengths = text_mask.sum(dim=1).long()
        starts = indices[..., 0]
        ends = indices[..., 1]
        legal = (
            (starts >= 0)
            & (ends > starts)
            & (ends <= text_lengths.reshape(indices.shape[0], 1, 1))
            & query_mask.unsqueeze(-1)
            & candidate_mask
        )
        compatibility = head.boundary_proposer.score_explicit_pairs(
            encoding.states, query_states, indices, legal
        )
        proposals = BoundaryProposals(
            indices=indices,
            logits=None,
            valid_mask=legal,
            compat_logits=compatibility,
        )
        composed_pair = head.pair_scorer(
            encoding.states,
            query_states,
            proposals,
            marginals.start_logits,
            marginals.end_logits,
            marginals.inside_prefix,
            text_lengths,
            text_states,
            text_mask,
            inside_prefix_mean=marginals.inside_prefix_mean,
        )
        full_pair = head.score_explicit_spans(
            text_states,
            text_mask,
            query_states,
            query_mask,
            indices,
            candidate_mask,
        )

    if not torch.equal(composed_pair, full_pair):
        maximum = (composed_pair - full_pair).abs().max().item()
        raise AssertionError(
            f"original modules differ from full score_explicit_spans: max_abs={maximum}"
        )
    if not torch.isfinite(full_pair).all() or not torch.isfinite(compatibility).all():
        raise AssertionError("untouched explicit scorer returned non-finite output")
    if not torch.equal(compatibility[~legal], torch.zeros_like(compatibility[~legal])):
        raise AssertionError("untouched scorer did not zero illegal compatibility")
    if not torch.equal(
        full_pair[~legal], torch.full_like(full_pair[~legal], MASK_LOGIT)
    ):
        raise AssertionError("untouched scorer did not mask illegal pair logits")

    return {
        "boundary_states": encoding.states,
        "text_states": text_states,
        "text_mask": text_mask,
        "query_states": query_states,
        "query_mask": query_mask,
        "start_logits": marginals.start_logits,
        "end_logits": marginals.end_logits,
        "inside_prefix": marginals.inside_prefix,
        "inside_prefix_mean": marginals.inside_prefix_mean,
        "candidate_indices": indices,
        "candidate_mask": candidate_mask,
        "pair_logits": full_pair,
        "compatibility": compatibility,
        "legal_mask": legal,
    }


def numpy_arrays(values: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        name: value.detach().cpu().contiguous().numpy()
        for name, value in values.items()
    }


def main() -> None:
    args = parse_args()
    configure_determinism(args.seed)
    source_hashes = assert_official_base_source(args.model_dir)
    model, config = load_reference_model(args.model_dir)
    if config.get("architecture") != "boundary" or config.get("architecture_version") != 1:
        raise RuntimeError("explicit vectors require boundary architecture version 1")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    specifications = [
        ("valid_b1_c1", 1, 5, 1, 1, False),
        ("mixed_b1_c7", 1, 11, 3, 7, False),
        ("dynamic_b2_c3", 2, 8, 2, 3, False),
        ("extreme_b2_c9", 2, 6, 3, 9, True),
    ]
    entries: list[dict[str, object]] = []
    for offset, (case_id, batch, length, queries, candidates, extremes) in enumerate(
        specifications
    ):
        inputs = make_inputs(
            model,
            seed=args.seed + offset,
            batch=batch,
            length=length,
            queries=queries,
            candidates=candidates,
        )
        if case_id != "valid_b1_c1":
            add_standard_invalids(inputs, extremes=extremes)
        arrays = numpy_arrays(source_outputs(model, inputs))
        path = args.out_dir / f"{case_id}.npz"
        write_deterministic_npz(path, arrays)
        legal = arrays["legal_mask"]
        entries.append(
            {
                "case_id": case_id,
                "batch": batch,
                "text_length": length,
                "query_count": queries,
                "candidate_count": candidates,
                "includes_i64_extremes": extremes,
                "legal_pairs": int(legal.sum()),
                "illegal_pairs": int((~legal).sum()),
                "vector": path.name,
                "vector_bytes": path.stat().st_size,
                "vector_sha256": sha256_file(path),
            }
        )
        print(f"wrote {path}", flush=True)

    manifest = {
        "format_version": 1,
        "oracle": (
            "untouched pinned GLiNER2 BoundaryHead.score_explicit_spans; "
            "compatibility/legal from its original component modules"
        ),
        "onnx_used_as_reference": False,
        "case_count": len(entries),
        "atol": 1e-4,
        "rtol": 1e-3,
        "provenance": {
            "gliner2_commit": GLINER2_COMMIT,
            "model_id": BASE_MODEL_ID,
            "hf_revision": BASE_HF_REVISION,
            "source_file_sha256": source_hashes,
            "generator_sha256": sha256_file(Path(__file__)),
            "seed": args.seed,
            "dtype": "float32",
            "device": "cpu",
            "dependencies": package_versions(),
        },
        "entries": entries,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
