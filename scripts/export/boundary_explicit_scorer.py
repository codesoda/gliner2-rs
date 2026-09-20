"""Learned GLiNER2.5 scorer for caller-supplied explicit spans.

This graph starts after boundary encoding and marginal computation.  It reuses
only the checkpoint's original ``SparseBoundaryProposer.score_explicit_pairs``
and ``SparseBoundaryPairScorer`` modules; it is intentionally separate from the
query-agnostic shared-pool scorer used by ordinary extraction.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from gliner2.models.boundary.proposal import BoundaryProposals

INPUT_NAMES = (
    "boundary_states",
    "text_states",
    "text_mask",
    "query_states",
    "query_mask",
    "start_logits",
    "end_logits",
    "inside_prefix",
    "inside_prefix_mean",
    "candidate_indices",
    "candidate_mask",
)
OUTPUT_NAMES = ("pair_logits", "compatibility", "legal_mask")


class BoundaryExplicitScorerGraph(torch.nn.Module):
    """Run the original learned explicit proposer and sparse pair scorer."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        head = model.boundary_head
        if not head.use_inside_evidence:
            raise ValueError("explicit scorer graph requires inside evidence")
        self.boundary_proposer = head.boundary_proposer
        self.pair_scorer = head.pair_scorer

    def forward(
        self,
        boundary_states: torch.Tensor,
        text_states: torch.Tensor,
        text_mask: torch.Tensor,
        query_states: torch.Tensor,
        query_mask: torch.Tensor,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        inside_prefix: torch.Tensor,
        inside_prefix_mean: torch.Tensor,
        candidate_indices: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = candidate_indices.shape[0]
        text_lengths = text_mask.sum(dim=1).long()
        starts = candidate_indices[..., 0]
        ends = candidate_indices[..., 1]
        legal = (
            (starts >= 0)
            & (ends > starts)
            & (ends <= text_lengths.reshape(batch, 1, 1))
            & query_mask.unsqueeze(-1)
            & candidate_mask
        )
        compatibility = self.boundary_proposer.score_explicit_pairs(
            boundary_states, query_states, candidate_indices, legal
        )
        proposals = BoundaryProposals(
            indices=candidate_indices,
            logits=None,
            valid_mask=legal,
            compat_logits=compatibility,
        )
        pair_logits = self.pair_scorer(
            boundary_states,
            query_states,
            proposals,
            start_logits,
            end_logits,
            inside_prefix,
            text_lengths,
            text_states,
            text_mask,
            inside_prefix_mean=inside_prefix_mean,
        )
        return pair_logits, compatibility, legal


def independently_composed_outputs(
    model: torch.nn.Module,
    boundary_states: torch.Tensor,
    text_states: torch.Tensor,
    text_mask: torch.Tensor,
    query_states: torch.Tensor,
    query_mask: torch.Tensor,
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    inside_prefix: torch.Tensor,
    inside_prefix_mean: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose the untouched upstream learned modules without using the wrapper."""
    head = model.boundary_head
    batch = candidate_indices.shape[0]
    text_lengths = text_mask.sum(dim=1).long()
    starts = candidate_indices[..., 0]
    ends = candidate_indices[..., 1]
    legal = (
        (starts >= 0)
        & (ends > starts)
        & (ends <= text_lengths.reshape(batch, 1, 1))
        & query_mask.unsqueeze(-1)
        & candidate_mask
    )
    compatibility = head.boundary_proposer.score_explicit_pairs(
        boundary_states, query_states, candidate_indices, legal
    )
    proposals = BoundaryProposals(
        indices=candidate_indices,
        logits=None,
        valid_mask=legal,
        compat_logits=compatibility,
    )
    pair_logits = head.pair_scorer(
        boundary_states,
        query_states,
        proposals,
        start_logits,
        end_logits,
        inside_prefix,
        text_lengths,
        text_states,
        text_mask,
        inside_prefix_mean=inside_prefix_mean,
    )
    return pair_logits, compatibility, legal


def full_method_outputs(
    model: torch.nn.Module,
    text_states: torch.Tensor,
    text_mask: torch.Tensor,
    query_states: torch.Tensor,
    query_mask: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Invoke the untouched full BoundaryHead explicit-span oracle."""
    return model.boundary_head.score_explicit_spans(
        text_states,
        text_mask,
        query_states,
        query_mask,
        candidate_indices,
        candidate_mask,
    )


def assert_output_parity(
    expected: Sequence[torch.Tensor],
    actual: Sequence[torch.Tensor],
    *,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> None:
    if len(expected) != len(OUTPUT_NAMES) or len(actual) != len(OUTPUT_NAMES):
        raise AssertionError("explicit scorer output count mismatch")
    for name, reference, observed in zip(OUTPUT_NAMES, expected, actual):
        if reference.dtype == torch.bool:
            if not torch.equal(reference, observed):
                raise AssertionError(f"{name}: wrapper boolean output differs")
        else:
            torch.testing.assert_close(
                observed,
                reference,
                atol=atol,
                rtol=rtol,
                msg=lambda message: f"{name}: wrapper differs from oracle: {message}",
            )
