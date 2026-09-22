"""Shared-pool GLiNER2.5 boundary scorer export wrapper.

Candidate selection stays outside this graph.  The caller supplies one explicit
query-agnostic document pool, including the compatibility prior produced by the
shared pool builder.  The wrapper preserves the upstream scorer arithmetic and
adds only the public candidate-major to query-major transpose plus the
endpoint-derived candidate, null, and count heads used by ``BoundaryModel``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from gliner2.models.boundary.indexing import gather_rows
from gliner2.models.boundary.pool import PooledCandidates

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
    "candidate_compat",
)
OUTPUT_NAMES = (
    "pair_logits",
    "candidate_states",
    "null_logits",
    "count_log_rates",
)


class BoundaryScorerGraph(torch.nn.Module):
    """Run the learned shared scorer and the source-required auxiliary heads."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        head = model.boundary_head
        if head.candidate_encoder is None:
            raise ValueError("boundary scorer requires the learned candidate_encoder")
        if head.null_projection is None:
            raise ValueError("boundary scorer requires the learned null_projection")
        if head.count_head is None:
            raise ValueError("boundary scorer requires the learned count_head")
        self.shared_pool_scorer = head.shared_pool_scorer
        self.candidate_encoder = head.candidate_encoder
        self.null_projection = head.null_projection
        self.count_head = head.count_head

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
        candidate_compat: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        pooled = PooledCandidates(
            indices=candidate_indices,
            mask=candidate_mask,
            proposal_logits=None,
            gold_mask=None,
            compat_logits=candidate_compat,
        )
        text_lengths = text_mask.sum(dim=1).long()
        candidate_major_logits, _feature_states = self.shared_pool_scorer(
            boundary_states,
            query_states,
            query_mask,
            pooled,
            start_logits,
            end_logits,
            inside_prefix,
            text_lengths,
            text_states,
            text_mask,
            inside_prefix_mean=inside_prefix_mean,
        )

        # These H-wide states are the record/relation-facing source contract.
        # The scorer's returned feature_states are pair_dim-wide and must not be
        # substituted for this separate learned endpoint projection.
        starts = candidate_indices[..., 0]
        ends = candidate_indices[..., 1]
        start_states = gather_rows(boundary_states, starts)
        end_states = gather_rows(boundary_states, ends)
        candidate_states = self.candidate_encoder(
            torch.cat((start_states, end_states), dim=-1)
        )
        candidate_states = candidate_states.masked_fill(
            ~candidate_mask.unsqueeze(-1), 0.0
        )

        pair_logits = candidate_major_logits.transpose(1, 2)
        null_logits = self.null_projection(query_states).squeeze(-1)
        count_log_rates = self.count_head(query_states).squeeze(-1)
        return pair_logits, candidate_states, null_logits, count_log_rates


def make_pool(
    candidate_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
    candidate_compat: torch.Tensor,
) -> PooledCandidates:
    """Build the untouched upstream scorer's explicit document-pool input."""
    return PooledCandidates(
        indices=candidate_indices,
        mask=candidate_mask,
        proposal_logits=None,
        gold_mask=None,
        compat_logits=candidate_compat,
    )


def oracle_outputs(
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
    candidate_compat: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Compose only the original, unmodified upstream modules."""
    head = model.boundary_head
    pooled = make_pool(candidate_indices, candidate_mask, candidate_compat)
    text_lengths = text_mask.sum(dim=1).long()
    candidate_major_logits, _feature_states = head.shared_pool_scorer(
        boundary_states,
        query_states,
        query_mask,
        pooled,
        start_logits,
        end_logits,
        inside_prefix,
        text_lengths,
        text_states,
        text_mask,
        inside_prefix_mean=inside_prefix_mean,
    )
    starts = candidate_indices[..., 0]
    ends = candidate_indices[..., 1]
    start_states = gather_rows(boundary_states, starts)
    end_states = gather_rows(boundary_states, ends)
    candidate_states = head.candidate_encoder(
        torch.cat((start_states, end_states), dim=-1)
    ).masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
    return (
        candidate_major_logits.transpose(1, 2),
        candidate_states,
        head.null_projection(query_states).squeeze(-1),
        head.count_head(query_states).squeeze(-1),
    )


def assert_wrapper_matches_oracle(
    names: Sequence[str],
    expected: Sequence[torch.Tensor],
    actual: Sequence[torch.Tensor],
    *,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> None:
    if len(names) != len(expected) or len(names) != len(actual):
        raise AssertionError("boundary scorer output count mismatch")
    for name, reference, wrapped in zip(names, expected, actual):
        torch.testing.assert_close(
            wrapped,
            reference,
            atol=atol,
            rtol=rtol,
            msg=lambda message: f"{name}: wrapper differs from oracle: {message}",
        )
