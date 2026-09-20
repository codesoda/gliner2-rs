"""Inference-only GLiNER2.5 learned sparse relation scorer.

The graph ABI receives actual encoder text states (the upstream scorer's
``boundary_states`` name is misleading), directional head-then-tail relation
query states, and flattened pair routing. Pair generation and public decoding
remain outside ONNX. Invalid routing is clamped for safe gathers and masked to
positive zero; the public Rust ABI rejects it before native inference.
"""

from __future__ import annotations

from typing import Sequence

import torch

INPUT_NAMES = (
    "text_states",
    "relation_query_states",
    "batch_index",
    "relation_index",
    "head_start",
    "head_end",
    "tail_start",
    "tail_end",
    "pair_mask",
)
OUTPUT_NAMES = ("relation_logits",)


class BoundaryRelationsGraph(torch.nn.Module):
    """Learned math from ``SparseRelationScorer.forward`` with dynamic shapes.

    The length divisor deliberately comes from ``torch._shape_as_tensor``.
    Using ``float(max(text_states.shape[1], 1))`` as the original eager method
    does would cause legacy tracing to bake the example length into ONNX.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        scorer = model.relation_scorer
        self.hidden_size = int(scorer.hidden_size)
        self.relation_query_dim = int(scorer.relation_query_dim)
        if not bool(scorer.use_biaffine_content):
            raise ValueError("relation export requires learned biaffine content")
        # Reference the loaded child modules directly: no copied or replaced weights.
        self.mlp = scorer.mlp
        self.head_content_projection = scorer.head_content_projection
        self.tail_content_projection = scorer.tail_content_projection
        self.relation_content_gate = scorer.relation_content_gate
        self.content_linear = scorer.content_linear

    def forward(
        self,
        text_states: torch.Tensor,             # [B,L,H], actual encoder states
        relation_query_states: torch.Tensor,   # [B,R,2H], head then tail
        batch_index: torch.Tensor,              # [P]
        relation_index: torch.Tensor,           # [P]
        head_start: torch.Tensor,               # [P], half-open word coordinates
        head_end: torch.Tensor,                 # [P]
        tail_start: torch.Tensor,               # [P]
        tail_end: torch.Tensor,                 # [P]
        pair_mask: torch.Tensor,                # [P]
    ) -> torch.Tensor:
        text_shape = torch._shape_as_tensor(text_states)
        relation_shape = torch._shape_as_tensor(relation_query_states)
        batch_count = torch.minimum(text_shape[0], relation_shape[0])
        length = text_shape[1]
        relation_count = relation_shape[1]

        zero = torch.zeros((), dtype=batch_index.dtype, device=batch_index.device)
        safe_batch = torch.minimum(
            torch.maximum(batch_index, zero),
            torch.maximum(batch_count - 1, zero),
        )
        safe_relation = torch.minimum(
            torch.maximum(relation_index, zero),
            torch.maximum(relation_count - 1, zero),
        )
        routing_valid = (
            (batch_index >= 0)
            & (batch_index < batch_count)
            & (relation_index >= 0)
            & (relation_index < relation_count)
            & pair_mask
        )

        def gather(position: torch.Tensor) -> torch.Tensor:
            safe_position = torch.minimum(
                torch.maximum(position, zero), torch.maximum(length - 1, zero)
            )
            return text_states[safe_batch, safe_position]

        h_start_state = gather(head_start)
        h_end_state = gather(head_end - 1)
        t_start_state = gather(tail_start)
        t_end_state = gather(tail_end - 1)
        relation_state = relation_query_states[safe_batch, safe_relation]

        delta = (tail_start - head_start).to(text_states.dtype)
        order = torch.sign(delta).unsqueeze(-1)
        # This tensor scalar, unlike Python float(max(length, 1)), remains dynamic.
        dynamic_length = length.to(text_states.dtype).clamp_min(1.0)
        distance = (delta.abs() / dynamic_length).unsqueeze(-1)
        features = torch.cat(
            (
                h_start_state,
                h_end_state,
                t_start_state,
                t_end_state,
                relation_state,
                order,
                distance,
            ),
            dim=-1,
        )
        score = self.mlp(features).squeeze(-1)

        # Upstream accumulates the prefix in f32, then returns to activation dtype.
        prefix = torch.cat(
            (
                torch.zeros_like(text_states[:, :1, :]),
                text_states.float().cumsum(1).to(text_states.dtype),
            ),
            dim=1,
        )

        def pool(start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
            safe_start = torch.minimum(torch.maximum(start, zero), length)
            safe_end = torch.minimum(torch.maximum(end, zero), length)
            span_sum = prefix[safe_batch, safe_end] - prefix[safe_batch, safe_start]
            width = (end - start).clamp_min(1).unsqueeze(-1).to(span_sum.dtype)
            return span_sum / width

        head_content = self.head_content_projection(pool(head_start, head_end))
        tail_content = self.tail_content_projection(pool(tail_start, tail_end))
        gate = torch.sigmoid(self.relation_content_gate(relation_state))
        biaffine = (head_content * gate * tail_content).sum(-1) / (
            self.hidden_size**0.5
        )
        linear = self.content_linear(
            torch.cat((head_content, tail_content, relation_state), dim=-1)
        ).squeeze(-1)
        score = score + biaffine + linear
        return score.masked_fill(~routing_valid, 0.0)


def validate_abi_inputs(inputs: Sequence[torch.Tensor]) -> None:
    """Validate the positive, legal-coordinate public relation ABI."""
    if len(inputs) != len(INPUT_NAMES):
        raise ValueError(f"expected {len(INPUT_NAMES)} relation inputs, got {len(inputs)}")
    text, relation, batch, relation_id, hs, he, ts, te, mask = inputs
    if text.dim() != 3 or relation.dim() != 3:
        raise ValueError("relation states must be [B,L,H] and [B,R,2H]")
    b, length, hidden = text.shape
    rb, relations, relation_width = relation.shape
    pair_count = batch.shape[0] if batch.dim() == 1 else -1
    if min(b, length, hidden, relations, pair_count) < 1:
        raise ValueError("relation graph requires positive B/L/H/R/P dimensions")
    if rb != b:
        raise ValueError(f"relation batch mismatch: text={b}, relation={rb}")
    if relation_width != 2 * hidden:
        raise ValueError(
            f"relation query width must equal 2H: got {relation_width}, expected {2 * hidden}"
        )
    vectors = (batch, relation_id, hs, he, ts, te, mask)
    if any(value.dim() != 1 or value.shape[0] != pair_count for value in vectors):
        raise ValueError("all routing, coordinate, and mask inputs must have shape [P]")
    if text.dtype != torch.float32 or relation.dtype != torch.float32:
        raise TypeError("relation state inputs must be fp32")
    if any(value.dtype != torch.int64 for value in (batch, relation_id, hs, he, ts, te)):
        raise TypeError("relation routing and coordinate inputs must be int64")
    if mask.dtype != torch.bool:
        raise TypeError("pair_mask must be bool")
    if not torch.isfinite(text).all() or not torch.isfinite(relation).all():
        raise ValueError("relation state inputs contain non-finite values")
    if not ((batch >= 0) & (batch < b)).all():
        raise ValueError("batch_index contains out-of-range routing")
    if not ((relation_id >= 0) & (relation_id < relations)).all():
        raise ValueError("relation_index contains out-of-range routing")
    for name, start, end in (("head", hs, he), ("tail", ts, te)):
        if not ((start >= 0) & (start < end) & (end <= length)).all():
            raise ValueError(
                f"{name} coordinates must satisfy 0 <= start < end <= L, even for masked pairs"
            )


def source_logits(
    scorer: torch.nn.Module,
    inputs: Sequence[torch.Tensor],
    *,
    enforce_public_abi: bool = True,
) -> torch.Tensor:
    """Run the untouched upstream scorer for oracle comparisons.

    Validation normally enforces the public ABI. Tests may disable it only to
    prove the raw graph's source-faithful safe routing mask; that is not an
    inference-API promise for invalid inputs.
    """
    from gliner2.models.boundary.relations import RelationPairBatch

    if enforce_public_abi:
        validate_abi_inputs(inputs)
    text, relation, batch, relation_id, hs, he, ts, te, mask = inputs
    zeros = text.new_zeros(batch.shape[0])
    pairs = RelationPairBatch(
        batch_index=batch,
        relation_index=relation_id,
        head_start=hs,
        head_end=he,
        tail_start=ts,
        tail_end=te,
        head_prob=zeros,
        tail_prob=zeros,
        pair_mask=mask,
    )
    return scorer(text, relation, None, pairs)
