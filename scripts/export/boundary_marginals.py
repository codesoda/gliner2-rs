"""Export-only finite-attention wrapper for the GLiNER2.5 marginal graph.

The upstream boundary encoder uses PyTorch's boolean-mask SDPA.  The legacy
opset-17 exporter lowers that path through non-finite mask constants.  This
module leaves the loaded oracle untouched and reuses its exact learned modules
while spelling out scaled dot-product attention with the bundle's finite
``-1e4`` mask sentinel.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from common import MASK_LOGIT
from boundary_reductions import sum_last_axis

OUTPUT_NAMES = (
    "boundary_states",
    "boundary_mask",
    "start_logits",
    "end_logits",
    "inside_logits",
    "inside_prefix",
    "inside_prefix_mean",
    "start_all",
    "end_all",
)


class FiniteBoundaryAttentionBlock(torch.nn.Module):
    """Numerically equivalent eval-mode attention with finite additive masks."""

    def __init__(self, block: torch.nn.Module) -> None:
        super().__init__()
        self.num_heads = block.num_heads
        self.head_dim = block.head_dim
        self.window = block.window
        self.norm = block.norm
        self.qkv_projection = block.qkv_projection
        self.output_projection = block.output_projection
        self.dropout = block.dropout

    def forward(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, boundaries, width = states.shape
        qkv = self.qkv_projection(self.norm(states)).view(
            batch, boundaries, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        allowed = mask.view(batch, 1, 1, boundaries).expand(
            batch, 1, boundaries, boundaries
        )
        if self.window > 0:
            positions = torch.arange(boundaries, device=states.device)
            local = (
                positions.view(boundaries, 1) - positions.view(1, boundaries)
            ).abs() <= self.window
            allowed = allowed & local.view(1, 1, boundaries, boundaries)
        diagonal_positions = torch.arange(boundaries, device=states.device)
        diagonal = diagonal_positions.view(boundaries, 1) == diagonal_positions.view(
            1, boundaries
        )
        allowed = allowed | diagonal.view(1, 1, boundaries, boundaries)

        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )
        logits = torch.where(
            allowed,
            logits,
            torch.full_like(logits, MASK_LOGIT),
        )
        attended = torch.matmul(torch.softmax(logits, dim=-1), value)
        attended = attended.transpose(1, 2).reshape(batch, boundaries, width)
        update = self.dropout(self.output_projection(attended))
        return (states + update) * mask.unsqueeze(-1).to(states.dtype)


class FiniteBoundaryEncoder(torch.nn.Module):
    """BoundaryEncoder using the oracle parameters and finite attention blocks."""

    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.left_projection = encoder.left_projection
        self.right_projection = encoder.right_projection
        self.output_projection = encoder.output_projection
        self.layer_norm = encoder.layer_norm
        self.dropout = encoder.dropout
        self.attention_blocks = torch.nn.ModuleList(
            FiniteBoundaryAttentionBlock(block)
            for block in encoder.attention_blocks
        )
        self.refinement_blocks = encoder.refinement_blocks
        self.bos_state = encoder.bos_state
        self.eos_state = encoder.eos_state

    def forward(
        self, text_states: torch.Tensor, text_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, width = text_states.shape
        text_lengths = text_mask.sum(dim=1).long()

        bos = self.bos_state.to(text_states.dtype).view(1, 1, width)
        left = torch.cat((bos.expand(batch, 1, width), text_states), dim=1)

        eos = self.eos_state.to(text_states.dtype).view(1, 1, width)
        right = torch.cat((text_states, eos.expand(batch, 1, width)), dim=1)
        eos_indices = text_lengths.clamp(max=length).view(batch, 1, 1)
        eos_indices = eos_indices.expand(batch, 1, width)
        right = right.scatter(1, eos_indices, eos.expand(batch, 1, width))

        left_projected = self.left_projection(left)
        right_projected = self.right_projection(right)
        states = self.output_projection(
            torch.cat((left_projected, right_projected), dim=-1)
        )
        states = self.dropout(self.layer_norm(states))
        positions = torch.arange(length + 1, device=text_states.device).unsqueeze(0)
        boundary_mask = positions <= text_lengths.unsqueeze(1)
        for block in self.attention_blocks:
            states = block(states, boundary_mask)
        for block in self.refinement_blocks:
            states = block(states)
        states = states * boundary_mask.unsqueeze(-1).to(states.dtype)
        return states, boundary_mask


class BoundaryMarginalGraph(torch.nn.Module):
    """Boundary encoder, marginal query head, and shared-pool projections."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        head = model.boundary_head
        self.boundary_encoder = FiniteBoundaryEncoder(head.boundary_encoder)
        self.query_head = head.boundary_query_head
        # These are deliberately the shared pool projections, not the sparse
        # proposer/scorer endpoint projections.
        self.pool_start_projection = head.shared_pool_builder.start_projection
        self.pool_end_projection = head.shared_pool_builder.end_projection

    def forward(
        self,
        text_states: torch.Tensor,
        text_mask: torch.Tensor,
        query_states: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        boundary_states, boundary_mask = self.boundary_encoder(
            text_states, text_mask
        )
        marginals = self.query_head(
            boundary_states,
            boundary_mask,
            text_states,
            text_mask,
            query_states,
            query_mask,
        )
        start_all = self.pool_start_projection(boundary_states)
        end_all = self.pool_end_projection(boundary_states)
        batch, queries, _ = query_states.shape
        # Preserve the pinned CPU oracle's fp32 sum order explicitly. ORT's
        # ReduceSum ordering perturbs the mean, which accumulates across long
        # centered prefixes (especially the multilingual checkpoint). Nothing
        # about the original masking, division, centering or scan is changed.
        token_keep = text_mask.unsqueeze(1) & query_mask.unsqueeze(-1)
        inside_values = marginals.inside_logits.masked_fill(~token_keep, 0.0).float()
        valid_count = token_keep.sum(-1, keepdim=True).clamp_min(1)
        inside_prefix_mean = (sum_last_axis(inside_values) / valid_count).detach()
        # Keep the explicit [B,Q,1] ABI for supported Q=0 diagnostic cases.
        inside_prefix_mean = inside_prefix_mean.reshape(batch, queries, 1)
        centered = (inside_values - inside_prefix_mean) * token_keep.to(torch.float32)
        zeros = torch.zeros(
            batch, queries, 1, dtype=torch.float32, device=inside_values.device
        )
        inside_prefix = torch.cat((zeros, centered.cumsum(dim=-1)), dim=-1)
        return (
            boundary_states,
            boundary_mask,
            marginals.start_logits,
            marginals.end_logits,
            marginals.inside_logits,
            inside_prefix,
            inside_prefix_mean,
            start_all,
            end_all,
        )


def oracle_outputs(
    model: torch.nn.Module,
    text_states: torch.Tensor,
    text_mask: torch.Tensor,
    query_states: torch.Tensor,
    query_mask: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Run the unmodified upstream modules for wrapper-parity checks."""
    head = model.boundary_head
    encoding = head.boundary_encoder(text_states, text_mask)
    marginals = head.boundary_query_head(
        encoding.states,
        encoding.mask,
        text_states,
        text_mask,
        query_states,
        query_mask,
    )
    return (
        encoding.states,
        encoding.mask,
        marginals.start_logits,
        marginals.end_logits,
        marginals.inside_logits,
        marginals.inside_prefix,
        marginals.inside_prefix_mean,
        head.shared_pool_builder.start_projection(encoding.states),
        head.shared_pool_builder.end_projection(encoding.states),
    )


def assert_wrapper_matches_oracle(
    names: Sequence[str],
    expected: Sequence[torch.Tensor],
    actual: Sequence[torch.Tensor],
    *,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> None:
    """Check every wrapper output, including exact boolean boundary masks."""
    if len(names) != len(expected) or len(names) != len(actual):
        raise AssertionError("marginal output count mismatch")
    for name, reference, wrapped in zip(names, expected, actual):
        if reference.dtype == torch.bool:
            if not torch.equal(reference, wrapped):
                raise AssertionError(f"{name}: finite wrapper boolean output differs")
        else:
            torch.testing.assert_close(
                wrapped,
                reference,
                atol=atol,
                rtol=rtol,
                msg=lambda message: f"{name}: finite wrapper differs from oracle: {message}",
            )
