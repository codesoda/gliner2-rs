"""Inference-only GLiNER2.5 record graph and padded ABI helpers.

Ragged candidate gathering, seed formation, record metadata, and decoding stay in
Rust.  This module contains only the learned ``RecordHead.forward_group`` math.
The wrapper references the checkpoint's existing modules and parameters directly;
it does not copy, replace, or reinitialize oracle weights.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

MASK_LOGIT = -1.0e4
MODE_NATURAL = 0
MODE_LATENT = 1
MODE_ANCHORLESS = 2

INPUT_NAMES = (
    "field_query_states",
    "field_candidate_states",
    "field_candidate_mask",
    "context_states",
    "context_mask",
    "seed_states",
    "seed_object_logits",
    "mode",
)
OUTPUT_NAMES = (
    "instance_states",
    "object_logits",
    "assignment_logits",
)


class BoundaryRecordsGraph(torch.nn.Module):
    """Learned inference kernel corresponding to ``RecordHead.forward_group``.

    Inputs are batch-one, with the batch axis deliberately removed.  ``mode`` is
    a scalar int64 tensor: natural=0, latent=1, anchorless=2.  The scripted
    tensor-dependent branches are retained as ONNX ``If`` nodes so one graph
    supports all three modes and a dynamic instance count.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        head = model.record_decoder
        self.record_dim = int(head.record_dim)
        self.instance_queries = int(head.instance_queries)

        self.inst_proj = head.inst_proj
        self.field_proj = head.field_proj
        self.cand_proj = head.cand_proj
        self.object_head = head.object_head
        self.latent_seed_head = head.latent_seed_head
        self.q_proj = head.q_proj
        self.k_proj = head.k_proj
        self.v_proj = head.v_proj
        self.instance_embed = head.instance_embed
        self.null_embed = head.null_embed

    def forward(
        self,
        field_query_states: torch.Tensor,      # [F,H]
        field_candidate_states: torch.Tensor,  # [F,C,H]
        field_candidate_mask: torch.Tensor,    # [F,C]
        context_states: torch.Tensor,          # [M,H]
        context_mask: torch.Tensor,            # [M]
        seed_states: torch.Tensor,             # [Ni,H]
        seed_object_logits: torch.Tensor,      # [Ni]
        mode: torch.Tensor,                    # scalar int64
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mode == 2:
            instance_states = self.instance_embed
            q = self.q_proj(instance_states)
            k = self.k_proj(context_states)
            v = self.v_proj(context_states)
            attention = torch.matmul(q, k.transpose(0, 1))
            attention = attention / math.sqrt(self.record_dim)
            attention = attention.masked_fill(~context_mask.unsqueeze(0), -1.0e4)
            weights = torch.softmax(attention, dim=-1)
            # Finite masking plus this explicit zeroing makes an all-invalid,
            # one-row context contribute exactly zero (no unsafe M=0 tensor).
            weights = weights.masked_fill(~context_mask.unsqueeze(0), 0.0)
            pooled = torch.matmul(weights, v)
            instance_states = instance_states + pooled
            object_logits = self.object_head(instance_states).squeeze(-1)
        else:
            instance_states = seed_states
            if mode == 0:
                object_logits = seed_object_logits
            else:
                object_logits = self.latent_seed_head(instance_states).squeeze(-1)

        instance_query = self.inst_proj(instance_states)             # [J,D]
        field_query = self.field_proj(field_query_states)            # [F,D]
        query = instance_query.unsqueeze(1) + field_query.unsqueeze(0)  # [J,F,D]
        null_logits = torch.matmul(query, self.null_embed)            # [J,F]
        candidate_states = self.cand_proj(field_candidate_states)     # [F,C,D]
        candidate_logits = torch.einsum("jfd,fcd->jfc", query, candidate_states)
        candidate_logits = candidate_logits.masked_fill(
            ~field_candidate_mask.unsqueeze(0), -1.0e4
        )
        assignment = torch.cat(
            (null_logits.unsqueeze(-1), candidate_logits), dim=-1
        )
        return instance_states, object_logits, assignment.permute(1, 0, 2)


def validate_abi_inputs(inputs: Sequence[torch.Tensor]) -> None:
    """Validate the positive-dimension caller contract before invoking ONNX."""
    if len(inputs) != len(INPUT_NAMES):
        raise ValueError(f"expected {len(INPUT_NAMES)} record inputs, got {len(inputs)}")
    fq, fc, fm, ctx, cm, seeds, seed_logits, mode = inputs
    if fq.dim() != 2 or fc.dim() != 3 or fm.dim() != 2:
        raise ValueError("field tensors must be [F,H], [F,C,H], and [F,C]")
    if ctx.dim() != 2 or cm.dim() != 1:
        raise ValueError("context tensors must be [M,H] and [M]")
    if seeds.dim() != 2 or seed_logits.dim() != 1 or mode.dim() != 0:
        raise ValueError("seed tensors must be [Ni,H], [Ni], with scalar mode")
    f, c, hidden = fc.shape
    if min(f, c, ctx.shape[0], seeds.shape[0]) < 1:
        raise ValueError("record graph requires positive F/C/M/seed-Ni dimensions")
    if fq.shape != (f, hidden) or fm.shape != (f, c):
        raise ValueError("field input dimensions disagree")
    if ctx.shape[1] != hidden or seeds.shape[1] != hidden:
        raise ValueError("record hidden dimensions disagree")
    if cm.shape[0] != ctx.shape[0] or seed_logits.shape[0] != seeds.shape[0]:
        raise ValueError("record mask/seed dimensions disagree")
    if fq.dtype != torch.float32 or fc.dtype != torch.float32:
        raise TypeError("record state inputs must be fp32")
    if ctx.dtype != torch.float32 or seeds.dtype != torch.float32:
        raise TypeError("context and seed state inputs must be fp32")
    if seed_logits.dtype != torch.float32:
        raise TypeError("seed_object_logits must be fp32")
    if fm.dtype != torch.bool or cm.dtype != torch.bool:
        raise TypeError("record masks must be bool")
    if mode.dtype != torch.int64:
        raise TypeError("mode must be scalar int64")
    mode_value = int(mode.item())
    if mode_value not in (MODE_NATURAL, MODE_LATENT, MODE_ANCHORLESS):
        raise ValueError(f"unknown record mode {mode_value}")


def mode_id(mode: str) -> int:
    try:
        return {"natural": 0, "latent": 1, "anchorless": 2}[mode]
    except KeyError as exc:
        raise ValueError(f"unknown record mode {mode!r}") from exc


def prepare_group_inputs(
    spec: Any,
    query_states: torch.Tensor,
    candidates: Any,
    sample_index: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Perform the source-faithful ragged/discrete preparation kept outside ONNX.

    This mirrors only the gathers and seed construction around the learned
    kernel.  Valid candidates are compacted in field order; duplicates across
    fields are intentionally retained in latent seeds and anchorless context.
    """
    field_query_ids = [field.query_id for field in spec.fields]
    query_count = min(
        query_states.shape[0],
        candidates.valid_mask.shape[1],
        candidates.pair_logits.shape[1],
        candidates.candidate_states.shape[1],
    )
    if query_count <= 0:
        raise ValueError("record routing requires at least one boundary query")
    safe_ids = [min(max(query_id, 0), query_count - 1) for query_id in field_query_ids]
    valid_ids = [0 <= query_id < query_count for query_id in field_query_ids]
    field_queries = query_states[safe_ids].to(dtype=torch.float32)

    compact_states: list[torch.Tensor] = []
    compact_logits: list[torch.Tensor] = []
    for query_id, query_valid in zip(safe_ids, valid_ids):
        valid = candidates.valid_mask[sample_index, query_id] & query_valid
        keep = torch.nonzero(valid, as_tuple=False).flatten()
        compact_states.append(
            candidates.candidate_states[sample_index, query_id][keep].to(torch.float32)
        )
        compact_logits.append(
            candidates.pair_logits[sample_index, query_id][keep].to(torch.float32)
        )

    hidden = int(query_states.shape[-1])
    field_count = len(compact_states)
    if field_count < 1:
        raise ValueError("record graph requires at least one field")
    candidate_count = max(max((value.shape[0] for value in compact_states), default=0), 1)
    field_candidates = query_states.new_zeros(
        (field_count, candidate_count, hidden), dtype=torch.float32
    )
    field_mask = torch.zeros(
        (field_count, candidate_count), dtype=torch.bool, device=query_states.device
    )
    for field_index, states in enumerate(compact_states):
        count = states.shape[0]
        if count:
            field_candidates[field_index, :count] = states
            field_mask[field_index, :count] = True

    nonempty = [states for states in compact_states if states.shape[0] > 0]
    if nonempty:
        context_states = torch.cat(nonempty, dim=0)
        context_mask = torch.ones(
            context_states.shape[0], dtype=torch.bool, device=query_states.device
        )
    else:
        context_states = query_states.new_zeros((1, hidden), dtype=torch.float32)
        context_mask = torch.zeros(1, dtype=torch.bool, device=query_states.device)

    selected_mode = mode_id(spec.mode)
    if selected_mode == MODE_NATURAL:
        try:
            anchor_field = field_query_ids.index(spec.anchor_query_id)
        except ValueError as exc:
            raise ValueError("natural record anchor is not a field") from exc
        seed_states = compact_states[anchor_field]
        seed_object_logits = compact_logits[anchor_field]
        if seed_states.shape[0] == 0:
            raise ValueError("natural record has no instances; caller must bypass ONNX")
    elif selected_mode == MODE_LATENT:
        if not nonempty:
            raise ValueError("latent record has no instances; caller must bypass ONNX")
        seed_states = torch.cat(nonempty, dim=0)
        seed_object_logits = query_states.new_zeros(
            seed_states.shape[0], dtype=torch.float32
        )
    else:
        # Positive seed-Ni is part of the graph contract even though anchorless
        # mode selects the 32 learned queries dynamically and ignores these rows.
        seed_states = query_states.new_zeros((1, hidden), dtype=torch.float32)
        seed_object_logits = query_states.new_zeros(1, dtype=torch.float32)

    result = (
        field_queries,
        field_candidates,
        field_mask,
        context_states,
        context_mask,
        seed_states,
        seed_object_logits,
        torch.tensor(selected_mode, dtype=torch.int64, device=query_states.device),
    )
    validate_abi_inputs(result)
    return result


def padded_source_assignments(
    assign_logits: Sequence[torch.Tensor], candidate_count: int
) -> torch.Tensor:
    """Pad source ragged ``List[Ni,Cf+1]`` to ABI ``[F,Ni,C+1]``."""
    if not assign_logits:
        raise ValueError("source record group has no fields")
    instances = int(assign_logits[0].shape[0])
    output = assign_logits[0].new_full(
        (len(assign_logits), instances, candidate_count + 1), MASK_LOGIT
    )
    for field_index, logits in enumerate(assign_logits):
        output[field_index, :, : logits.shape[1]] = logits
    return output


def assert_wrapper_matches_group(
    source_group: Any,
    source_instance_states: torch.Tensor,
    inputs: Sequence[torch.Tensor],
    actual: Sequence[torch.Tensor],
    *,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> None:
    """Compare wrapper outputs with untouched ``RecordHead.forward_group`` output."""
    instance_states, object_logits, assignments = actual
    candidate_count = int(inputs[1].shape[1])
    expected_assignments = padded_source_assignments(
        source_group.assign_logits, candidate_count
    )
    expected = (
        source_instance_states,
        source_group.object_logits,
        expected_assignments,
    )
    for name, reference, observed in zip(OUTPUT_NAMES, expected, actual):
        torch.testing.assert_close(
            observed,
            reference,
            atol=atol,
            rtol=rtol,
            msg=lambda message: f"{name}: wrapper differs from forward_group: {message}",
        )

    mask = inputs[2]
    invalid = ~mask[:, None, :].expand(
        mask.shape[0], instance_states.shape[0], mask.shape[1]
    )
    invalid_values = assignments[..., 1:][invalid]
    if invalid_values.numel() and not torch.equal(
        invalid_values, torch.full_like(invalid_values, MASK_LOGIT)
    ):
        raise AssertionError("invalid padded assignment columns are not exactly -1e4")
