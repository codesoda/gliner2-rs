"""Source-ordered fp32 reduction for exported centered marginal prefixes.

This is the four-lane/four-partial/four-level inner sum used by the pinned
PyTorch 2.8 CPU oracle, mirroring ``src/boundary/pool.rs::compatibility``
without its products or final scale. Host SIMD width must not change the order.
Only the masked-logit sum is replaced: masking, integer valid
counts, fp32 division/subtraction and cumsum remain the caller's responsibility.

``sum_last_axis`` is scripted, not traced: all length-dependent loops and
cascade decisions survive legacy ONNX export as dynamic control flow. Export
``AArch64SumLastAxis`` with opset 17, dynamo=False and dynamic B/Q/L axes.
The ONNX ABI is float32 [B,Q,L] -> float32 [B,Q,1]; do not feed other dtypes.
Empty axes work in Torch, but are deliberately not certified for ONNX Runtime.
"""

import torch


@torch.jit.script
def _level_step(cascade_size: int) -> int:
    # Exact integer ceil(log2(cascade_size)); no floating logarithms/powers.
    remaining = cascade_size - 1
    ceil_log2 = 0
    while remaining > 0:
        remaining = remaining // 2
        ceil_log2 += 1
    level_power = max(4, ceil_log2 // 4)
    level_step = 1
    exponent = 0
    while exponent < level_power:
        level_step *= 2
        exponent += 1

    return level_step


@torch.jit.script
def sum_last_axis(values: torch.Tensor) -> torch.Tensor:
    """Sum contiguous logical rows in fixed AArch64 fp32 order, keeping dim -1."""
    assert values.dim() == 3, "expected [B,Q,L]"
    assert values.dtype == torch.float32, "expected fp32"
    batch, queries, length = values.shape
    vector_count = length // 4
    cascade_size = vector_count // 4
    level_step = _level_step(cascade_size)

    zeros = torch.zeros((batch, queries, 4, 4), dtype=torch.float32, device=values.device)
    level0 = zeros
    level1 = zeros
    level2 = zeros
    level3 = zeros
    index = 0
    while index + level_step <= cascade_size:
        stop = index + level_step
        while index < stop:
            # [partial,lane]: adjacent vectors go to independent partials.
            block = values[:, :, index * 16 : (index + 1) * 16]
            level0 = level0 + block.reshape(batch, queries, 4, 4)
            index += 1
        level1 = level1 + level0
        level0 = zeros
        # Equivalent to index & ((level_step-1) << level_power) == 0,
        # using integer division/modulo, which opset 17 can represent directly.
        if (index // level_step) % level_step == 0:
            level2 = level2 + level1
            level1 = zeros
            if (index // (level_step * level_step)) % level_step == 0:
                level3 = level3 + level2
                level2 = zeros

    while index < cascade_size:
        block = values[:, :, index * 16 : (index + 1) * 16]
        level0 = level0 + block.reshape(batch, queries, 4, 4)
        index += 1

    level0 = level0 + level1
    level0 = level0 + level2
    level0 = level0 + level3
    lanes = level0[:, :, 0, :]
    vector = cascade_size * 4
    while vector < vector_count:
        lanes = lanes + values[:, :, vector * 4 : (vector + 1) * 4]
        vector += 1
    # Deliberately not ReduceSum: each addition is a rounding boundary.
    lanes = lanes + level0[:, :, 1, :]
    lanes = lanes + level0[:, :, 2, :]
    lanes = lanes + level0[:, :, 3, :]

    total = torch.zeros((batch, queries, 1), dtype=torch.float32, device=values.device)
    element = vector_count * 4
    while element < length:
        total = total + values[:, :, element : element + 1]
        element += 1
    total = total + lanes[:, :, 0:1]
    total = total + lanes[:, :, 1:2]
    total = total + lanes[:, :, 2:3]
    total = total + lanes[:, :, 3:4]
    return total


class AArch64SumLastAxis(torch.nn.Module):
    """Weightless export shell around the scripted reduction."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return sum_last_axis(values)
