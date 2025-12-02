# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional, Tuple

import torch


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: Optional[bool] = False,
    pos_encoding_mode: Optional[str] = "NONE",
    logits_soft_cap: Optional[float] = None,
    return_lse: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Naive PyTorch implementation of attention for reference.

    Args:
        q: query tensor, shape: [qo_len, num_qo_heads, head_dim]
        k: key tensor, shape: [kv_len, num_kv_heads, head_dim], NHD layout
        v: value tensor, shape: [kv_len, num_kv_heads, head_dim], NHD layout
    Optional Args:
        causal: whether to apply causal masking
        pos_encoding_mode: the position encoding mode to use
        logits_soft_cap: if not None, applies soft cap to logits: soft_cap * tanh(logits / soft_cap)
        return_lse: whether to return the log sum exp value of the attention logits
    Returns:
        A tuple of two tensors: (o, lse), where:
        - o: output tensor, shape: [qo_len, num_qo_heads, head_dim]
        - lse: log sum exp value of the attention logits, shape: [qo_len, num_qo_heads], or None if return_lse is False
    """

    # The current validation only supports simple cases without RoPE/ALiBi
    if pos_encoding_mode != "NONE":
        raise ValueError(
            f"Only pos_encoding_mode == NONE is supported for this validation, got {pos_encoding_mode}"
        )

    qo_len, num_qo_heads, head_dim = q.shape
    kv_len, num_kv_heads, _ = k.shape

    sm_scale = 1.0 / math.sqrt(head_dim)  # softmax scale

    # Handle grouped query attention (GQA)
    group_size = num_qo_heads // num_kv_heads

    # Expand k and v to match q's head dimension if using GQA
    if group_size > 1:
        k = k.repeat_interleave(group_size, dim=1)  # [kv_len, num_qo_heads, head_dim]
        v = v.repeat_interleave(group_size, dim=1)  # [kv_len, num_qo_heads, head_dim]

    # Transpose for batch matrix multiply: [num_heads, seq_len, head_dim]
    q_t = q.transpose(0, 1)  # [num_qo_heads, qo_len, head_dim]
    k_t = k.transpose(0, 1)  # [num_qo_heads, kv_len, head_dim]
    v_t = v.transpose(0, 1)  # [num_qo_heads, kv_len, head_dim]

    # DEBUG: Print raw QK scores (before scaling, soft cap, or masking)
    raw_scores = torch.matmul(
        q_t, k_t.transpose(1, 2)
    )  # [num_qo_heads, qo_len, kv_len]
    print(f"\n{'='*80}")
    print(f"REFERENCE: Raw QK scores (Q @ K^T, before sm_scale)")
    print(f"{'='*80}")
    print(f"Shape: {raw_scores.shape}")
    print(f"Head 0, raw QK scores [qo_len={qo_len}, kv_len={kv_len}]:")
    print(raw_scores[0].cpu().numpy())
    print(f"{'='*80}\n")

    # Compute attention scores: [num_qo_heads, qo_len, kv_len]
    # When soft cap is used: compute raw scores WITHOUT sm_scale
    # When soft cap is NOT used: apply sm_scale directly
    if logits_soft_cap is not None:
        scores = torch.matmul(q_t, k_t.transpose(1, 2))
        scores = logits_soft_cap * torch.tanh(scores * sm_scale / logits_soft_cap)
    else:
        scores = torch.matmul(q_t, k_t.transpose(1, 2)) * sm_scale

    # Apply causal mask if needed (AFTER soft cap)
    if causal:
        mask = torch.tril(
            torch.ones((qo_len, kv_len), device=q.device, dtype=torch.bool),
            diagonal=(kv_len - qo_len),
        )
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))

    # Compute LSE on the final scores (after soft cap is applied) and before softmax
    lse = None
    if return_lse:
        lse = torch.logsumexp(scores, dim=-1)  # [num_qo_heads, qo_len]
        lse = lse / math.log(2)  # to match FlashInfer implementation
        lse = lse.transpose(0, 1)  # [qo_len, num_qo_heads]

    # Softmax
    attn = torch.softmax(scores, dim=-1)

    # Apply attention to values: [num_qo_heads, qo_len, head_dim]
    out = torch.matmul(attn, v_t)

    # Transpose back: [qo_len, num_qo_heads, head_dim]
    out = out.transpose(0, 1)

    return out, lse
