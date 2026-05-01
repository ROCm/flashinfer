# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import math
import time
from typing import Optional, Tuple

import torch

# HIPBLAS error markers that indicate transient handle-pool exhaustion under
# concurrent xdist load. PyTorch wraps these as plain RuntimeError, so we have
# to substring-match the message — there is no typed exception to catch.
_HIPBLAS_TRANSIENT_MARKERS = ("HIPBLAS_STATUS_ALLOC_FAILED", "hipblasCreate")
_HIPBLAS_RETRY_ATTEMPTS = 4
_HIPBLAS_RETRY_BACKOFF_S = 0.1  # exhaustion clears in tens of ms


def _hipblas_safe_call(fn, *args, **kwargs):
    """Retry any callable on transient HIPBLAS handle-pool exhaustion."""
    for _ in range(_HIPBLAS_RETRY_ATTEMPTS - 1):
        try:
            return fn(*args, **kwargs)
        except RuntimeError as e:
            if not any(m in str(e) for m in _HIPBLAS_TRANSIENT_MARKERS):
                raise
            time.sleep(_HIPBLAS_RETRY_BACKOFF_S)
    return fn(*args, **kwargs)


def _hipblas_safe_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``torch.matmul`` with retry on transient HIPBLAS handle-pool exhaustion.

    Under heavy concurrent xdist load on AMD CPX systems, ``hipblasCreate``
    occasionally returns ``HIPBLAS_STATUS_ALLOC_FAILED``. The kernel itself
    is fine — the failure is in the library's resource management. Retry
    with a short fixed back-off; non-transient RuntimeErrors propagate.
    """
    for _ in range(_HIPBLAS_RETRY_ATTEMPTS - 1):
        try:
            return torch.matmul(a, b)
        except RuntimeError as e:
            if not any(m in str(e) for m in _HIPBLAS_TRANSIENT_MARKERS):
                raise
            time.sleep(_HIPBLAS_RETRY_BACKOFF_S)
    return torch.matmul(a, b)  # final attempt; error propagates if it fails


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

    # Compute attention scores: [num_qo_heads, qo_len, kv_len]
    # When soft cap is used: compute raw scores WITHOUT sm_scale
    # When soft cap is NOT used: apply sm_scale directly
    if logits_soft_cap is not None:
        scores = _hipblas_safe_matmul(q_t, k_t.transpose(1, 2))
        scores = logits_soft_cap * torch.tanh(scores * sm_scale / logits_soft_cap)
    else:
        scores = _hipblas_safe_matmul(q_t, k_t.transpose(1, 2)) * sm_scale

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
    out = _hipblas_safe_matmul(attn, v_t)

    # Transpose back: [qo_len, num_qo_heads, head_dim]
    out = out.transpose(0, 1)

    return out, lse
