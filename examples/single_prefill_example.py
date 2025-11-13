# SPDX-FileCopyrightText : 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier : Apache-2.0

import math
from typing import Optional

import torch

import flashinfer


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    logits_soft_cap: Optional[float] = None,
) -> torch.Tensor:
    """
    Naive PyTorch implementation of attention for reference.

    Args:
        q: query tensor, shape: [qo_len, num_qo_heads, head_dim]
        k: key tensor, shape: [kv_len, num_kv_heads, head_dim]
        v: value tensor, shape: [kv_len, num_kv_heads, head_dim]
        causal: whether to apply causal masking
        sm_scale: softmax scale (default: 1/sqrt(head_dim))
        logits_soft_cap: if not None, applies soft cap to logits: soft_cap * tanh(logits / soft_cap)

    Returns:
        o: output tensor, shape: [qo_len, num_qo_heads, head_dim]
    """
    qo_len, num_qo_heads, head_dim = q.shape
    kv_len, num_kv_heads, _ = k.shape

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

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
    scores = torch.matmul(q_t, k_t.transpose(1, 2)) * sm_scale

    # Apply logits soft cap if specified
    # Formula: soft_cap * tanh(logits / soft_cap)
    if logits_soft_cap is not None and logits_soft_cap > 0:
        scores = logits_soft_cap * torch.tanh(scores / logits_soft_cap)

    # Apply causal mask if needed
    if causal:
        # Create causal mask
        mask = torch.tril(
            torch.ones((qo_len, kv_len), device=q.device, dtype=torch.bool),
            diagonal=(kv_len - qo_len),
        )
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))

    # Softmax
    attn = torch.softmax(scores, dim=-1)

    # Apply attention to values: [num_qo_heads, qo_len, head_dim]
    out = torch.matmul(attn, v_t)

    # Transpose back: [qo_len, num_qo_heads, head_dim]
    out = out.transpose(0, 1)

    return out


def single_prefill_with_kv_cache_example(
    qo_len: int,
    kv_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    kv_layout: str,
    pos_encoding_mode: str,
    logits_soft_cap: float,
    causal: bool,
    return_lse: bool,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
):
    """
    Run single_prefill_with_kv_cache and verify the output against a naive PyTorch reference implementation.

    This function creates random Q, K, V tensors and compares the output
    of flashinfer's single_prefill_with_kv_cache against a naive PyTorch reference implementation.
    """
    print("\nRunning configuration:")
    print(f"  qo_len={qo_len}")
    print(f"  kv_len={kv_len}")
    print(f"  num_qo_heads={num_qo_heads}")
    print(f"  num_kv_heads={num_kv_heads}")
    print(f"  head_dim={head_dim}")
    print(f"  kv_layout={kv_layout}")
    print(f"  pos_encoding_mode={pos_encoding_mode}")
    print(f"  logits_soft_cap={logits_soft_cap}")
    print(f"  causal={causal}")
    print(f"  return_lse={return_lse}")
    print(f"  q_dtype={q_dtype}")
    print(f"  kv_dtype={kv_dtype}")

    # Create query tensor: [qo_len, num_qo_heads, head_dim]
    q = torch.randn(qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=q_dtype)

    # Create key and value tensors based on layout
    if kv_layout == "HND":
        # [num_kv_heads, kv_len, head_dim]
        k = torch.randn(num_kv_heads, kv_len, head_dim, device="cuda:0", dtype=kv_dtype)
        v = torch.randn(num_kv_heads, kv_len, head_dim, device="cuda:0", dtype=kv_dtype)
        # Convert to NHD for reference implementation
        k_ref = k.transpose(0, 1).contiguous()  # [kv_len, num_kv_heads, head_dim]
        v_ref = v.transpose(0, 1).contiguous()  # [kv_len, num_kv_heads, head_dim]
    else:  # NHD layout
        # [kv_len, num_kv_heads, head_dim]
        k = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=kv_dtype)
        v = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=kv_dtype)
        k_ref = k
        v_ref = v

    # Call flashinfer API
    if return_lse:
        o, lse = flashinfer.single_prefill_with_kv_cache(
            q,
            k,
            v,
            causal=causal,
            kv_layout=kv_layout,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap if logits_soft_cap > 0 else None,
            backend="auto",
            return_lse=True,
        )
        print(f"  FlashInfer output shape: {o.shape}, LSE shape: {lse.shape}")
    else:
        o = flashinfer.single_prefill_with_kv_cache(
            q,
            k,
            v,
            causal=causal,
            kv_layout=kv_layout,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap if logits_soft_cap > 0 else None,
            backend="auto",
            return_lse=False,
        )
        print(f"  FlashInfer output shape: {o.shape}")

    # For simple cases without RoPE/ALiBi, verify against naive implementation
    if pos_encoding_mode == "NONE":
        o_ref = naive_attention(
            q.float(),
            k_ref.float(),
            v_ref.float(),
            causal=causal,
            logits_soft_cap=logits_soft_cap if logits_soft_cap > 0 else None,
        ).to(q_dtype)
        print(f"  Reference output shape: {o_ref.shape}")

        try:
            torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
            print("  ✓ PASS: Output matches reference implementation")
        except AssertionError as e:
            print("  ✗ FAIL: Output does not match reference implementation")
            max_diff = (o - o_ref).abs().max().item()
            print(f"    Max absolute difference: {max_diff}")


if __name__ == "__main__":
    print("=" * 60)
    print("FlashInfer Single Prefill Example")
    print("=" * 60)

    # Basic test with logits soft cap
    single_prefill_with_kv_cache_example(
        qo_len=128,
        kv_len=128,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim=64,
        kv_layout="NHD",
        pos_encoding_mode="NONE",  # No RoPE/ALiBi
        logits_soft_cap=8.0,  # soft cap enabled
        causal=False,  # no causal mask
        return_lse=False,  # no log sum exp
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
    )

    # Without logits soft cap
    single_prefill_with_kv_cache_example(
        qo_len=128,
        kv_len=128,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim=64,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        logits_soft_cap=0.0,  # soft cap disabled
        causal=False,
        return_lse=False,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
    )

    # GQA with num_qo_heads = num_kv_heads > 1
    single_prefill_with_kv_cache_example(
        qo_len=128,
        kv_len=128,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=64,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        logits_soft_cap=8.0,
        causal=False,
        return_lse=False,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
    )

    # GQA with num_qo_heads > num_kv_heads
    single_prefill_with_kv_cache_example(
        qo_len=128,
        kv_len=128,
        num_qo_heads=8,
        num_kv_heads=4,
        head_dim=64,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        logits_soft_cap=8.0,
        causal=False,
        return_lse=False,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
    )

    # GQA with qo_len < kv_len (typical prefill)
    single_prefill_with_kv_cache_example(
        qo_len=15,
        kv_len=127,
        num_qo_heads=32,
        num_kv_heads=4,
        head_dim=64,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        logits_soft_cap=8.0,
        causal=False,
        return_lse=False,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
    )
