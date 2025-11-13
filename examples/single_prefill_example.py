# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional, Tuple, Union

import torch

import flashinfer


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: Optional[bool] = False,
    return_lse: Optional[bool] = False,
    logits_soft_cap: Optional[float] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Naive PyTorch implementation of attention for reference.

    Args:
        q: query tensor, shape: [qo_len, num_qo_heads, head_dim]
        k: key tensor, shape: [kv_len, num_kv_heads, head_dim]
        v: value tensor, shape: [kv_len, num_kv_heads, head_dim]
    Optional Args:
        causal: whether to apply causal masking
        return_lse: whether to return the log sum exp value of the attention logits
        logits_soft_cap: if not None, applies soft cap to logits: soft_cap * tanh(logits / soft_cap)
    Returns:
        o: output tensor, shape: [qo_len, num_qo_heads, head_dim] if return_lse is None or False,
        otherwise a tuple of two tensors: (o, lse), where o is the output tensor, shape: [qo_len, num_qo_heads, head_dim],
        and lse is the log sum exp value of the attention logits, shape: [qo_len, num_qo_heads]
    """
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
    if return_lse:
        lse = torch.logsumexp(scores, dim=-1) / math.log(2)  # [num_qo_heads, qo_len]
        lse = lse.transpose(0, 1)  # [qo_len, num_qo_heads]

    # Softmax
    attn = torch.softmax(scores, dim=-1)

    # Apply attention to values: [num_qo_heads, qo_len, head_dim]
    out = torch.matmul(attn, v_t)

    # Transpose back: [qo_len, num_qo_heads, head_dim]
    out = out.transpose(0, 1)

    if return_lse:
        return out, lse
    else:
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
    q_dtype: Optional[torch.dtype] = torch.float16,
    kv_dtype: Optional[torch.dtype] = torch.float16,
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

    # The current validation only supports simple cases without RoPE/ALiBi
    if pos_encoding_mode != "NONE":
        print(
            "Only pos_encoding_mode == NONE is supported for this validation. Skipping..."
        )
        return

    logits_soft_cap = logits_soft_cap if logits_soft_cap > 0 else None

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
        o, lse = flashinfer.single_prefill_with_kv_cache_return_lse(
            q,
            k,
            v,
            causal=causal,
            kv_layout=kv_layout,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
            backend="auto",
        )
        print(f"  FlashInfer output shape: {o.shape}, LSE shape: {lse.shape}")
        # Compute reference in FP32 for better accuracy
        o_ref, lse_ref = naive_attention(
            q.float(),
            k_ref.float(),
            v_ref.float(),
            causal=causal,
            return_lse=True,
            logits_soft_cap=logits_soft_cap,
        )
        # Convert reference back to match FlashInfer dtype
        o_ref = o_ref.to(o.dtype)
        lse_ref = lse_ref.to(lse.dtype)
        print(f"  Reference output shape: {o_ref.shape}, LSE shape: {lse_ref.shape}")
        try:
            torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-3)
            print("  ✓ PASS: Output and LSE match reference implementation")
        except AssertionError:
            print("  ✗ FAIL: Output or LSE does not match reference implementation")
            max_diff_o = (o - o_ref).abs().max().item()
            max_diff_lse = (lse - lse_ref).abs().max().item()
            print(f"    Max absolute difference for output: {max_diff_o}")
            print(f"    Max absolute difference for LSE: {max_diff_lse}")
    else:
        o = flashinfer.single_prefill_with_kv_cache(
            q,
            k,
            v,
            causal=causal,
            kv_layout=kv_layout,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
            backend="auto",
        )
        print(f"  FlashInfer output shape: {o.shape}")

        # Compute reference in FP32 for better accuracy
        o_ref = naive_attention(
            q.float(),
            k_ref.float(),
            v_ref.float(),
            causal=causal,
            return_lse=False,
            logits_soft_cap=logits_soft_cap,
        )
        # Convert reference back to match FlashInfer dtype
        o_ref = o_ref.to(q.dtype)
        print(f"  Reference output shape: {o_ref.shape}")

        try:
            torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
            print("  ✓ PASS: Output matches reference implementation")
        except AssertionError:
            print("  ✗ FAIL: Output does not match reference implementation")
            max_diff_o = (o - o_ref).abs().max().item()
            print(f"    Max absolute difference for output: {max_diff_o}")


if __name__ == "__main__":
    print("=" * 60)
    print("FlashInfer Single Prefill Example")
    print("=" * 60)

    # Self-attention with logits soft cap
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
    )

    # Self-attention without logits soft cap
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
    )

    # Multi-head attention (MHA)
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
    )

    # Grouped query attention (GQA)
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
    )

    # GQA with LSE enabled
    single_prefill_with_kv_cache_example(
        qo_len=15,
        kv_len=127,
        num_qo_heads=8,
        num_kv_heads=4,
        head_dim=64,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        logits_soft_cap=0.0,
        causal=False,
        return_lse=True,
    )

    # GQA with soft cap and LSE enabled
    single_prefill_with_kv_cache_example(
        qo_len=15,
        kv_len=127,
        num_qo_heads=8,
        num_kv_heads=4,
        head_dim=64,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        logits_soft_cap=8.0,
        causal=False,
        return_lse=True,
    )
