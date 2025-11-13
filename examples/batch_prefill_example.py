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


def batch_prefill_with_paged_kv_cache_example(
    batch_size: int,
    qo_len: int,
    kv_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    kv_layout: str,
    pos_encoding_mode: str,
    logits_soft_cap: float,
    causal: bool,
    return_lse: bool,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    backend: str,
):
    """
    Run batch_prefill_with_paged_kv_cache and verify the output against:
    1. Naive PyTorch reference implementation
    2. single_prefill_with_kv_cache implementation

    This function creates random Q, K, V tensors organized in a paged KV cache format
    and compares the output of flashinfer's batch_prefill_with_paged_kv_cache against
    both reference implementations.
    """
    print("\nRunning configuration:")
    print(f"  batch_size={batch_size}")
    print(f"  qo_len={qo_len}")
    print(f"  kv_len={kv_len}")
    print(f"  num_qo_heads={num_qo_heads}")
    print(f"  num_kv_heads={num_kv_heads}")
    print(f"  head_dim={head_dim}")
    print(f"  page_size={page_size}")
    print(f"  kv_layout={kv_layout}")
    print(f"  pos_encoding_mode={pos_encoding_mode}")
    print(f"  logits_soft_cap={logits_soft_cap}")
    print(f"  causal={causal}")
    print(f"  return_lse={return_lse}")
    print(f"  q_dtype={q_dtype}")
    print(f"  kv_dtype={kv_dtype}")
    print(f"  backend={backend}")

    # Create flattened query tensor: [batch_size * qo_len, num_qo_heads, head_dim]
    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=q_dtype,
    )

    # Create query indptr (index pointers for batch boundaries)
    q_indptr_cpu = torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len

    # Setup paged KV cache
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    # Create KV cache data
    if kv_layout == "HND":
        kv_shape = [total_num_pages, 2, num_kv_heads, page_size, head_dim]
    else:  # NHD
        kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]

    kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
    kv_data = kv_data_fp32.to(kv_dtype)

    # Create KV indptr and indices
    kv_indptr_cpu = (
        torch.arange(0, batch_size + 1, dtype=torch.int32) * num_pages_per_seq
    )
    kv_indices_cpu = torch.arange(0, total_num_pages, dtype=torch.int32)
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )

    # Move tensors to GPU
    q_indptr_gpu = q_indptr_cpu.to("cuda:0")
    kv_indptr_gpu = kv_indptr_cpu.to("cuda:0")
    kv_indices_gpu = kv_indices_cpu.to("cuda:0")
    kv_last_page_len_gpu = kv_last_page_len_cpu.to("cuda:0")

    # Create workspace buffer and wrapper
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend=backend
    )

    # Plan the batch prefill operation
    wrapper.plan(
        q_indptr_gpu,
        kv_indptr_gpu,
        kv_indices_gpu,
        kv_last_page_len_gpu,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
    )

    # Run batch prefill
    if return_lse:
        o, lse = wrapper.run(q, kv_data, return_lse=True)
        print(f"  FlashInfer batch output shape: {o.shape}, LSE shape: {lse.shape}")
    else:
        o = wrapper.run(q, kv_data)
        print(f"  FlashInfer batch output shape: {o.shape}")

    # Verify each sequence in the batch
    print("\n  Verifying individual sequences:")
    all_passed = True

    for i in range(batch_size):
        # Extract the i-th sequence from batch output
        o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]

        # Extract K and V for the i-th sequence from paged KV cache
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]

        # Reconstruct full pages and last page for K
        ki = torch.cat(
            [
                kv_data_fp32[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 0]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 0, :, : kv_last_page_len_cpu[i]
                    ]
                    if kv_layout == "HND"
                    else kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 0, : kv_last_page_len_cpu[i], :
                    ]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).to(kv_dtype)

        # Reconstruct full pages and last page for V
        vi = torch.cat(
            [
                kv_data_fp32[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 1, :, : kv_last_page_len_cpu[i]
                    ]
                    if kv_layout == "HND"
                    else kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 1, : kv_last_page_len_cpu[i], :
                    ]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).to(kv_dtype)

        # Compare with single_prefill_with_kv_cache
        o_ref_single = flashinfer.prefill.single_prefill_with_kv_cache(
            qi,
            ki,
            vi,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )

        try:
            torch.testing.assert_close(o_i, o_ref_single, rtol=1e-3, atol=1e-3)
            single_pass = True
        except AssertionError:
            single_pass = False
            max_diff = (o_i - o_ref_single).abs().max().item()
            print(f"    Sequence {i}: ✗ FAIL vs single_prefill (max diff: {max_diff})")

        # For simple cases without RoPE/ALiBi, also compare with naive implementation
        if pos_encoding_mode == "NONE":
            o_ref_naive = naive_attention(
                qi.float(),
                ki.float(),
                vi.float(),
                causal=causal,
                logits_soft_cap=logits_soft_cap if logits_soft_cap > 0 else None,
            ).to(q_dtype)

            try:
                torch.testing.assert_close(o_i, o_ref_naive, rtol=1e-3, atol=1e-3)
                naive_pass = True
            except AssertionError:
                naive_pass = False
                max_diff = (o_i - o_ref_naive).abs().max().item()
                print(f"    Sequence {i}: ✗ FAIL vs naive (max diff: {max_diff})")

            if single_pass and naive_pass:
                print(f"    Sequence {i}: ✓ PASS")
            all_passed = all_passed and single_pass and naive_pass
        else:
            if single_pass:
                print(f"    Sequence {i}: ✓ PASS")
            all_passed = all_passed and single_pass

    if all_passed:
        print("\n  ✓ ALL SEQUENCES PASSED")
    else:
        print("\n  ✗ SOME SEQUENCES FAILED")


if __name__ == "__main__":
    print("=" * 60)
    print("FlashInfer Batch Prefill Example")
    print("=" * 60)

    # Basic test with small batch
    batch_prefill_with_paged_kv_cache_example(
        batch_size=4,
        qo_len=128,
        kv_len=128,
        num_qo_heads=8,
        num_kv_heads=8,
        head_dim=64,
        page_size=16,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        logits_soft_cap=0.0,
        causal=False,
        return_lse=False,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        backend="fa2",
    )

    # Test with logits soft cap
    batch_prefill_with_paged_kv_cache_example(
        batch_size=4,
        qo_len=128,
        kv_len=128,
        num_qo_heads=8,
        num_kv_heads=8,
        head_dim=64,
        page_size=16,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        logits_soft_cap=8.0,
        causal=False,
        return_lse=False,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        backend="fa2",
    )

    # Test with causal masking
    batch_prefill_with_paged_kv_cache_example(
        batch_size=4,
        qo_len=128,
        kv_len=128,
        num_qo_heads=8,
        num_kv_heads=8,
        head_dim=64,
        page_size=16,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        logits_soft_cap=0.0,
        causal=True,
        return_lse=False,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        backend="fa2",
    )

    # Test with GQA (num_qo_heads > num_kv_heads)
    batch_prefill_with_paged_kv_cache_example(
        batch_size=4,
        qo_len=128,
        kv_len=128,
        num_qo_heads=32,
        num_kv_heads=4,
        head_dim=64,
        page_size=16,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        logits_soft_cap=8.0,
        causal=False,
        return_lse=False,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        backend="fa2",
    )

    # Test with different qo_len and kv_len
    batch_prefill_with_paged_kv_cache_example(
        batch_size=8,
        qo_len=64,
        kv_len=256,
        num_qo_heads=16,
        num_kv_heads=4,
        head_dim=128,
        page_size=16,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        logits_soft_cap=0.0,
        causal=False,
        return_lse=False,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        backend="fa2",
    )

    # Test with smaller page size
    batch_prefill_with_paged_kv_cache_example(
        batch_size=4,
        qo_len=127,
        kv_len=127,
        num_qo_heads=8,
        num_kv_heads=8,
        head_dim=64,
        page_size=5,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        logits_soft_cap=8.0,
        causal=False,
        return_lse=False,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        backend="fa2",
    )

    # Test with return_lse=True
    batch_prefill_with_paged_kv_cache_example(
        batch_size=4,
        qo_len=128,
        kv_len=128,
        num_qo_heads=8,
        num_kv_heads=8,
        head_dim=64,
        page_size=16,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        logits_soft_cap=0.0,
        causal=False,
        return_lse=True,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        backend="fa2",
    )

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
