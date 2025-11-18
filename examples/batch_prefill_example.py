# SPDX-FileCopyrightText : 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier : Apache-2.0

from typing import Optional

import torch

import flashinfer
from tests.attention_reference import naive_attention


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
    backend: Optional[str] = "fa2",
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
    print(f"  backend={backend}")

    # Create flattened query tensor: [batch_size * qo_len, num_qo_heads, head_dim]
    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
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
    kv_data = kv_data_fp32.to(torch.float16)

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
        ).to(torch.float16)

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
        ).to(torch.float16)

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
            ).to(torch.float16)

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


def batch_prefill_with_ragged_kv_cache_example(
    batch_size: int,
    kv_len: int,
    qo_len: int,
    num_kv_heads: int,
    num_qo_heads: int,
    head_dim: int,
    causal: bool,
    pos_encoding_mode: str,
    logits_soft_cap: float,
    return_lse: bool,
):
    if qo_len > kv_len and causal:
        raise ValueError("qo_len > kv_len and causal is not supported")
    kv_layout = "NHD"
    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    q_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    )

    k = torch.randn(
        batch_size * kv_len,
        num_kv_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    v = torch.randn(
        batch_size * kv_len,
        num_kv_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * kv_len
    )

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper.plan(
        q_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
    )
    if return_lse:
        o, _ = wrapper.run(q, k, v, return_lse=True)
    else:
        o = wrapper.run(q, k, v)

    all_passed, single_pass, naive_pass = False, False, False
    for i in range(batch_size):
        o_ref_single = flashinfer.prefill.single_prefill_with_kv_cache(
            q[q_indptr[i] : q_indptr[i + 1]],
            k[kv_indptr[i] : kv_indptr[i + 1]],
            v[kv_indptr[i] : kv_indptr[i + 1]],
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        o_single = o[q_indptr[i] : q_indptr[i + 1]]

        try:
            torch.testing.assert_close(o_single, o_ref_single, rtol=1e-3, atol=1e-3)
            single_pass = True
        except AssertionError:
            single_pass = False
            max_diff = (o_single - o_ref_single).abs().max().item()
            print(f"    Sequence {i}: ✗ FAIL vs single_prefill (max diff: {max_diff})")

        # For simple cases without RoPE/ALiBi, also compare with naive implementation
        if pos_encoding_mode == "NONE":
            o_ref_naive = naive_attention(
                q[q_indptr[i] : q_indptr[i + 1]].float(),
                k[kv_indptr[i] : kv_indptr[i + 1]].float(),
                v[kv_indptr[i] : kv_indptr[i + 1]].float(),
                causal=causal,
                logits_soft_cap=logits_soft_cap if logits_soft_cap > 0 else None,
            ).to(torch.float16)

            try:
                torch.testing.assert_close(o_single, o_ref_naive, rtol=1e-3, atol=1e-3)
                naive_pass = True
            except AssertionError:
                naive_pass = False
                max_diff = (o_single - o_ref_naive).abs().max().item()
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

    batch_prefill_with_ragged_kv_cache_example(
        12, 54, 37, 8, 8, 128, False, "NONE", 0.0, True
    )

    # # Basic test with small batch
    # batch_prefill_with_paged_kv_cache_example(
    #     4, 128, 128, 8, 8, 64, 16, "NHD", "NONE", 0.0, False, False
    # )

    # # Test with logits soft cap
    # batch_prefill_with_paged_kv_cache_example(
    #     4, 128, 128, 8, 8, 64, 16, "NHD", "NONE", 8.0, False, False
    # )

    # # Test with causal masking
    # batch_prefill_with_paged_kv_cache_example(
    #     4, 128, 128, 8, 8, 64, 16, "NHD", "NONE", 0.0, True, False
    # )

    # # Test with GQA (num_qo_heads > num_kv_heads)
    # batch_prefill_with_paged_kv_cache_example(
    #     4, 128, 128, 32, 4, 64, 16, "NHD", "NONE", 8.0, False, False
    # )

    # # Test with different qo_len and kv_len
    # batch_prefill_with_paged_kv_cache_example(
    #     8, 64, 256, 16, 4, 128, 16, "NHD", "NONE", 0.0, False, False
    # )

    # # Test with smaller page size
    # batch_prefill_with_paged_kv_cache_example(
    #     4, 127, 127, 8, 8, 64, 5, "NHD", "NONE", 8.0, False, False
    # )

    # # Test with return_lse=True
    # batch_prefill_with_paged_kv_cache_example(
    #     4, 128, 128, 8, 8, 64, 16, "NHD", "NONE", 0.0, False, True
    # )

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("=" * 60)
