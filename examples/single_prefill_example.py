# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import flashinfer
from tests.attention_reference import naive_attention


def single_prefill_with_kv_cache_example(
    qo_len: int,
    kv_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    kv_layout: str,
    pos_encoding_mode: str,
    logits_soft_cap: float,
    return_lse: bool,
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
    print(f"  causal={causal}")
    print(f"  kv_layout={kv_layout}")
    print(f"  pos_encoding_mode={pos_encoding_mode}")
    print(f"  logits_soft_cap={logits_soft_cap}")
    print(f"  return_lse={return_lse}")

    q = torch.randn(
        qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16
    )

    if kv_layout == "HND":
        k = torch.randn(
            num_kv_heads, kv_len, head_dim, device="cuda:0", dtype=torch.float16
        )
        v = torch.randn(
            num_kv_heads, kv_len, head_dim, device="cuda:0", dtype=torch.float16
        )
        # Convert to NHD for reference implementation
        k_ref = k.transpose(0, 1).contiguous()  # [kv_len, num_kv_heads, head_dim]
        v_ref = v.transpose(0, 1).contiguous()  # [kv_len, num_kv_heads, head_dim]
    else:  # NHD layout
        k = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16
        )
        v = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16
        )
        k_ref = k
        v_ref = v

    # Call flashinfer API
    logits_soft_cap = logits_soft_cap if logits_soft_cap > 0 else None
    if return_lse:
        o, lse = flashinfer.single_prefill_with_kv_cache_return_lse(
            q,
            k,
            v,
            causal=causal,
            kv_layout=kv_layout,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        print(f"  FlashInfer output shape: {o.shape}, LSE shape: {lse.shape}")
        # Compute reference in FP32 for better accuracy
        o_ref, lse_ref = naive_attention(
            q.float(),
            k_ref.float(),
            v_ref.float(),
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
            return_lse=True,
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
        )
        print(f"  FlashInfer output shape: {o.shape}")

        # Compute reference in FP32 for better accuracy
        o_ref, _ = naive_attention(
            q.float(),
            k_ref.float(),
            v_ref.float(),
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
            return_lse=False,
        )
        # Convert reference back to match FlashInfer dtype
        o_ref = o_ref.to(o.dtype)
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
        128, 128, 1, 1, 64, False, "NHD", "NONE", 8.0, False
    )
    # Self-attention without logits soft cap
    single_prefill_with_kv_cache_example(
        128, 128, 1, 1, 64, False, "NHD", "NONE", 0.0, False
    )
    # Multi-head attention (MHA)
    single_prefill_with_kv_cache_example(
        128, 128, 4, 4, 64, False, "NHD", "NONE", 8.0, False
    )
    # Grouped query attention (GQA)
    single_prefill_with_kv_cache_example(
        128, 128, 8, 4, 64, False, "NHD", "NONE", 8.0, False
    )
    # GQA with qo_len < kv_len (typical prefill)
    single_prefill_with_kv_cache_example(
        15, 127, 32, 4, 64, False, "NHD", "NONE", 8.0, False
    )
    # GQA with LSE enabled
    single_prefill_with_kv_cache_example(
        15, 127, 8, 4, 64, False, "NHD", "NONE", 0.0, True
    )
    # GQA with soft cap and LSE enabled
    single_prefill_with_kv_cache_example(
        15, 127, 8, 4, 64, False, "NHD", "NONE", 8.0, True
    )

    # Test case specifically for threadblock_sync_mdo_states validation
    # This config triggers CTA_TILE_Q=16, NUM_WARPS_KV=4, calling threadblock_sync_mdo_states
    print("\n" + "=" * 60)
    print("Testing threadblock_sync_mdo_states (CTA_TILE_Q=16, NUM_WARPS_KV=4)")
    print("=" * 60)
    single_prefill_with_kv_cache_example(
        16, 128, 1, 1, 64, False, "NHD", "NONE", 0.0, False
    )
    single_prefill_with_kv_cache_example(
        16, 128, 1, 1, 64, False, "NHD", "NONE", 0.0, True
    )
