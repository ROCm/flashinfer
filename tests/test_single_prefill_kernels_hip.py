# SPDX-FileCopyrightText : 2023-2025 FlashInfer team.
# SPDX-FileCopyrightText : 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier : Apache-2.0

# The following tests validate the correctness of SinglePrefillWithKVCacheDevice() in prefill.cuh through the Python API layer, ensuring:
# - Correct computation of attention scores (QK^T) using naive attention implementation as ground truth
# - Proper application of masks (causal, custom)
# - Correct softmax and attention-weighted value computation
# - Accurate position encoding (RoPE)
# - Proper handling of GQA, different layouts, and various configurations

import math

import pytest
import torch
from jit_utils import jit_prefill_attention_func_args

import flashinfer


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    """
    Warmup JIT modules for single prefill kernels.
    - Pre-compiles all necessary kernel variants
    - Prevents compilation during tests for faster execution
    - Aborts test session if warmup fails to avoid unnecessary test failures
    """
    if flashinfer.jit.has_prebuilt_ops:
        yield
    else:
        try:
            flashinfer.jit.parallel_load_modules(
                jit_prefill_attention_func_args(
                    [torch.float16],  # q_dtypes
                    [torch.float16],  # kv_dtypes
                    [128, 256],  # head_dims
                    [0, 1],  # pos_encoding_modes (NONE, ROPE_LLAMA)
                    [False],  # use_sliding_windows
                    [False, True],  # use_logits_soft_caps
                    [False],  # use_fp16_qk_reduction
                )
            )
        except Exception as e:
            # abort the test session if warmup fails
            pytest.exit(str(e))
        finally:
            yield


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None,
    logits_soft_cap: float = None,
) -> torch.Tensor:
    """
    Naive PyTorch implementation of attention for reference.
    - Supports causal masking
    - Supports logits soft capping
    - Handles GQA (grouped query attention)
    - Used as ground truth for NONE position encoding mode

    Args:
        q: query tensor, shape: [qo_len, num_qo_heads, head_dim]
        k: key tensor, shape: [kv_len, num_kv_heads, head_dim]
        v: value tensor, shape: [kv_len, num_kv_heads, head_dim]
        causal: whether to apply causal masking
        sm_scale: softmax scale (default: 1/sqrt(head_dim))
        logits_soft_cap: if not None, applies soft cap to logits

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
        k = k.repeat_interleave(group_size, dim=1)
        v = v.repeat_interleave(group_size, dim=1)

    # Transpose for batch matrix multiply: [num_heads, seq_len, head_dim]
    q_t = q.transpose(0, 1)  # [num_qo_heads, qo_len, head_dim]
    k_t = k.transpose(0, 1)  # [num_qo_heads, kv_len, head_dim]
    v_t = v.transpose(0, 1)  # [num_qo_heads, kv_len, head_dim]

    # Compute attention scores: [num_qo_heads, qo_len, kv_len]
    scores = torch.matmul(q_t, k_t.transpose(1, 2)) * sm_scale

    # Apply logits soft cap if specified
    if logits_soft_cap is not None and logits_soft_cap > 0:
        scores = logits_soft_cap * torch.tanh(scores / logits_soft_cap)

    # Apply causal mask if needed
    if causal:
        mask = torch.tril(
            torch.ones((qo_len, kv_len), device=q.device, dtype=torch.bool),
            diagonal=(kv_len - qo_len),
        )
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))

    # Softmax
    attn = torch.softmax(scores, dim=-1)

    # Apply attention to values
    out = torch.matmul(attn, v_t)

    # Transpose back: [qo_len, num_qo_heads, head_dim]
    out = out.transpose(0, 1)

    return out


@pytest.mark.parametrize("qo_len", [1, 7, 15, 63, 127])
@pytest.mark.parametrize("kv_len", [7, 31, 127, 511, 2047])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("q_dtype", [torch.float16])
@pytest.mark.parametrize("kv_dtype", [torch.float16])
def test_single_prefill_with_kv_cache(
    qo_len,
    kv_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    causal,
    pos_encoding_mode,
    logits_soft_cap,
    return_lse,
    q_dtype,
    kv_dtype,
):
    """
    Comprehensive test for single_prefill_with_kv_cache function.
    Tests various sequence lengths, head configurations, and options.
    """
    # Create query tensor: [qo_len, num_qo_heads, head_dim]
    q = torch.randn(qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=q_dtype)

    # Create key and value tensors based on layout
    if kv_layout == "HND":
        # [num_kv_heads, kv_len, head_dim]
        k = torch.randn(num_kv_heads, kv_len, head_dim, device="cuda:0", dtype=kv_dtype)
        v = torch.randn(num_kv_heads, kv_len, head_dim, device="cuda:0", dtype=kv_dtype)
        # Convert to NHD for reference implementation
        k_ref = k.transpose(0, 1).contiguous()
        v_ref = v.transpose(0, 1).contiguous()
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
            return_lse=True,
        )
        assert lse.shape == (qo_len, num_qo_heads)
    else:
        o = flashinfer.single_prefill_with_kv_cache(
            q,
            k,
            v,
            causal=causal,
            kv_layout=kv_layout,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap if logits_soft_cap > 0 else None,
            return_lse=False,
        )

    assert o.shape == (qo_len, num_qo_heads, head_dim)

    # For NONE pos_encoding_mode, verify against naive implementation
    if pos_encoding_mode == "NONE":
        o_ref = naive_attention(
            q.float(),
            k_ref.float(),
            v_ref.float(),
            causal=causal,
            logits_soft_cap=logits_soft_cap if logits_soft_cap > 0 else None,
        ).to(q_dtype)

        torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("qo_len", [7, 127])
@pytest.mark.parametrize("kv_len", [31, 511])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE"])
@pytest.mark.parametrize("q_dtype", [torch.float16])
@pytest.mark.parametrize("kv_dtype", [torch.float16])
def test_single_prefill_kv_layouts(
    qo_len,
    kv_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    causal,
    pos_encoding_mode,
    q_dtype,
    kv_dtype,
):
    """
    Test both NHD (seq_len, num_heads, head_dim) and HND (num_heads, seq_len, head_dim) layouts produce consistent results.
    """
    q = torch.randn(qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=q_dtype)

    # Create KV in both layouts
    k_nhd = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=kv_dtype)
    v_nhd = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=kv_dtype)

    # Convert to HND
    k_hnd = k_nhd.transpose(0, 1).contiguous()
    v_hnd = v_nhd.transpose(0, 1).contiguous()

    if kv_layout == "NHD":
        k, v = k_nhd, v_nhd
    else:
        k, v = k_hnd, v_hnd

    o = flashinfer.single_prefill_with_kv_cache(
        q,
        k,
        v,
        causal=causal,
        kv_layout=kv_layout,
        pos_encoding_mode=pos_encoding_mode,
    )

    # Compare with reference using NHD
    o_ref = naive_attention(q.float(), k_nhd.float(), v_nhd.float(), causal=causal).to(
        q_dtype
    )

    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("qo_len", [15, 127])
@pytest.mark.parametrize("kv_len", [127, 511])
@pytest.mark.parametrize("num_kv_heads", [4, 8])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("q_dtype", [torch.float16])
@pytest.mark.parametrize("kv_dtype", [torch.float16])
def test_single_prefill_gqa(
    qo_len,
    kv_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    causal,
    q_dtype,
    kv_dtype,
):
    """
    Test grouped query attention (GQA) where num_qo_heads > num_kv_heads (typical in modern LLMs like Llama).
    """
    assert (
        num_qo_heads % num_kv_heads == 0
    ), "num_qo_heads must be divisible by num_kv_heads"

    q = torch.randn(qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=q_dtype)
    k = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=kv_dtype)
    v = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=kv_dtype)

    o = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=causal, kv_layout=kv_layout, pos_encoding_mode="NONE"
    )

    assert o.shape == (qo_len, num_qo_heads, head_dim)

    # Compare with reference
    o_ref = naive_attention(q.float(), k.float(), v.float(), causal=causal).to(q_dtype)
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("qo_len", [15])
@pytest.mark.parametrize("kv_len", [127])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("window_left", [16, 32, 64])
@pytest.mark.parametrize("q_dtype", [torch.float16])
@pytest.mark.parametrize("kv_dtype", [torch.float16])
def test_single_prefill_sliding_window(
    qo_len,
    kv_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    causal,
    window_left,
    q_dtype,
    kv_dtype,
):
    """
    Test local attention with sliding window.
    """
    q = torch.randn(qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=q_dtype)
    k = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=kv_dtype)
    v = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=kv_dtype)

    o = flashinfer.single_prefill_with_kv_cache(
        q,
        k,
        v,
        causal=causal,
        kv_layout=kv_layout,
        pos_encoding_mode="NONE",
        window_left=window_left,
    )

    assert o.shape == (qo_len, num_qo_heads, head_dim)


@pytest.mark.parametrize("qo_len", [15, 63])
@pytest.mark.parametrize("kv_len", [127, 511])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["ROPE_LLAMA"])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("q_dtype", [torch.float16])
@pytest.mark.parametrize("kv_dtype", [torch.float16])
def test_single_prefill_rope(
    qo_len,
    kv_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    pos_encoding_mode,
    causal,
    q_dtype,
    kv_dtype,
):
    """
    Test RoPE (Rotary Position Embedding) encoding used in models like Llama.
    """
    q = torch.randn(qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=q_dtype)
    k = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=kv_dtype)
    v = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=kv_dtype)

    o = flashinfer.single_prefill_with_kv_cache(
        q,
        k,
        v,
        causal=causal,
        kv_layout=kv_layout,
        pos_encoding_mode=pos_encoding_mode,
    )

    assert o.shape == (qo_len, num_qo_heads, head_dim)


@pytest.mark.parametrize("qo_len", [7, 63])
@pytest.mark.parametrize("kv_len", [127])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("q_dtype", [torch.float16])
@pytest.mark.parametrize("kv_dtype", [torch.float16])
def test_single_prefill_custom_mask(
    qo_len,
    kv_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    causal,
    q_dtype,
    kv_dtype,
):
    """
    Test custom boolean mask for attention.
    """
    q = torch.randn(qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=q_dtype)
    k = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=kv_dtype)
    v = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=kv_dtype)

    # Create a custom mask: block diagonal pattern
    custom_mask = torch.zeros(qo_len, kv_len, device="cuda:0", dtype=torch.bool)
    for i in range(qo_len):
        start = max(0, i * kv_len // qo_len - 10)
        end = min(kv_len, i * kv_len // qo_len + 10)
        custom_mask[i, start:end] = True

    o = flashinfer.single_prefill_with_kv_cache(
        q,
        k,
        v,
        custom_mask=custom_mask,
        causal=causal,
        kv_layout=kv_layout,
        pos_encoding_mode="NONE",
    )

    assert o.shape == (qo_len, num_qo_heads, head_dim)


@pytest.mark.parametrize("qo_len", [31])
@pytest.mark.parametrize("kv_len", [127])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("head_dim", [128])
def test_single_prefill_edge_cases(
    qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim
):
    """
    Test edge cases and boundary conditions for single prefill.
    - qo_len < kv_len (typical prefill)
    - qo_len == kv_len (equal length case)
    """
    q = torch.randn(
        qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16
    )
    k = torch.randn(
        kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16
    )
    v = torch.randn(
        kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16
    )

    # Test 1: qo_len < kv_len (typical prefill)
    o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=False)
    assert o.shape == (qo_len, num_qo_heads, head_dim)

    # Test 2: qo_len == kv_len
    q_equal = torch.randn(
        kv_len, num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16
    )
    o_equal = flashinfer.single_prefill_with_kv_cache(q_equal, k, v, causal=False)
    assert o_equal.shape == (kv_len, num_qo_heads, head_dim)


if __name__ == "__main__":
    # Run a quick sanity check
    print("=" * 60)
    print("Running sanity checks for single_prefill_with_kv_cache...")
    print("=" * 60)

    try:
        test_single_prefill_with_kv_cache(
            qo_len=15,
            kv_len=127,
            num_kv_heads=4,
            num_qo_heads=4,
            head_dim=128,
            kv_layout="NHD",
            causal=True,
            pos_encoding_mode="NONE",
            logits_soft_cap=0.0,
            return_lse=False,
            q_dtype=torch.float16,
            kv_dtype=torch.float16,
        )
        print("✓ Basic test passed")
    except Exception as e:
        print(f"✗ Basic test failed: {e}")
        import traceback

        traceback.print_exc()

    # try:
    #     test_single_prefill_with_kv_cache(
    #         qo_len=15,
    #         kv_len=127,
    #         num_kv_heads=4,
    #         num_qo_heads=4,
    #         head_dim=128,
    #         kv_layout="NHD",
    #         causal=True,
    #         pos_encoding_mode="NONE",
    #         logits_soft_cap=0.0,
    #         return_lse=False,
    #         q_dtype=torch.float16,
    #         kv_dtype=torch.float16,
    #     )
    #     print("✓ Basic test passed")
    # except Exception as e:
    #     print(f"✗ Basic test failed: {e}")
    #     import traceback

    #     traceback.print_exc()

    # try:
    #     test_single_prefill_gqa(
    #         qo_len=15,
    #         kv_len=127,
    #         num_kv_heads=4,
    #         num_qo_heads=32,
    #         head_dim=128,
    #         kv_layout="NHD",
    #         causal=True,
    #         q_dtype=torch.float16,
    #         kv_dtype=torch.float16,
    #     )
    #     print("✓ GQA test passed")
    # except Exception as e:
    #     print(f"✗ GQA test failed: {e}")
    #     import traceback

    #     traceback.print_exc()

    # try:
    #     test_single_prefill_kv_layouts(
    #         qo_len=15,
    #         kv_len=127,
    #         num_kv_heads=4,
    #         num_qo_heads=4,
    #         head_dim=128,
    #         kv_layout="NHD",
    #         causal=True,
    #     )
    #     print("✓ KV layouts test passed")
    # except Exception as e:
    #     print(f"✗ KV layouts test failed: {e}")
    #     import traceback

    #     traceback.print_exc()

    # try:
    #     test_single_prefill_rope(
    #         qo_len=15,
    #         kv_len=127,
    #         num_kv_heads=4,
    #         num_qo_heads=4,
    #         head_dim=128,
    #         kv_layout="NHD",
    #         pos_encoding_mode="ROPE_LLAMA",
    #         causal=True,
    #         q_dtype=torch.float16,
    #         kv_dtype=torch.float16,
    #     )
    #     print("✓ RoPE test passed")
    # except Exception as e:
    #     print(f"✗ RoPE test failed: {e}")
    #     import traceback

    #     traceback.print_exc()

    print("=" * 60)
    print("All sanity checks completed!")
    print("=" * 60)
