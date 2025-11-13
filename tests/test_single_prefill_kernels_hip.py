# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional, Tuple

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
    causal: Optional[bool] = False,
    return_lse: Optional[bool] = False,
    logits_soft_cap: Optional[float] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        o: output tensor, shape: [qo_len, num_qo_heads, head_dim] if return_lse is False,
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


@pytest.mark.parametrize("qo_len", [1, 7, 15, 63, 127])
@pytest.mark.parametrize("kv_len", [7, 31, 127, 511, 2047])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 8.0])
@pytest.mark.parametrize("return_lse", [False, True])
def test_single_prefill_with_kv_cache(
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
):
    """
    Comprehensive test for single_prefill_with_kv_cache function.
    Tests various sequence lengths, head configurations, and options.
    """
    # The current validation only supports simple cases without RoPE/ALiBi
    if pos_encoding_mode != "NONE":
        pytest.skip("Only pos_encoding_mode == NONE is supported for this validation")

    logits_soft_cap = logits_soft_cap if logits_soft_cap > 0.0 else None

    # Create query tensor: [qo_len, num_qo_heads, head_dim]
    q = torch.randn(
        qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16
    )

    # Create key and value tensors based on layout
    if kv_layout == "HND":
        # [num_kv_heads, kv_len, head_dim]
        k = torch.randn(
            num_kv_heads, kv_len, head_dim, device="cuda:0", dtype=torch.float16
        )
        v = torch.randn(
            num_kv_heads, kv_len, head_dim, device="cuda:0", dtype=torch.float16
        )
        # Convert to NHD for reference implementation
        k_ref = k.transpose(0, 1).contiguous()
        v_ref = v.transpose(0, 1).contiguous()
    else:  # NHD layout
        # [kv_len, num_kv_heads, head_dim]
        k = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16
        )
        v = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16
        )
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
            logits_soft_cap=logits_soft_cap,
            return_lse=False,
        )

    assert o.shape == (qo_len, num_qo_heads, head_dim)

    # Compute reference in FP32 for better accuracy
    o_ref, lse_ref = naive_attention(
        q.float(),  #
        k_ref.float(),
        v_ref.float(),
        causal=causal,
        return_lse=return_lse,
        logits_soft_cap=logits_soft_cap,
    )
    torch.testing.assert_close(o, o_ref.to(o.dtype), rtol=1e-3, atol=1e-3)
    if return_lse:
        torch.testing.assert_close(
            lse, lse_ref.to(lse.dtype), rtol=1e-3, atol=1e-3
        )  # lse is in fp32


if __name__ == "__main__":
    # Self-attention with logits soft cap
    test_single_prefill_with_kv_cache(
        128, 128, 1, 1, 64, "NHD", "NONE", 8.0, False, False
    )
    # Self-attention without logits soft cap
    test_single_prefill_with_kv_cache(
        128, 128, 1, 1, 64, "NHD", "NONE", 0.0, False, False
    )
    # Multi-head attention (MHA)
    test_single_prefill_with_kv_cache(
        128, 128, 4, 4, 64, "NHD", "NONE", 8.0, False, False
    )
    # Grouped query attention (GQA)
    test_single_prefill_with_kv_cache(
        128, 128, 8, 4, 64, "NHD", "NONE", 8.0, False, False
    )
    # GQA with qo_len < kv_len (typical prefill)
    test_single_prefill_with_kv_cache(
        15, 127, 32, 4, 64, "NHD", "NONE", 8.0, False, False
    )
    # GQA with LSE enabled
    test_single_prefill_with_kv_cache(
        15, 127, 8, 4, 64, "NHD", "NONE", 0.0, False, True
    )
    # GQA with soft cap and LSE enabled
    test_single_prefill_with_kv_cache(
        15, 127, 8, 4, 64, "NHD", "NONE", 8.0, False, True
    )
    # GQA with HND layout
    test_single_prefill_with_kv_cache(
        15, 127, 32, 4, 64, "HND", "NONE", 8.0, False, True
    )

    print("All tests passed!")
