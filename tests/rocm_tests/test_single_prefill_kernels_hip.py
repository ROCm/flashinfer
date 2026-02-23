# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from attention_reference import naive_attention
from jit_utils import gen_prefill_attention_modules

import flashinfer

from flashinfer.jit.core import logger
from flashinfer.aiter_utils import HAS_AITER
import logging

logger.setLevel(logging.ERROR)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_prefill_attention_modules(
            [torch.float16],  # q_dtypes
            [torch.float16],  # kv_dtypes
            [64, 128, 256],  # head_dims
            [0],  # pos_encoding_modes (NONE)
            [False],  # use_sliding_windows
            [False, True],  # use_logits_soft_caps
            [False],  # use_fp16_qk_reduction_options
        ),
        verbose=False,
    )
    yield


@pytest.mark.parametrize("qo_len", [37, 17, 127, 577])
@pytest.mark.parametrize("kv_len", [54, 97, 128, 512, 2048])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 8.0])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("backend", ["fa2", "aiter"])
def test_single_prefill_with_kv_cache(
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
    backend: str,
):
    q = torch.randn(
        qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16
    )

    if backend == "aiter" and not HAS_AITER:
        pytest.skip("AITER is not available")

    if backend == "aiter" and kv_layout == "HND":
        pytest.skip("AITER does not support HND layout")

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
            backend=backend,
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
            backend=backend,
        )

    assert o.shape == (qo_len, num_qo_heads, head_dim)

    # Compute reference in FP32 for better accuracy
    o_ref, lse_ref = naive_attention(
        q.float(),
        k_ref.float(),
        v_ref.float(),
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
        return_lse=return_lse,
    )
    torch.testing.assert_close(o, o_ref.to(o.dtype), rtol=1e-3, atol=1e-3)
    if return_lse:
        torch.testing.assert_close(
            lse, lse_ref.to(lse.dtype), rtol=1e-3, atol=1e-3
        )  # lse is in fp32


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("return_lse", [False, True])
def test_single_prefill_threadblock_sync_mdo_states(
    head_dim: int,
    return_lse: bool,
):
    """
    Test case specifically for threadblock_sync_mdo_states validation.
    This config triggers CTA_TILE_Q=16, NUM_WARPS_KV=4, calling threadblock_sync_mdo_states.
    """
    qo_len = 16
    kv_len = 128
    num_qo_heads = 1
    num_kv_heads = 1
    causal = False
    kv_layout = "NHD"
    pos_encoding_mode = "NONE"
    logits_soft_cap = None

    q = torch.randn(
        qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16
    )
    k = torch.randn(
        kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16
    )
    v = torch.randn(
        kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16
    )

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
            backend="fa2",
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
            backend="fa2",
        )

    assert o.shape == (qo_len, num_qo_heads, head_dim)

    # Compute reference in FP32 for better accuracy
    o_ref, lse_ref = naive_attention(
        q.float(),
        k.float(),
        v.float(),
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
        return_lse=return_lse,
    )
    torch.testing.assert_close(o, o_ref.to(o.dtype), rtol=1e-3, atol=1e-3)
    if return_lse:
        torch.testing.assert_close(lse, lse_ref.to(lse.dtype), rtol=1e-3, atol=1e-3)
