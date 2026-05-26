# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from jit_utils import gen_prefill_attention_modules

import flashinfer
from flashinfer.jit.attention import gen_batch_pod_module


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_prefill_attention_modules(
            [torch.float16],
            [torch.float16],
            [128],
            [0],
            [False],
            [False],
            [False],
        )
        + [
            gen_batch_pod_module(
                torch.float16,  # dtype_q
                torch.float16,  # dtype_kv
                torch.float16,  # dtype_o
                128,  # head_dim
                0,  # pos_encoding_mode_p (NONE)
                False,  # use_sliding_window_p
                False,  # use_logits_soft_cap_p
                False,  # use_fp16_qk_reduction
                torch.int32,  # dtype_idx
                0,  # pos_encoding_mode_d (NONE)
                False,  # use_sliding_window_d
                False,  # use_logits_soft_cap_d
            )
        ],
        verbose=False,
    )
    yield


def _make_paged_kv(
    batch_size, kv_len, page_size, num_kv_heads, head_dim, dtype, device
):
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_pages = num_pages_per_seq * batch_size
    kv_data = torch.randn(
        total_pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_pages, device=device, dtype=torch.int32)
    last_page_len = torch.full(
        (batch_size,),
        (kv_len - 1) % page_size + 1,
        device=device,
        dtype=torch.int32,
    )
    return kv_data, kv_indptr, kv_indices, last_page_len


@pytest.mark.parametrize("batch_size_p", [2, 4])
@pytest.mark.parametrize("qo_len_p", [64, 256])
@pytest.mark.parametrize("kv_len_p", [128, 512])
@pytest.mark.parametrize("batch_size_d", [4, 16])
@pytest.mark.parametrize("kv_len_d", [256, 1024])
@pytest.mark.parametrize("page_size", [16])
@pytest.mark.parametrize("num_qo_heads", [8, 32])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("q_dtype", [torch.float16])
def test_batch_pod_with_paged_kv_cache(
    batch_size_p,
    qo_len_p,
    kv_len_p,
    batch_size_d,
    kv_len_d,
    page_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    q_dtype,
):
    device = "cuda:0"
    kv_dtype = q_dtype
    kv_layout = "NHD"
    pos_encoding_mode = "NONE"

    # Prefill: each request has qo_len_p tokens attending to kv_len_p tokens
    total_qo_p = batch_size_p * qo_len_p
    q_p = torch.randn(total_qo_p, num_qo_heads, head_dim, device=device, dtype=q_dtype)
    kv_data_p, kv_indptr_p, kv_indices_p, last_page_len_p = _make_paged_kv(
        batch_size_p, kv_len_p, page_size, num_kv_heads, head_dim, kv_dtype, device
    )
    qo_indptr_p = (
        torch.arange(0, batch_size_p + 1, device=device, dtype=torch.int32) * qo_len_p
    )

    # Decode: each request has 1 token
    q_d = torch.randn(
        batch_size_d, num_qo_heads, head_dim, device=device, dtype=q_dtype
    )
    kv_data_d, kv_indptr_d, kv_indices_d, last_page_len_d = _make_paged_kv(
        batch_size_d, kv_len_d, page_size, num_kv_heads, head_dim, kv_dtype, device
    )
    qo_indptr_d = torch.arange(0, batch_size_d + 1, device=device, dtype=torch.int32)

    # Reference: batch prefill for the prefill arm
    float_buf_ref = torch.empty(64 * 1024 * 1024, device=device, dtype=torch.uint8)
    prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        float_buf_ref, kv_layout
    )
    prefill_wrapper.plan(
        qo_indptr_p,
        kv_indptr_p,
        kv_indices_p,
        last_page_len_p,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        pos_encoding_mode=pos_encoding_mode,
        kv_data_type=kv_dtype,
        q_data_type=q_dtype,
    )
    o_ref_p = prefill_wrapper.run(q_p, kv_data_p)

    # Reference: batch prefill for the decode arm (1 token per seq)
    float_buf_ref_d = torch.empty(64 * 1024 * 1024, device=device, dtype=torch.uint8)
    decode_prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        float_buf_ref_d, kv_layout
    )
    decode_prefill_wrapper.plan(
        qo_indptr_d,
        kv_indptr_d,
        kv_indices_d,
        last_page_len_d,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        pos_encoding_mode=pos_encoding_mode,
        kv_data_type=kv_dtype,
        q_data_type=q_dtype,
    )
    o_ref_d = decode_prefill_wrapper.run(q_d, kv_data_d)

    # BatchPOD wrapper
    workspace_buffer = torch.empty(128 * 1024 * 1024, device=device, dtype=torch.uint8)
    batch_pod_wrapper = flashinfer.BatchPODWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    batch_pod_wrapper.plan(
        qo_indptr_p,
        kv_indptr_p,
        kv_indices_p,
        last_page_len_p,
        qo_indptr_d,
        kv_indptr_d,
        kv_indices_d,
        last_page_len_d,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pos_encoding_mode=pos_encoding_mode,
        q_data_type=q_dtype,
        kv_data_type=kv_dtype,
    )
    o_p, o_d = batch_pod_wrapper.run(
        q_p,
        kv_data_p,
        q_d,
        kv_data_d,
        causal_p=False,
    )

    torch.testing.assert_close(
        o_p, o_ref_p, rtol=1e-2, atol=1e-2, msg="BatchPOD prefill mismatch"
    )
    torch.testing.assert_close(
        o_d, o_ref_d, rtol=1e-2, atol=1e-2, msg="BatchPOD decode mismatch"
    )


@pytest.mark.parametrize("q_dtype", [torch.bfloat16])
def test_batch_pod_bf16(q_dtype):
    device = "cuda:0"
    kv_dtype = q_dtype
    kv_layout = "NHD"
    pos_encoding_mode = "NONE"
    batch_size_p, qo_len_p, kv_len_p = 2, 64, 128
    batch_size_d, kv_len_d, page_size = 4, 256, 16
    num_qo_heads, num_kv_heads, head_dim = 8, 8, 128

    total_qo_p = batch_size_p * qo_len_p
    q_p = torch.randn(total_qo_p, num_qo_heads, head_dim, device=device, dtype=q_dtype)
    kv_data_p, kv_indptr_p, kv_indices_p, last_page_len_p = _make_paged_kv(
        batch_size_p, kv_len_p, page_size, num_kv_heads, head_dim, kv_dtype, device
    )
    qo_indptr_p = (
        torch.arange(0, batch_size_p + 1, device=device, dtype=torch.int32) * qo_len_p
    )

    q_d = torch.randn(
        batch_size_d, num_qo_heads, head_dim, device=device, dtype=q_dtype
    )
    kv_data_d, kv_indptr_d, kv_indices_d, last_page_len_d = _make_paged_kv(
        batch_size_d, kv_len_d, page_size, num_kv_heads, head_dim, kv_dtype, device
    )
    qo_indptr_d = torch.arange(0, batch_size_d + 1, device=device, dtype=torch.int32)

    float_buf = torch.empty(64 * 1024 * 1024, device=device, dtype=torch.uint8)
    prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        float_buf, kv_layout, backend="fa2"
    )
    prefill_wrapper.plan(
        qo_indptr_p,
        kv_indptr_p,
        kv_indices_p,
        last_page_len_p,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        pos_encoding_mode=pos_encoding_mode,
        kv_data_type=kv_dtype,
        q_data_type=q_dtype,
    )
    o_ref_p = prefill_wrapper.run(q_p, kv_data_p)

    float_buf_d = torch.empty(64 * 1024 * 1024, device=device, dtype=torch.uint8)
    decode_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        float_buf_d, kv_layout, backend="fa2"
    )
    decode_wrapper.plan(
        qo_indptr_d,
        kv_indptr_d,
        kv_indices_d,
        last_page_len_d,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        pos_encoding_mode=pos_encoding_mode,
        kv_data_type=kv_dtype,
        q_data_type=q_dtype,
    )
    o_ref_d = decode_wrapper.run(q_d, kv_data_d)

    workspace_buffer = torch.empty(128 * 1024 * 1024, device=device, dtype=torch.uint8)
    batch_pod_wrapper = flashinfer.BatchPODWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    batch_pod_wrapper.plan(
        qo_indptr_p,
        kv_indptr_p,
        kv_indices_p,
        last_page_len_p,
        qo_indptr_d,
        kv_indptr_d,
        kv_indices_d,
        last_page_len_d,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pos_encoding_mode=pos_encoding_mode,
        q_data_type=q_dtype,
        kv_data_type=kv_dtype,
    )
    o_p, o_d = batch_pod_wrapper.run(
        q_p,
        kv_data_p,
        q_d,
        kv_data_d,
        causal_p=False,
    )

    torch.testing.assert_close(
        o_p, o_ref_p, rtol=1e-2, atol=1e-2, msg="BatchPOD bf16 prefill mismatch"
    )
    torch.testing.assert_close(
        o_d, o_ref_d, rtol=1e-2, atol=1e-2, msg="BatchPOD bf16 decode mismatch"
    )


def test_batch_pod_causal_p():
    """Exercises MaskMode::kCausal on the prefill arm of BatchPOD."""
    device = "cuda:0"
    q_dtype = kv_dtype = torch.float16
    kv_layout = "NHD"
    pos_encoding_mode = "NONE"
    batch_size_p, qo_len_p, kv_len_p = 2, 64, 128
    batch_size_d, kv_len_d, page_size = 4, 256, 16
    num_qo_heads, num_kv_heads, head_dim = 8, 8, 128

    total_qo_p = batch_size_p * qo_len_p
    q_p = torch.randn(total_qo_p, num_qo_heads, head_dim, device=device, dtype=q_dtype)
    kv_data_p, kv_indptr_p, kv_indices_p, last_page_len_p = _make_paged_kv(
        batch_size_p, kv_len_p, page_size, num_kv_heads, head_dim, kv_dtype, device
    )
    qo_indptr_p = (
        torch.arange(0, batch_size_p + 1, device=device, dtype=torch.int32) * qo_len_p
    )

    q_d = torch.randn(
        batch_size_d, num_qo_heads, head_dim, device=device, dtype=q_dtype
    )
    kv_data_d, kv_indptr_d, kv_indices_d, last_page_len_d = _make_paged_kv(
        batch_size_d, kv_len_d, page_size, num_kv_heads, head_dim, kv_dtype, device
    )
    qo_indptr_d = torch.arange(0, batch_size_d + 1, device=device, dtype=torch.int32)

    float_buf_p = torch.empty(64 * 1024 * 1024, device=device, dtype=torch.uint8)
    ref_p = flashinfer.BatchPrefillWithPagedKVCacheWrapper(float_buf_p, kv_layout)
    ref_p.plan(
        qo_indptr_p,
        kv_indptr_p,
        kv_indices_p,
        last_page_len_p,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        pos_encoding_mode=pos_encoding_mode,
        kv_data_type=kv_dtype,
        q_data_type=q_dtype,
    )
    o_ref_p = ref_p.run(q_p, kv_data_p)

    float_buf_d = torch.empty(64 * 1024 * 1024, device=device, dtype=torch.uint8)
    ref_d = flashinfer.BatchPrefillWithPagedKVCacheWrapper(float_buf_d, kv_layout)
    ref_d.plan(
        qo_indptr_d,
        kv_indptr_d,
        kv_indices_d,
        last_page_len_d,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        pos_encoding_mode=pos_encoding_mode,
        kv_data_type=kv_dtype,
        q_data_type=q_dtype,
    )
    o_ref_d = ref_d.run(q_d, kv_data_d)

    workspace_buffer = torch.empty(128 * 1024 * 1024, device=device, dtype=torch.uint8)
    batch_pod_wrapper = flashinfer.BatchPODWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    batch_pod_wrapper.plan(
        qo_indptr_p,
        kv_indptr_p,
        kv_indices_p,
        last_page_len_p,
        qo_indptr_d,
        kv_indptr_d,
        kv_indices_d,
        last_page_len_d,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pos_encoding_mode=pos_encoding_mode,
        q_data_type=q_dtype,
        kv_data_type=kv_dtype,
    )
    o_p, o_d = batch_pod_wrapper.run(q_p, kv_data_p, q_d, kv_data_d, causal_p=True)

    torch.testing.assert_close(
        o_p, o_ref_p, rtol=1e-2, atol=1e-2, msg="BatchPOD causal prefill mismatch"
    )
    torch.testing.assert_close(
        o_d, o_ref_d, rtol=1e-2, atol=1e-2, msg="BatchPOD causal decode mismatch"
    )
