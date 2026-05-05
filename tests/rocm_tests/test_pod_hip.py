# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from jit_utils import gen_prefill_attention_modules

import flashinfer
from flashinfer.jit.attention import gen_pod_module


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
            gen_pod_module(
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


@pytest.mark.parametrize("kv_len_p", [127, 4096])
@pytest.mark.parametrize("qo_len_p", [127, 512])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("batch_size_d", [1, 17])
@pytest.mark.parametrize("kv_len_d", [127, 2048])
@pytest.mark.parametrize("page_size_d", [16])
@pytest.mark.parametrize("kv_layout_d", ["NHD"])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("num_qo_heads", [8, 32])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE"])
@pytest.mark.parametrize("q_dtype", [torch.float16])
@pytest.mark.parametrize("kv_dtype", [torch.float16])
def test_pod_with_paged_kv_cache(
    kv_len_p,
    qo_len_p,
    causal,
    batch_size_d,
    kv_len_d,
    page_size_d,
    kv_layout_d,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    pos_encoding_mode,
    q_dtype,
    kv_dtype,
):
    if causal and qo_len_p > kv_len_p:
        pytest.skip("Causal prefill with qo_len_p > kv_len_p is not supported")

    device = "cuda:0"

    # Prefill inputs (single-request, NHD layout)
    q_p = torch.randn(qo_len_p, num_qo_heads, head_dim, device=device, dtype=q_dtype)
    k_p = torch.randn(kv_len_p, num_kv_heads, head_dim, device=device, dtype=kv_dtype)
    v_p = torch.randn(kv_len_p, num_kv_heads, head_dim, device=device, dtype=kv_dtype)

    # Reference prefill output
    o_ref_p = flashinfer.single_prefill_with_kv_cache(
        q_p,
        k_p,
        v_p,
        causal=causal,
        pos_encoding_mode=pos_encoding_mode,
    )

    # Decode inputs (paged KV cache)
    q_d = torch.randn(
        batch_size_d, num_qo_heads, head_dim, device=device, dtype=q_dtype
    )
    num_pages_per_seq = (kv_len_d + page_size_d - 1) // page_size_d
    total_num_pages = num_pages_per_seq * batch_size_d

    kv_data = torch.randn(
        total_num_pages,
        2,
        page_size_d,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=kv_dtype,
    )
    kv_indptr_d = (
        torch.arange(0, batch_size_d + 1, device=device, dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices_d = torch.arange(0, total_num_pages, device=device, dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size_d,),
        (kv_len_d - 1) % page_size_d + 1,
        device=device,
        dtype=torch.int32,
    )

    # Reference decode output
    decode_workspace = torch.empty(32 * 1024 * 1024, device=device, dtype=torch.int8)
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        decode_workspace, kv_layout_d
    )
    decode_wrapper.plan(
        kv_indptr_d,
        kv_indices_d,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size_d,
        pos_encoding_mode=pos_encoding_mode,
        data_type=kv_dtype,
        q_data_type=q_dtype,
    )
    o_ref_d = decode_wrapper.run(q_d, kv_data)

    # POD wrapper
    workspace_buffer = torch.empty(32 * 1024 * 1024, device=device, dtype=torch.int8)
    pod_wrapper = flashinfer.PODWithPagedKVCacheWrapper(workspace_buffer, kv_layout_d)
    pod_wrapper.plan(
        kv_indptr_d,
        kv_indices_d,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size_d,
        pos_encoding_mode=pos_encoding_mode,
        data_type=kv_dtype,
        q_data_type=q_dtype,
    )

    o_p, o_d = pod_wrapper.run(
        q_p,
        k_p,
        v_p,
        q_d,
        kv_data,
        pos_encoding_mode_p=pos_encoding_mode,
        causal_p=causal,
    )

    torch.testing.assert_close(
        o_p, o_ref_p, rtol=1e-2, atol=1e-2, msg="Prefill output mismatch"
    )
    torch.testing.assert_close(
        o_d, o_ref_d, rtol=1e-2, atol=1e-2, msg="Decode output mismatch"
    )


@pytest.mark.parametrize("q_dtype", [torch.bfloat16])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16])
def test_pod_bf16(q_dtype, kv_dtype):
    device = "cuda:0"
    qo_len_p, kv_len_p = 64, 128
    batch_size_d, kv_len_d, page_size_d = 4, 256, 16
    num_qo_heads, num_kv_heads, head_dim = 8, 8, 128
    kv_layout_d = "NHD"
    pos_encoding_mode = "NONE"

    q_p = torch.randn(qo_len_p, num_qo_heads, head_dim, device=device, dtype=q_dtype)
    k_p = torch.randn(kv_len_p, num_kv_heads, head_dim, device=device, dtype=kv_dtype)
    v_p = torch.randn(kv_len_p, num_kv_heads, head_dim, device=device, dtype=kv_dtype)

    o_ref_p = flashinfer.single_prefill_with_kv_cache(
        q_p,
        k_p,
        v_p,
        causal=False,
        pos_encoding_mode=pos_encoding_mode,
    )

    q_d = torch.randn(
        batch_size_d, num_qo_heads, head_dim, device=device, dtype=q_dtype
    )
    num_pages_per_seq = (kv_len_d + page_size_d - 1) // page_size_d
    total_num_pages = num_pages_per_seq * batch_size_d
    kv_data = torch.randn(
        total_num_pages,
        2,
        page_size_d,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=kv_dtype,
    )
    kv_indptr_d = (
        torch.arange(0, batch_size_d + 1, device=device, dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices_d = torch.arange(0, total_num_pages, device=device, dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size_d,),
        (kv_len_d - 1) % page_size_d + 1,
        device=device,
        dtype=torch.int32,
    )

    decode_workspace = torch.empty(32 * 1024 * 1024, device=device, dtype=torch.int8)
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        decode_workspace, kv_layout_d
    )
    decode_wrapper.plan(
        kv_indptr_d,
        kv_indices_d,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size_d,
        pos_encoding_mode=pos_encoding_mode,
        data_type=kv_dtype,
        q_data_type=q_dtype,
    )
    o_ref_d = decode_wrapper.run(q_d, kv_data)

    workspace_buffer = torch.empty(32 * 1024 * 1024, device=device, dtype=torch.int8)
    pod_wrapper = flashinfer.PODWithPagedKVCacheWrapper(workspace_buffer, kv_layout_d)
    pod_wrapper.plan(
        kv_indptr_d,
        kv_indices_d,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size_d,
        pos_encoding_mode=pos_encoding_mode,
        data_type=kv_dtype,
        q_data_type=q_dtype,
    )
    o_p, o_d = pod_wrapper.run(
        q_p,
        k_p,
        v_p,
        q_d,
        kv_data,
        pos_encoding_mode_p=pos_encoding_mode,
        causal_p=False,
    )

    torch.testing.assert_close(
        o_p, o_ref_p, rtol=1e-2, atol=1e-2, msg="bf16 prefill mismatch"
    )
    torch.testing.assert_close(
        o_d, o_ref_d, rtol=1e-2, atol=1e-2, msg="bf16 decode mismatch"
    )
