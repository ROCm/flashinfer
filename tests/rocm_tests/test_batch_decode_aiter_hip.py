# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Parity tests for the AITER paged_attention_v1 decode backend vs the FA2 reference.

import logging

import pytest
import torch

import flashinfer
from flashinfer.aiter_utils import is_aiter_supported
from flashinfer.jit.core import logger

logger.setLevel(logging.ERROR)


def _build_paged_kv(
    batch_size, kv_lens, page_size, num_kv_heads, head_dim, dtype, device
):
    """Build a NHD paged KV cache + indptr/indices/last_page_len for the given per-seq lengths.

    KV cache shape: [num_pages_total, page_size, num_kv_heads, head_dim].
    """
    total_pages = sum((L + page_size - 1) // page_size for L in kv_lens) + 4
    k = (
        torch.randn(
            total_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        * 0.1
    ).contiguous()
    v = (
        torch.randn(
            total_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        * 0.1
    ).contiguous()

    indptr = [0]
    indices = []
    last_page_len = []
    pg = 0
    for L in kv_lens:
        rem = L % page_size
        n = (L // page_size) + (1 if rem > 0 else 0)
        indices.extend(range(pg, pg + n))
        pg += n
        indptr.append(len(indices))
        last_page_len.append(rem if rem > 0 else page_size)

    return (
        k,
        v,
        torch.tensor(indptr, dtype=torch.int32, device=device),
        torch.tensor(indices, dtype=torch.int32, device=device),
        torch.tensor(last_page_len, dtype=torch.int32, device=device),
    )


@pytest.mark.skipif(
    not is_aiter_supported(torch.device("cuda:0")),
    reason="AITER backend requires gfx942/gfx950",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 4, 17])
@pytest.mark.parametrize("page_size", [16, 32])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(8, 8), (16, 4), (32, 8)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("max_kv_len", [64, 1024, 2048])
def test_batch_decode_aiter_vs_fa2(
    dtype,
    batch_size,
    page_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    max_kv_len,
):
    torch.manual_seed(0xA17E2)
    device = torch.device("cuda:0")

    # Mix of short/long sequences within the batch.
    kv_lens = [max(1, max_kv_len - 23 * (i % 7)) for i in range(batch_size)]

    k_cache, v_cache, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = (
        _build_paged_kv(
            batch_size, kv_lens, page_size, num_kv_heads, head_dim, dtype, device
        )
    )
    q = (
        torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device)
        * 0.1
    ).contiguous()
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    ref = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    ref.plan(
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o_ref = ref.run(q, (k_cache, v_cache))

    cand = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", backend="aiter"
    )
    cand.plan(
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o_cand = cand.run(q, (k_cache, v_cache))

    rtol, atol = (5e-3, 5e-3) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(o_cand.float(), o_ref.float(), rtol=rtol, atol=atol)


@pytest.mark.skipif(
    not is_aiter_supported(torch.device("cuda:0")),
    reason="AITER backend requires gfx942/gfx950",
)
def test_batch_decode_aiter_rejects_invalid_config():
    """plan() should reject unsupported configs with a clear error."""
    device = torch.device("cuda:0")
    workspace = torch.zeros(8 * 1024 * 1024, dtype=torch.uint8, device=device)
    indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    indices = torch.tensor([0], dtype=torch.int32, device=device)
    last_page_len = torch.tensor([1], dtype=torch.int32, device=device)

    # HND layout not supported.
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "HND", backend="aiter")
    with pytest.raises(ValueError, match="NHD"):
        w.plan(
            indptr,
            indices,
            last_page_len,
            8,
            8,
            128,
            16,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
        )

    # ROPE not supported.
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", backend="aiter")
    with pytest.raises(ValueError, match="pos_encoding_mode"):
        w.plan(
            indptr,
            indices,
            last_page_len,
            8,
            8,
            128,
            16,
            pos_encoding_mode="ROPE_LLAMA",
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
        )

    # tensor cores not supported.
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", use_tensor_cores=True, backend="aiter"
    )
    with pytest.raises(ValueError, match="use_tensor_cores"):
        w.plan(
            indptr,
            indices,
            last_page_len,
            8,
            8,
            128,
            16,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
        )
