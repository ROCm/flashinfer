# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# HIP/ROCm test for append_paged_mla_kv_cache (MLA KV-cache write path).
# Mirrors tests/attention/test_mla_page.py.

import math

import pytest
import torch

import flashinfer

CKV_DIM = 512
KPE_DIM = 64


def _last_page_lens(kv_lens, page_size):
    return [kl % page_size if kl % page_size != 0 else page_size for kl in kv_lens]


_KV_LEN_CONFIGS = [
    [45],
    [4096],
    [45, 8, 25],
    [45, 8, 25, 22],
    [45, 8, 25, 22, 400],
    [45, 8, 25, 22, 100],
]


@pytest.mark.parametrize("kv_lens", _KV_LEN_CONFIGS)
@pytest.mark.parametrize("page_size", [1, 16, 64])
def test_append_mla_paged_kv_cache(kv_lens, page_size):
    device = torch.device("cuda")
    nnz_kv = sum(kv_lens)
    ckv_append = torch.randn(nnz_kv, CKV_DIM, dtype=torch.float16, device=device)
    kpe_append = torch.randn(nnz_kv, KPE_DIM, dtype=torch.float16, device=device)

    num_pages_per_req = torch.tensor(
        [math.ceil(kl / page_size) for kl in kv_lens], dtype=torch.int32, device=device
    )
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    kv_append_indptr = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), kv_lens_t.cumsum(0).int()]
    )

    max_num_pages = int(num_pages_per_req.sum().item())
    ckv_cache = torch.zeros(
        max_num_pages, page_size, CKV_DIM, dtype=torch.float16, device=device
    )
    kpe_cache = torch.zeros(
        max_num_pages, page_size, KPE_DIM, dtype=torch.float16, device=device
    )

    kv_page_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            num_pages_per_req.cumsum(0).int(),
        ]
    )
    kv_page_indices = torch.arange(max_num_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor(
        _last_page_lens(kv_lens, page_size), dtype=torch.int32, device=device
    )

    batch_indices, positions = flashinfer.get_batch_indices_positions(
        kv_append_indptr,
        flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, page_size),
        nnz_kv,
    )
    flashinfer.append_paged_mla_kv_cache(
        ckv_append,
        kpe_append,
        batch_indices,
        positions,
        ckv_cache,
        kpe_cache,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_len,
    )

    ckv_flat = ckv_cache.view(-1, CKV_DIM)
    kpe_flat = kpe_cache.view(-1, KPE_DIM)

    acc_kv = 0
    acc_pad = 0
    for i, kl in enumerate(kv_lens):
        pages_i = int(num_pages_per_req[i].item())
        torch.testing.assert_close(
            ckv_append[acc_kv : acc_kv + kl], ckv_flat[acc_pad : acc_pad + kl]
        )
        torch.testing.assert_close(
            kpe_append[acc_kv : acc_kv + kl], kpe_flat[acc_pad : acc_pad + kl]
        )
        # Padding slots must remain zero.
        assert torch.all(ckv_flat[acc_pad + kl : acc_pad + pages_i * page_size] == 0)
        assert torch.all(kpe_flat[acc_pad + kl : acc_pad + pages_i * page_size] == 0)
        acc_kv += kl
        acc_pad += pages_i * page_size
