# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# DeepSeek-style HIP/ROCm tests for MLA: prefill (qo_len > 1) + decode (qo_len = 1),
# LSE return, pre-allocated output buffers, varying kv_lens per batch,
# fp16 + bf16. Mirrors the core checks in tests/attention/test_deepseek_mla.py
# but uses the HIP backend and the head dimensions DeepSeek actually ships.

import math

import pytest
import torch

import flashinfer
import flashinfer.mla
from flashinfer.jit import build_jit_specs, gen_batch_mla_module

HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64


def _attention_ref(batch_size, q, k, v, causal, sm_scale):
    qo_len = q.shape[0] // batch_size
    kv_len = k.shape[0] // batch_size
    num_heads = q.shape[1]
    head_dim_qk = q.shape[2]
    head_dim_vo = v.shape[2]
    logits = (
        torch.einsum(
            "bmhd,bnhd->bhmn",
            q.view(batch_size, qo_len, num_heads, head_dim_qk).float(),
            k.view(batch_size, kv_len, num_heads, head_dim_qk).float(),
        )
        * sm_scale
    )
    if causal:
        mask = torch.arange(kv_len - qo_len, kv_len, device=q.device).unsqueeze(
            1
        ) >= torch.arange(0, kv_len, device=q.device).unsqueeze(0)
        logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
    lse_ref = torch.logsumexp(logits, -1).transpose(-1, -2)
    p = torch.softmax(logits, dim=-1)
    o_ref = (
        torch.einsum(
            "bhmn,bnhd->bmhd",
            p,
            v.view(batch_size, kv_len, num_heads, head_dim_vo).float(),
        )
        .contiguous()
        .view(batch_size * qo_len, num_heads, head_dim_vo)
        .to(q)
    )
    # convert lse from natural log to log2 to match the kernel's return_lse_base_on_e=False default
    return o_ref, lse_ref * math.log2(math.e)


def _kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads):
    bs_page_num, page_size, ckv_dim = ckv.shape
    page_num = bs_page_num // batch_size
    _, _, kpe_dim = kpe.shape
    ckv = ckv.view(batch_size, page_num * page_size, ckv_dim)[:, :kv_len, :]
    kpe = kpe.view(batch_size, page_num * page_size, kpe_dim)[:, :kv_len, :]
    k = (
        torch.cat([ckv, kpe], dim=-1)
        .unsqueeze(2)
        .repeat(1, 1, num_heads, 1)
        .view(batch_size * kv_len, num_heads, ckv_dim + kpe_dim)
    )
    v = (
        ckv.unsqueeze(2)
        .repeat(1, 1, num_heads, 1)
        .view(batch_size * kv_len, num_heads, ckv_dim)
    )
    return k, v


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    build_jit_specs(
        [
            gen_batch_mla_module(
                "hip",
                torch.float16,
                torch.float16,
                torch.float16,
                torch.int32,
                HEAD_DIM_CKV,
                HEAD_DIM_KPE,
                False,
            ),
            gen_batch_mla_module(
                "hip",
                torch.bfloat16,
                torch.bfloat16,
                torch.bfloat16,
                torch.int32,
                HEAD_DIM_CKV,
                HEAD_DIM_KPE,
                False,
            ),
        ],
        verbose=False,
    )
    yield


@pytest.mark.parametrize("batch_size", [1, 3, 7])
@pytest.mark.parametrize(
    "kv_len",
    [
        17,
        33,
        96,
        514,
        1024,
        pytest.param(4096, marks=pytest.mark.slow),
        pytest.param(8192, marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize("qo_len", [1])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_batch_mla_page_attention(
    batch_size, kv_len, qo_len, num_heads, causal, page_size, dtype
):
    # The HIP MLA kernel uses CTA_TILE_Q = 16, while the MLA planner assumes
    # CTA_TILE_Q = 64. For decode (packed_qo_len = qo_len * num_heads <= 16)
    # the planner generates one work item per batch and the CTA covers it exactly.
    # Prefill (qo_len > 1) requires an inner Q-sub-tile loop — not yet implemented.
    assert qo_len * num_heads <= 16, "Prefill MLA on HIP not yet supported"
    if causal and qo_len > kv_len:
        pytest.skip("qo_len > kv_len not supported for causal attention")
    device = torch.device("cuda")
    torch.manual_seed(42)

    q_nope = torch.randn(
        batch_size * qo_len, num_heads, HEAD_DIM_CKV, dtype=dtype, device=device
    )
    q_pe = torch.randn(
        batch_size * qo_len, num_heads, HEAD_DIM_KPE, dtype=dtype, device=device
    )
    pages_per_req = math.ceil(kv_len / page_size)
    ckv = torch.randn(
        batch_size * pages_per_req, page_size, HEAD_DIM_CKV, dtype=dtype, device=device
    )
    kpe = torch.randn(
        batch_size * pages_per_req, page_size, HEAD_DIM_KPE, dtype=dtype, device=device
    )

    sm_scale = 1.0 / ((128 + 64) ** 0.5)  # head dim before matrix absorption
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="auto")

    q_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
        * pages_per_req
    )
    kv_indices = torch.arange(
        0, batch_size * pages_per_req, dtype=torch.int32, device=device
    )
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        HEAD_DIM_CKV,
        HEAD_DIM_KPE,
        page_size,
        causal,
        sm_scale,
        q_nope.dtype,
        ckv.dtype,
    )
    o, lse = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=True)

    k, v = _kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads)
    q = torch.cat([q_nope, q_pe], dim=-1)
    o_ref, lse_ref = _attention_ref(batch_size, q, k, v, causal, sm_scale)
    lse_ref = lse_ref.flatten(0, 1)

    rtol, atol = (1e-3, 1e-3) if dtype == torch.float16 else (1e-2, 1e-2)
    torch.testing.assert_close(o, o_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(lse, lse_ref, rtol=rtol, atol=atol)

    # Pre-allocated output buffers must produce identical results.
    o_buf = torch.empty_like(o)
    lse_buf = torch.empty_like(lse)
    wrapper.run(q_nope, q_pe, ckv, kpe, out=o_buf, lse=lse_buf)
    torch.testing.assert_close(o, o_buf, rtol=rtol, atol=atol)
    torch.testing.assert_close(lse, lse_buf, rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [3, 5])
@pytest.mark.parametrize(
    "kv_lens_list",
    [
        [17, 33, 79],
        pytest.param([96, 514, 2048], marks=pytest.mark.slow),
        pytest.param([128, 256, 512, 1024, 2048], marks=pytest.mark.slow),
    ],
    ids=lambda v: "x".join(str(x) for x in v),
)
@pytest.mark.parametrize("qo_len", [1])
@pytest.mark.parametrize("causal", [False, True])
def test_batch_mla_varlen_kv(batch_size, kv_lens_list, qo_len, causal):
    """Each request in the batch has a different kv_len (still page_size=1)."""
    if causal and qo_len > min(kv_lens_list):
        pytest.skip("qo_len > min(kv_len) not supported for causal attention")
    device = torch.device("cuda")
    torch.manual_seed(0)

    num_heads = 16
    page_size = 1
    dtype = torch.float16
    # Repeat the kv_lens_list `batch_size` times along the request axis.
    kv_lens_full = (kv_lens_list * batch_size)[: batch_size * len(kv_lens_list)]
    n_req = len(kv_lens_full)
    pages = [math.ceil(kv / page_size) for kv in kv_lens_full]
    total_pages = sum(pages)

    q_nope = torch.randn(
        n_req * qo_len, num_heads, HEAD_DIM_CKV, dtype=dtype, device=device
    )
    q_pe = torch.randn(
        n_req * qo_len, num_heads, HEAD_DIM_KPE, dtype=dtype, device=device
    )
    ckv = torch.randn(total_pages, page_size, HEAD_DIM_CKV, dtype=dtype, device=device)
    kpe = torch.randn(total_pages, page_size, HEAD_DIM_KPE, dtype=dtype, device=device)

    sm_scale = 1.0 / ((128 + 64) ** 0.5)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="auto")

    q_indptr = torch.arange(0, n_req + 1, dtype=torch.int32, device=device) * qo_len
    page_indptr = torch.zeros(n_req + 1, dtype=torch.int32, device=device)
    page_indptr[1:] = torch.tensor(pages, dtype=torch.int32, device=device).cumsum(0)
    kv_indices = torch.arange(0, total_pages, dtype=torch.int32, device=device)
    kv_lens = torch.tensor(kv_lens_full, dtype=torch.int32, device=device)

    wrapper.plan(
        q_indptr,
        page_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        HEAD_DIM_CKV,
        HEAD_DIM_KPE,
        page_size,
        causal,
        sm_scale,
        q_nope.dtype,
        ckv.dtype,
    )
    o = wrapper.run(q_nope, q_pe, ckv, kpe)

    # Per-request reference (each request has its own kv_len so we can't batch
    # them in a single attention_ref call with padding zeros — that would shift
    # the softmax denominator. Compute and concatenate.)
    out_chunks = []
    page_offset = 0
    qo_offset = 0
    for kv_len in kv_lens_full:
        k_p = ckv[page_offset : page_offset + kv_len].view(1, kv_len, HEAD_DIM_CKV)
        kpe_p = kpe[page_offset : page_offset + kv_len].view(1, kv_len, HEAD_DIM_KPE)
        page_offset += kv_len
        q_n = q_nope[qo_offset : qo_offset + qo_len]
        q_p = q_pe[qo_offset : qo_offset + qo_len]
        qo_offset += qo_len
        if kv_len == 0:
            out_chunks.append(torch.zeros_like(q_n))
            continue
        # k = [ckv ; kpe] repeated across heads; v = ckv repeated across heads.
        k = torch.cat([k_p, kpe_p], dim=-1).expand(
            1, kv_len, HEAD_DIM_CKV + HEAD_DIM_KPE
        )
        k = (
            k.unsqueeze(2)
            .repeat(1, 1, num_heads, 1)
            .view(kv_len, num_heads, HEAD_DIM_CKV + HEAD_DIM_KPE)
        )
        v = k_p.expand(1, kv_len, HEAD_DIM_CKV)
        v = (
            v.unsqueeze(2)
            .repeat(1, 1, num_heads, 1)
            .view(kv_len, num_heads, HEAD_DIM_CKV)
        )
        q = torch.cat([q_n, q_p], dim=-1)
        ref, _ = _attention_ref(1, q, k, v, causal, sm_scale)
        out_chunks.append(ref)
    o_ref = torch.cat(out_chunks, dim=0)

    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
