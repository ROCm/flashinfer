# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# HIP/ROCm tests for MLA (Multi-head Latent Attention) batch paged attention.

import math

import pytest
import torch

import flashinfer.mla
from flashinfer.jit import build_jit_specs, gen_batch_mla_module
from flashinfer.utils import determine_mla_backend

HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64


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
            )
        ],
        verbose=False,
    )
    yield


def _make_wrapper():
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
    return flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="auto")


def _plan(wrapper, batch_size, kv_len, page_size, causal, dtype):
    qo_len = 1  # decode: one query token per request
    pages_per_req = math.ceil(kv_len / page_size)
    q_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
        * pages_per_req
    )
    kv_indices = torch.arange(
        0, batch_size * pages_per_req, dtype=torch.int32, device="cuda"
    )
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device="cuda")
    sm_scale = 1.0 / ((HEAD_DIM_CKV + HEAD_DIM_KPE) ** 0.5)
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        16,
        HEAD_DIM_CKV,
        HEAD_DIM_KPE,
        page_size,
        causal,
        sm_scale,
        dtype,
        dtype,
    )
    return kv_lens, pages_per_req


def _mla_reference(q_nope, q_pe, ckv, kpe, kv_lens, page_size, sm_scale, causal):
    """Pure-PyTorch MLA reference: S = q_pe @ kpe^T + q_nope @ ckv^T; O = softmax(S) @ ckv."""
    batch_size, num_heads, _ = q_nope.shape
    dtype = q_nope.dtype

    # ckv/kpe: [num_pages, page_size, head_dim]
    # flatten to [batch_size, kv_len, head_dim] using kv_lens
    max_kv_len = int(kv_lens.max().item())
    # pages are laid out sequentially per request
    pages_per_req = math.ceil(max_kv_len / page_size)

    ckv_flat = ckv.reshape(batch_size, pages_per_req * page_size, HEAD_DIM_CKV)[
        :, :max_kv_len, :
    ]
    kpe_flat = kpe.reshape(batch_size, pages_per_req * page_size, HEAD_DIM_KPE)[
        :, :max_kv_len, :
    ]

    # q: [batch, heads, dim]  K/V: [batch, kv_len, dim]  scores: [batch, heads, kv_len]
    # decode is qo_len=1 so causal masking is a no-op; only kv-length padding matters.
    del causal
    scores = torch.einsum("bhd,bsd->bhs", q_pe.float(), kpe_flat.float())
    scores += torch.einsum("bhd,bsd->bhs", q_nope.float(), ckv_flat.float())
    scores = scores * sm_scale

    for b in range(batch_size):
        kl = int(kv_lens[b].item())
        if kl < max_kv_len:
            scores[b, :, kl:] = float("-inf")

    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhs,bsd->bhd", weights, ckv_flat.float())
    return out.to(dtype)


def test_determine_mla_backend():
    device = torch.device("cuda")
    backend = determine_mla_backend(device)
    assert backend == "hip", f"expected 'hip', got {backend!r}"


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("kv_len", [1, 64, 512])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_batch_mla_plan(batch_size, kv_len, page_size, causal, dtype):
    if causal and kv_len == 0:
        pytest.skip("causal with kv_len=0 unsupported")
    wrapper = _make_wrapper()
    _plan(wrapper, batch_size, kv_len, page_size, causal, dtype)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_len", [64, 256])
@pytest.mark.parametrize("page_size", [16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_batch_mla_correctness(batch_size, kv_len, page_size, causal, dtype):
    torch.manual_seed(42)
    wrapper = _make_wrapper()
    sm_scale = 1.0 / ((HEAD_DIM_CKV + HEAD_DIM_KPE) ** 0.5)
    kv_lens, pages_per_req = _plan(
        wrapper, batch_size, kv_len, page_size, causal, dtype
    )

    q_nope = torch.randn(batch_size, 16, HEAD_DIM_CKV, dtype=dtype, device="cuda")
    q_pe = torch.randn(batch_size, 16, HEAD_DIM_KPE, dtype=dtype, device="cuda")
    ckv = torch.randn(
        batch_size * pages_per_req, page_size, HEAD_DIM_CKV, dtype=dtype, device="cuda"
    )
    kpe = torch.randn(
        batch_size * pages_per_req, page_size, HEAD_DIM_KPE, dtype=dtype, device="cuda"
    )

    out = wrapper.run(q_nope, q_pe, ckv, kpe)

    ref = _mla_reference(
        q_nope,
        q_pe,
        ckv,
        kpe,
        kv_lens,
        page_size,
        sm_scale,
        causal,
    )

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
