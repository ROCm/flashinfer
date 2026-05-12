# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# HIP/ROCm tests for MLA (Multi-head Latent Attention) batch paged attention.
#
# Phase 1 coverage (plan-only):
# - test_determine_mla_backend: verify determine_mla_backend returns "hip"
# - test_batch_mla_plan: verify BatchMLAPagedAttentionPlan runs without error
#   for typical DeepSeek-v2/v3 head dims across batch sizes and dtypes
# - test_batch_mla_run_raises: verify BatchMLAPagedAttentionRun raises
#   RuntimeError with a clear "not yet implemented" message (Phase 2 stub)

import math

import pytest
import torch

import flashinfer
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


def _make_wrapper(batch_size, dtype):
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
    sm_scale = 1.0 / ((128 + 64) ** 0.5)
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        16,  # num_heads
        HEAD_DIM_CKV,
        HEAD_DIM_KPE,
        page_size,
        causal,
        sm_scale,
        dtype,
        dtype,
    )
    return kv_lens, pages_per_req


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
    wrapper = _make_wrapper(batch_size, dtype)
    # plan must not raise
    _plan(wrapper, batch_size, kv_len, page_size, causal, dtype)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_len", [64])
@pytest.mark.parametrize("page_size", [16])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_batch_mla_run_raises(batch_size, kv_len, page_size, dtype):
    wrapper = _make_wrapper(batch_size, dtype)
    kv_lens, pages_per_req = _plan(wrapper, batch_size, kv_len, page_size, False, dtype)

    q_nope = torch.randn(batch_size, 16, HEAD_DIM_CKV, dtype=dtype, device="cuda")
    q_pe = torch.randn(batch_size, 16, HEAD_DIM_KPE, dtype=dtype, device="cuda")
    ckv = torch.randn(
        batch_size * pages_per_req, page_size, HEAD_DIM_CKV, dtype=dtype, device="cuda"
    )
    kpe = torch.randn(
        batch_size * pages_per_req, page_size, HEAD_DIM_KPE, dtype=dtype, device="cuda"
    )

    with pytest.raises(RuntimeError, match="not yet implemented on HIP"):
        wrapper.run(q_nope, q_pe, ckv, kpe)
