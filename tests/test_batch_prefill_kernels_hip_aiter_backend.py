# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from jit_utils import gen_prefill_attention_modules

import flashinfer
from flashinfer.jit.core import logger
import logging

logger.setLevel(logging.ERROR)


def _manual_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Compute multi-head attention with an optional bottom-right causal mask.

    Supports GQA: when ``q`` has more heads than ``k``/``v`` the KV heads are
    repeated to match.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor, shape ``[qo_len, num_qo_heads, head_dim]``.
    k : torch.Tensor
        Key tensor, shape ``[kv_len, num_kv_heads, head_dim]``.
    v : torch.Tensor
        Value tensor, shape ``[kv_len, num_kv_heads, head_dim]``.
    causal : bool
        If ``True``, apply a bottom-right aligned causal mask so that
        ``q[i]`` (mapped to position ``kv_len - qo_len + i``) may only attend
        to ``k[j]`` with ``j <= kv_len - qo_len + i``.
    sm_scale : float, optional
        Softmax scale; defaults to ``1 / sqrt(head_dim)``.

    Returns
    -------
    torch.Tensor
        Output tensor, shape ``[qo_len, num_qo_heads, head_dim]`` in the same
        dtype as *q*.
    """
    qo_len = q.size(0)
    kv_len = k.size(0)
    num_qo_heads = q.size(1)
    num_kv_heads = k.size(1)
    head_dim = q.size(-1)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # GQA: repeat KV heads to match query heads.
    if num_qo_heads != num_kv_heads:
        assert num_qo_heads % num_kv_heads == 0
        reps = num_qo_heads // num_kv_heads
        k = k.repeat_interleave(reps, dim=1)
        v = v.repeat_interleave(reps, dim=1)

    # [num_heads, seq_len, head_dim]
    q_t = q.float().permute(1, 0, 2)
    k_t = k.float().permute(1, 0, 2)
    v_t = v.float().permute(1, 0, 2)

    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * sm_scale  # [H, qo, kv]

    if causal:
        # Bottom-right aligned: q[i] at position (kv_len - qo_len + i) can
        # attend to k[j] where j <= kv_len - qo_len + i.
        offset = kv_len - qo_len
        mask = torch.zeros(qo_len, kv_len, device=q.device, dtype=torch.bool)
        for i in range(qo_len):
            mask[i, i + offset + 1 :] = True
        scores.masked_fill_(mask.unsqueeze(0), float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_t)  # [H, qo, head_dim]
    return out.permute(1, 0, 2).to(q.dtype)  # [qo, H, head_dim]


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_prefill_attention_modules(
            [torch.float16],  # q_dtypes
            [
                torch.float16,
            ],  # kv_dtypes
            [128, 256],  # head_dims
            [0],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False],  # use_logits_soft_caps
            [False],  # use_fp16_qk_reductions
        ),
        verbose=False,
    )
    yield


@pytest.mark.parametrize("batch_size", [12, 17, 30])
@pytest.mark.parametrize("kv_len", [54, 97, 512, 2048])
@pytest.mark.parametrize("qo_len", [37, 17, 127])
@pytest.mark.parametrize("page_size", [1, 5, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE"])
@pytest.mark.parametrize("use_cuda_graph", [False, True])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
@pytest.mark.parametrize("return_lse", [True])
@pytest.mark.parametrize("contiguous_kv", [True])
def test_batch_prefill_with_paged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    kv_layout,
    pos_encoding_mode,
    use_cuda_graph,
    logits_soft_cap,
    return_lse,
    contiguous_kv,
):
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")

    max_num_batched_tokens = 4096

    if batch_size * qo_len > max_num_batched_tokens:
        pytest.skip(
            f"batch_size * qo_len ({batch_size * qo_len}) exceeds max_num_batched_tokens ({max_num_batched_tokens}). You may see OOM errors."
        )

    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    q_indptr_cpu = torch.arange(0, batch_size + 1).int() * qo_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "HND":
        kv_shape = [total_num_pages, 2, num_kv_heads, page_size, head_dim]
    else:
        kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v)
        kv_shape = tmp
        kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
        kv_data = kv_data_fp32.half()
        kv_data = kv_data[:, 1, :, 1, :, 1, :, 1, :]
        kv_data_fp32 = kv_data_fp32[:, 1, :, 1, :, 1, :, 1, :]
        # actual data is stored in non-contiguous memory
        assert (
            kv_data.stride(-4)
            != kv_data.shape[-3] * kv_data.shape[-2] * kv_data.shape[-1]
        )
    else:
        kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
        kv_data = kv_data_fp32.half()
    kv_indptr_cpu = torch.arange(0, batch_size + 1).int() * num_pages_per_seq
    kv_indices_cpu = torch.arange(0, total_num_pages).int()
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )

    workspace_buffer = torch.empty(
        1024 * 1024 * 1024, dtype=torch.int8, device="cuda:0"
    )
    if not use_cuda_graph:
        q_indptr_gpu = q_indptr_cpu.to(0)
        kv_indptr_gpu = kv_indptr_cpu.to(0)
        kv_indices_gpu = kv_indices_cpu.to(0)
        kv_last_page_len_gpu = kv_last_page_len_cpu.to(0)
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout, backend="aiter"
        )
        wrapper.plan(
            q_indptr_gpu,
            kv_indptr_gpu,
            kv_indices_gpu,
            kv_last_page_len_gpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        if return_lse:
            o, _ = wrapper.run(q, kv_data, return_lse=True)
        else:
            o = wrapper.run(q, kv_data)

        # test with pre-allocated output
        o_buffer = torch.empty_like(o)
        wrapper.run(q, kv_data, out=o_buffer)
        torch.testing.assert_close(o, o_buffer, rtol=1e-3, atol=1e-3)
    else:
        q_indptr_buffer = torch.empty(
            batch_size + 1, device="cuda:0", dtype=torch.int32
        )
        kv_indptr_buffer = torch.empty(
            batch_size + 1, device="cuda:0", dtype=torch.int32
        )
        kv_indices_buffer = torch.empty(
            total_num_pages, device="cuda:0", dtype=torch.int32
        )
        kv_last_page_len_buffer = torch.empty(
            batch_size, device="cuda:0", dtype=torch.int32
        )
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer,
            kv_layout,
            use_cuda_graph=True,
            backend="aiter",
            qo_indptr_buf=q_indptr_buffer,
            paged_kv_indptr_buf=kv_indptr_buffer,
            paged_kv_indices_buf=kv_indices_buffer,
            paged_kv_last_page_len_buf=kv_last_page_len_buffer,
        )
        q_indptr_warmup = torch.arange(0, batch_size + 1).int() * qo_len
        kv_indptr_warmup = torch.arange(0, batch_size + 1).int()
        kv_indices_warmup = torch.arange(0, batch_size).int()
        kv_last_page_len_warmup = torch.full(
            (batch_size,), page_size, dtype=torch.int32
        )

        wrapper.plan(
            q_indptr_warmup,
            kv_indptr_warmup,
            kv_indices_warmup,
            kv_last_page_len_warmup,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )

        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                if return_lse:
                    o, _ = wrapper.run(q, kv_data, return_lse=True)
                else:
                    o = wrapper.run(q, kv_data)
        torch.cuda.current_stream().wait_stream(s)
        # capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            if return_lse:
                o, _ = wrapper.run(q, kv_data, return_lse=True)
            else:
                o = wrapper.run(q, kv_data)

        wrapper.plan(
            q_indptr_cpu,
            kv_indptr_cpu,
            kv_indices_cpu,
            kv_last_page_len_cpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )

        g.replay()

    for i in range(batch_size):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        ki = torch.cat(
            [
                kv_data_fp32[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 0]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 0, :, : kv_last_page_len_cpu[i]
                    ]
                    if kv_layout == "HND"
                    else kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 0, : kv_last_page_len_cpu[i], :
                    ]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        vi = torch.cat(
            [
                kv_data_fp32[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 1, :, : kv_last_page_len_cpu[i]
                    ]
                    if kv_layout == "HND"
                    else kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 1, : kv_last_page_len_cpu[i], :
                    ]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        o_ref_i = _manual_attention_reference(qi, ki, vi, causal=causal)
        o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        torch.testing.assert_close(o_i, o_ref_i, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [12, 17, 30])
@pytest.mark.parametrize("kv_len", [54, 97, 512, 2048])
@pytest.mark.parametrize("qo_len", [37, 17, 127])
@pytest.mark.parametrize("page_size", [1, 5, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE"])
@pytest.mark.parametrize("use_cuda_graph", [False, True])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
@pytest.mark.parametrize("return_lse", [True])
@pytest.mark.parametrize("contiguous_kv", [True])
def test_batch_prefill_with_tuple_paged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    kv_layout,
    pos_encoding_mode,
    use_cuda_graph,
    logits_soft_cap,
    return_lse,
    contiguous_kv,
):
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")
        
    max_num_batched_tokens = 4096

    if batch_size * qo_len > max_num_batched_tokens:
        pytest.skip(
            f"batch_size * qo_len ({batch_size * qo_len}) exceeds max_num_batched_tokens ({max_num_batched_tokens}). You may see OOM errors."
        )

    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    q_indptr_cpu = torch.arange(0, batch_size + 1).int() * qo_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "HND":
        kv_shape = [total_num_pages, num_kv_heads, page_size, head_dim]
    else:
        kv_shape = [total_num_pages, page_size, num_kv_heads, head_dim]
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v)
        kv_shape = tmp
        kv_data_fp32 = [
            torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
            for _ in range(2)
        ]
        kv_data = [kv_data_fp32[i].half() for i in range(2)]
        for i in range(2):
            kv_data_fp32[i] = kv_data_fp32[i][:, 1, :, 1, :, 1, :]
            kv_data[i] = kv_data[i][:, 1, :, 1, :, 1, :]
            # actual data is stored in non-contiguous memory
            assert (
                kv_data[i].stride(-4)
                != kv_data[i].shape[-3] * kv_data[i].shape[-2] * kv_data[i].shape[-1]
            )
    else:
        kv_data_fp32 = [
            torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
            for _ in range(2)
        ]
        kv_data = [kv_data_fp32[i].half() for i in range(2)]
    kv_data = tuple(kv_data)
    kv_indptr_cpu = torch.arange(0, batch_size + 1).int() * num_pages_per_seq
    kv_indices_cpu = torch.arange(0, total_num_pages).int()
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )

    workspace_buffer = torch.empty(
        1024 * 1024 * 1024, dtype=torch.int8, device="cuda:0"
    )
    if not use_cuda_graph:
        q_indptr_gpu = q_indptr_cpu.to(0)
        kv_indptr_gpu = kv_indptr_cpu.to(0)
        kv_indices_gpu = kv_indices_cpu.to(0)
        kv_last_page_len_gpu = kv_last_page_len_cpu.to(0)
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout, backend="aiter"
        )
        wrapper.plan(
            q_indptr_gpu,
            kv_indptr_gpu,
            kv_indices_gpu,
            kv_last_page_len_gpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        if return_lse:
            o, _ = wrapper.run(q, kv_data, return_lse=True)
        else:
            o = wrapper.run(q, kv_data)
    else:
        q_indptr_buffer = torch.empty(
            batch_size + 1, device="cuda:0", dtype=torch.int32
        )
        kv_indptr_buffer = torch.empty(
            batch_size + 1, device="cuda:0", dtype=torch.int32
        )
        kv_indices_buffer = torch.empty(
            total_num_pages, device="cuda:0", dtype=torch.int32
        )
        kv_last_page_len_buffer = torch.empty(
            batch_size, device="cuda:0", dtype=torch.int32
        )
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer,
            kv_layout,
            use_cuda_graph=True,
            backend="aiter",
            qo_indptr_buf=q_indptr_buffer,
            paged_kv_indptr_buf=kv_indptr_buffer,
            paged_kv_indices_buf=kv_indices_buffer,
            paged_kv_last_page_len_buf=kv_last_page_len_buffer,
        )
        q_indptr_warmup = torch.arange(0, batch_size + 1).int() * qo_len
        kv_indptr_warmup = torch.arange(0, batch_size + 1).int()
        kv_indices_warmup = torch.arange(0, batch_size).int()
        kv_last_page_len_warmup = torch.full(
            (batch_size,), page_size, dtype=torch.int32
        )
        wrapper.plan(
            q_indptr_warmup,
            kv_indptr_warmup,
            kv_indices_warmup,
            kv_last_page_len_warmup,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )

        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                if return_lse:
                    o, _ = wrapper.run(q, kv_data, return_lse=True)
                else:
                    o = wrapper.run(q, kv_data)
        torch.cuda.current_stream().wait_stream(s)
        # capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            if return_lse:
                o, _ = wrapper.run(q, kv_data, return_lse=True)
            else:
                o = wrapper.run(q, kv_data)

        wrapper.plan(
            q_indptr_cpu,
            kv_indptr_cpu,
            kv_indices_cpu,
            kv_last_page_len_cpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )

        g.replay()

    k_cache, v_cache = kv_data_fp32
    for i in range(batch_size):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        ki = torch.cat(
            [
                k_cache[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    k_cache[kv_indptr_cpu[i + 1] - 1, :, : kv_last_page_len_cpu[i]]
                    if kv_layout == "HND"
                    else k_cache[kv_indptr_cpu[i + 1] - 1, : kv_last_page_len_cpu[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        vi = torch.cat(
            [
                v_cache[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    v_cache[kv_indptr_cpu[i + 1] - 1, :, : kv_last_page_len_cpu[i]]
                    if kv_layout == "HND"
                    else v_cache[kv_indptr_cpu[i + 1] - 1, : kv_last_page_len_cpu[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        o_ref_i = _manual_attention_reference(qi, ki, vi, causal=causal)
        o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        torch.testing.assert_close(o_i, o_ref_i, rtol=1e-3, atol=1e-3)




if __name__ == "__main__":
    test_batch_prefill_with_paged_kv_cache(
        12, 54, 37, 16, 8, 8, 128, True, "NHD", "NONE", True, 0.0, False, True
    )
    test_batch_prefill_with_tuple_paged_kv_cache(
        12, 54, 37, 16, 8, 8, 128, True, "NHD", "NONE", True, 0.0, False, True
    )
    test_batch_prefill_with_paged_kv_cache(
        12, 54, 37, 1, 8, 8, 128, True, "NHD", "NONE", False, 0.0, False, True
    )
