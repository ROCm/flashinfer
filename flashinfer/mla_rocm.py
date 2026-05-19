# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# ROCm MLA (Multi-head Latent Attention) wrapper backed by AITER.
# Mirrors the public API of flashinfer.mla.BatchMLAPagedAttentionWrapper
# and routes through aiter.mla.mla_decode_fwd / mla_prefill_fwd.

import functools
from typing import Optional, Tuple, Union

import torch

from .aiter_utils import is_aiter_supported


@functools.cache
def _aiter_mla():
    import aiter.mla as _m

    return _m


def _require_aiter_mla(device: torch.device) -> None:
    if not is_aiter_supported(device):
        try:
            arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        except Exception:
            arch = "unknown"
        raise RuntimeError(f"AITER MLA requires a gfx942/gfx950 GPU; got '{arch}'.")
    try:
        _aiter_mla()
    except ImportError as exc:
        raise ImportError(
            "The 'aiter' package is required for MLA on ROCm. "
            "Install it via:\n"
            "  git clone --recursive https://github.com/ROCm/aiter.git\n"
            "  cd aiter && python3 setup.py develop"
        ) from exc


class BatchMLAPagedAttentionWrapper:
    r"""ROCm MLA paged attention wrapper backed by AITER.

    Mirrors the public API of :class:`flashinfer.mla.BatchMLAPagedAttentionWrapper`
    for use on AMD gfx942/gfx950 GPUs.  Implements the Matrix Absorption variant
    (absorbed W_UQ·W_UK and W_UV·W_O) where the KV-cache stores compressed-KV
    (``ckv``) and rope-key (``kpe``) tensors concatenated into a single buffer.

    KV-cache layout expected by AITER:
        ``kv_buffer[num_pages, page_size, 1, head_dim_ckv + head_dim_kpe]``

    This wrapper accepts the FlashInfer-style separate ``(ckv_cache, kpe_cache)``
    tuple and concatenates them at run time.  For zero-copy, pre-allocate a single
    buffer of shape ``[num_pages, page_size, 1, head_dim_ckv + head_dim_kpe]`` and
    pass it as both arguments (sliced).

    Parameters
    ----------
    float_workspace_buffer : torch.Tensor
        Reserved workspace.  Size is ignored; only the device is used.
    backend : str
        Must be ``"aiter"`` (only supported backend on ROCm).
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        backend: str = "aiter",
    ) -> None:
        if backend != "aiter":
            raise ValueError(
                f"Only backend='aiter' is supported on ROCm; got {backend!r}."
            )
        self.device = float_workspace_buffer.device
        _require_aiter_mla(self.device)

        self._qo_indptr: Optional[torch.Tensor] = None
        self._kv_indptr: Optional[torch.Tensor] = None
        self._kv_indices: Optional[torch.Tensor] = None
        self._kv_last_page_len: Optional[torch.Tensor] = None
        self._sm_scale: float = 1.0
        self._max_seqlen_q: int = 1

    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        use_profiler: bool = False,
    ) -> None:
        r"""Plan MLA attention.

        Parameters
        ----------
        qo_indptr : torch.IntTensor
            Query/output indptr, shape ``[batch_size + 1]``.
            For decode, content is ``[0, 1, …, batch_size]``.
        kv_indptr : torch.IntTensor
            Paged KV indptr, shape ``[batch_size + 1]``.
        kv_indices : torch.IntTensor
            Page indices, shape ``[kv_indptr[-1]]``.
        kv_len_arr : torch.IntTensor
            Per-request KV sequence lengths, shape ``[batch_size]``.
        num_heads : int
            Number of query/output heads.
        head_dim_ckv : int
            Compressed-KV head dimension (512 for DeepSeek V2/V3).
        head_dim_kpe : int
            Rope-key head dimension (64 for DeepSeek V2/V3).
        page_size : int
            Page size of the paged KV-cache.
        causal : bool
            Whether to apply causal masking (no-op for single-token decode).
        sm_scale : float
            Softmax scale (typically ``1 / sqrt(head_dim_ckv + head_dim_kpe)``).
        q_data_type : torch.dtype
            Query dtype; must be ``torch.float16`` or ``torch.bfloat16``.
        kv_data_type : torch.dtype
            KV dtype; must match ``q_data_type``.
        use_profiler : bool
            Ignored (AITER does not expose a per-kernel profiler through this API).
        """
        if use_profiler:
            raise ValueError(
                "use_profiler=True is not supported with the AITER MLA backend."
            )
        if q_data_type not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"AITER MLA requires q_data_type in {{fp16, bf16}}; got {q_data_type}."
            )
        if q_data_type != kv_data_type:
            raise ValueError(
                f"AITER MLA requires q_data_type == kv_data_type; "
                f"got {q_data_type} vs {kv_data_type}."
            )
        for t, name in [
            (qo_indptr, "qo_indptr"),
            (kv_indptr, "kv_indptr"),
            (kv_indices, "kv_indices"),
            (kv_len_arr, "kv_len_arr"),
        ]:
            if t.dtype != torch.int32:
                raise ValueError(
                    f"Expected {name}.dtype == torch.int32, got {t.dtype}."
                )

        self._qo_indptr = qo_indptr.to(self.device, non_blocking=True)
        self._kv_indptr = kv_indptr.to(self.device, non_blocking=True)
        self._kv_indices = kv_indices.to(self.device, non_blocking=True)
        self._kv_last_page_len = kv_len_arr.to(self.device, non_blocking=True)
        self._sm_scale = sm_scale
        qo_lens = qo_indptr[1:] - qo_indptr[:-1]
        self._max_seqlen_q = int(qo_lens.max().item()) if len(qo_lens) > 0 else 1

    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        return_lse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Run MLA attention.

        Parameters
        ----------
        q_nope : torch.Tensor
            Query without rope, shape ``[total_q, num_heads, head_dim_ckv]``.
        q_pe : torch.Tensor
            Rope part of query, shape ``[total_q, num_heads, head_dim_kpe]``.
        ckv_cache : torch.Tensor
            Compressed-KV cache (without rope), shape
            ``[num_pages, page_size, head_dim_ckv]``.
        kpe_cache : torch.Tensor
            Rope-key cache, shape ``[num_pages, page_size, head_dim_kpe]``.
        out : Optional[torch.Tensor]
            Pre-allocated output, shape ``[total_q, num_heads, head_dim_ckv]``.
        return_lse : bool
            Not supported; raises ``NotImplementedError`` if ``True``.

        Returns
        -------
        out : torch.Tensor
            Attention output, shape ``[total_q, num_heads, head_dim_ckv]``.
        """
        if return_lse:
            raise NotImplementedError(
                "return_lse is not currently supported by the AITER MLA backend."
            )
        if self._qo_indptr is None:
            raise RuntimeError("plan() must be called before run().")

        total_q, num_heads, head_dim_ckv = q_nope.shape
        if out is None:
            out = torch.empty(
                (total_q, num_heads, head_dim_ckv),
                dtype=q_nope.dtype,
                device=q_nope.device,
            )

        q = torch.cat([q_nope, q_pe], dim=-1)
        # kpe_cache is concatenated each call; pre-allocate a combined [pages, size, 1, ckv+kpe]
        # buffer and pass sliced views to avoid this copy on hot paths.
        kv_buffer = torch.cat([ckv_cache.unsqueeze(2), kpe_cache.unsqueeze(2)], dim=-1)

        if self._max_seqlen_q == 1:
            _aiter_mla().mla_decode_fwd(
                q,
                kv_buffer,
                out,
                self._qo_indptr,
                self._kv_indptr,
                self._kv_indices,
                self._kv_last_page_len,
                self._max_seqlen_q,
                sm_scale=self._sm_scale,
            )
        else:
            _aiter_mla().mla_prefill_fwd(
                q,
                kv_buffer,
                out,
                self._qo_indptr,
                self._kv_indptr,
                self._kv_indices,
                self._kv_last_page_len,
                self._max_seqlen_q,
                sm_scale=self._sm_scale,
            )

        return out
