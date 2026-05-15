// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <flashinfer/attention/aiter/single_prefill.cuh>
#include <gpu_iface/enums.hpp>
#include <gpu_iface/layout.cuh>
#include <optional>

#include "pytorch_extension_utils.h"
#include "single_prefill_config.inc"

using flashinfer::MaskMode;
using flashinfer::QKVLayout;

void single_prefill_with_kv_cache(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor tmp,
                                  at::Tensor o, std::optional<at::Tensor> maybe_lse,
                                  int64_t mask_mode_code, int64_t layout,
                                  int64_t window_left ADDITIONAL_FUNC_PARAMS) {
  const auto device = q.device();
  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device);

  const QKVLayout kv_layout = static_cast<QKVLayout>(layout);
  TORCH_CHECK(kv_layout == QKVLayout::kNHD,
              "AITER backend only supports NHD kv_layout; got layout=", layout);

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);
  TORCH_CHECK(mask_mode != MaskMode::kCustom, "AITER backend does not support custom mask");

  const auto q_dtype = q.scalar_type();
  TORCH_CHECK(q_dtype == at::kHalf || q_dtype == at::kBFloat16,
              "AITER backend supports fp16/bf16 only; got dtype=", q_dtype);
  TORCH_CHECK(k.scalar_type() == q_dtype && v.scalar_type() == q_dtype, "q, k, v must share dtype");
  TORCH_CHECK(o.is_contiguous(), "AITER backend requires a contiguous output tensor");

  const hipStream_t stream = c10::hip::getCurrentHIPStream();
  const bool causal = (mask_mode == MaskMode::kCausal);

  const char* dtype_str = (q_dtype == at::kHalf) ? "fp16" : "bf16";
  const auto dtype_enum = (q_dtype == at::kHalf) ? flashinfer::aiter::VariantKey::Dtype::kFp16
                                                 : flashinfer::aiter::VariantKey::Dtype::kBf16;

  Params params;
  params.q = static_cast<DTypeQ*>(q.data_ptr());
  params.k = static_cast<DTypeKV*>(k.data_ptr());
  params.v = static_cast<DTypeKV*>(v.data_ptr());
  params.o = static_cast<DTypeO*>(o.data_ptr());

  // AITER LSE layout: [num_qo_heads, qo_len] in nats; FlashInfer expects [qo_len, num_qo_heads] in log2.
  at::Tensor aiter_lse_scratch;
  if (maybe_lse) {
    aiter_lse_scratch = at::empty({q.size(1), q.size(0)}, maybe_lse->options().dtype(at::kFloat));
    params.lse = static_cast<float*>(aiter_lse_scratch.data_ptr());
  } else {
    params.lse = nullptr;
  }

  params.num_qo_heads = static_cast<uint32_t>(q.size(1));
  params.num_kv_heads = static_cast<uint32_t>(k.size(1));
  params.qo_len = static_cast<uint32_t>(q.size(0));
  params.kv_len = static_cast<uint32_t>(k.size(0));
  params.q_stride_n = static_cast<uint32_t>(q.stride(0));
  params.q_stride_h = static_cast<uint32_t>(q.stride(1));
  params.k_stride_n = static_cast<uint32_t>(k.stride(0));
  params.k_stride_h = static_cast<uint32_t>(k.stride(1));
  params.v_stride_n = static_cast<uint32_t>(v.stride(0));
  params.v_stride_h = static_cast<uint32_t>(v.stride(1));
  params.window_left = static_cast<int32_t>(window_left);
  ADDITIONAL_PARAMS_SETTER

  // AITER JIT variants use group mode; encode batch=1 as seqstart arrays [0, seqlen].
  int32_t cu_q_host[2] = {0, static_cast<int32_t>(params.qo_len)};
  int32_t cu_k_host[2] = {0, static_cast<int32_t>(params.kv_len)};
  at::Tensor cu_seqlens_q_dev = at::empty({2}, q.options().dtype(at::kInt));
  at::Tensor cu_seqlens_k_dev = at::empty({2}, k.options().dtype(at::kInt));
  TORCH_CHECK(hipMemcpyAsync(cu_seqlens_q_dev.data_ptr(), cu_q_host, 2 * sizeof(int32_t),
                             hipMemcpyHostToDevice, stream) == hipSuccess,
              "hipMemcpyAsync failed for cu_seqlens_q");
  TORCH_CHECK(hipMemcpyAsync(cu_seqlens_k_dev.data_ptr(), cu_k_host, 2 * sizeof(int32_t),
                             hipMemcpyHostToDevice, stream) == hipSuccess,
              "hipMemcpyAsync failed for cu_seqlens_k");

  hipError_t status = flashinfer::SinglePrefillWithKVCacheDispatched<HEAD_DIM_QK, HEAD_DIM_VO>(
      params, causal, dtype_str, dtype_enum,
      static_cast<const int32_t*>(cu_seqlens_q_dev.data_ptr()),
      static_cast<const int32_t*>(cu_seqlens_k_dev.data_ptr()),
      static_cast<DTypeO*>(tmp.data_ptr()), stream);
  TORCH_CHECK(status == hipSuccess,
              "AITER SinglePrefill kernel launch failed: ", hipGetErrorString(status));

  if (maybe_lse) {
    aiter_lse_scratch.div_(std::log(2.0));
    maybe_lse->copy_(aiter_lse_scratch.t());
  }
}
