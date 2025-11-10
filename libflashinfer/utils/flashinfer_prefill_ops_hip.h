// SPDX-FileCopyrightText: 2023-2025 Flashinfer team
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "flashinfer/attention/generic/allocator.h"
#include "flashinfer/attention/generic/default_prefill_params.cuh"
#include "flashinfer/attention/generic/exception.h"
#include "flashinfer/attention/generic/prefill.cuh"
#include "flashinfer/attention/generic/scheduler.cuh"
#include "flashinfer/attention/generic/variants.cuh"
#include "gpu_iface/enums.hpp"
#include "gpu_iface/layout.cuh"
#include "utils_hip.h"

namespace flashinfer {

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, PosEncodingMode POS_ENCODING_MODE,
          bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE, typename AttentionVariant,
          typename Params>
hipError_t SinglePrefillWithKVCacheDispatched(Params params, typename Params::DTypeO* tmp,
                                              hipStream_t stream);

template <typename DTypeIn, typename DTypeO>
hipError_t SinglePrefillWithKVCacheCustomMask(
    DTypeIn* q, DTypeIn* k, DTypeIn* v, uint8_t* custom_mask, DTypeO* o, DTypeO* tmp, float* lse,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    uint32_t head_dim, QKVLayout kv_layout = QKVLayout::kNHD,
    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone, bool use_fp16_qk_reduction = false,
    std::optional<float> maybe_sm_scale = std::nullopt, float rope_scale = 1.f,
    float rope_theta = 1e4, hipStream_t stream = nullptr) {
  const float sm_scale = 1.f;
  auto [qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h] =
      get_qkv_strides(kv_layout, kv_len, num_qo_heads, num_kv_heads, head_dim);
  DISPATCH_use_fp16_qk_reduction(
      static_cast<int>(use_fp16_qk_reduction), USE_FP16_QK_REDUCTION,
      {DISPATCH_head_dim(
          head_dim, HEAD_DIM, {DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
            using Params = SinglePrefillParams<DTypeIn, DTypeIn, DTypeO>;
            using AttentionVariant = DefaultAttention<
                /*use_custom_mask=*/true, /*use_sliding_window=*/false,
                /*use_logits_soft_cap=*/false, /*use_alibi=*/false>;
            Params params(q, k, v, custom_mask, o, lse,
                          /*alibi_slopes=*/nullptr, num_qo_heads, num_kv_heads, qo_len, kv_len,
                          qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h, head_dim,
                          /*window_left=*/-1,
                          /*logits_soft_cap=*/0.f, sm_scale, rope_scale, rope_theta);
            return SinglePrefillWithKVCacheDispatched<HEAD_DIM, HEAD_DIM, POS_ENCODING_MODE,
                                                      USE_FP16_QK_REDUCTION, MaskMode::kCustom,
                                                      AttentionVariant>(params, tmp, stream);
          })})});
  return hipSuccess;
}

/*!
 * \brief FlashAttention prefill hip function for a single request.
 * \tparam DTypeIn The data type of input
 * \tparam DTypeO The data type of output
 * \param q The query tensor.
 * \param k The key tensor.
 * \param v The value tensor.
 * \param o The output tensor.
 * \param tmp The temporary storage (only used for cooperative kernel).
 * \param lse The logsumexp values.
 * \param num_qo_heads The number of query and output heads.
 * \param num_kv_heads The number of key and value heads.
 * \param qo_len The length of query and output.
 * \param kv_len The length of key and value.
 * \param head_dim The dimension of each head.
 * \param causal Whether to use causal attention.
 * \param kv_layout The layout of input and output.
 * \param pos_encoding_mode The positional encoding mode.
 * \param use_fp16_qk_reduction Whether to allow accumulating q*k^T with fp16.
 * \param rope_scale The scaling factor used in RoPE interpolation.
 * \param rope_theta The theta used in RoPE.
 * \param stream The hip stream to execute the kernel on.
 * \return status Indicates whether hip calls are successful
 */
template <typename DTypeQ, typename DTypeKV, typename DTypeO>
hipError_t SinglePrefillWithKVCache(DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeO* o, DTypeO* tmp,
                                    float* lse, uint32_t num_qo_heads, uint32_t num_kv_heads,
                                    uint32_t qo_len, uint32_t kv_len, uint32_t head_dim,
                                    bool causal = true, QKVLayout kv_layout = QKVLayout::kNHD,
                                    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
                                    bool use_fp16_qk_reduction = false,
                                    std::optional<float> maybe_sm_scale = std::nullopt,
                                    float rope_scale = 1.f, float rope_theta = 1e4,
                                    hipStream_t stream = nullptr) {
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));
  const MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;
  auto [qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h] =
      get_qkv_strides(kv_layout, kv_len, num_qo_heads, num_kv_heads, head_dim);
  DISPATCH_use_fp16_qk_reduction(
      static_cast<int>(use_fp16_qk_reduction), USE_FP16_QK_REDUCTION,
      {DISPATCH_mask_mode(
          mask_mode, MASK_MODE,
          {DISPATCH_head_dim(
              head_dim, HEAD_DIM,
              {DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
                using Params = SinglePrefillParams<DTypeQ, DTypeKV, DTypeO>;
                using AttentionVariant = DefaultAttention<
                    /*use_custom_mask=*/(MASK_MODE == MaskMode::kCustom),
                    /*use_sliding_window=*/false,
                    /*use_logits_soft_cap=*/true, /*use_alibi=*/false>;
                Params params(q, k, v, /*custom_mask=*/nullptr, o, lse,
                              /*alibi_slopes=*/nullptr, num_qo_heads, num_kv_heads, qo_len, kv_len,
                              qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h, head_dim,
                              /*window_left=*/-1,
                              /*logits_soft_cap=*/8.f, sm_scale, rope_scale, rope_theta);
                return SinglePrefillWithKVCacheDispatched<HEAD_DIM, HEAD_DIM, POS_ENCODING_MODE,
                                                          USE_FP16_QK_REDUCTION, MASK_MODE,
                                                          AttentionVariant, Params>(params, tmp,
                                                                                    stream);
              })})})});
  return hipSuccess;
}

// Version without logits soft cap for testing
template <typename DTypeQ, typename DTypeKV, typename DTypeO>
hipError_t SinglePrefillWithKVCacheNoSoftCap(
    DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeO* o, DTypeO* tmp, float* lse, uint32_t num_qo_heads,
    uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len, uint32_t head_dim, bool causal = true,
    QKVLayout kv_layout = QKVLayout::kNHD,
    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone, bool use_fp16_qk_reduction = false,
    std::optional<float> maybe_sm_scale = std::nullopt, float rope_scale = 1.f,
    float rope_theta = 1e4, hipStream_t stream = nullptr) {
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));
  const MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;
  auto [qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h] =
      get_qkv_strides(kv_layout, kv_len, num_qo_heads, num_kv_heads, head_dim);
  DISPATCH_use_fp16_qk_reduction(
      static_cast<int>(use_fp16_qk_reduction), USE_FP16_QK_REDUCTION,
      {DISPATCH_mask_mode(
          mask_mode, MASK_MODE,
          {DISPATCH_head_dim(
              head_dim, HEAD_DIM,
              {DISPATCH_pos_encoding_mode(pos_encoding_mode, POS_ENCODING_MODE, {
                using Params = SinglePrefillParams<DTypeQ, DTypeKV, DTypeO>;
                using AttentionVariant = DefaultAttention<
                    /*use_custom_mask=*/(MASK_MODE == MaskMode::kCustom),
                    /*use_sliding_window=*/false,
                    /*use_logits_soft_cap=*/false, /*use_alibi=*/false>;
                Params params(q, k, v, /*custom_mask=*/nullptr, o, lse,
                              /*alibi_slopes=*/nullptr, num_qo_heads, num_kv_heads, qo_len, kv_len,
                              qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h, head_dim,
                              /*window_left=*/-1,
                              /*logits_soft_cap=*/0.f, sm_scale, rope_scale, rope_theta);
                return SinglePrefillWithKVCacheDispatched<HEAD_DIM, HEAD_DIM, POS_ENCODING_MODE,
                                                          USE_FP16_QK_REDUCTION, MASK_MODE,
                                                          AttentionVariant, Params>(params, tmp,
                                                                                    stream);
              })})})});
  return hipSuccess;
}

}  // namespace flashinfer
