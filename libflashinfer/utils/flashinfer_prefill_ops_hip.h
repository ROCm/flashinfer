// SPDX-FileCopyrightText: 2023-2025 FlashInfer team.
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

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
hipError_t BatchPrefillWithRaggedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                   float* tmp_s, hipStream_t stream);

template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename Params>
hipError_t BatchPrefillWithPagedKVCacheDispatched(Params params, typename Params::DTypeO* tmp_v,
                                                  float* tmp_s, hipStream_t stream);

class BatchPrefillHandler {
 public:
  void UpdatePageLockedBufferSize(size_t int_workspace_size_in_bytes) {
    hipFreeHost(page_locked_buffer_);
    hipMallocHost(&page_locked_buffer_, int_workspace_size_in_bytes);
  }

  template <typename DTypeO, typename IdType>
  hipError_t Plan(void* float_buffer, size_t float_workspace_size_in_bytes, void* int_buffer,
                  size_t int_workspace_size_in_bytes, IdType* qo_indptr_h, IdType* kv_indptr_h,
                  uint32_t total_num_rows, uint32_t batch_size, uint32_t num_qo_heads,
                  uint32_t num_kv_heads, uint32_t head_dim, uint32_t page_size) {
    int_buffer_ = int_buffer;
    float_buffer_ = float_buffer;
    return PrefillPlan<IdType>(float_buffer, float_workspace_size_in_bytes, int_buffer,
                               page_locked_buffer_, int_workspace_size_in_bytes, plan_info_,
                               qo_indptr_h, kv_indptr_h, total_num_rows, batch_size, num_qo_heads,
                               num_kv_heads, head_dim, head_dim, page_size, enable_cuda_graph_,
                               sizeof(DTypeO), stream_);
  }

  hipStream_t GetCUDAStream() const { return stream_; }

  void SetCUDAStream(hipStream_t stream) { stream_ = stream; }

  bool IsCUDAGraphEnabled() const { return enable_cuda_graph_; }

  BatchPrefillHandler(bool enable_cuda_graph = false)
      : enable_cuda_graph_(enable_cuda_graph), stream_(nullptr) {
    hipMallocHost(&page_locked_buffer_, 8 * 1024 * 1024);
  }
  ~BatchPrefillHandler() {
    if (page_locked_buffer_ != nullptr) {
      (page_locked_buffer_);
    }
  }

  PrefillPlanInfo GetPlanInfo() const { return plan_info_; }

  template <typename IdType>
  IdType* GetRequestIndices() {
    return GetPtrFromBaseOffset<IdType>(int_buffer_, plan_info_.request_indices_offset);
  }

  template <typename IdType>
  IdType* GetQOTileIndices() {
    return GetPtrFromBaseOffset<IdType>(int_buffer_, plan_info_.qo_tile_indices_offset);
  }

  template <typename IdType>
  IdType* GetKVTileIndices() {
    return GetPtrFromBaseOffset<IdType>(int_buffer_, plan_info_.kv_tile_indices_offset);
  }

  template <typename IdType>
  IdType* GetOIndptr() {
    return GetPtrFromBaseOffset<IdType>(int_buffer_, plan_info_.o_indptr_offset);
  }

  template <typename IdType>
  IdType* GetKVChunkSizePtr() {
    return GetPtrFromBaseOffset<IdType>(int_buffer_, plan_info_.kv_chunk_size_ptr_offset);
  }

  template <typename IdType>
  IdType* GetMergeIndptr() {
    if (plan_info_.split_kv) {
      return GetPtrFromBaseOffset<IdType>(int_buffer_, plan_info_.merge_indptr_offset);
    }
    return nullptr;
  }

  template <typename DTypeO>
  DTypeO* GetTmpV() {
    if (plan_info_.split_kv) {
      return GetPtrFromBaseOffset<DTypeO>(float_buffer_, plan_info_.v_offset);
    }
    return nullptr;
  }

  float* GetTmpS() {
    if (plan_info_.split_kv) {
      return GetPtrFromBaseOffset<float>(float_buffer_, plan_info_.s_offset);
    }
    return nullptr;
  }

  uint32_t* GetTotalNumRows() {
    if (plan_info_.enable_cuda_graph) {
      return GetPtrFromBaseOffset<uint32_t>(int_buffer_, plan_info_.total_num_rows_offset);
    }
    return nullptr;
  }

  bool* GetBlockValidMask() {
    if (plan_info_.split_kv && plan_info_.enable_cuda_graph) {
      return GetPtrFromBaseOffset<bool>(int_buffer_, plan_info_.block_valid_mask_offset);
    }
    return nullptr;
  }

 protected:
  void* page_locked_buffer_{nullptr};
  void* int_buffer_;
  void* float_buffer_;
  PrefillPlanInfo plan_info_;
  bool enable_cuda_graph_;
  hipStream_t stream_;
};

template <typename DTypeQ, typename DTypeKV, typename DTypeO, typename IdType>
hipError_t BatchPrefillWithRaggedKVCacheWrapper(
    BatchPrefillHandler* handler, DTypeQ* q, IdType* qo_indptr, DTypeKV* k, DTypeKV* v,
    IdType* kv_indptr, IdType* q_rope_offset, IdType* k_rope_offset, DTypeO* o, float* lse,
    const uint32_t batch_size, const uint32_t num_qo_heads, const uint32_t num_kv_heads,
    const uint32_t head_dim, bool causal = true, QKVLayout kv_layout = QKVLayout::kNHD,
    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone, bool use_fp16_qk_reduction = false,
    std::optional<float> maybe_sm_scale = std::nullopt, const float rope_scale = 1.f,
    const float rope_theta = 1e4, hipStream_t stream = nullptr) {
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(head_dim)));
  const MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;
  auto [qo_stride_n, qo_stride_h, kv_stride_n, kv_stride_h] =
      get_qkv_strides(kv_layout, 0, num_qo_heads, num_kv_heads, head_dim);
  auto plan_info = handler->GetPlanInfo();
  DISPATCH_head_dim(
      head_dim, HEAD_DIM,
      {DISPATCH_mask_mode(
          mask_mode, MASK_MODE,
          {DISPATCH_pos_encoding_mode(
              pos_encoding_mode, POS_ENCODING_MODE,
              {DISPATCH_use_fp16_qk_reduction(use_fp16_qk_reduction, USE_FP16_QK_REDUCTION, {
                using Params = BatchPrefillRaggedParams<DTypeQ, DTypeKV, DTypeO, IdType>;
                using AttentionVariant = DefaultAttention<
                    /*use_custom_mask=*/(MASK_MODE == MaskMode::kCustom),
                    /*use_sliding_window=*/false,
                    /*use_logits_soft_cap=*/false, /*use_alibi=*/false>;
                Params params(q, k, v, /*custom_mask=*/nullptr, qo_indptr, kv_indptr,
                              /*mask_indptr=*/nullptr, q_rope_offset, k_rope_offset, o, lse,
                              /*alibi_slopes=*/nullptr, num_qo_heads, num_kv_heads, qo_stride_n,
                              qo_stride_h, kv_stride_n, kv_stride_h,
                              /*window_left=*/-1,
                              /*logits_soft_cap=*/0.f, sm_scale, rope_scale, rope_theta);
                params.request_indices = handler->GetRequestIndices<IdType>();
                params.qo_tile_indices = handler->GetQOTileIndices<IdType>();
                params.kv_tile_indices = handler->GetKVTileIndices<IdType>();
                params.o_indptr = handler->GetOIndptr<IdType>();
                params.kv_chunk_size_ptr = handler->GetKVChunkSizePtr<IdType>();
                params.merge_indptr = handler->GetMergeIndptr<IdType>();
                params.block_valid_mask = handler->GetBlockValidMask();
                params.max_total_num_rows = plan_info.total_num_rows;
                params.total_num_rows = handler->GetTotalNumRows();
                params.padded_batch_size = plan_info.padded_batch_size;

                DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
                  BatchPrefillWithRaggedKVCacheDispatched<CTA_TILE_Q, HEAD_DIM, HEAD_DIM,
                                                          POS_ENCODING_MODE, USE_FP16_QK_REDUCTION,
                                                          MASK_MODE, AttentionVariant>(
                      params, handler->GetTmpV<DTypeO>(), handler->GetTmpS(), stream);
                });
              })})})});
  return hipSuccess;
}

template <typename DTypeQ, typename DTypeKV, typename DTypeO, typename IdType>
hipError_t BatchPrefillWithPagedKVCacheWrapper(
    BatchPrefillHandler* handler, DTypeQ* q, IdType* qo_indptr, IdType* q_rope_offset,
    paged_kv_t<DTypeKV, IdType> paged_kv, DTypeO* o, float* lse, uint32_t num_qo_heads,
    bool causal = true, PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
    bool use_fp16_qk_reduction = false, std::optional<float> maybe_sm_scale = std::nullopt,
    float rope_scale = 1.f, float rope_theta = 1e4, hipStream_t stream = nullptr) {
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(paged_kv.head_dim)));
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t head_dim = paged_kv.head_dim;
  const MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;
  auto plan_info = handler->GetPlanInfo();
  DISPATCH_head_dim(
      head_dim, HEAD_DIM,
      {DISPATCH_mask_mode(
          mask_mode, MASK_MODE,
          {DISPATCH_pos_encoding_mode(
              pos_encoding_mode, POS_ENCODING_MODE,
              {DISPATCH_use_fp16_qk_reduction(use_fp16_qk_reduction, USE_FP16_QK_REDUCTION, {
                using Params = BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeO, IdType>;
                using AttentionVariant = DefaultAttention<
                    /*use_custom_mask=*/(MASK_MODE == MaskMode::kCustom),
                    /*use_sliding_window=*/false,
                    /*use_logits_soft_cap=*/false,
                    /*use_alibi=*/false>;
                Params params(q, paged_kv, /*custom_mask=*/nullptr, qo_indptr,
                              /*mask_indptr=*/nullptr, q_rope_offset, o, lse,
                              /*alibi_slopes=*/nullptr, num_qo_heads,
                              /*q_stride_n*/ num_qo_heads * HEAD_DIM,
                              /*q_stride_h*/ HEAD_DIM,
                              /*window_left=*/-1, /*logits_soft_cap=*/0.f, sm_scale, rope_scale,
                              rope_theta);
                params.request_indices = handler->GetRequestIndices<IdType>();
                params.qo_tile_indices = handler->GetQOTileIndices<IdType>();
                params.kv_tile_indices = handler->GetKVTileIndices<IdType>();
                params.o_indptr = handler->GetOIndptr<IdType>();
                params.kv_chunk_size_ptr = handler->GetKVChunkSizePtr<IdType>();
                params.merge_indptr = handler->GetMergeIndptr<IdType>();
                params.block_valid_mask = handler->GetBlockValidMask();
                params.max_total_num_rows = plan_info.total_num_rows;
                params.total_num_rows = handler->GetTotalNumRows();
                params.padded_batch_size = plan_info.padded_batch_size;
                DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
                  return BatchPrefillWithPagedKVCacheDispatched<
                      CTA_TILE_Q, HEAD_DIM, HEAD_DIM, POS_ENCODING_MODE, USE_FP16_QK_REDUCTION,
                      MASK_MODE, AttentionVariant>(params, handler->GetTmpV<DTypeO>(),
                                                   handler->GetTmpS(), stream);
                })
              })})})});
  return hipSuccess;
}

}  // namespace flashinfer
