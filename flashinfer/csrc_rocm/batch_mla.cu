// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

#include <flashinfer/attention/mla_hip.cuh>

#include "batch_mla_config.inc"
#include "pytorch_conversion_utils.h"
#include "pytorch_extension_utils.h"

using namespace flashinfer;

at::Tensor BatchMLAPagedAttentionPlan(at::Tensor float_workspace_buffer,
                                      at::Tensor int_workspace_buffer,
                                      at::Tensor pin_memory_int_workspace_buffer,
                                      at::Tensor qo_indptr, at::Tensor kv_indptr,
                                      at::Tensor kv_len_arr, int64_t num_heads, int64_t head_dim_o,
                                      bool causal) {
  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * float_workspace_buffer.element_size();
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * int_workspace_buffer.element_size();

  MLAPlanInfo plan_info;
  int64_t batch_size = kv_len_arr.size(0);

  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(float_workspace_buffer.device());
  const hipStream_t stream = c10::hip::getCurrentHIPStream();

  hipError_t status =
      MLAPlan(float_workspace_buffer.data_ptr(), float_workspace_size_in_bytes,
              int_workspace_buffer.data_ptr(), pin_memory_int_workspace_buffer.data_ptr(),
              int_workspace_size_in_bytes, plan_info, static_cast<IdType*>(qo_indptr.data_ptr()),
              static_cast<IdType*>(kv_indptr.data_ptr()),
              static_cast<IdType*>(kv_len_arr.data_ptr()), static_cast<uint32_t>(batch_size),
              static_cast<uint32_t>(num_heads), static_cast<uint32_t>(head_dim_o), causal, stream);

  TORCH_CHECK(status == hipSuccess,
              "BatchMLAPagedAttentionPlan failed with error: ", hipGetErrorString(status));

  return vec_to_tensor(plan_info.ToVector());
}

void BatchMLAPagedAttentionRun(at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
                               at::Tensor plan_info_vec, at::Tensor q_nope, at::Tensor q_pe,
                               at::Tensor ckv_cache, at::Tensor kpe_cache, at::Tensor kv_indices,
                               at::Tensor o, std::optional<at::Tensor> maybe_lse,
                               int64_t mask_mode_code, int64_t num_heads, int64_t page_size,
                               double sm_scale, bool return_lse_base_on_e ADDITIONAL_FUNC_PARAMS) {
  MLAPlanInfo plan_info;
  plan_info.FromVector(tensor_to_vec(plan_info_vec));

  void* float_buffer_ptr = float_workspace_buffer.data_ptr();
  void* int_buffer_ptr = int_workspace_buffer.data_ptr();

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(q_nope.device());
  const hipStream_t stream = c10::hip::getCurrentHIPStream();

  unsigned int q_nope_stride_n = q_nope.stride(0);
  unsigned int q_nope_stride_h = q_nope.stride(1);
  unsigned int q_pe_stride_n = q_pe.stride(0);
  unsigned int q_pe_stride_h = q_pe.stride(1);
  unsigned int ckv_stride_page = ckv_cache.stride(0);
  unsigned int ckv_stride_n = ckv_cache.stride(1);
  unsigned int kpe_stride_page = kpe_cache.stride(0);
  unsigned int kpe_stride_n = kpe_cache.stride(1);
  unsigned int o_stride_n = o.stride(0);
  unsigned int o_stride_h = o.stride(1);

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_CKV, HEAD_DIM_KPE, Params, [&] {
        Params params;

        params.q_nope = static_cast<DTypeQ*>(q_nope.data_ptr());
        params.q_pe = static_cast<DTypeQ*>(q_pe.data_ptr());
        params.ckv = static_cast<DTypeKV*>(ckv_cache.data_ptr());
        params.kpe = static_cast<DTypeKV*>(kpe_cache.data_ptr());

        params.q_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.q_indptr_offset);
        params.kv_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_indptr_offset);
        params.partial_indptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.partial_indptr_offset);
        params.kv_indices = static_cast<IdType*>(kv_indices.data_ptr());
        params.q_len = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.q_len_offset);
        params.kv_len = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_len_offset);
        params.q_start = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.q_start_offset);
        params.kv_start = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_start_offset);
        params.kv_end = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_end_offset);
        params.work_indptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.work_indptr_offset);
        params.merge_packed_offset_start = GetPtrFromBaseOffset<IdType>(
            int_buffer_ptr, plan_info.merge_packed_offset_start_offset);
        params.merge_packed_offset_end =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_packed_offset_end_offset);
        params.merge_partial_packed_offset_start = GetPtrFromBaseOffset<IdType>(
            int_buffer_ptr, plan_info.merge_partial_packed_offset_start_offset);
        params.merge_partial_packed_offset_end = GetPtrFromBaseOffset<IdType>(
            int_buffer_ptr, plan_info.merge_partial_packed_offset_end_offset);
        params.merge_partial_stride =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_partial_stride_offset);

        params.final_o = static_cast<DTypeO*>(o.data_ptr());
        params.final_lse =
            maybe_lse.has_value() ? static_cast<float*>(maybe_lse.value().data_ptr()) : nullptr;
        params.partial_o =
            GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr, plan_info.partial_o_offset);
        params.partial_lse =
            GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.partial_lse_offset);

        params.num_heads = uint_fastdiv(num_heads);
        params.block_size = uint_fastdiv(page_size);

        params.q_nope_stride_n = q_nope_stride_n;
        params.q_nope_stride_h = q_nope_stride_h;
        params.q_pe_stride_n = q_pe_stride_n;
        params.q_pe_stride_h = q_pe_stride_h;
        params.ckv_stride_page = ckv_stride_page;
        params.ckv_stride_n = ckv_stride_n;
        params.kpe_stride_page = kpe_stride_page;
        params.kpe_stride_n = kpe_stride_n;
        params.o_stride_n = o_stride_n;
        params.o_stride_h = o_stride_h;

        params.sm_scale = static_cast<float>(sm_scale);
        params.return_lse_base_on_e = return_lse_base_on_e;

        ADDITIONAL_PARAMS_SETTER

        hipError_t status = mla::BatchMLAPagedAttentionHIP<MASK_MODE, HEAD_DIM_CKV, HEAD_DIM_KPE>(
            params, plan_info.num_blks_x, plan_info.num_blks_y, stream);
        TORCH_CHECK(status == hipSuccess,
                    "BatchMLAPagedAttentionRun failed: ", hipGetErrorString(status));
      });
}
