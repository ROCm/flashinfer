// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

#include <flashinfer/attention/mla_params.cuh>

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
  TORCH_CHECK(false, "BatchMLAPagedAttentionRun: not yet implemented on HIP/ROCm");
}
