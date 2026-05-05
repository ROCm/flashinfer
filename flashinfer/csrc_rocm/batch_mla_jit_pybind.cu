// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "batch_mla_config.inc"
#include "pytorch_extension_utils.h"

at::Tensor BatchMLAPagedAttentionPlan(at::Tensor float_workspace_buffer,
                                      at::Tensor int_workspace_buffer,
                                      at::Tensor pin_memory_int_workspace_buffer,
                                      at::Tensor qo_indptr, at::Tensor kv_indptr,
                                      at::Tensor kv_len_arr, int64_t num_heads, int64_t head_dim_o,
                                      bool causal);

void BatchMLAPagedAttentionRun(at::Tensor float_workspace_buffer, at::Tensor int_workspace_buffer,
                               at::Tensor plan_info_vec, at::Tensor q_nope, at::Tensor q_pe,
                               at::Tensor ckv_cache, at::Tensor kpe_cache, at::Tensor kv_indices,
                               at::Tensor o, std::optional<at::Tensor> maybe_lse,
                               int64_t mask_mode_code, int64_t num_heads, int64_t page_size,
                               double sm_scale, bool return_lse_base_on_e ADDITIONAL_FUNC_PARAMS);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("plan", BatchMLAPagedAttentionPlan);
  m.def("run", BatchMLAPagedAttentionRun);
}
