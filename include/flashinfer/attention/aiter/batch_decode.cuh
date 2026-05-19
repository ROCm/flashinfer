// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Framework-agnostic helper that invokes AITER's PA v1 (paged_attention_v1) extern-C
// JIT entry via a cached dlopen+dlsym handle. The .so is per-variant (template params
// baked in by AITER's compile_template_op) — the Python plan() side resolves the
// (so_path, func_name) by calling aiter.csrc.cpp_itfs.pa.pa_v1.compile() and passes
// both strings down here.

#pragma once

#include <flashinfer/attention/aiter/aiter_loader.h>
#include <hip/hip_runtime.h>

#include <string>

namespace flashinfer {

// Extern-C signature emitted by aiter/csrc/cpp_itfs/pa/pa_v1.cpp.jinja.
// Pinned to amd-aiter 0.1.10 — regenerate alongside any AITER pa_v1 template change.
using AiterPaV1Fn = void (*)(void* out_ptr, void* workspace_buffer_ptr, void* query_ptr,
                             void* key_cache_ptr, void* value_cache_ptr, int* block_tables_ptr,
                             int* cu_query_lens_ptr, int* context_lens_ptr,
                             const float* alibi_slopes_ptr, const float* q_scale_ptr,
                             const float* k_scale_ptr, const float* v_scale_ptr,
                             const float* fp8_out_scale_ptr, float scale,
                             int max_num_blocks_per_seq, int max_num_partitions,
                             float logits_soft_cap, int num_seqs, int num_kv_heads, int num_heads,
                             int q_stride, int kv_block_stride, int kv_head_stride,
                             int kv_seq_stride, int sliding_window, void* stream);

// Bytes of workspace needed by PA v1:
//   exp_sums   : float [num_seqs, num_heads, max_num_partitions]
//   max_logits : float [num_seqs, num_heads, max_num_partitions]
//   tmp_out    : dtype [num_seqs, num_heads, max_num_partitions, head_size]
inline std::size_t AiterPaV1WorkspaceBytes(int num_seqs, int num_heads, int max_num_partitions,
                                           int head_size, std::size_t dtype_bytes) {
  const std::size_t per_part = static_cast<std::size_t>(num_seqs) * num_heads * max_num_partitions;
  return per_part * (2 * sizeof(float) + dtype_bytes * static_cast<std::size_t>(head_size));
}

inline hipError_t BatchDecodeAiterPaV1Run(
    const std::string& so_path, const std::string& func_name, void* out_ptr,
    void* workspace_buffer_ptr, const void* query_ptr, const void* key_cache_ptr,
    const void* value_cache_ptr, const int32_t* block_tables_ptr, const int32_t* cu_query_lens_ptr,
    const int32_t* context_lens_ptr, const float* alibi_slopes_ptr, const float* q_scale_ptr,
    const float* k_scale_ptr, const float* v_scale_ptr, const float* fp8_out_scale_ptr, float scale,
    int max_num_blocks_per_seq, int max_num_partitions, float logits_soft_cap, int num_seqs,
    int num_kv_heads, int num_heads, int q_stride, int kv_block_stride, int kv_head_stride,
    int kv_seq_stride, int sliding_window, hipStream_t stream) {
  auto fn = reinterpret_cast<AiterPaV1Fn>(
      flashinfer::aiter::get_aiter_extern_c_handle(so_path, func_name));
  fn(out_ptr, workspace_buffer_ptr, const_cast<void*>(query_ptr), const_cast<void*>(key_cache_ptr),
     const_cast<void*>(value_cache_ptr), const_cast<int*>(block_tables_ptr),
     const_cast<int*>(cu_query_lens_ptr), const_cast<int*>(context_lens_ptr), alibi_slopes_ptr,
     q_scale_ptr, k_scale_ptr, v_scale_ptr, fp8_out_scale_ptr, scale, max_num_blocks_per_seq,
     max_num_partitions, logits_soft_cap, num_seqs, num_kv_heads, num_heads, q_stride,
     kv_block_stride, kv_head_stride, kv_seq_stride, sliding_window, static_cast<void*>(stream));
  return hipGetLastError();
}

}  // namespace flashinfer
