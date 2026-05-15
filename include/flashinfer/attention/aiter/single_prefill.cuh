// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Framework-agnostic (raw pointers + hipStream_t, no at::Tensor) template that
// calls AITER's mha_fwd C++ symbol directly via a cached dlopen function pointer.

#pragma once

#include <hip/hip_runtime.h>
#include <ck_tile/host/stream_config.hpp>

#include <flashinfer/attention/aiter/aiter_loader.h>
#include <flashinfer/attention/aiter/mha_fwd_args.h>

namespace flashinfer {

// CK Tile mask_type codes (from CK example mask.hpp):
//   no_mask=0, mask_top_left=1 (causal), mask_bottom_right=2, window_generic=3
inline constexpr int32_t kAiterMaskNone = 0;
inline constexpr int32_t kAiterMaskTopLeft = 1;  // standard causal

// params.lse: [num_qo_heads, qo_len] float32 scratch in natural-log scale; nullptr to skip.
// tmp: unused; accepted for API parity with the FA2 template.
template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, typename Params>
hipError_t SinglePrefillWithKVCacheDispatched(Params const& params, bool causal,
                                              const char* dtype_str,
                                              flashinfer::aiter::VariantKey::Dtype dtype_enum,
                                              const int32_t* cu_seqlens_q,
                                              const int32_t* cu_seqlens_k,
                                              void* /* tmp */, hipStream_t stream) {
  static_assert(HEAD_DIM_QK == HEAD_DIM_VO,
                "AITER backend requires HEAD_DIM_QK == HEAD_DIM_VO");

  const bool has_lse = (params.lse != nullptr);
  const bool has_logits_cap = (params.logits_soft_cap > 0.0);

  const flashinfer::aiter::VariantKey key{
      .dtype = dtype_enum,
      .causal = causal,
      .has_lse = has_lse,
      .has_alibi = false,
      .has_logits_cap = has_logits_cap,
  };

  using mha_fwd_fn = float (*)(::aiter::mha_fwd_args, ::ck_tile::stream_config const&);
  auto fn = reinterpret_cast<mha_fwd_fn>(flashinfer::aiter::get_aiter_mha_fwd_handle(key));

  ::aiter::mha_fwd_args args{};
  // Runtime traits baked into mha_fwd_args (select pipeline within the variant .so)
  args.use_asm_v3 = false;
  args.v3_api_check = false;
  args.how_v3_bf16_cvt = 0;
  args.data_type = dtype_str;
  // AITER JIT variants only contain group-mode kernel specializations.
  // Use group mode with seqstart arrays [0, seqlen] to represent batch=1.
  args.is_group_mode = true;
  args.bias_type = 0;          // no bias / no alibi
  args.has_lse = has_lse;
  args.qscale_type = 0;
  args.has_sink = false;

  args.q_ptr = static_cast<const void*>(params.q);
  args.k_ptr = static_cast<const void*>(params.k);
  args.v_ptr = static_cast<const void*>(params.v);
  args.o_ptr = static_cast<void*>(params.o);
  args.lse_ptr = static_cast<void*>(params.lse);

  // Group mode with seqstart arrays [0, seqlen] encodes a single-sequence batch.
  args.seqstart_q_ptr = static_cast<const void*>(cu_seqlens_q);
  args.seqstart_k_ptr = static_cast<const void*>(cu_seqlens_k);

  args.seqlen_q = static_cast<int32_t>(params.qo_len);
  args.seqlen_k = static_cast<int32_t>(params.kv_len);
  args.batch = 1;
  args.max_seqlen_q = static_cast<int32_t>(params.qo_len);
  args.hdim_q = static_cast<int32_t>(HEAD_DIM_QK);
  args.hdim_v = static_cast<int32_t>(HEAD_DIM_VO);
  args.nhead_q = static_cast<int32_t>(params.num_qo_heads);
  args.nhead_k = static_cast<int32_t>(params.num_kv_heads);

  args.scale_s = static_cast<float>(params.sm_scale);
  args.logits_soft_cap = static_cast<float>(params.logits_soft_cap);

  args.stride_q = static_cast<int32_t>(params.q_stride_n);
  args.stride_k = static_cast<int32_t>(params.k_stride_n);
  args.stride_v = static_cast<int32_t>(params.v_stride_n);
  // Output is always contiguous NHD [qo_len, num_qo_heads, HEAD_DIM_VO]
  args.stride_o = static_cast<int32_t>(params.num_qo_heads * HEAD_DIM_VO);

  args.nhead_stride_q = static_cast<int32_t>(params.q_stride_h);
  args.nhead_stride_k = static_cast<int32_t>(params.k_stride_h);
  args.nhead_stride_v = static_cast<int32_t>(params.v_stride_h);
  // LSE layout is [num_qo_heads, qo_len] in natural-log — nhead stride = qo_len
  args.nhead_stride_lse = static_cast<int32_t>(params.qo_len);
  args.nhead_stride_o = static_cast<int32_t>(HEAD_DIM_VO);

  args.mask_type = causal ? kAiterMaskTopLeft : kAiterMaskNone;
  args.window_size_left = static_cast<int32_t>(params.window_left);
  args.window_size_right = -1;

  ::ck_tile::stream_config sconfig{};
  sconfig.stream_id_ = stream;

  fn(args, sconfig);
  return hipSuccess;
}

}  // namespace flashinfer
