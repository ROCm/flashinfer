// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Framework-agnostic (raw pointers + hipStream_t, no at::Tensor) template that
// calls AITER's mha_fwd C++ symbol directly via a cached dlopen function pointer.

#pragma once

#include <flashinfer/rocm/attention/aiter/aiter_loader.h>
#include <flashinfer/rocm/attention/aiter/mha_fwd_args.h>
#include <hip/hip_runtime.h>

#include <ck_tile/host/stream_config.hpp>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <flashinfer/rocm/gpu_runtime_compat.hpp>

namespace flashinfer {

// CK Tile mask_type codes (from CK example mask.hpp):
//   no_mask=0, mask_top_left=1 (causal), mask_bottom_right=2, window_generic=3
inline constexpr int32_t kAiterMaskNone = 0;
inline constexpr int32_t kAiterMaskTopLeft = 1;  // standard causal (qo_len == kv_len)
inline constexpr int32_t kAiterMaskBottomRight =
    2;  // prefill-with-history causal (kv_len > qo_len)

// AITER's ASM v3 fast path is restricted to the (hdim_q, hdim_v) pairs that ship as
// precompiled .co files in aiter_meta/hsa/gfx9{42,50}/fmha_v3_fwd/. AITER also ships
// hd192/hd128 .co files, but the single-prefill entry point hard-requires
// HEAD_DIM_QK == HEAD_DIM_VO (see static_assert below), so only the equal-hdim pair
// is reachable here.
inline constexpr bool AiterAsmV3HdimSupported(uint32_t hdim_q, uint32_t hdim_v) {
  return hdim_q == 128 && hdim_v == 128;
}

// Cheap pre-filter for the ASM v3 arm, mirroring the guard in aiter::fmha_fwd_v3.
// AITER's own v3_api_check probe is the authoritative test, but it needs the asm .so
// dlopened and it logs a warning on every rejection, so screen the obvious misses here
// rather than probing for each fp16 or hd64 caller.
inline constexpr bool AiterAsmV3Eligible(uint32_t hdim_q, uint32_t hdim_v,
                                         flashinfer::aiter::VariantKey::Dtype dtype,
                                         bool has_logits_cap, int32_t window_left) {
  return AiterAsmV3HdimSupported(hdim_q, hdim_v) &&
         dtype == flashinfer::aiter::VariantKey::Dtype::kBf16 && !has_logits_cap && window_left < 0;
}

// Smallest qo_len at which the asm arm is worth taking, by architecture. Mirrors
// _AITER_ASM_PREFILL_MIN_QO_LEN in flashinfer/rocm/arch_caps.py, which carries the
// measurement; 0 means never. The line below is the one
// tests/rocm/test_aiter_asm_routing.py reads, so keep the two in step.
//   arch_caps: gfx942=None gfx950=2048
inline uint32_t AiterAsmPrefillMinQoLen(const char* arch) {
  if (arch != nullptr && std::strcmp(arch, "gfx950") == 0) return 2048u;
  return 0u;  // gfx942 loses non-monotonically; an unknown arch is treated the same
}

// FLASHINFER_AITER_ASM_PREFILL=0 pins the CK Tile arm. AITER aborts the process on
// several asm failure modes rather than returning, so an operator-side off switch is
// part of the contract, not a convenience.
inline bool AiterAsmPrefillEnabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("FLASHINFER_AITER_ASM_PREFILL");
    return v == nullptr || std::strcmp(v, "0") != 0;
  }();
  return enabled;
}

// params.lse: [num_qo_heads, qo_len] float32 scratch in natural-log scale; nullptr to skip.
// tmp: unused; accepted for API parity with the FA2 template.
// cu_seqlens_q / cu_seqlens_k: [0, seqlen] arrays on device; consumed only when
// logits_soft_cap > 0 (which forces the varlen/group-mode .so because the mha_fwd
// template has no _logits arm). May be nullptr otherwise.
template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, typename Params>
hipError_t SinglePrefillWithKVCacheDispatched(Params const& params, bool causal,
                                              const char* dtype_str,
                                              flashinfer::aiter::VariantKey::Dtype dtype_enum,
                                              const int32_t* cu_seqlens_q,
                                              const int32_t* cu_seqlens_k, void* /* tmp */,
                                              hipStream_t stream) {
  static_assert(HEAD_DIM_QK == HEAD_DIM_VO, "AITER backend requires HEAD_DIM_QK == HEAD_DIM_VO");

  const bool has_lse = (params.lse != nullptr);
  const bool has_logits_cap = (params.logits_soft_cap > 0.0f);

  // A window wider than the sequence masks nothing; normalizing it here (as AITER's
  // own torch path does) keeps such a call off the _mask variant and its cold build.
  const int32_t window_left = (params.window_left >= static_cast<int32_t>(params.kv_len))
                                  ? -1
                                  : static_cast<int32_t>(params.window_left);
  const bool needs_mask = causal || window_left >= 0;

  const flashinfer::aiter::VariantKey key{
      .dtype = dtype_enum,
      .needs_mask = needs_mask,
      .has_lse = has_lse,
      .has_alibi = false,
      .has_logits_cap = has_logits_cap,
  };

  // AITER ships two dispatcher .so templates: mha_fwd (batch-mode CK + ASM v3) and
  // mha_varlen_fwd (group-mode CK + ASM v3). Single-prefill is batch=1, so the
  // mha_fwd .so is the natural fit — except it has no _logits arm, so logits_soft_cap
  // forces us onto the varlen path.
  using mha_fwd_fn = float (*)(::aiter::mha_fwd_args, ::ck_tile::stream_config const&);
  auto fn = reinterpret_cast<mha_fwd_fn>(
      has_logits_cap ? flashinfer::aiter::get_aiter_mha_varlen_fwd_handle(key)
                     : flashinfer::aiter::get_aiter_mha_fwd_handle(key));

  // Defensive: the varlen pipeline dereferences seqstart_*_ptr in CK Tile. Surface
  // missing pointers as hipErrorInvalidValue here rather than as a memory fault
  // inside AITER.
  if (has_logits_cap && (cu_seqlens_q == nullptr || cu_seqlens_k == nullptr)) {
    return hipErrorInvalidValue;
  }

  ::aiter::mha_fwd_args args{};
  // Set per arm below: the CK Tile .so ignores this field entirely (it is built
  // -DFAV2_ON=1 with no -DFAV3_ON), and the asm .so returns -1 unless it is true.
  args.use_asm_v3 = false;
  args.v3_api_check = false;
  args.how_v3_bf16_cvt = 0;
  args.data_type = dtype_str;
  args.is_group_mode = has_logits_cap;
  args.bias_type = 0;  // no bias / no alibi
  args.has_lse = has_lse;
  args.qscale_type = 0;
  args.has_sink = false;

  args.q_ptr = static_cast<const void*>(params.q);
  args.k_ptr = static_cast<const void*>(params.k);
  args.v_ptr = static_cast<const void*>(params.v);
  args.o_ptr = static_cast<void*>(params.o);
  args.lse_ptr = static_cast<void*>(params.lse);

  // Group mode encodes batch=1 as seqstart arrays [0, seqlen] on device.
  // Batch mode addresses tensors via stride/nhead_stride only — batch_stride_*
  // are unused at batch=1 and left at their default 0.
  args.seqstart_q_ptr = args.is_group_mode ? static_cast<const void*>(cu_seqlens_q) : nullptr;
  args.seqstart_k_ptr = args.is_group_mode ? static_cast<const void*>(cu_seqlens_k) : nullptr;

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

  // mask_bottom_right: q[i] attends to kv[kv_len−qo_len+i], correct for prefill-with-history.
  // When qo_len == kv_len, mask_bottom_right degenerates to mask_top_left.
  // A window needs it too: right=-1 saturates to the full extent, leaving the
  // left-bound-only band FlashInfer defines. right=0 is the causal convention.
  args.mask_type = needs_mask ? kAiterMaskBottomRight : kAiterMaskNone;
  args.window_size_left = window_left;
  args.window_size_right = causal ? 0 : -1;

  ::ck_tile::stream_config sconfig{};
  sconfig.stream_id_ = stream;

  int device = 0;
  uint32_t asm_min_qo_len = 0;
  if (hipGetDevice(&device) == hipSuccess) {
    try {
      asm_min_qo_len = AiterAsmPrefillMinQoLen(getGcnArchName(device));
    } catch (const std::exception&) {
      asm_min_qo_len = 0;  // FI_HIP_CALL throws; an unreadable arch stays on CK Tile
    }
  }
  const bool asm_wanted =
      asm_min_qo_len > 0 && params.qo_len >= asm_min_qo_len && AiterAsmPrefillEnabled() &&
      AiterAsmV3Eligible(HEAD_DIM_QK, HEAD_DIM_VO, dtype_enum, has_logits_cap, window_left);

  if (asm_wanted) {
    // Everything here is best-effort: any failure leaves the output untouched and
    // falls through to CK Tile below. The asm module is built -DENABLE_CK=0, so it
    // reports a miss as a negative return having launched nothing, and reports a
    // failed launch by throwing out of ck_tile_shim rather than returning.
    try {
      auto asm_fn = reinterpret_cast<mha_fwd_fn>(flashinfer::aiter::get_aiter_mha_fwd_asm_handle());

      // Probe once per trait set. v3_api_check resolves AITER's config table and
      // returns without launching, so this costs one lookup per process rather than
      // a wasted dispatch per call. Only needs_mask varies the lookup here: head dim
      // is a template parameter, dtype is fixed by the eligibility check above, and
      // is_group_mode is false whenever there is no soft cap.
      static thread_local int probe[2] = {0, 0};  // 0 unknown, 1 supported, -1 not
      const int slot = needs_mask ? 1 : 0;
      if (probe[slot] == 0) {
        ::aiter::mha_fwd_args probe_args = args;
        probe_args.use_asm_v3 = true;
        probe_args.v3_api_check = true;
        probe[slot] = asm_fn(probe_args, sconfig) > 0.f ? 1 : -1;
      }

      if (probe[slot] == 1) {
        args.use_asm_v3 = true;
        if (asm_fn(args, sconfig) >= 0.f) {
          // A failed launch would have thrown above, and the shim consumed
          // hipGetLastError() on its way out, so there is nothing left to report.
          return hipSuccess;
        }
        args.use_asm_v3 = false;
      }
    } catch (const std::exception&) {
      // Missing .so, unset AITER_ASM_DIR, or a launch failure inside AITER. CK Tile
      // serves the call instead; surfacing AITER's message here would replace a
      // working fallback with a hard error.
    }
  }

  // A negative return means no kernel instance matched and nothing was launched,
  // which would otherwise leave the caller's output buffer untouched and unflagged.
  if (fn(args, sconfig) < 0.f) return hipErrorNoBinaryForGpu;
  return hipGetLastError();
}

}  // namespace flashinfer
