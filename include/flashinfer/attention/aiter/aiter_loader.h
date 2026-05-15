// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>

namespace flashinfer::aiter {

struct VariantKey {
  enum class Dtype : uint8_t { kFp16, kBf16 };
  Dtype dtype;
  bool causal;
  bool has_lse;
  bool has_alibi;
  bool has_logits_cap;

  bool operator==(VariantKey const& o) const noexcept {
    return dtype == o.dtype && causal == o.causal && has_lse == o.has_lse &&
           has_alibi == o.has_alibi && has_logits_cap == o.has_logits_cap;
  }
};

struct VariantKeyHash {
  std::size_t operator()(VariantKey const& k) const noexcept {
    std::size_t h = static_cast<std::size_t>(k.dtype);
    h = h * 31 + k.causal;
    h = h * 31 + k.has_lse;
    h = h * 31 + k.has_alibi;
    h = h * 31 + k.has_logits_cap;
    return h;
  }
};

// Returns the raw dlsym function pointer for aiter::mha_fwd(mha_fwd_args, stream_config const&).
// The variant .so matching `key` is loaded via dlopen on first call; cached thereafter.
// Throws std::runtime_error if the variant .so is not found or the symbol is missing.
void* get_aiter_mha_fwd_handle(VariantKey const& key);

// Batch-prefill variants share the same key fields as mha_fwd variants.
// page_size is NOT in the key — the .so dispatches all native page sizes at runtime.
using BatchPrefillVariantKey     = VariantKey;
using BatchPrefillVariantKeyHash = VariantKeyHash;

// Returns the raw dlsym function pointer for aiter::mha_batch_prefill(...).
// The JIT directory is resolved from FLASHINFER_AITER_JIT_DIR / AITER_JIT_DIR env vars at
// runtime; no jit_dir argument is needed. The .so is built lazily by AITER — bootstrap
// by calling aiter.ops.mha.mha_batch_prefill_func once before invoking this function.
// Throws std::runtime_error if the variant .so is not found or the symbol is missing.
void* get_aiter_mha_batch_prefill_handle(BatchPrefillVariantKey const& key);

}  // namespace flashinfer::aiter
