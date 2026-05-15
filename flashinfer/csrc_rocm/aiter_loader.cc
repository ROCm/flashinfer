// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

// No AITER or CK Tile headers needed here — we work with void* function pointers.

#include <flashinfer/attention/aiter/aiter_loader.h>

#include <dlfcn.h>

#include <cstdlib>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace flashinfer::aiter {

namespace {

std::string get_jit_dir() {
  if (const char* env = std::getenv("AITER_JIT_DIR")) return env;
#ifdef FLASHINFER_AITER_JIT_DIR
  return FLASHINFER_AITER_JIT_DIR;
#else
  throw std::runtime_error(
      "AITER_JIT_DIR env var not set and FLASHINFER_AITER_JIT_DIR not compiled in. "
      "Set AITER_JIT_DIR=<path to aiter/jit/> or rebuild the FlashInfer JIT cache "
      "after installing AITER (rm -rf ~/.cache/flashinfer/).");
#endif
}

// Build a variant .so filename from a key.
// Segments: {prefix}{dtype}_{logits}_{bias}_{mask}_{lse}{suffix}
// mha_varlen_fwd:    prefix="mha_varlen_fwd_",   suffix="_ndropout_nskip_nqscale.so"
// mha_batch_prefill: prefix="mha_batch_prefill_", suffix="_ndropout_nqscale.so"
std::string build_so_name(VariantKey const& key, std::string_view prefix, std::string_view suffix) {
  std::string name(prefix);
  name += (key.dtype == VariantKey::Dtype::kFp16) ? "fp16" : "bf16";
  name += key.has_logits_cap ? "_logits" : "_nlogits";
  name += key.has_alibi ? "_alibi" : "_nbias";
  name += key.causal ? "_mask" : "_nmask";
  name += key.has_lse ? "_lse" : "_nlse";
  name += suffix;
  return name;
}

std::string variant_so_name(VariantKey const& key) {
  return build_so_name(key, "mha_varlen_fwd_", "_ndropout_nskip_nqscale.so");
}

// Mangled symbol for aiter::mha_fwd(aiter::mha_fwd_args, ck_tile::stream_config const&).
// Stable across GCC/Clang Itanium ABI; verified by `nm -D` on all shipped variants.
// Pinned to amd-aiter 0.1.10. Regenerate with: nm -D <variant.so> | grep mha_fwd
constexpr const char* kMhaFwdSymbol =
    "_ZN5aiter7mha_fwdENS_12mha_fwd_argsERKN7ck_tile13stream_configE";

std::shared_mutex s_mu;
std::unordered_map<VariantKey, void*, VariantKeyHash> s_cache;

}  // namespace

void* get_aiter_mha_fwd_handle(VariantKey const& key) {
  {
    std::shared_lock rd(s_mu);
    auto it = s_cache.find(key);
    if (it != s_cache.end()) return it->second;
  }

  std::unique_lock wr(s_mu);
  auto it = s_cache.find(key);
  if (it != s_cache.end()) return it->second;

  std::string jit_dir = get_jit_dir();
  std::string so_path = jit_dir + "/" + variant_so_name(key);

  // RTLD_LOCAL prevents symbol clashes: every variant .so defines the same
  // mangled name aiter::mha_fwd — RTLD_GLOBAL would let later loads override it.
  void* handle = dlopen(so_path.c_str(), RTLD_LOCAL | RTLD_LAZY);
  if (!handle) {
    const char* err = dlerror();
    throw std::runtime_error(
        "AITER variant not found: " + so_path + "\n  dlerror: " +
        (err ? err : "unknown") +
        "\n  Hint: trigger AITER's lazy JIT build by importing aiter.ops.mha and "
        "calling mha_varlen_fwd with matching (dtype=" +
        std::string(key.dtype == VariantKey::Dtype::kFp16 ? "fp16" : "bf16") +
        ", causal=" + (key.causal ? "true" : "false") +
        ", has_lse=" + (key.has_lse ? "true" : "false") + ").");
  }

  dlerror();  // clear any pre-existing error before dlsym
  void* sym = dlsym(handle, kMhaFwdSymbol);
  if (!sym) {
    const char* err = dlerror();
    dlclose(handle);
    throw std::runtime_error("dlsym(" + std::string(kMhaFwdSymbol) + ") failed in " + so_path +
                             ": " + (err ? err : "unknown"));
  }

  s_cache.emplace(key, sym);
  return sym;
}

// ----- batch-prefill variant loader -----

namespace {

std::string batch_prefill_variant_so_name(BatchPrefillVariantKey const& key) {
  return build_so_name(key, "mha_batch_prefill_", "_ndropout_nqscale.so");
}

// Itanium-ABI mangled symbol for aiter::mha_batch_prefill(...).
// Pinned to amd-aiter 0.1.10. Regenerate with: nm -D <.so> | grep mha_batch_prefill
// Note: '23' encodes len("fmha_batch_prefill_args") == 23.
constexpr const char* kMhaBatchPrefillSymbol =
    "_ZN5aiter17mha_batch_prefillE23fmha_batch_prefill_argsRKN7ck_tile13stream_configE"
    "NSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEb9mask_enum9bias_enumb"
    "16quant_scale_enumb";

std::shared_mutex s_bp_mu;
std::unordered_map<BatchPrefillVariantKey, void*, BatchPrefillVariantKeyHash> s_bp_cache;

}  // namespace

void* get_aiter_mha_batch_prefill_handle(BatchPrefillVariantKey const& key) {
  {
    std::shared_lock rd(s_bp_mu);
    auto it = s_bp_cache.find(key);
    if (it != s_bp_cache.end()) return it->second;
  }

  std::unique_lock wr(s_bp_mu);
  auto it = s_bp_cache.find(key);
  if (it != s_bp_cache.end()) return it->second;

  std::string jit_dir = get_jit_dir();
  std::string so_path = jit_dir + "/" + batch_prefill_variant_so_name(key);

  void* handle = dlopen(so_path.c_str(), RTLD_LOCAL | RTLD_LAZY);
  if (!handle) {
    const char* err = dlerror();
    throw std::runtime_error(
        "AITER batch-prefill variant not found: " + so_path + "\n  dlerror: " +
        (err ? err : "unknown") +
        "\n  Hint: trigger AITER's lazy JIT build by calling "
        "aiter.ops.mha.mha_batch_prefill_func() once with matching (dtype=" +
        std::string(key.dtype == VariantKey::Dtype::kFp16 ? "fp16" : "bf16") +
        ", causal=" + (key.causal ? "true" : "false") +
        ", has_lse=" + (key.has_lse ? "true" : "false") + ") before this C++ path.");
  }

  dlerror();  // clear any pre-existing error before dlsym
  void* sym = dlsym(handle, kMhaBatchPrefillSymbol);
  if (!sym) {
    const char* err = dlerror();
    dlclose(handle);
    throw std::runtime_error("dlsym(" + std::string(kMhaBatchPrefillSymbol) + ") failed in " +
                             so_path + ": " + (err ? err : "unknown") +
                             "\n  This usually means the AITER ABI changed. "
                             "Run: nm -D " + so_path + " | grep mha_batch_prefill");
  }

  s_bp_cache.emplace(key, sym);
  return sym;
}

}  // namespace flashinfer::aiter
