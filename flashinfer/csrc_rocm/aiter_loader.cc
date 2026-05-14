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

// Convention: mha_varlen_fwd_{dtype}_{logits}_{bias}_{mask}_{lse}_ndropout_nskip_nqscale.so
std::string variant_so_name(VariantKey const& key) {
  std::string name = "mha_varlen_fwd_";
  name += (key.dtype == VariantKey::Dtype::kFp16) ? "fp16" : "bf16";
  name += key.has_logits_cap ? "_logits" : "_nlogits";
  name += key.has_alibi ? "_alibi" : "_nbias";
  name += key.causal ? "_mask" : "_nmask";
  name += key.has_lse ? "_lse" : "_nlse";
  name += "_ndropout_nskip_nqscale.so";
  return name;
}

// Mangled symbol for aiter::mha_fwd(aiter::mha_fwd_args, ck_tile::stream_config const&).
// Stable across GCC/Clang Itanium ABI; verified by `nm -D` on all shipped variants.
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

}  // namespace flashinfer::aiter
