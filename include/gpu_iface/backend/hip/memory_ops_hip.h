// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace flashinfer {
namespace gpu_iface {
namespace memory {
namespace detail {
namespace hip {

// ─── Pipeline fence primitives ───────────────────────────────────────────────

/**
 * @brief No-op on HIP: vmcnt is a continuous hardware counter, so no explicit
 *        group flush is required before s_waitcnt vmcnt(N).
 */
__device__ __forceinline__ void commit_group() {}

/**
 * @brief Stall until at most N outstanding async VMEM operations remain.
 *
 * With synchronous register-to-LDS stores, vmcnt is always 0 at the point
 * this is called, so the instruction would return immediately regardless.
 * The s_waitcnt line is disabled until async buffer_load_dword_lds is wired
 * into the KV tile load path.
 *
 * @tparam N Number of in-flight VMEM ops to allow through (0 = drain all).
 */
template <size_t N>
__device__ __forceinline__ void wait_group() {
  // TODO: Uncomment once async buffer_load_dword_lds is wired into produce_kv.
  // asm volatile("s_waitcnt vmcnt(%0)" : : "n"(N) : "memory");
}

// ─── Async GMEM → LDS primitives ─────────────────────────────────────────────
//
// GFX9 / CDNA3 buffer_load_dword_lds semantics:
//   LDS[ M0 + lane_k * size ]  ←  global[ rsrc.base + voffset[k] + soffset ]
// where M0 = readfirstlane(lds_ptr).
//
// IMPORTANT: lds_ptr must be uniform (the same value in every lane of the
// wavefront).  Use to_sgpr_u32() to enforce this before passing the pointer.

/// 4-SGPR buffer resource descriptor (V#/SRD) for MUBUF and buffer_load_lds.
/// The ext_vector_type attribute keeps the four words in consecutive SGPRs.
using srsrc_t = int __attribute__((ext_vector_type(4)));

/// LDS address-space pointer (AMDGPU address space 3 = local/shared memory).
using lds_ptr_t = uint32_t __attribute__((address_space(3)))*;

/// V# descriptor word 3 for raw MUBUF access on the GFX9 family (gfx942 included).
/// Matches CK_TILE_BUFFER_RESOURCE_3RD_DWORD.
static constexpr uint32_t kBufferResource3rdDword = 0x00020000u;

/// Wavefront width for GFX9 / CDNA3.
static constexpr uint32_t kWarpSize = 64u;

/**
 * @brief LLVM intrinsic that emits a buffer_load_dword_lds instruction.
 *
 * @param rsrc    Buffer resource descriptor (V#).
 * @param lds_ptr Uniform LDS destination pointer.
 * @param size    Element size in bytes (1, 2, or 4 on GFX9).
 * @param voffset Per-lane global byte offset.
 * @param soffset Scalar byte offset added to voffset.
 * @param offset  Immediate byte offset.
 * @param aux     Auxiliary flags (GLC/SLC/etc.).
 */
__device__ extern void _fi_async_load_to_lds(srsrc_t rsrc, lds_ptr_t lds_ptr, int size, int voffset,
                                             int soffset, int offset,
                                             int aux) __asm("llvm.amdgcn.raw.buffer.load.lds");

/**
 * @brief Promote a value to the SGPR register class.
 *
 * readfirstlane makes the value uniform across the wavefront; the asm volatile
 * constraint prevents LLVM from re-classifying it as a VGPR after the fact.
 * Required when passing an LDS pointer to async_load_dword_to_lds().
 *
 * @param x  Value to pin to an SGPR.
 * @return   The same value, guaranteed to reside in an SGPR.
 */
__device__ __forceinline__ uint32_t to_sgpr_u32(uint32_t x) {
  x = __builtin_amdgcn_readfirstlane(x);
  asm volatile("" : "+s"(x));
  return x;
}

/**
 * @brief Build a V# buffer resource descriptor for a contiguous device buffer.
 *
 * Typically computed once per warp and reused across inner-loop iterations.
 *
 * @param base      Base device pointer.
 * @param num_bytes Byte extent of the region in bytes.  Pass 0xFFFFFFFF to
 *                  disable hardware bounds checking when per-lane offsets are
 *                  already validated by the caller.
 * @return          4-SGPR buffer resource descriptor.
 */
__device__ __forceinline__ srsrc_t make_srsrc(const void* base, uint32_t num_bytes) {
  // V# layout: words 0-1 = 64-bit base address, word 2 = byte count, word 3 = format.
  struct __attribute__((packed)) BufRsrc {
    uint64_t ptr;
    uint32_t range;
    uint32_t config;
  };
  BufRsrc res{reinterpret_cast<uint64_t>(base), num_bytes, kBufferResource3rdDword};
  return __builtin_bit_cast(srsrc_t, res);
}

/**
 * @brief Issue one buffer_load_dword_lds for the whole wavefront (vmcnt += 1).
 *
 * Each lane loads one dword from global memory into its corresponding LDS slot:
 * @code
 *   LDS[ lds_base_uniform + lane_k * 4 ]  ←  global[ rsrc.base + voffset[k] ]
 * @endcode
 *
 * @param lds_base_uniform  Uniform LDS byte offset for the tile.  Must be
 *                          pinned to an SGPR with to_sgpr_u32() before this call.
 * @param rsrc              Buffer descriptor from make_srsrc().
 * @param voffset           Per-lane global byte offset into the buffer.
 */
__device__ __forceinline__ void async_load_dword_to_lds(uint32_t lds_base_uniform, srsrc_t rsrc,
                                                        uint32_t voffset) {
  _fi_async_load_to_lds(rsrc, (lds_ptr_t)(uintptr_t)lds_base_uniform, 4, static_cast<int>(voffset),
                        0, 0, 0);
}

/**
 * @brief Async-load a 64-thread (256-byte) tile from global memory to LDS.
 *
 * Covers one full wavefront of dwords in a single call.  For larger tiles,
 * advance lds_cur and global_base in steps of kWarpSize * sizeof(uint32_t).
 *
 * @param lds_cur      Uniform LDS byte offset (to_sgpr_u32()-pinned).
 * @param rsrc         Buffer descriptor from make_srsrc().
 * @param global_base  Uniform byte offset of the tile start within the buffer.
 */
__device__ __forceinline__ void async_load_tile64_to_lds(uint32_t lds_cur, srsrc_t rsrc,
                                                         uint32_t global_base) {
  uint32_t lane = threadIdx.x & 0x3fu;
  async_load_dword_to_lds(lds_cur, rsrc,
                          global_base + lane * static_cast<uint32_t>(sizeof(uint32_t)));
}

/**
 * @brief Deprecated overload with a divergent LDS pointer — do not use.
 *
 * The original signature accepted a per-lane lds_ptr_t.  This is incorrect:
 * M0 is derived from readfirstlane(lds_ptr), so all lanes write to the lane-0
 * offset regardless of their own pointer value.  The overload is preserved to
 * catch remaining call-sites at compile time via the [[deprecated]] attribute.
 *
 * @param lds_dst            Per-lane LDS destination (divergent; see note above).
 * @param rsrc               Buffer descriptor from make_srsrc().
 * @param global_byte_offset Per-lane byte offset from rsrc base.
 */
// TODO: Remove once all call-sites are migrated to async_load_tile64_to_lds.
[[deprecated("Use async_load_tile64_to_lds with a uniform LDS base instead.")]]
__device__ __forceinline__ void async_load_64b_to_lds(lds_ptr_t lds_dst, srsrc_t rsrc,
                                                      uint32_t global_byte_offset) {
  _fi_async_load_to_lds(rsrc, lds_dst, 4, static_cast<int>(global_byte_offset), 0, 0, 0);
  _fi_async_load_to_lds(rsrc, lds_dst + 1, 4, static_cast<int>(global_byte_offset + 4), 0, 0, 0);
}

// ─── Synchronous load functions ───────────────────────────────────────────────

/**
 * @brief Load 128 bits from global to shared memory.
 *
 * @tparam PrefetchOpt Prefetch hint (unused on HIP; kept for API parity with the CUDA path).
 * @tparam T           Element type.
 * @param smem_ptr     Shared memory destination.
 * @param gmem_ptr     Global memory source.
 */
template <PrefetchMode PrefetchOpt, typename T>
__device__ __forceinline__ void load_128b(T* smem_ptr, const T* gmem_ptr) {
  *reinterpret_cast<uint4*>(smem_ptr) = *reinterpret_cast<const uint4*>(gmem_ptr);
}

/**
 * @brief Load 64 bits from global to shared memory.
 *
 * @tparam PrefetchOpt Prefetch hint (unused on HIP).
 * @tparam T           Element type.
 * @param smem_ptr     Shared memory destination.
 * @param gmem_ptr     Global memory source.
 */
template <PrefetchMode PrefetchOpt, typename T>
__device__ __forceinline__ void load_64b(T* smem_ptr, const T* gmem_ptr) {
  *reinterpret_cast<uint2*>(smem_ptr) = *reinterpret_cast<const uint2*>(gmem_ptr);
}

/**
 * @brief Predicated 128-bit load from global to shared memory.
 *
 * When predicate is false and FillOpt is kFillZero, the destination is zeroed;
 * with kNoFill it is left unchanged.
 *
 * @tparam PrefetchOpt Prefetch hint (unused on HIP).
 * @tparam FillOpt     Fill mode applied when predicate is false.
 * @tparam T           Element type.
 * @param smem_ptr     Shared memory destination.
 * @param gmem_ptr     Global memory source.
 * @param predicate    When false, the global load is skipped.
 */
template <PrefetchMode PrefetchOpt, SharedMemFillMode FillOpt, typename T>
__device__ __forceinline__ void pred_load_128b(T* smem_ptr, const T* gmem_ptr, bool predicate) {
  if (predicate) {
    *reinterpret_cast<uint4*>(smem_ptr) = *reinterpret_cast<const uint4*>(gmem_ptr);
  } else {
    if constexpr (FillOpt == SharedMemFillMode::kFillZero) {
      *reinterpret_cast<uint4*>(smem_ptr) = make_uint4(0, 0, 0, 0);
    }
  }
}

/**
 * @brief Predicated 64-bit load from global to shared memory.
 *
 * @tparam PrefetchOpt Prefetch hint (unused on HIP).
 * @tparam FillOpt     Fill mode applied when predicate is false.
 * @tparam T           Element type.
 * @param smem_ptr     Shared memory destination.
 * @param gmem_ptr     Global memory source.
 * @param predicate    When false, the global load is skipped.
 */
template <PrefetchMode PrefetchOpt, SharedMemFillMode FillOpt, typename T>
__device__ __forceinline__ void pred_load_64b(T* smem_ptr, const T* gmem_ptr, bool predicate) {
  if (predicate) {
    *reinterpret_cast<uint2*>(smem_ptr) = *reinterpret_cast<const uint2*>(gmem_ptr);
  } else {
    if constexpr (FillOpt == SharedMemFillMode::kFillZero) {
      *reinterpret_cast<uint2*>(smem_ptr) = make_uint2(0, 0);
    }
  }
}

/**
 * @brief Load NumBits bits from global to shared memory.
 *
 * @tparam NumBits     Transfer width in bits (128 or 256).
 * @tparam PrefetchOpt Prefetch hint (unused on HIP).
 * @tparam T           Element type.
 * @param smem_ptr     Shared memory destination.
 * @param gmem_ptr     Global memory source.
 */
template <size_t NumBits, PrefetchMode PrefetchOpt, typename T>
__device__ __forceinline__ void load(T* smem_ptr, const T* gmem_ptr) {
  static_assert(NumBits == 128 || NumBits == 256, "NumBits must be 128 or 256");
  if constexpr (NumBits == 128) {
    load_128b<PrefetchOpt>(smem_ptr, gmem_ptr);
  } else {
    load_128b<PrefetchOpt>(smem_ptr, gmem_ptr);
    load_128b<PrefetchOpt>(smem_ptr + 16 / sizeof(T), gmem_ptr + 16 / sizeof(T));
  }
}

/**
 * @brief Predicated load of NumBits bits from global to shared memory.
 *
 * @tparam NumBits     Transfer width in bits (64, 128, or 256).
 * @tparam PrefetchOpt Prefetch hint (unused on HIP).
 * @tparam FillOpt     Fill mode applied when predicate is false.
 * @tparam T           Element type.
 * @param smem_ptr     Shared memory destination.
 * @param gmem_ptr     Global memory source.
 * @param predicate    When false, the global load is skipped.
 */
template <size_t NumBits, PrefetchMode PrefetchOpt, SharedMemFillMode FillOpt, typename T>
__device__ __forceinline__ void pred_load(T* smem_ptr, const T* gmem_ptr, bool predicate) {
  static_assert(NumBits == 64 || NumBits == 128 || NumBits == 256,
                "NumBits must be 64, 128 or 256");
  if constexpr (NumBits == 64) {
    pred_load_64b<PrefetchOpt, FillOpt>(smem_ptr, gmem_ptr, predicate);
  } else if constexpr (NumBits == 128) {
    pred_load_128b<PrefetchOpt, FillOpt>(smem_ptr, gmem_ptr, predicate);
  } else {
    pred_load_128b<PrefetchOpt, FillOpt>(smem_ptr, gmem_ptr, predicate);
    pred_load_128b<PrefetchOpt, FillOpt>(smem_ptr + 16 / sizeof(T), gmem_ptr + 16 / sizeof(T),
                                         predicate);
  }
}

}  // namespace hip
}  // namespace detail
}  // namespace memory
}  // namespace gpu_iface
}  // namespace flashinfer
