// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "platform.hpp"

namespace flashinfer {
namespace gpu_iface {
namespace memory {

/**
 * @brief Control options for shared memory fill behavior
 */
enum class SharedMemFillMode {
  kFillZero,  // Fill zero to shared memory when predicate is false
  kNoFill     // Do not fill zero to shared memory when predicate is false
};

/**
 * @brief Control options for memory prefetch behavior
 */
enum class PrefetchMode {
  kNoPrefetch,  // Do not fetch additional data from global memory to L2
  kPrefetch     // Fetch additional data from global memory to L2
};

// Include platform-specific implementations
#if defined(PLATFORM_CUDA_DEVICE)
#include "backend/cuda/memory_ops.cuh"
namespace mem_detail = flashinfer::gpu_iface::memory::detail::cuda;
#elif defined(PLATFORM_HIP_DEVICE)
#include "backend/hip/memory_ops_hip.h"
namespace mem_detail = flashinfer::gpu_iface::memory::detail::hip;
#endif

/**
 * @brief Commits pending asynchronous memory operations to a group
 */
__device__ __forceinline__ void commit_group() { mem_detail::commit_group(); }

/**
 * @brief Waits until N most recent groups of async operations are complete
 *
 * @tparam N Number of most recent groups to wait for (0-7)
 */
template <size_t N>
__device__ __forceinline__ void wait_group() {
  mem_detail::wait_group<N>();
}

/**
 * @brief Asynchronously loads 128 bits from global to shared memory
 *
 * @tparam PrefetchOpt Prefetch option
 * @tparam T Data type
 * @param smem_ptr Destination shared memory pointer
 * @param gmem_ptr Source global memory pointer
 */
template <PrefetchMode PrefetchOpt, typename T>
__device__ __forceinline__ void load_128b(T* smem_ptr, const T* gmem_ptr) {
  mem_detail::load_128b<PrefetchOpt>(smem_ptr, gmem_ptr);
}

template <PrefetchMode PrefetchOpt, typename T>
__device__ __forceinline__ void load_64b(T* smem_ptr, const T* gmem_ptr) {
#if defined(PLATFORM_HIP_DEVICE)
  mem_detail::load_64b<PrefetchOpt>(smem_ptr, gmem_ptr);
#else
#error "load_64b not implemented for this platform"
#endif
}

/**
 * @brief Conditionally loads 128 bits from global to shared memory
 *
 * @tparam PrefetchOpt Prefetch option
 * @tparam FillOpt Memory fill option
 * @tparam T Data type
 * @param smem_ptr Destination shared memory pointer
 * @param gmem_ptr Source global memory pointer
 * @param predicate Condition for executing the load
 */
template <PrefetchMode PrefetchOpt, SharedMemFillMode FillOpt, typename T>
__device__ __forceinline__ void pred_load_128b(T* smem_ptr, const T* gmem_ptr, bool predicate) {
  mem_detail::pred_load_128b<PrefetchOpt, FillOpt>(smem_ptr, gmem_ptr, predicate);
}

template <PrefetchMode PrefetchOpt, SharedMemFillMode FillOpt, typename T>
__device__ __forceinline__ void pred_load_64b(T* smem_ptr, const T* gmem_ptr, bool predicate) {
#if defined(PLATFORM_HIP_DEVICE)
  mem_detail::pred_load_64b<PrefetchOpt, FillOpt>(smem_ptr, gmem_ptr, predicate);
#else
#error "pred_load_64b not implemented for this platform"
#endif
}

/**
 * @brief Loads N bits (128 or 256) from global to shared memory
 *
 * @tparam NumBits Number of bits to load (128 or 256)
 * @tparam PrefetchOpt Prefetch option
 * @tparam T Data type
 * @param smem_ptr Destination shared memory pointer
 * @param gmem_ptr Source global memory pointer
 */
template <size_t NumBits, PrefetchMode PrefetchOpt, typename T>
__device__ __forceinline__ void load(T* smem_ptr, const T* gmem_ptr) {
  mem_detail::load<NumBits, PrefetchOpt>(smem_ptr, gmem_ptr);
}

/**
 * @brief Conditionally loads N bits from global to shared memory
 *
 * @tparam NumBits Number of bits to load (128 or 256)
 * @tparam PrefetchOpt Prefetch option
 * @tparam FillOpt Memory fill option
 * @tparam T Data type
 * @param smem_ptr Destination shared memory pointer
 * @param gmem_ptr Source global memory pointer
 * @param predicate Condition for executing the load
 */
template <size_t NumBits, PrefetchMode PrefetchOpt, SharedMemFillMode FillOpt, typename T>
__device__ __forceinline__ void pred_load(T* smem_ptr, const T* gmem_ptr, bool predicate) {
  mem_detail::pred_load<NumBits, PrefetchOpt, FillOpt>(smem_ptr, gmem_ptr, predicate);
}

#if defined(PLATFORM_HIP_DEVICE)
// === HIP-only async GMEM → LDS primitives

/**
 * @brief Build a buffer resource descriptor (V#) for async GMEM→LDS copies.
 *
 * @param base      Tensor base pointer (K or V head pointer)
 * @param num_bytes Byte size of the region (use 0xFFFFFFFF to skip bounds check)
 * @return srsrc_t  4-SGPR buffer resource descriptor
 */
__device__ __forceinline__ mem_detail::srsrc_t make_srsrc(const void* base, uint32_t num_bytes) {
  return mem_detail::make_srsrc(base, num_bytes);
}

/**
 * @brief Async load 64 bits from global buffer to LDS.
 *
 * Issues two buffer_load_dword lds instructions (vmcnt += 2).
 * The wavefront does not stall; the caller must later call
 * wait_group<N>() + __syncthreads() before reading lds_dst.
 *
 * @param lds_dst           LDS destination (2 consecutive uint32 slots)
 * @param rsrc              Buffer resource from make_srsrc()
 * @param global_byte_offset Per-thread byte offset from rsrc base
 */
__device__ __forceinline__ void async_load_64b_to_lds(mem_detail::lds_ptr_t lds_dst,
                                                      mem_detail::srsrc_t rsrc,
                                                      uint32_t global_byte_offset) {
  mem_detail::async_load_64b_to_lds(lds_dst, rsrc, global_byte_offset);
}
#endif  // PLATFORM_HIP_DEVICE

}  // namespace memory
}  // namespace gpu_iface
}  // namespace flashinfer
