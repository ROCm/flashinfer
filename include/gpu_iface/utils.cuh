// SPDX-FileCopyrightText: 2023-2025 FlashInfer team.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <vector>

#include "gpu_runtime_compat.hpp"

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// macro to turn off fp16 qk reduction to reduce binary
#ifndef FLASHINFER_ALWAYS_DISUSE_FP16_QK_REDUCTION
#define FLASHINFER_ALWAYS_DISUSE_FP16_QK_REDUCTION 0
#endif

namespace flashinfer {

template <typename T1, typename T2>
__forceinline__ __device__ __host__ T1 ceil_div(const T1 x, const T2 y) {
  return (x + y - 1) / y;
}

template <typename T1, typename T2>
__forceinline__ __device__ __host__ T1 round_up(const T1 x, const T2 y) {
  return ceil_div(x, y) * y;
}

#if defined(PLATFORM_CUDA_DEVICE)
inline std::pair<int, int> GetCudaComputeCapability() {
  int device_id = 0;
  cudaGetDevice(&device_id);
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);
  return std::make_pair(major, minor);
}
#elif defined(PLATFORM_HIP_DEVICE)
inline std::pair<int, int> GetCudaComputeCapability() {
  int device_id = 0;
  FI_GPU_CALL(hipGetDevice(&device_id));
  int major = 0, minor = 0;
  FI_GPU_CALL(hipDeviceGetAttribute(&major, hipDeviceAttributeComputeCapabilityMajor, device_id));
  FI_GPU_CALL(hipDeviceGetAttribute(&minor, hipDeviceAttributeComputeCapabilityMinor, device_id));
  return std::make_pair(major, minor);
}
#endif

template <typename T>
inline void DebugPrintCUDAArray(T* device_ptr, size_t size, std::string prefix = "") {
  std::vector<T> host_array(size);
  std::cout << prefix;
  gpuMemcpy(host_array.data(), device_ptr, size * sizeof(T), gpuMemcpyDeviceToHost);
  for (size_t i = 0; i < size; ++i) {
    std::cout << host_array[i] << " ";
  }
  std::cout << std::endl;
}

inline uint32_t FA2DetermineCtaTileQ(int64_t avg_packed_qo_len, uint32_t head_dim) {
#if defined(PLATFORM_CUDA_DEVICE)
  if (avg_packed_qo_len > 64 && head_dim < 256) {
    return 128;
  } else {
    auto compute_capacity = GetCudaComputeCapability();
    if (compute_capacity.first >= 8) {
      // Ampere or newer
      if (avg_packed_qo_len > 16) {
        // avg_packed_qo_len <= 64
        return 64;
      } else {
        // avg_packed_qo_len <= 16
        return 16;
      }
    } else {
      // NOTE(Zihao): not enough shared memory on Turing for 1x4 warp
      // layout
      return 64;
    }
  }
#elif defined(PLATFORM_HIP_DEVICE)
  // CDNA3 (MI300X) occupancy-aware tile selection.
  //
  // LDS per CU on gfx942 is 64 KB. SharedStorageQKVO with CTA_TILE_Q=128 and
  // head_dim=128 occupies 48 KB, fitting only 1 block/CU → 4 wavefronts/CU
  // (12.5% of the 32-wavefront HW maximum), leaving MFMA units idle >93% of
  // the time.
  //
  // CTA_TILE_Q=64 with head_dim=128 → 32 KB smem → 2 blocks/CU → 8 wavefronts,
  // doubling latency-hiding capacity and MFMA utilization.
  //
  // For head_dim >= 256, CTA_TILE_Q=16 is non-viable on CDNA3: get_num_warps_q(16)=1
  // forces NUM_WARPS_KV=4, so even the minimum NUM_MMA_KV=1 configuration produces
  // smem = Q(8 KB) + K(32 KB) + V(32 KB) = 72 KB, exceeding the 64 KB LDS ceiling.
  // 2 blocks/CU is also impossible since the Q tile alone occupies 32 KB = half of LDS,
  // so CTA_TILE_Q=64 is the correct choice for all sequence lengths at head_dim >= 256.
  if (head_dim >= 256) return 64;
  // CTA_TILE_Q=16 is retained for very short sequences (avg ≤ 16 rows) with head_dim < 256
  // to avoid launching an excessive number of near-empty threadblocks.
  return avg_packed_qo_len <= 16 ? 16 : 64;
#endif
}

/*!
 * \brief Return x - y if x > y, otherwise return 0.
 */
__device__ __forceinline__ uint32_t sub_if_greater_or_zero(uint32_t x, uint32_t y) {
  return (x > y) ? x - y : 0U;
}

__device__ __forceinline__ void swap(uint32_t& a, uint32_t& b) {
  uint32_t tmp = a;
  a = b;
  b = tmp;
}

__device__ __forceinline__ uint32_t dim2_offset(const uint32_t& dim_a, const uint32_t& idx_b,
                                                const uint32_t& idx_a) {
  return idx_b * dim_a + idx_a;
}

__device__ __forceinline__ uint32_t dim3_offset(const uint32_t& dim_b, const uint32_t& dim_a,
                                                const uint32_t& idx_c, const uint32_t& idx_b,
                                                const uint32_t& idx_a) {
  return (idx_c * dim_b + idx_b) * dim_a + idx_a;
}

__device__ __forceinline__ uint32_t dim4_offset(const uint32_t& dim_c, const uint32_t& dim_b,
                                                const uint32_t& dim_a, const uint32_t& idx_d,
                                                const uint32_t& idx_c, const uint32_t& idx_b,
                                                const uint32_t& idx_a) {
  return ((idx_d * dim_c + idx_c) * dim_b + idx_b) * dim_a + idx_a;
}

#define DEFINE_HAS_MEMBER(member)                                                              \
  template <typename T, typename = void>                                                       \
  struct has_##member : std::false_type {};                                                    \
  template <typename T>                                                                        \
  struct has_##member<T, std::void_t<decltype(std::declval<T>().member)>> : std::true_type {}; \
  template <typename T>                                                                        \
  inline constexpr bool has_##member##_v = has_##member<T>::value;

}  // namespace flashinfer
