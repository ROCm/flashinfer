// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gpu_iface/fragment.hpp"
#include "gpu_iface/mma_types.hpp"
#include "gpu_iface/platform.hpp"

namespace flashinfer
{
namespace gpu_iface
{
namespace mma_impl
{
namespace hip
{

using flashinfer::gpu_iface::mma::accumulator_fragment_m16n16k16;
using flashinfer::gpu_iface::mma::col_major_fragment_m16n16k16;
using flashinfer::gpu_iface::mma::MMAMode;
using flashinfer::gpu_iface::mma::row_major_fragment_m16n16k16;

// Architecture detection for MI300
#if defined(__gfx942__)
#define FLASHINFER_MMA_F16F16F32_M16N16K16_ENABLED
#define FLASHINFER_MMA_BF16BF16F32_M16N16K16_ENABLED
#define FLASHINFER_MMA_F16F16F16_M16N16K16_ENABLED
#define FLASHINFER_LDMATRIX_M8N8X4_ENABLED
#define FLASHINFER_STMATRIX_M8N8X4_ENABLED
#endif

#define FLASHINFER_RUNTIME_ASSERT(x) assert(0 && x)
// Single unified load function for all fragment types
template <typename FragmentType>
__device__ __forceinline__ void
load_fragment_m16n16(FragmentType &frag,
                     const typename FragmentType::value_type *ptr,
                     uint32_t stride)
{
#ifdef FLASHINFER_LDMATRIX_M8N8X4_ENABLED
    if constexpr (std::is_same_v<FragmentType,
                                 accumulator_fragment_m16n16k16<
                                     typename FragmentType::value_type>>)
    {
        // Accumulator fragments need the layout parameter
        rocwmma::load_matrix_sync(frag.frag, ptr, stride,
                                  rocwmma::mem_row_major);
    }
    else {
        // Row-major and col-major fragments already have layout baked in
        rocwmma::load_matrix_sync(frag.frag, ptr, stride);
    }
#else
    FLASHINFER_RUNTIME_ASSERT("ldmatrix emulation not supported");
#endif
}

// Single unified store function for all fragment types
template <typename FragmentType>
__device__ __forceinline__ void
store_fragment_m16n16(typename FragmentType::value_type *ptr,
                      const FragmentType &frag,
                      uint32_t stride)
{
#ifdef FLASHINFER_STMATRIX_M8N8X4_ENABLED
    if constexpr (std::is_same_v<FragmentType,
                                 accumulator_fragment_m16n16k16<
                                     typename FragmentType::value_type>>)
    {
        // Accumulator fragments need the layout parameter
        rocwmma::store_matrix_sync(ptr, frag.frag, stride,
                                   rocwmma::mem_row_major);
    }
    else {
        // Row-major and col-major fragments already have layout baked in
        rocwmma::store_matrix_sync(ptr, frag.frag, stride);
    }
#else
    FLASHINFER_RUNTIME_ASSERT("stmatrix emulation not supported");
#endif
}

// MMA operation for FP16 inputs with FP32 accumulator
template <typename T, MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f32(
    accumulator_fragment_m16n16k16<float> &c_frag,
    const row_major_fragment_m16n16k16<T> &a_frag,
    const col_major_fragment_m16n16k16<T> &b_frag)
{
#if defined(FLASHINFER_MMA_F16F16F32_M16N16K16_ENABLED)
    // Ensure T is either __half or __hip_bfloat16
    static_assert(std::is_same_v<T, __half> ||
                      std::is_same_v<T, __hip_bfloat16>,
                  "T must be __half or __hip_bfloat16");

    // Initialize C if requested
    if constexpr (mma_mode == MMAMode::kInit) {
        rocwmma::fill_fragment(c_frag.frag, 0.0f);
    }

    // Perform MMA operation directly with fragments
    rocwmma::mma_sync(c_frag.frag, a_frag.frag, b_frag.frag, c_frag.frag);
#else
    FLASHINFER_RUNTIME_ASSERT(
        "MMA f16f16f32 not supported on this architecture");
#endif
}

// MMA operation for FP16 inputs with FP16 accumulator
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f16(
    accumulator_fragment_m16n16k16<__half> &c_frag,
    const row_major_fragment_m16n16k16<__half> &a_frag,
    const col_major_fragment_m16n16k16<__half> &b_frag)
{
#if defined(FLASHINFER_MMA_F16F16F16_M16N16K16_ENABLED)
    // Initialize C if requested
    if constexpr (mma_mode == MMAMode::kInit) {
        rocwmma::fill_fragment(c_frag.frag, __float2half(0.0f));
    }

    // Perform MMA
    rocwmma::mma_sync(c_frag.frag, a_frag.frag, b_frag.frag, c_frag.frag);
#else
    FLASHINFER_RUNTIME_ASSERT(
        "MMA f16f16f16 not supported on this architecture");
#endif
}

// Rowsum operation using MMA
template <typename DType>
__device__ __forceinline__ void
m16k16_rowsum_f16f16f32(accumulator_fragment_m16n16k16<float> &d_frag,
                        const row_major_fragment_m16n16k16<DType> &s_frag)
{
    static_assert(sizeof(DType) == 2, "DType must be 16bit");

    // Create a ones fragment
    col_major_fragment_m16n16k16<DType> ones_frag;

    // Fill with ones
    if constexpr (std::is_same_v<DType, __half>) {
        ones_frag.fill(__float2half(1.0f));
    }
    else if constexpr (std::is_same_v<DType, hip_bfloat16>) {
        ones_frag.fill(__float2bfloat16(1.0f));
    }

    // Use MMA to compute rowsum
    mma_sync_m16n16k16_row_col_f16f16f32<DType, MMAMode::kInplaceUpdate>(
        d_frag, s_frag, ones_frag);
}

// FP8 operations - not implemented for MI300 yet
template <typename T>
__device__ __forceinline__ void mma_sync_m16n16k32_row_col_f8f8f32(
    accumulator_fragment_m16n16k16<float> &c_frag,
    const row_major_fragment_m16n16k16<T> &a_frag,
    const col_major_fragment_m16n16k16<T> &b_frag)
{
    FLASHINFER_RUNTIME_ASSERT("FP8 MMA not implemented for AMD");
}

template <typename DType>
__device__ __forceinline__ void
m16k32_rowsum_f8f8f32(accumulator_fragment_m16n16k16<float> &d_frag,
                      const row_major_fragment_m16n16k16<DType> &s_frag)
{
    FLASHINFER_RUNTIME_ASSERT("FP8 rowsum not implemented for AMD");
}

} // namespace hip
} // namespace mma_impl
} // namespace gpu_iface
} // namespace flashinfer
