// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache - 2.0

#pragma once

#include "gpu_iface/fragment.hpp"
#include "gpu_iface/mma_types.hpp"

// Include platform-specific implementations
#if defined(PLATFORM_CUDA_DEVICE)
#include "backend/cuda/mma.cuh"
namespace detail = flashinfer::gpu_iface::mma_impl::cuda;
#elif defined(PLATFORM_HIP_DEVICE)
#include "backend/hip/mma_hip.h"
namespace detail = flashinfer::gpu_iface::mma_impl::hip;
#endif

namespace flashinfer
{
namespace gpu_iface
{
namespace mma
{

/*!
 * \brief Loads data from shared memory to fragment
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename FragmentType>
__device__ __forceinline__ void
load_fragment_m16n16(FragmentType &frag,
                     const typename FragmentType::value_type *ptr,
                     uint32_t stride)
{
    detail::load_fragment_m16n16(frag, ptr, stride);
}

/*!
 * \brief Stores data from fragment to shared memory
 * \tparam T data type of the fragment
 * \param R pointer to the fragment
 * \param smem_ptr pointer to the shared memory
 */
template <typename FragmentType>
__device__ __forceinline__ void
store_fragment_m16n16(typename FragmentType::value_type *ptr,
                      const FragmentType &frag,
                      uint32_t stride)
{
    detail::store_fragment_m16n16(ptr, frag, stride);
}

/*!
 * \brief Wrapper of two mma m16n16k16 instructions for row major and column
 * major f16 matrix multiplication, accumulated in f32.
 * \tparam T data type of the fragment
 * \tparam mma_mode whether we are initializing the accumulator or updating it
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <typename T, MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void
mma_sync_m16n16k16_row_col_f16f16f32(accumulator_fragment_m16n16k16<float> &d,
                                     const row_major_fragment_m16n16k16<T> &a,
                                     const col_major_fragment_m16n16k16<T> &b)
{
    detail::mma_sync_m16n16k16_row_col_f16f16f32<T, mma_mode>(d, a, b);
}

/*!
 * \brief Use mma instructions to compute rowsum.
 */
template <typename DType>
__device__ __forceinline__ void
m16k16_rowsum_f16f16f32(accumulator_fragment_m16n16k16<float> &d,
                        const row_major_fragment_m16n16k16<DType> &s)
{
    detail::m16k16_rowsum_f16f16f32(d, s);
}

/*!
 * \brief Wrapper of two mma m16n16k16 instructions for row major and column
 * major f16 matrix multiplication, accumulated in f16.
 * \tparam mma_mode whether we are initializing the accumulator or updating it
 * \param C pointer to the accumulator
 * \param A pointer to the fragment of matrix A
 * \param B pointer to the fragment of matrix B
 */
template <MMAMode mma_mode = MMAMode::kInplaceUpdate>
__device__ __forceinline__ void mma_sync_m16n16k16_row_col_f16f16f16(
    accumulator_fragment_m16n16k16<__half> &d,
    const row_major_fragment_m16n16k16<__half> &a,
    const col_major_fragment_m16n16k16<__half> &b)
{
    detail::mma_sync_m16n16k16_row_col_f16f16f16<mma_mode>(d, a, b);
}

} // namespace mma
} // namespace gpu_iface
} // namespace flashinfer
