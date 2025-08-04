// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gpu_iface/mma_types.hpp"
#include "gpu_iface/platform.hpp"

#ifdef PLATFORM_HIP_DEVICE
#include <rocwmma/rocwmma.hpp>
#endif

namespace flashinfer
{
namespace gpu_iface
{
namespace mma
{

enum class FragmentType
{
    row_major,  // Row-major matrix layout
    col_major,  // Column-major matrix layout
    accumulator // Accumulator (no layout)
};

template <typename T, int M, int N, int K, FragmentType frag_type>
struct fragment_t
{
    using value_type = T;
#ifdef PLATFORM_CUDA_DEVICE
    // flashinfer's generic CUDA implementation uses raw arrays for matrix
    // fragments and the interface is designed to accomodate use of raw arrays
    // for such use cases.
    static constexpr int elements_per_thread =
        (frag_type == FragmentType::accumulator) ? 8
        : (sizeof(T) == 1)                       ? 8
                                                 : 4;

    // Number of 32-bit registers needed
    static constexpr int num_regs = (elements_per_thread * sizeof(T) + 3) / 4;

    uint32_t data[num_regs];

    // Provide array-like access
    __device__ __forceinline__ T &operator[](int i)
    {
        return reinterpret_cast<T *>(data)[i];
    }
    __device__ __forceinline__ const T &operator[](int i) const
    {
        return reinterpret_cast<const T *>(data)[i];
    }

    // Get number of elements this thread holds
    __device__ __forceinline__ constexpr int size() const
    {
        return elements_per_thread;
    }

    // Get raw pointer for MMA operations
    __device__ __forceinline__ uint32_t *raw_ptr() { return data; }
    __device__ __forceinline__ const uint32_t *raw_ptr() const { return data; }

#elif defined(PLATFORM_HIP_DEVICE)
    // AMD: Use rocWMMA fragments
    using rocwmma_layout = typename std::conditional<
        frag_type == FragmentType::row_major,
        rocwmma::row_major,
        typename std::conditional<frag_type == FragmentType::col_major,
                                  rocwmma::col_major,
                                  void>::type>::type;

    using rocwmma_matrix_t = typename std::conditional<
        frag_type == FragmentType::row_major,
        rocwmma::matrix_a,
        typename std::conditional<frag_type == FragmentType::col_major,
                                  rocwmma::matrix_b,
                                  rocwmma::accumulator>::type>::type;

    // Select appropriate fragment type based on whether it's accumulator or not
    using rocwmma_frag_t = typename std::conditional<
        frag_type == FragmentType::accumulator,
        rocwmma::fragment<rocwmma_matrix_t, M, N, K, T>,
        rocwmma::fragment<rocwmma_matrix_t, M, N, K, T, rocwmma_layout>>::type;

    rocwmma_frag_t frag;

    // Provide array-like access that maps to rocWMMA fragment
    __device__ __forceinline__ T operator[](int i) const { return frag.x[i]; }

    // For non-const access, we need to provide a setter since we can't return a
    // reference
    __device__ __forceinline__ void set(int i, T value) { frag.x[i] = value; }

    // Get number of elements this thread holds
    __device__ __forceinline__ int size() const { return frag.num_elements; }

    // Get raw pointer for operations that need it
    __device__ __forceinline__ rocwmma_frag_t *raw_ptr() { return &frag; }
    __device__ __forceinline__ const rocwmma_frag_t *raw_ptr() const
    {
        return &frag;
    }
#endif

    // Common interface - update fill method to use setter for HIP
    __device__ __forceinline__ void fill(T value)
    {
#ifdef PLATFORM_CUDA_DEVICE
#pragma unroll
        for (int i = 0; i < elements_per_thread; ++i) {
            (*this)[i] = value;
        }
#elif defined(PLATFORM_HIP_DEVICE)
        rocwmma::fill_fragment(frag, value);
#endif
    }
};

// Convenience typedefs for common fragment types
template <typename T>
using row_major_fragment_m16n16k16 =
    fragment_t<T, 16, 16, 16, FragmentType::row_major>;

template <typename T>
using col_major_fragment_m16n16k16 =
    fragment_t<T, 16, 16, 16, FragmentType::col_major>;

template <typename T>
using accumulator_fragment_m16n16k16 =
    fragment_t<T, 16, 16, 16, FragmentType::accumulator>;

// Helper to get compile-time fragment size
template <typename Fragment> struct fragment_traits
{
#ifdef PLATFORM_CUDA_DEVICE
    static constexpr int size = Fragment::elements_per_thread;
#elif defined(PLATFORM_HIP_DEVICE)
    // For HIP, we can't make this constexpr, so provide a device function
    __device__ static int get_size(const Fragment &f) { return f.size(); }
#endif
};

} // namespace mma
} // namespace gpu_iface
} // namespace flashinfer
