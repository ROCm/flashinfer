// SPDX-FileCopyrightText: 2023-2025 FlashInfer team.
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#ifndef FLASHINFER_PERMUTED_SMEM_CUH_
#define FLASHINFER_PERMUTED_SMEM_CUH_

#include "gpu_iface/memory_ops.hpp"
#include "gpu_iface/mma_ops.hpp"
#include "gpu_iface/platform.hpp"

namespace gpu_mem = flashinfer::gpu_iface::memory;

namespace flashinfer {

enum class SwizzleMode {
  k64B,
  k128B,
  k128B_16Row,  // Period-16 XOR swizzle for CDNA3 MI300x (64-thread wavefront, 16-thread LDS
                // phases)
  kLinear,
};

// Use 128bit as the granularity to fetch/store data per thread to maximize
// memory bandwidth
using b128_t = uint4;
// 64b type to support 16-bit CDNA3 WMMA ops where each thread in a 64 thread
// wavefront loads a four element fragment.
using b64_t = uint2;

/*!
 * \brief Compute the number of elements that can be stored in a b128_t.
 * \tparam T The data type of the elements.
 * \tparam VectorWidthBits The width in bits for vector operations (64 or 128).
 */
template <typename T, size_t VectorWidthBits>
constexpr __host__ __device__ __forceinline__ uint32_t upcast_size() {
  static_assert(VectorWidthBits == 128 || VectorWidthBits == 64,
                "Only 64 and 128 bits are supported");
  if constexpr (VectorWidthBits == 128) {
    return sizeof(b128_t) / sizeof(T);
  } else if constexpr (VectorWidthBits == 64) {
    return sizeof(b64_t) / sizeof(T);
  }
}

/*!
 * \brief Pure arithmetic layout policy for XOR-swizzled shared memory tiles.
 *
 * Contains no pointer and no memory — only the coordinate arithmetic that maps
 * logical (row, col) to physical cell index and vice-versa.  Its methods are
 * static device functions implementing pure arithmetic that the compiler can
 * typically eliminate entirely at compile time.
 *
 * This type is passed as a template parameter to smem_t (composition), giving
 * the caller an explicit bijection handle: any code that derives global-memory
 * coordinates from smem_t::Layout is structurally guaranteed to use the same
 * swizzle pattern as the LDS read/write path.
 */
template <SwizzleMode swizzle_mode>
struct SwizzleLayout {
  // ── Primitive ──────────────────────────────────────────────────────────────
  // XOR mask applied to column bits for a given row.
  // XOR is self-inverse, so the same mask is used for both the forward
  // (LDS write) and inverse (global read) directions.
  template <uint32_t stride>
  static __device__ __forceinline__ uint32_t col_swizzle_xor(uint32_t row) {
    if constexpr (swizzle_mode == SwizzleMode::k128B_16Row) {
      constexpr uint32_t period = (stride >= 16u) ? 16u : 8u;
      return row & (period - 1u);
    } else if constexpr (swizzle_mode == SwizzleMode::k128B) {
      return row & 7u;
    } else if constexpr (swizzle_mode == SwizzleMode::k64B) {
      return (row >> 1u) & 3u;
    } else {
      return 0u;  // kLinear
    }
  }

  // ── Derived from the primitive ─────────────────────────────────────────────

  /*!
   * \brief Compute the element offset given coordinates in a permuted shared memory.
   * \tparam stride The stride (in terms of BasePtrTy elements) in the permuted shared memory.
   * \param i The row index.
   * \param j The column index.
   */
  template <uint32_t stride>
  static __device__ __forceinline__ uint32_t get_permuted_offset(uint32_t i, uint32_t j) {
    if constexpr (swizzle_mode == SwizzleMode::k64B) {
      static_assert(stride == 4, "k64B swizzle requires stride == 4");
    }
    return i * stride + (j ^ col_swizzle_xor<stride>(i));
  }

  /*!
   * \brief Inverse of get_permuted_offset: recover (row, col) from a physical LDS cell index.
   *
   * XOR is self-inverse ((x ^ mask) ^ mask == x), so the same col_swizzle_xor
   * expression serves both the forward and inverse direction in all supported
   * XOR-based swizzle modes.
   *
   * \tparam stride  The BasePtrTy-unit row stride (UPCAST_STRIDE_K / UPCAST_STRIDE_Q / etc.).
   * \param  cell    Physical cell index (0 .. smem_size-1).
   * \returns b64_t{row, col} — the logical (row, column) that maps to this cell.
   */
  template <uint32_t stride>
  static __device__ __forceinline__ b64_t get_inverse_offset(uint32_t cell) {
    const uint32_t row = cell / stride;
    const uint32_t col_sw = cell % stride;
    return b64_t{row, col_sw ^ col_swizzle_xor<stride>(row)};
  }

  // advance_offset_by_column
  //
  // The optional `stride` template parameter and `col_idx` runtime parameter are
  // required for k128B_16Row with step_size ∈ {2, 4}.  All other modes and the
  // step_size % 8 == 0 fast-path ignore them, so existing call sites that do not
  // supply them continue to compile unchanged.
  //
  // k128B_16Row, step ∈ {2,4} — exact formula:
  //   L         = offset & (stride-1)          // swizzled lower bits = j ^ (i%16)
  //   i_mod     = col_idx ^ L                  // recover i % 16
  //   new_lower = (col_idx + step_size) ^ i_mod // target lower bits = (j+step) ^ (i%16)
  //   result    = (offset & ~(stride-1)) + new_lower
  //
  // This requires the caller to track col_idx (the current column j before the
  // advance), similar to how row_idx is tracked for advance_offset_by_row<4>.
  template <uint32_t step_size, uint32_t stride = 0>
  static __device__ __forceinline__ uint32_t advance_offset_by_column(uint32_t offset,
                                                                      uint32_t step_idx,
                                                                      uint32_t col_idx = 0) {
    if constexpr (swizzle_mode == SwizzleMode::k128B) {
      static_assert(step_size == 2 || step_size == 4 || step_size % 8 == 0,
                    "Unsupported step size");
      if constexpr (step_size == 2) {
        return (offset ^ (0x2 + (0x4 * (step_idx % 2 == 1)))) + (step_idx % 4 == 3) * 8;
      } else if constexpr (step_size == 4) {
        return (offset ^ 0x4) + (step_idx % 2 == 1) * 8;
      } else {
        // step_size % 8 == 0
        return offset + step_size;
      }
    } else if constexpr (swizzle_mode == SwizzleMode::k128B_16Row) {
      static_assert(step_size == 2 || step_size == 4 || step_size % 8 == 0,
                    "Unsupported step size for k128B_16Row");
      if constexpr (step_size % 8 == 0) {
        // No swizzle correction needed; also covers the write path (step_size=16).
        return offset + step_size;
      } else {
        // step_size == 2 or 4.
        // (col_idx + step_size) ^ j can equal 4, 12, or 28 (for step=4) depending on
        // the carry pattern, so the simple "+8 / ^8" shortcut used by k128B is not
        // universally correct here.  The formula below is exact for all j values.
        static_assert(stride > 0u && (stride & (stride - 1u)) == 0u,
                      "k128B_16Row advance_offset_by_column<2/4> requires "
                      "a non-zero power-of-2 stride template argument");
        constexpr uint32_t mask = stride - 1u;
        const uint32_t L = offset & mask;    // j ^ (i%period)
        const uint32_t i_mod = col_idx ^ L;  // recover i % period
        return (offset & ~mask) + ((col_idx + step_size) ^ i_mod);
      }
    } else if constexpr (swizzle_mode == SwizzleMode::k64B) {
      static_assert(step_size == 2 || step_size == 4, "Unsupported step size");
      return (offset ^ 0x2) + (step_idx % 2 == 1) * 4;
    } else {
      // swizzle_mode == SwizzleMode::kLinear
      return offset + step_size;
    }
  }

  // advance_offset_by_row
  //
  // The optional `row_idx` runtime parameter is required for k128B_16Row with
  // step_size == 4.  It must be the logical row index immediately before the
  // advance (i.e. the current row, not the target row).  All other modes and
  // the step_size % 8 == 0 path ignore it, so existing call sites are unaffected.
  //
  // k128B_16Row, step == 4:
  //   (i+4)%16 ^ i%16 alternates between 4 and 12 every 4 rows:
  //     rows  0-3  and  8-11  →  xor_mask = 0x4
  //     rows  4-7  and 12-15  →  xor_mask = 0xC
  //   xor_mask = 0x4 | (0x8 * ((row_idx / 4) % 2))
  template <uint32_t step_size, uint32_t row_stride>
  static __device__ __forceinline__ uint32_t advance_offset_by_row(uint32_t offset,
                                                                   uint32_t row_idx = 0) {
    if constexpr (swizzle_mode == SwizzleMode::k128B) {
      static_assert(step_size == 4 || step_size % 8 == 0, "Unsupported step size");
      if constexpr (step_size == 4) {
        return (offset ^ 0x4) + step_size * row_stride;
      } else {
        // step_size % 8 == 0
        return offset + step_size * row_stride;
      }
    } else if constexpr (swizzle_mode == SwizzleMode::k128B_16Row) {
      static_assert(step_size == 4 || step_size % 8 == 0, "Unsupported step size");
      if constexpr (step_size == 4) {
        // Use the same period as get_permuted_offset: period=16 when row_stride>=16,
        // period=8 otherwise.  For period-16: (i+4)%16 ^ i%16 alternates between 4
        // and 12 every 4 rows.  For period-8 (fallback): (i+4)%8 ^ i%8 = 4 always.
        if constexpr (row_stride >= 16u) {
          const uint32_t xor_mask = 0x4u | (0x8u * ((row_idx / 4u) % 2u));
          return (offset ^ xor_mask) + step_size * row_stride;
        } else {
          return (offset ^ 0x4u) + step_size * row_stride;
        }
      } else {
        // step_size % 8 == 0 (e.g. step=16 in the read path)
        return offset + step_size * row_stride;
      }
    } else if constexpr (swizzle_mode == SwizzleMode::k64B) {
      static_assert(step_size == 4 || step_size % 8 == 0, "Unsupported step size");
      if constexpr (step_size == 4) {
        return (offset ^ 0x2) + step_size * row_stride;
      } else {
        // step_size % 8 == 0
        return offset + step_size * row_stride;
      }
    } else {
      // swizzle_mode == SwizzleMode::kLinear
      return offset + step_size * row_stride;
    }
  }
};

/*!
 * \brief Shared memory wrapper parameterized over a layout policy (composition).
 *
 * The LayoutPolicy template parameter (typically SwizzleLayout<swizzle_mode>) owns
 * all coordinate arithmetic.  smem_t exposes thin __forceinline__ wrappers that
 * forward to LayoutPolicy, so existing call sites — smem.get_permuted_offset<stride>(i,j),
 * smem->advance_offset_by_column<step,...>(...), etc. — compile without changes.
 *
 * The Layout type alias is the bijection handle: any caller that derives global-memory
 * coordinates via smem_t::Layout is structurally tied to the same swizzle pattern
 * as the LDS read/write path.  It is impossible to accidentally mix modes.
 */
template <typename LayoutPolicy, typename BasePtrTy = b128_t>
struct smem_t {
  using Layout = LayoutPolicy;

  // The base pointer.
  BasePtrTy* base;
  __device__ __forceinline__ smem_t() : base(nullptr) {}
  template <typename T>
  __device__ __forceinline__ smem_t(T* base) : base((BasePtrTy*)base) {}

  // ── Thin wrappers forwarding to Layout ────────────────────────────────────
  // All are static __forceinline__ — zero runtime cost, zero binary bloat.

  template <uint32_t stride>
  static __device__ __forceinline__ uint32_t get_permuted_offset(uint32_t i, uint32_t j) {
    return Layout::template get_permuted_offset<stride>(i, j);
  }

  template <uint32_t stride>
  static __device__ __forceinline__ b64_t get_inverse_offset(uint32_t cell) {
    return Layout::template get_inverse_offset<stride>(cell);
  }

  template <uint32_t step_size, uint32_t stride = 0>
  static __device__ __forceinline__ uint32_t advance_offset_by_column(uint32_t offset,
                                                                      uint32_t step_idx,
                                                                      uint32_t col_idx = 0) {
    return Layout::template advance_offset_by_column<step_size, stride>(offset, step_idx, col_idx);
  }

  template <uint32_t step_size, uint32_t row_stride>
  static __device__ __forceinline__ uint32_t advance_offset_by_row(uint32_t offset,
                                                                   uint32_t row_idx = 0) {
    return Layout::template advance_offset_by_row<step_size, row_stride>(offset, row_idx);
  }

  // ── LDS memory operations ─────────────────────────────────────────────────

  template <typename T = uint32_t>
  __device__ __forceinline__ void load_fragment(uint32_t offset, T* frag) {
#if defined(PLATFORM_HIP_DEVICE)
    static_assert(sizeof(T) == 4, "Only 32-bit fragment loading supported");
    reinterpret_cast<uint2*>(frag)[0] = *reinterpret_cast<const uint2*>(base + offset);
#else
    ldmatrix_m8n8x4(offset, frag);
#endif
  }

#if defined(PLATFORM_HIP_DEVICE)
  /*!
   * \brief Loads a fragment from shared memory and performs an in-register transpose across a quad.
   * \details This function is designed to prepare the B-matrix operand for a CDNA3 MFMA
   *          instruction.
   *          It performs two actions in sequence for a quad of 4 threads:
   *          1. Each thread loads a row-oriented fragment (e.g., 4 `half` values) from shared
   *             memory.
   *          2. It then calls `transpose_intra_quad_fragments` to perform an in-register transpose
   *             of this data among the 4 threads.
   *
   *          The result is that each thread's registers are populated with a column-oriented
   *          fragment, which is the required layout for the B-operand in a
   *          row-major(A) x col-major(B) MFMA.
   *
   *          Visual Representation:
   *          If `[a,b,c,d]` are the 4 `half` values loaded by Thread 0:
   *
   *          Data in Shared Memory (conceptually):
   *          Row 0: [a, b, c, d]
   *          Row 1: [e, f, g, h]
   *          Row 2: [i, j, k, l]
   *          Row 3: [m, n, o, p]
   *
   *          After this function, registers hold:
   *          Thread 0: [a, e, i, m] (Column 0)
   *          Thread 1: [b, f, j, n] (Column 1)
   *          Thread 2: [c, g, k, o] (Column 2)
   *          Thread 3: [d, h, l, p] (Column 3)
   *
   * \tparam T The type of the register fragment (e.g., uint32_t).
   * \param offset The starting offset in shared memory for the quad to begin loading.
   * \param frag A pointer to the thread's local registers to store the resulting column fragment.
   */
  template <typename T = uint32_t>
  __device__ __forceinline__ void load_matrix_m16n16_trans(uint32_t offset, T* frag) {
    load_fragment(offset, frag);
    gpu_iface::mma::transpose_mma_tile(frag);
  }
#endif

  template <typename T = uint32_t>
  __device__ __forceinline__ void store_fragment(uint32_t offset, const T* frag) {
#if defined(PLATFORM_HIP_DEVICE)
    static_assert(sizeof(T) == 4, "Only 32-bit fragment storing supported");
    *reinterpret_cast<uint2*>(base + offset) = reinterpret_cast<const uint2*>(frag)[0];
#else
    stmatrix_m8n8x4(offset, frag);
#endif
  }

#if defined(PLATFORM_CUDA_DEVICE)
  __device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4(R, smem_ptr);
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4_left_half(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4_left_half(R, smem_ptr);
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4_right_half(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4_right_half(R, smem_ptr);
  }

  __device__ __forceinline__ void stmatrix_m8n8x4(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::stmatrix_m8n8x4(R, smem_ptr);
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4_trans(R, smem_ptr);
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4_trans_left_half(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4_trans_left_half(R, smem_ptr);
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4_trans_right_half(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4_trans_right_half(R, smem_ptr);
  }
#endif
  template <gpu_mem::SharedMemFillMode fill_mode, typename T>
  __device__ __forceinline__ void load_128b_async(uint32_t offset, const T* gptr, bool predicate) {
    b128_t* smem_ptr = base + offset;
    gpu_mem::pred_load_128b<gpu_mem::PrefetchMode::kPrefetch, fill_mode>(
        smem_ptr, reinterpret_cast<const b128_t*>(gptr), predicate);
  }

  template <typename T>
  __device__ __forceinline__ void load_128b_async(uint32_t offset, const T* gptr) {
    b128_t* smem_ptr = base + offset;
    gpu_mem::load_128b<gpu_mem::PrefetchMode::kPrefetch>(smem_ptr,
                                                         reinterpret_cast<const b128_t*>(gptr));
  }

  template <gpu_mem::SharedMemFillMode fill_mode, typename T>
  __device__ __forceinline__ void load_64b_async(uint32_t offset, const T* gptr, bool predicate) {
    b64_t* smem_ptr = base + offset;
    gpu_mem::pred_load_64b<gpu_mem::PrefetchMode::kPrefetch, fill_mode>(
        smem_ptr, reinterpret_cast<const b64_t*>(gptr), predicate);
  }

  template <typename T>
  __device__ __forceinline__ void load_64b_async(uint32_t offset, const T* gptr) {
    b64_t* smem_ptr = base + offset;
    gpu_mem::load_64b<gpu_mem::PrefetchMode::kPrefetch>(smem_ptr,
                                                        reinterpret_cast<const b64_t*>(gptr));
  }

  template <gpu_mem::SharedMemFillMode fill_mode, typename T>
  __device__ __forceinline__ void load_vector_async(uint32_t offset, const T* gptr,
                                                    bool predicate) {
#if defined(PLATFORM_HIP_DEVICE)
    load_64b_async<fill_mode>(offset, gptr, predicate);
#else
    load_128b_async<fill_mode>(offset, gptr, predicate);
#endif
  }

  template <typename T>
  __device__ __forceinline__ void load_vector_async(uint32_t offset, const T* gptr) {
#if defined(PLATFORM_HIP_DEVICE)
    load_64b_async(offset, gptr);
#else
    load_128b_async(offset, gptr);
#endif
  }

  template <typename T>
  __device__ __forceinline__ void store_128b(uint32_t offset, T* gptr) {
    *reinterpret_cast<b128_t*>(gptr) = *(base + offset);
  }

  template <typename T>
  __device__ __forceinline__ void store_64b(uint32_t offset, T* gptr) {
    *reinterpret_cast<b64_t*>(gptr) = *(base + offset);
  }

  template <typename T>
  __device__ __forceinline__ void store_vector(uint32_t offset, T* gptr) {
#if defined(PLATFORM_HIP_DEVICE)
    store_64b(offset, gptr);
#else
    store_128b(offset, gptr);
#endif
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_PERMUTED_SMEM_CUH_
