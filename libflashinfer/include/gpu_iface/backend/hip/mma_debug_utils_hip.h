// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gpu_iface/backend/hip/mma_hip.h"
#include "gpu_iface/gpu_runtime_compat.hpp"

namespace {
constexpr uint32_t MMA_COLS = 16;
constexpr uint32_t MMA_ROWS_PER_THREAD = 4;
}  // namespace

namespace flashinfer::gpu_iface::debug_utils::hip {

enum class MatrixLayout { A, B };

/// @brief Initializes a 2D LDS array with lexicographical values (0, 1, 2, ...).
/// @param lds_array Pointer to the shared memory array.
/// @param dimY The height of the 2D array.
/// @param dimX The width of the 2D array.
__device__ void lexicographic_init_lds_array(half* lds_array, uint32_t dimY, uint32_t dimX) {
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int y = 0; y < dimY; ++y) {
      for (int x = 0; x < dimX; ++x) {
        lds_array[y * dimX + x] = __half(y * dimX + x);
      }
    }
  }
  __syncthreads();
}

/// @brief Loads a 16x16 tile from LDS into registers using the A-matrix layout pattern.
/// @details Each thread `T_(16*c + r)` loads a 1x4 horizontal fragment from `LDS[r, 4*c : 4*c+3]`.
/// @tparam T The data type of the LDS array, must be `__half`.
/// @param lds_array Pointer to the shared memory array.
/// @param R Pointer to the thread's registers (uint32_t[2]).
/// @param dimX The width of the LDS array.
template <typename T>
__device__ void load_amatrix_layout(T* lds_array, uint32_t* R, uint32_t dimX) {
  static_assert(std::is_same_v<T, __half>, "Only supported for __half types");
  const int lane_id = threadIdx.x % 64;
  const int row = lane_id % MMA_COLS;
  const int col_start = (lane_id / MMA_COLS) * MMA_ROWS_PER_THREAD;

  auto offset = lds_array + row * dimX + col_start;
  mma_impl::hip::load_fragment(R, offset);
}

/// @brief Loads a 16x16 tile from LDS into registers using the B-matrix layout pattern.
/// @details Uses an efficient load-and-transpose strategy. A 4x4 block of threads loads a
///          contiguous 4x4 tile from LDS and then performs an in-register transpose,
///          resulting in each thread holding a column fragment.
/// @tparam T The data type of the LDS array, must be `__half`.
/// @param arr Pointer to the shared memory array.
/// @param R Pointer to the thread's registers (uint32_t[2]).
/// @param dimY The height of the LDS array.
template <typename T>
__device__ void load_bmatrix_layout(T* arr, uint32_t* R, uint32_t dimY) {
  static_assert(std::is_same_v<T, __half>, "Only supported for __half types");
  const int lane_id = threadIdx.x % 64;
  int b_idx =
      ((lane_id % MMA_ROWS_PER_THREAD) + MMA_ROWS_PER_THREAD * (lane_id / MMA_COLS)) * dimY +
      ((lane_id % MMA_COLS) / MMA_ROWS_PER_THREAD) * MMA_ROWS_PER_THREAD;
  mma_impl::hip::load_quad_transposed_fragment<__half>(R, &arr[b_idx]);
}

/// @brief Prints the four `half` values held in a thread's registers.
/// @tparam T The data type to interpret the registers as, must be `__half`.
/// @param R Pointer to the thread's registers (uint32_t[2]).
template <typename T>
__device__ void print_register(uint32_t* R) {
  static_assert(std::is_same_v<T, __half>, "Only supported for __half types");
  auto values = reinterpret_cast<__half*>(R);
  printf("[%5.1f %5.1f %5.1f %5.1f]\n", __half2float(values[0]), __half2float(values[1]),
         __half2float(values[2]), __half2float(values[3]));
}

/// @brief Prints the full s_frag array from a single thread's registers.
template <typename T, uint32_t NUM_MMA_Q, uint32_t NUM_MMA_KV, uint32_t ELEMS_PER_FRAGMENT>
__device__ void print_s_frag_register(const T (*s_frag)[NUM_MMA_KV][ELEMS_PER_FRAGMENT],
                                      const dim3 tid = threadIdx) {
  if (tid.x == 0 && tid.y == 0 && tid.z == 0) {
    printf("Thread (0,0,0) s_frag registers:\n");
    for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
      for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
        const T* values = s_frag[mma_q][mma_kv];
        printf("  frag[%u][%u]: [%8.3f, %8.3f, %8.3f, %8.3f]\n", mma_q, mma_kv, float(values[0]),
               float(values[1]), float(values[2]), float(values[3]));
      }
    }
    printf("\n");
  }
  __syncthreads();
}

/// @brief Prints a 2D LDS array to the console from a single thread.
/// @tparam T The data type of the LDS array, must be `__half`.
/// @param lds_array Pointer to the shared memory array.
/// @param dimY The height of the 2D array.
/// @param dimX The width of the 2D array.
template <typename T>
__device__ void print_lds_array(T* lds_array, uint32_t dimY, uint32_t dimX,
                                const char* title = "LDS Array") {
  static_assert(std::is_same_v<T, __half>, "Only supported for __half types");
  if (threadIdx.x == 0) {
    printf("%s (%dx%d):\n", title, dimX, dimY);
    for (int y = 0; y < dimY; ++y) {
      for (int x = 0; x < dimX; ++x) {
        printf("%5.1f ", __half2float(lds_array[y * dimX + x]));
      }
      printf("\n");
    }
    printf("\n");
  }
  __syncthreads();
}

/// @brief Prints a 2D LDS array of floats to the console from a single thread.
__device__ void print_lds_array(float* lds_array, uint32_t dimY, uint32_t dimX,
                                const char* title = "LDS Array (float)") {
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    printf("%s (%dx%d):\n", title, dimX, dimY);
    for (int y = 0; y < dimY; ++y) {
      for (int x = 0; x < dimX; ++x) {
        printf("%10.6f ", lds_array[y * dimX + x]);
      }
      printf("\n");
    }
    printf("\n");
  }
  __syncthreads();
}

/// @brief Prints a 1D LDS array of floats to the console from a single thread.
/// @details Useful for printing row-wise statistics like m or d values.
__device__ void print_lds_array_1d(float* lds_array, uint32_t dim,
                                   const char* title = "LDS Array 1D (float)") {
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    printf("%s (%d elements):\n", title, dim);
    for (int i = 0; i < dim; ++i) {
      printf("%10.6f ", lds_array[i]);
      if ((i + 1) % 16 == 0) printf("\n");  // Line break every 16 elements
    }
    if (dim % 16 != 0) printf("\n");
    printf("\n");
  }
  __syncthreads();
}

/// @brief Generic function to materialize 2D fragment arrays into shared memory.
/// @details Works for both s_frag (attention scores) and o_frag (output accumulator).
///          Reconstructs a logical tile from distributed register fragments.
/// @tparam T The data type of the fragments and LDS array (e.g., float or half).
/// @tparam NUM_MMA_ROW The number of fragments along the rows dimension per thread.
/// @tparam NUM_MMA_COL The number of fragments along the column dimension per thread.
///                     For s_frag: NUM_MMA_KV (KV sequence length)
///                     For o_frag: NUM_MMA_D_VO (head dimension)
/// @tparam ELEMS_PER_FRAGMENT The number of elements per fragment (typically 4).
/// @param frag The 3D fragment array from the thread's registers.
/// @param lds_scratchpad Pointer to the shared memory array.
/// @param lds_stride The width/stride of the lds_scratchpad.
/// @param tid The thread's index within the block (threadIdx).
template <typename T, uint32_t NUM_MMA_ROW, uint32_t NUM_MMA_COL, uint32_t ELEMS_PER_FRAGMENT = 4>
__device__ void write_frag_to_lds(const T (*frag)[NUM_MMA_COL][ELEMS_PER_FRAGMENT],
                                  T* lds_scratchpad, const uint32_t lds_stride,
                                  const dim3 tid = threadIdx) {
  const int lane_id = tid.x % 64;
  const int warp_idx_q = tid.y;

  // Calculate the starting row in the LDS tile for this entire warp.
  const uint32_t warp_base_row = warp_idx_q * NUM_MMA_ROW * MMA_COLS;

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_ROW; ++mma_q) {
#pragma unroll
    for (uint32_t mma_col = 0; mma_col < NUM_MMA_COL; ++mma_col) {
      // -- Calculate the top-left corner of the 16x16 fragment this thread contributes to --
      const uint32_t frag_row_offset = mma_q * MMA_COLS;
      const uint32_t frag_col_offset = mma_col * MMA_COLS;

      // -- Calculate the specific 4x1 element strip this thread writes within that fragment --
      // This logic correctly materializes a B-layout fragment (column strip).
      // Each thread T_c handles column 'c' of the fragment.
      // The 4 threads in a "column" of the warp (e.g., lanes 0, 16, 32, 48)
      // handle the 4 rows of that column strip.
      const uint32_t thread_start_row_in_frag = (lane_id / MMA_COLS) * MMA_ROWS_PER_THREAD;
      const uint32_t thread_col_in_frag = (lane_id % MMA_COLS);

      // -- Combine all offsets and write the 4x1 column strip to LDS --
      const T* values = frag[mma_q][mma_col];
      for (int i = 0; i < MMA_ROWS_PER_THREAD; ++i) {
        // The row for this element is the thread's starting row + the element's index in the strip.
        const uint32_t final_row = warp_base_row + frag_row_offset + thread_start_row_in_frag + i;
        // The column is fixed for all 4 elements in the strip.
        const uint32_t final_col = frag_col_offset + thread_col_in_frag;

        // Calculate destination and write the value.
        T* dest = lds_scratchpad + final_row * lds_stride + final_col;
        *dest = values[i];
      }
    }
  }
}

/// @brief Convenience wrapper for s_frag (attention scores).
template <typename T, uint32_t NUM_MMA_Q, uint32_t NUM_MMA_KV, uint32_t ELEMS_PER_FRAGMENT = 4>
__device__ void write_s_frag_to_lds(const T (*s_frag)[NUM_MMA_KV][ELEMS_PER_FRAGMENT],
                                    T* lds_scratchpad, const uint32_t lds_stride,
                                    const dim3 tid = threadIdx) {
  write_frag_to_lds<T, NUM_MMA_Q, NUM_MMA_KV, ELEMS_PER_FRAGMENT>(s_frag, lds_scratchpad,
                                                                  lds_stride, tid);
}

/// @brief Convenience wrapper for o_frag (output accumulator).
template <typename T, uint32_t NUM_MMA_Q, uint32_t NUM_MMA_D_VO, uint32_t ELEMS_PER_FRAGMENT = 4>
__device__ void write_o_frag_to_lds(const T (*o_frag)[NUM_MMA_D_VO][ELEMS_PER_FRAGMENT],
                                    T* lds_scratchpad, const uint32_t lds_stride,
                                    const dim3 tid = threadIdx) {
  write_frag_to_lds<T, NUM_MMA_Q, NUM_MMA_D_VO, ELEMS_PER_FRAGMENT>(o_frag, lds_scratchpad,
                                                                    lds_stride, tid);
}

/// @brief Generic function to materialize 1D row-wise values (m or d) into shared memory.
/// @details Writes row-wise statistics (like max or denominator) from register arrays
///          to a 1D shared memory array, with one value per row.
/// @tparam T The data type (typically float).
/// @tparam NUM_MMA_Q The number of fragments along the Q dimension per thread.
/// @tparam NUM_ACCUM_ROWS_PER_THREAD The number of accumulator rows per thread (typically 4).
/// @param values The 2D array from registers [NUM_MMA_Q][NUM_ACCUM_ROWS_PER_THREAD].
/// @param lds_scratchpad Pointer to the 1D shared memory array.
/// @param tid The thread's index within the block (threadIdx).
template <typename T, uint32_t NUM_MMA_Q, uint32_t NUM_ACCUM_ROWS_PER_THREAD>
__device__ void write_row_values_to_lds(const T (*values)[NUM_ACCUM_ROWS_PER_THREAD],
                                        T* lds_scratchpad, const dim3 tid = threadIdx) {
  const int lane_idx = tid.x;
  const int warp_idx_q = tid.y;

  // Each group of 16 threads (a "row group") handles 4 rows.
  // We only need one thread from each group to write the results.
  if (lane_idx % MMA_COLS == 0) {
    // Base row index for this warp's Q tile
    const uint32_t warp_base_row = warp_idx_q * NUM_MMA_Q * MMA_COLS;

#pragma unroll
    for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
      // Base row for this specific MMA instruction within the warp's tile
      const uint32_t mma_base_row = mma_q * MMA_COLS;

#pragma unroll
      for (uint32_t j = 0; j < NUM_ACCUM_ROWS_PER_THREAD; ++j) {
        // The thread's lane_idx determines which group of 4 rows it is in.
        // e.g., lane 0 is in group 0, lane 16 is in group 1, etc.
        const uint32_t row_group_offset = (lane_idx / MMA_COLS) * NUM_ACCUM_ROWS_PER_THREAD;

        // The final row index in the logical matrix
        const uint32_t final_row_idx = warp_base_row + mma_base_row + row_group_offset + j;

        lds_scratchpad[final_row_idx] = values[mma_q][j];
      }
    }
  }
}

/// @brief Convenience wrapper for m (row-wise max) values.
template <typename T, uint32_t NUM_MMA_Q, uint32_t NUM_ACCUM_ROWS_PER_THREAD>
__device__ void write_m_to_lds(const T (*m)[NUM_ACCUM_ROWS_PER_THREAD], T* lds_scratchpad,
                               const dim3 tid = threadIdx) {
  write_row_values_to_lds<T, NUM_MMA_Q, NUM_ACCUM_ROWS_PER_THREAD>(m, lds_scratchpad, tid);
}

/// @brief Convenience wrapper for d (denominator) values.
template <typename T, uint32_t NUM_MMA_Q, uint32_t NUM_ACCUM_ROWS_PER_THREAD>
__device__ void write_d_to_lds(const T (*d)[NUM_ACCUM_ROWS_PER_THREAD], T* lds_scratchpad,
                               const dim3 tid = threadIdx) {
  write_row_values_to_lds<T, NUM_MMA_Q, NUM_ACCUM_ROWS_PER_THREAD>(d, lds_scratchpad, tid);
}

// Legacy alias for backward compatibility
template <typename T, uint32_t NUM_MMA_Q, uint32_t NUM_ACCUM_ROWS_PER_THREAD>
__device__ void write_m_new_to_lds(const T (*m)[NUM_ACCUM_ROWS_PER_THREAD], T* lds_scratchpad,
                                   const dim3 tid = threadIdx) {
  write_m_to_lds<T, NUM_MMA_Q, NUM_ACCUM_ROWS_PER_THREAD>(m, lds_scratchpad, tid);
}

}  // namespace flashinfer::gpu_iface::debug_utils::hip
