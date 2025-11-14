// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <cstdint>

namespace flashinfer {
namespace gpu_iface {
namespace mma {

// ============================================================================
// Generic M16N16K16 Layout (common traits across A/B/C)
// ============================================================================

/*!
 * \brief Generic M16N16K16 layout traits for HIP CDNA3
 *
 * Properties uniform across A/B/C matrices for the native CDNA3 16×16×16 operation.
 */
template <>
struct wmma_generic_layout<WmmaOp::M16N16K16, half> {
  using DataType = half;

  // Fragment properties (same for A/B/C on CDNA3)
  static constexpr uint32_t FRAG_SIZE = 4;
  static constexpr uint32_t FRAG_BITWIDTH = FRAG_SIZE * sizeof(DataType) * 8;
  static constexpr uint32_t INT32_WORDS_PER_FRAG = 2;

  // Tile dimensions
  static constexpr uint32_t TILE_M = 16;
  static constexpr uint32_t TILE_N = 16;
  static constexpr uint32_t TILE_K = 16;
};

// ============================================================================
// M16N16K16 Specializations (Native CDNA3 MFMA operation)
// ============================================================================

/*!
 * \brief HIP CDNA3 MFMA A-matrix layout for fp16 (16×16×16 operation)
 *
 * CDNA3 MFMA natively supports 16×16 tiles in a single instruction.
 * This is NOT a composite operation like CUDA's m16n16k16.
 *
 * - Fragment layout: RowMajor
 * - Tile size: 16 rows × 16 columns (k-dimension)
 * - Total: 64 threads (wavefront size)
 * - Each thread owns 4 elements from 1 row × 4 columns
 *
 * Thread-to-element ownership pattern:
 *
 *   Row\Col:  0   1   2   3 | 4   5   6   7  | 8   9  10  11  |12  13  14  15  |
 *          +----------------+----------------+----------------+----------------+
 *   Row  0 | T0  T0  T0  T0 |T16 T16 T16 T16 |T32 T32 T32 T32 |T48 T48 T48 T48 |
 *   Row  1 | T1  T1  T1  T1 |T17 T17 T17 T17 |T33 T33 T33 T33 |T49 T49 T49 T49 |
 *   Row  2 | T2  T2  T2  T2 |T18 T18 T18 T18 |T34 T34 T34 T34 |T50 T50 T50 T50 |
 *   Row  3 | T3  T3  T3  T3 |T19 T19 T19 T19 |T35 T35 T35 T35 |T51 T51 T51 T51 |
 *   Row  4 | T4  T4  T4  T4 |T20 T20 T20 T20 |T36 T36 T36 T36 |T52 T52 T52 T52 |
 *   Row  5 | T5  T5  T5  T5 |T21 T21 T21 T21 |T37 T37 T37 T37 |T53 T53 T53 T53 |
 *   Row  6 | T6  T6  T6  T6 |T22 T22 T22 T22 |T38 T38 T38 T38 |T54 T54 T54 T54 |
 *   Row  7 | T7  T7  T7  T7 |T23 T23 T23 T23 |T39 T39 T39 T39 |T55 T55 T55 T55 |
 *   Row  8 | T8  T8  T8  T8 |T24 T24 T24 T24 |T40 T40 T40 T40 |T56 T56 T56 T56 |
 *   Row  9 | T9  T9  T9  T9 |T25 T25 T25 T25 |T41 T41 T41 T41 |T57 T57 T57 T57 |
 *   Row 10 |T10 T10 T10 T10 |T26 T26 T26 T26 |T42 T42 T42 T42 |T58 T58 T58 T58 |
 *   Row 11 |T11 T11 T11 T11 |T27 T27 T27 T27 |T43 T43 T43 T43 |T59 T59 T59 T59 |
 *   Row 12 |T12 T12 T12 T12 |T28 T28 T28 T28 |T44 T44 T44 T44 |T60 T60 T60 T60 |
 *   Row 13 |T13 T13 T13 T13 |T29 T29 T29 T29 |T45 T45 T45 T45 |T61 T61 T61 T61 |
 *   Row 14 |T14 T14 T14 T14 |T30 T30 T30 T30 |T46 T46 T46 T46 |T62 T62 T62 T62 |
 *   Row 15 |T15 T15 T15 T15 |T31 T31 T31 T31 |T47 T47 T47 T47 |T63 T63 T63 T63 |
 *          +----------------+----------------+----------------+----------------+
 *
 * Thread T0 example owns:
 *   - Row 0: cols {0,1,2,3} (registers r0,r1,r2,r3)
 *   Total: 4 fp16 values from 1 row × 4 columns
 *
 * Key pattern:
 *   - Each thread owns elements from exactly 1 row
 *   - Within that row, owns 4 consecutive column positions
 *   - Thread grid: 16 rows × 4 columns = 64 threads
 *   - Threads collaborate in sets of 4 per row (THREADS_PER_ROW_SET = 4)
 */
template <>
struct wmma_a_layout<WmmaOp::M16N16K16, half> {
  using DataType = half;
  static constexpr FragmentLayout LAYOUT = FragmentLayout::RowMajor;

  static constexpr uint32_t TILE_ROWS = 16;
  static constexpr uint32_t TILE_COLS = 16;

  static constexpr uint32_t THREAD_GRID_ROWS = 16;
  static constexpr uint32_t THREAD_GRID_COLS = 4;

  /*! \brief Number of matrix rows owned by each thread in its register fragment */
  static constexpr uint32_t ROWS_PER_FRAGMENT = 1;
  /*! \brief Number of matrix columns owned by each thread in its register fragment */
  static constexpr uint32_t COLS_PER_FRAGMENT = 4;

  /*! \brief Total number of matrix elements owned by each thread (ROWS × COLS) */
  static constexpr uint32_t ELEMS_PER_FRAGMENT = 4;  // 1 row × 4 cols
  static constexpr uint32_t INT32_WORDS_PER_FRAGMENT = 2;
};

template <>
struct wmma_a_layout<WmmaOp::M16N16K16, hip_bfloat16> {};

/*!
 * \brief HIP CDNA3 MFMA B-matrix layout for fp16 (16×16×16 operation)
 *
 * CDNA3 MFMA natively supports 16×16 tiles in a single instruction.
 *
 * - Fragment layout: ColumnMajor
 * - Tile size: 16 rows (k-dimension) × 16 columns
 * - Total: 64 threads (wavefront size)
 * - Each thread owns 4 elements from 4 rows × 1 column
 *
 * Thread-to-element ownership pattern:
 *
 *   Row\Col:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
 *          +----------------------------------------------------------------+
 *   Row  0 | T0  T1  T2  T3  T4  T5  T6  T7  T8  T9 T10 T11 T12 T13 T14 T15 |
 *   Row  1 | T0  T1  T2  T3  T4  T5  T6  T7  T8  T9 T10 T11 T12 T13 T14 T15 |
 *   Row  2 | T0  T1  T2  T3  T4  T5  T6  T7  T8  T9 T10 T11 T12 T13 T14 T15 |
 *   Row  3 | T0  T1  T2  T3  T4  T5  T6  T7  T8  T9 T10 T11 T12 T13 T14 T15 |
 *          +----------------------------------------------------------------+
 *   Row  4 |T16 T17 T18 T19 T20 T21 T22 T23 T24 T25 T26 T27 T28 T29 T30 T31 |
 *   Row  5 |T16 T17 T18 T19 T20 T21 T22 T23 T24 T25 T26 T27 T28 T29 T30 T31 |
 *   Row  6 |T16 T17 T18 T19 T20 T21 T22 T23 T24 T25 T26 T27 T28 T29 T30 T31 |
 *   Row  7 |T16 T17 T18 T19 T20 T21 T22 T23 T24 T25 T26 T27 T28 T29 T30 T31 |
 *          +----------------------------------------------------------------+
 *   Row  8 |T32 T33 T34 T35 T36 T37 T38 T39 T40 T41 T42 T43 T44 T45 T46 T47 |
 *   Row  9 |T32 T33 T34 T35 T36 T37 T38 T39 T40 T41 T42 T43 T44 T45 T46 T47 |
 *   Row 10 |T32 T33 T34 T35 T36 T37 T38 T39 T40 T41 T42 T43 T44 T45 T46 T47 |
 *   Row 11 |T32 T33 T34 T35 T36 T37 T38 T39 T40 T41 T42 T43 T44 T45 T46 T47 |
 *          +----------------------------------------------------------------+
 *   Row 12 |T48 T49 T50 T51 T52 T53 T54 T55 T56 T57 T58 T59 T60 T61 T62 T63 |
 *   Row 13 |T48 T49 T50 T51 T52 T53 T54 T55 T56 T57 T58 T59 T60 T61 T62 T63 |
 *   Row 14 |T48 T49 T50 T51 T52 T53 T54 T55 T56 T57 T58 T59 T60 T61 T62 T63 |
 *   Row 15 |T48 T49 T50 T51 T52 T53 T54 T55 T56 T57 T58 T59 T60 T61 T62 T63 |
 *          +----------------------------------------------------------------+
 *
 * Thread T0 example owns:
 *   - Rows {0,1,2,3}: col 0 (registers r0,r1,r2,r3)
 *   Total: 4 fp16 values from 4 rows × 1 column
 *
 * Key pattern:
 *   - Each thread owns elements from exactly 1 column
 *   - Within that column, owns 4 consecutive row positions
 *   - Thread grid: 4 rows × 16 columns = 64 threads
 *   - Threads collaborate in sets of 16 per column (THREADS_PER_ROW_SET = 16)
 *
 * Note: B-layout and C-layout are IDENTICAL on CDNA3 (both column-major)
 */
template <>
struct wmma_b_layout<WmmaOp::M16N16K16, half> {
  using DataType = half;
  static constexpr FragmentLayout LAYOUT = FragmentLayout::ColumnMajor;

  static constexpr uint32_t TILE_ROWS = 16;  // K-dimension
  static constexpr uint32_t TILE_COLS = 16;

  static constexpr uint32_t THREAD_GRID_ROWS = 4;
  static constexpr uint32_t THREAD_GRID_COLS = 16;

  /*! \brief Number of matrix rows owned by each thread in its register fragment */
  static constexpr uint32_t ROWS_PER_FRAGMENT = 4;
  /*! \brief Number of matrix columns owned by each thread in its register fragment */
  static constexpr uint32_t COLS_PER_FRAGMENT = 1;

  /*! \brief Total number of matrix elements owned by each thread (ROWS × COLS) */
  static constexpr uint32_t ELEMS_PER_FRAGMENT = 4;  // 4 rows × 1 col
  static constexpr uint32_t INT32_WORDS_PER_FRAGMENT = 2;
};

template <>
struct wmma_b_layout<WmmaOp::M16N16K16, hip_bfloat16> {};

/*!
 * \brief HIP CDNA3 MFMA C/D-matrix (accumulator) layout for fp16 (16×16×16 operation)
 *
 * CDNA3 MFMA natively supports 16×16 tiles in a single instruction.
 *
 * - Fragment layout: ColumnMajor
 * - Tile size: 16 rows × 16 columns
 * - Total: 64 threads (wavefront size)
 * - Each thread owns 4 elements from 4 rows × 1 column
 *
 * Thread-to-element ownership pattern:
 * IDENTICAL to B-matrix layout on CDNA3.
 *
 *   Row\Col:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
 *          +----------------------------------------------------------------+
 *   Row  0 | T0  T1  T2  T3  T4  T5  T6  T7  T8  T9 T10 T11 T12 T13 T14 T15 |
 *   Row  1 | T0  T1  T2  T3  T4  T5  T6  T7  T8  T9 T10 T11 T12 T13 T14 T15 |
 *   Row  2 | T0  T1  T2  T3  T4  T5  T6  T7  T8  T9 T10 T11 T12 T13 T14 T15 |
 *   Row  3 | T0  T1  T2  T3  T4  T5  T6  T7  T8  T9 T10 T11 T12 T13 T14 T15 |
 *          +----------------------------------------------------------------+
 *   Row  4 |T16 T17 T18 T19 T20 T21 T22 T23 T24 T25 T26 T27 T28 T29 T30 T31 |
 *   Row  5 |T16 T17 T18 T19 T20 T21 T22 T23 T24 T25 T26 T27 T28 T29 T30 T31 |
 *   Row  6 |T16 T17 T18 T19 T20 T21 T22 T23 T24 T25 T26 T27 T28 T29 T30 T31 |
 *   Row  7 |T16 T17 T18 T19 T20 T21 T22 T23 T24 T25 T26 T27 T28 T29 T30 T31 |
 *          +----------------------------------------------------------------+
 *   Row  8 |T32 T33 T34 T35 T36 T37 T38 T39 T40 T41 T42 T43 T44 T45 T46 T47 |
 *   Row  9 |T32 T33 T34 T35 T36 T37 T38 T39 T40 T41 T42 T43 T44 T45 T46 T47 |
 *   Row 10 |T32 T33 T34 T35 T36 T37 T38 T39 T40 T41 T42 T43 T44 T45 T46 T47 |
 *   Row 11 |T32 T33 T34 T35 T36 T37 T38 T39 T40 T41 T42 T43 T44 T45 T46 T47 |
 *          +----------------------------------------------------------------+
 *   Row 12 |T48 T49 T50 T51 T52 T53 T54 T55 T56 T57 T58 T59 T60 T61 T62 T63 |
 *   Row 13 |T48 T49 T50 T51 T52 T53 T54 T55 T56 T57 T58 T59 T60 T61 T62 T63 |
 *   Row 14 |T48 T49 T50 T51 T52 T53 T54 T55 T56 T57 T58 T59 T60 T61 T62 T63 |
 *   Row 15 |T48 T49 T50 T51 T52 T53 T54 T55 T56 T57 T58 T59 T60 T61 T62 T63 |
 *          +----------------------------------------------------------------+
 *
 * Thread T0 example owns:
 *   - Rows {0,1,2,3}: col 0 (registers r0,r1,r2,r3)
 *   Total: 4 fp16 values from 4 rows × 1 column
 *
 * Key insight for attention:
 *   - Accumulator S (attention scores) and O (output) use this layout
 *   - Same as B-layout on CDNA3 (column-major distribution)
 *   - Each thread handles 4 output rows from 1 column
 */
template <>
struct wmma_c_layout<WmmaOp::M16N16K16, half> {
  using DataType = half;
  static constexpr FragmentLayout LAYOUT = FragmentLayout::ColumnMajor;

  static constexpr uint32_t TILE_ROWS = 16;
  static constexpr uint32_t TILE_COLS = 16;

  static constexpr uint32_t THREAD_GRID_ROWS = 4;
  static constexpr uint32_t THREAD_GRID_COLS = 16;

  /*! \brief Number of matrix rows owned by each thread in its register fragment */
  static constexpr uint32_t ROWS_PER_FRAGMENT = 4;
  /*! \brief Number of matrix columns owned by each thread in its register fragment */
  static constexpr uint32_t COLS_PER_FRAGMENT = 1;

  /*! \brief Total number of matrix elements owned by each thread (ROWS × COLS) */
  static constexpr uint32_t ELEMS_PER_FRAGMENT = 4;  // 4 rows × 1 col
  static constexpr uint32_t INT32_WORDS_PER_FRAGMENT = 2;
};

template <>
struct wmma_c_layout<WmmaOp::M16N16K16, hip_bfloat16> {};

}  // namespace mma
}  // namespace gpu_iface
}  // namespace flashinfer
