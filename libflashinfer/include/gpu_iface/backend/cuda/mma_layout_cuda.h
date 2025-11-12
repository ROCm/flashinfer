// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace flashinfer {
namespace gpu_iface {
namespace mma {

// ============================================================================
// M16N8K16 Specializations for fp16, bf16 (Base PTX operations)
// ============================================================================

/*!
 * \brief CUDA Tensor Core A-matrix layout for fp16 m16n8k16 operation
 *
 * Represents a A fragment layout for a SINGLE m16n8k16 PTX instruction.
 *
 * - Fragment layout: RowMajor-like with quadrant replication
 * - Tile size: 16 rows × 16 columns (k-dimension)
 * - Total: 32 threads (warp size)
 * - Each thread owns 8 elements from 2 rows spread across quadrants
 *
 * Thread-to-element ownership pattern:
 * Based on PTX documentation for m16n8k16 A-matrix.
 *
 * Notation: T0:{a0,a1} means the fp16 values stored in Thread 0's registers a0,a1
 *
 *   Row\Col:  0   1    |    2   3   |   4   5    |   6   7    | 8   9 |  10  11 | 12  13 | 14  15 |
 *          +-----------+------------+------------+------------+-------+---------+--------+--------+
 *   Row  0 |T0:{a0,a1} | T1:{a0,a1} | T2:{a0,a1} | T3:{a0,a1} | ...T0..T3:{a4,a5} ...             |
 *          +-----------+------------+------------+------------+-------+---------+--------+--------+
 *   Row  1 |T4:{a0,a1} | T5:{a0,a1} | T6:{a0,a1} | T7:{a0,a1} | ...T6..T7:{a4,a5} ...             |
 *          +-----------+------------+------------+------------+-------+---------+--------+--------+
 *          |         ...pattern continues for rows 2-6 (T8-T27)...                                |
 *          +-----------+------------+------------+------------+-------+---------+--------+--------+
 *   Row  7 |T28:{a0,a1}| T29:{a0,a1}| T30:{a0,a1}| T31:{a0,a1}| ...T28..T31:{a4,a5} ...           |
 *          +-----------+------------+------------+------------+-------+---------+--------+--------+
 *   Row  8 |T0:{a2,a3} | T1:{a2,a3} | T2:{a2,a3} | T3:{a2,a3} | ...T0..T3:{a6,a7} ...             |
 *          +-----------+------------+------------+------------+-------+---------+--------+--------+
 *   Row  9 |T4:{a2,a3} | T5:{a2,a3} | T6:{a2,a3} | T7:{a2,a3} | ...T6..T7:{a6,a7} ...             |
 *          +-----------+------------+------------+------------+-------+---------+--------+--------+
 *          |         ...pattern continues for rows 10-14 (T8-T27)...                              |
 *          +-----------+------------+------------+------------+-------+---------+--------+--------+
 *   Row 15 |T28:{a2,a3}| T29:{a2,a3}| T30:{a2,a3}| T31:{a2,a3}| ...T28..T31:{a6,a7} ...           |
 *          +-----------+------------+------------+------------+-------+---------+--------+--------+
 *
 * Thread T0 example owns:
 *   - Row 0: cols {0,1} (a0,a1) and cols {8,9} (a4,a5)
 *   - Row 8: cols {0,1} (a2,a3) and cols {8,9} (a6,a7)
 *   Total: 8 fp16 values from 2 rows, replicated across left/right halves
 *
 * Key pattern:
 *   - Each thread owns elements from 2 rows (row i and row i+8)
 *   - Within each row, owns 4 column positions (2 in cols 0-7, 2 in cols 8-15)
 *   - Thread i (0≤i<32) maps to rows: 2*(i/4) and 2*(i/4)+8
 */
template <>
struct wmma_a_layout<WmmaOp::M16N8K16, half> {
  using DataType = half;
  static constexpr FragmentLayout LAYOUT = FragmentLayout::RowMajor;

  static constexpr uint32_t TILE_ROWS = 16;
  static constexpr uint32_t TILE_COLS = 16;

  static constexpr uint32_t THREAD_GRID_ROWS = 8;
  static constexpr uint32_t THREAD_GRID_COLS = 4;

  /*! \brief Number of matrix rows owned by each thread in its register fragment */
  static constexpr uint32_t ROWS_PER_FRAGMENT = 2;
  /*! \brief Number of matrix columns owned by each thread in its register fragment */
  static constexpr uint32_t COLS_PER_FRAGMENT = 4;

  /*! \brief Total number of matrix elements owned by each thread (ROWS × COLS) */
  static constexpr uint32_t ELEMS_PER_FRAGMENT = 8;
  static constexpr uint32_t INT32_WORDS_PER_FRAGMENT = 4;
};

template <>
struct wmma_a_layout<WmmaOp::M16N8K16, __nv_bfloat16> {};

/*!
 * \brief CUDA Tensor Core B-matrix layout for fp16 m16n8k16 operation
 *
 * Represents a SINGLE m16n8k16 PTX instruction for the B-matrix.
 *
 * - Fragment layout: ColumnMajor with quadrant structure
 * - Tile size: 16 rows (k-dimension) × 8 columns
 * - Total: 32 threads (warp size)
 * - Each thread owns 2 elements from 4 rows × 1 column
 *
 * Thread-to-element ownership pattern:
 * Based on PTX documentation for m16n8k16 B-matrix.
 *
 *   Row\Col:   0   |   1   |  2  |  3  |  4 |  5 |  6 |    7    |
 *          +-------+-------+-----+-----+----+----+----+---------|
 *   Row  0 |T0:{b0}|T4:{b0}|     |     |    |    |    | T28:{b0}|
 *   Row  1 |T0:{b1}|T4:{b1}|     |     |    |    |    | T28:{b1}|
 *          +-------+-------+-----+-----+----+----+----+---------+
 *          |         ...continues for T1,T2...                  |
 *          +-------+-------+-----+-----+----+----+----+---------+
 *   Row  6 |T3:{b0}|T7:{b0}|     |     |    |    |    | T31:{b0}|
 *   Row  7 |T3:{b1}|T7:{b1}|     |     |    |    |    | T31:{b1}|
 *          +-------+-------+-----+-----+----+----+----+---------+
 *   Row  8 |T0:{b2}|T4:{b2}|     |     |    |    |    | T28:{b2}|
 *   Row  9 |T0:{b3}|T4:{b3}|     |     |    |    |    | T28:{b3}|
 *          +-------+-------+-----+-----+----+----+----+---------+
 *          |         ...continues for T1,T2...                  |
 *          +-------+-------+-----+-----+----+----+----+---------+
 *   Row 14 |T3:{b2}|T7:{b2}|     |     |    |    |    | T31:{b2}|
 *   Row 15 |T3:{b3}|T7:{b3}|     |     |    |    |    | T31:{b3}|
 *          +-------+-------+-----+-----+----+----+----+---------+
 *
 * Thread T0 example owns:
 *   - Rows {0,1,8,9}: col 0 (b0,b1,b2,b3)
 *   Total: 4 fp16 values from 4 rows × 1 column
 */
template <>
struct wmma_b_layout<WmmaOp::M16N8K16, half> {
  using DataType = half;
  static constexpr FragmentLayout LAYOUT = FragmentLayout::ColumnMajor;

  static constexpr uint32_t TILE_ROWS = 16;  // K-dimension for m16n8k16
  static constexpr uint32_t TILE_COLS = 8;

  static constexpr uint32_t THREAD_GRID_ROWS = 4;
  static constexpr uint32_t THREAD_GRID_COLS = 8;

  /*! \brief Number of matrix rows owned by each thread in its register fragment */
  static constexpr uint32_t ROWS_PER_FRAGMENT = 4;  // 4 rows from quadrants
  /*! \brief Number of matrix columns owned by each thread in its register fragment */
  static constexpr uint32_t COLS_PER_FRAGMENT = 1;  // 1 column

  /*! \brief Total number of matrix elements owned by each thread (ROWS × COLS) */
  static constexpr uint32_t ELEMS_PER_FRAGMENT = 4;  // 4 rows × 1 col
  static constexpr uint32_t INT32_WORDS_PER_FRAGMENT = 2;
};

template <>
struct wmma_b_layout<WmmaOp::M16N8K16, __nv_bfloat16> {};

/*!
 * \brief CUDA Tensor Core C/D-matrix (accumulator) layout for fp16 m16n8k16 operation
 *
 * Represents a SINGLE m16n8k16 PTX instruction for the C/D-matrix.
 *
 * - Fragment layout: ColumnMajor with quadrant structure
 * - Tile size: 16 rows × 8 columns
 * - Total: 32 threads (warp size)
 * - Each thread owns 4 elements from 2 rows × 2 columns
 *
 * Thread-to-element ownership pattern:
 * Based on PTX documentation for m16n8k16 C/D-matrix (accumulator).
 *
 *   Row\Col:  0   1    |   2   3    |    4  5    |    6  7    |
 *          +-----------+------------+------------+------------+
 *   Row  0 |T0:{c0,c1} | T1:{c0,c1} | T2:{c0,c1} | T3:{c0,c1} |
 *          +-----------+------------+------------+------------+
 *   Row  1 |T4:{c0,c1} | T5:{c0,c1} | T6:{c0,c1} | T7:{c0,c1} |
 *          +-----------+------------+------------+------------+
 *          |         ...continues for T8..T27...              |
 *   Row  7 |T28:{c0,c1}|T29:{c0,c1} |T30:{c0,c1} |T31:{c0,c1} |
 *          +-----------+------------+------------+------------+
 *   Row  8 |T0:{c2,c3} | T1:{c2,c3} | T2:{c2,c3} | T3:{c2,c3} |
 *          +-----------+------------+------------+------------+
 *   Row  9 |T4:{c2,c3} | T5:{c2,c3} | T6:{c2,c3} | T7:{c2,c3} |
 *          +-----------+------------+------------+------------+
 *          |         ...continues for T8..T27...              |
 *   Row 15 |T28:{c2,c3}|T29:{c2,c3} |T30:{c2,c3} |T31:{c2,c3} |
 *          +-----------+------------+------------+------------+
 *
 * Thread T0 example owns:
 *   - Row 0: cols {0,1} (c0,c1)
 *   - Row 8: cols {0,1} (c2,c3)
 *   Total: 4 fp16 values from 2 rows × 2 columns
 */
template <>
struct wmma_c_layout<WmmaOp::M16N8K16, half> {
  using DataType = half;
  static constexpr FragmentLayout LAYOUT = FragmentLayout::ColumnMajor;

  static constexpr uint32_t TILE_ROWS = 16;
  static constexpr uint32_t TILE_COLS = 8;

  static constexpr uint32_t THREAD_GRID_ROWS = 4;
  static constexpr uint32_t THREAD_GRID_COLS = 8;

  /*! \brief Number of matrix rows owned by each thread in its register fragment */
  static constexpr uint32_t ROWS_PER_FRAGMENT = 2;
  /*! \brief Number of matrix columns owned by each thread in its register fragment */
  static constexpr uint32_t COLS_PER_FRAGMENT = 2;

  /*! \brief Total number of matrix elements owned by each thread (ROWS × COLS) */
  static constexpr uint32_t ELEMS_PER_FRAGMENT = 4;  // 2 rows × 2 cols
  static constexpr uint32_t INT32_WORDS_PER_FRAGMENT = 2;
};

template <>
struct wmma_c_layout<WmmaOp::M16N8K16, __nv_bfloat16> {};

// M16N16K16 Specializations (Composite of two M16N8K16 operations)
// ============================================================================

/*!
 * \brief Generic M16N16K16 layout traits for CUDA Tensor Cores
 *
 * Properties uniform across A/B/C matrices for the composite 16×16×16 operation.
 * Note: This is a composite of 2× m16n8k16 PTX operations.
 */
template <>
struct wmma_generic_layout<WmmaOp::M16N16K16, half> {
  using DataType = half;

  // Fragment properties (same for A/B/C on CUDA M16N16K16)
  static constexpr uint32_t FRAG_SIZE = 8;
  static constexpr uint32_t FRAG_BITWIDTH = FRAG_SIZE * sizeof(DataType) * 8;
  static constexpr uint32_t INT32_WORDS_PER_FRAG = 4;

  // Tile dimensions
  static constexpr uint32_t TILE_M = 16;
  static constexpr uint32_t TILE_N = 16;
  static constexpr uint32_t TILE_K = 16;
};

template <>
struct wmma_generic_layout<WmmaOp::M16N16K16, __nv_bfloat16> {};

/*!
 * \brief CUDA Tensor Core A-matrix layout for fp16 m16n16k16 (composite operation)
 *
 * Represents TWO chained m16n8k16 PTX instructions for the A-matrix.
 * FlashInfer composes two m16n8k16 ops to achieve a logical 16×16 output tile.
 *
 * - Fragment layout: RowMajor-like with quadrant replication
 * - Logical tile size: 16 rows × 16 columns (k-dimension)
 * - Total: 32 threads (warp size)
 * - Each thread owns 8 elements from 2 rows × 4 columns
 *
 * See M16N8K16 specialization for detailed thread-to-element mapping.
 * NOTE: A-matrix is IDENTICAL to M16N8K16. The k-dimension is already 16
 * in the base operation. Chaining two m16n8k16 ops doubles the N-dimension
 * (output width 8→16) but does not affect the A-matrix fragment.
 */
template <>
struct wmma_a_layout<WmmaOp::M16N16K16, half> {
  using DataType = half;
  static constexpr FragmentLayout LAYOUT = FragmentLayout::RowMajor;

  static constexpr uint32_t TILE_ROWS = 16;
  static constexpr uint32_t TILE_COLS = 16;  // 2× m16n8k16 k-dimension

  static constexpr uint32_t THREAD_GRID_ROWS = 8;
  static constexpr uint32_t THREAD_GRID_COLS = 4;

  /*! \brief Number of matrix rows owned by each thread in its register fragment */
  static constexpr uint32_t ROWS_PER_FRAGMENT = 2;
  /*! \brief Number of matrix columns owned by each thread in its register fragment */
  static constexpr uint32_t COLS_PER_FRAGMENT = 4;  // 2 base

  /*! \brief Total number of matrix elements owned by each thread (ROWS × COLS) */
  static constexpr uint32_t ELEMS_PER_FRAGMENT = 8;        // 2 rows × 4 cols
  static constexpr uint32_t INT32_WORDS_PER_FRAGMENT = 4;  // 2× base
};

template <>
struct wmma_a_layout<WmmaOp::M16N16K16, __nv_bfloat16> {};

/*!
 * \brief CUDA Tensor Core B-matrix layout for fp16 m16n16k16 (composite operation)
 *
 * Represents TWO chained m16n8k16 PTX instructions for the B-matrix.
 * FlashInfer composes two m16n8k16 ops to achieve a logical 16×16 output tile.
 *
 * - Fragment layout: ColumnMajor with quadrant structure
 * - Logical tile size: 16 rows (k-dimension) × 16 columns
 * - Total: 32 threads (warp size)
 * - Each thread owns 4 elements from 4 rows × 2 columns
 *
 * See M16N8K16 specialization for detailed thread-to-element mapping.
 * The M16N16K16 version uses fragments from two adjacent columns.
 */
template <>
struct wmma_b_layout<WmmaOp::M16N16K16, half> {
  using DataType = half;
  static constexpr FragmentLayout LAYOUT = FragmentLayout::ColumnMajor;

  static constexpr uint32_t TILE_ROWS = 16;  // K-dimension (same as base)
  static constexpr uint32_t TILE_COLS = 16;  // 2× m16n8k16 n-dimension

  static constexpr uint32_t THREAD_GRID_ROWS = 4;
  static constexpr uint32_t THREAD_GRID_COLS = 8;

  /*! \brief Number of matrix rows owned by each thread in its register fragment */
  static constexpr uint32_t ROWS_PER_FRAGMENT = 4;  // 4 rows (same as base)
  /*! \brief Number of matrix columns owned by each thread in its register fragment */
  static constexpr uint32_t COLS_PER_FRAGMENT = 2;  // 2× base

  /*! \brief Total number of matrix elements owned by each thread (ROWS × COLS) */
  static constexpr uint32_t ELEMS_PER_FRAGMENT = 8;        // 4 rows × 2 cols
  static constexpr uint32_t INT32_WORDS_PER_FRAGMENT = 4;  // 2× base
};

template <>
struct wmma_b_layout<WmmaOp::M16N16K16, __nv_bfloat16> {};

/*!
 * \brief CUDA Tensor Core C/D-matrix (accumulator) layout for fp16 m16n16k16 (composite)
 *
 * Represents TWO chained m16n8k16 PTX instructions for the C/D-matrix.
 * FlashInfer composes two m16n8k16 ops to achieve a logical 16×16 output tile.
 *
 * - Fragment layout: ColumnMajor with quadrant structure
 * - Logical tile size: 16 rows × 16 columns
 * - Total: 32 threads (warp size)
 * - Each thread owns 8 elements from 2 rows × 4 columns
 *
 * See M16N8K16 specialization for detailed thread-to-element mapping.
 * The M16N16K16 version doubles the column coverage.
 *
 * Key insight for attention:
 *   - Accumulator S (attention scores) and O (output) use this layout
 *   - Each thread handles 2 output rows with 4 columns total
 */
template <>
struct wmma_c_layout<WmmaOp::M16N16K16, half> {
  using DataType = half;
  static constexpr FragmentLayout LAYOUT = FragmentLayout::ColumnMajor;

  static constexpr uint32_t TILE_ROWS = 16;
  static constexpr uint32_t TILE_COLS = 16;  // 2× m16n8k16 n-dimension

  static constexpr uint32_t THREAD_GRID_ROWS = 4;
  static constexpr uint32_t THREAD_GRID_COLS = 8;

  /*! \brief Number of matrix rows owned by each thread in its register fragment */
  static constexpr uint32_t ROWS_PER_FRAGMENT = 2;
  /*! \brief Number of matrix columns owned by each thread in its register fragment */
  static constexpr uint32_t COLS_PER_FRAGMENT = 4;  // 2× base

  /*! \brief Total number of matrix elements owned by each thread (ROWS × COLS) */
  static constexpr uint32_t ELEMS_PER_FRAGMENT = 8;        // 2 rows × 4 cols
  static constexpr uint32_t INT32_WORDS_PER_FRAGMENT = 4;  // 2× base
};

template <>
struct wmma_c_layout<WmmaOp::M16N16K16, __nv_bfloat16> {};

}  // namespace mma
}  // namespace gpu_iface
}  // namespace flashinfer
