// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace flashinfer {
namespace gpu_iface {
namespace mma {

/*!
 * \brief Fragment layout enumeration
 *
 * Describes how matrix elements are distributed across threads in a fragment.
 * - RowMajor: Threads primarily own elements from the same row(s)
 * - ColumnMajor: Threads primarily own elements from the same column(s)
 */
enum class FragmentLayout {
  RowMajor,
  ColumnMajor,
};

/*!
 * \brief WMMA operation type enumeration
 *
 * Specifies the logical MMA operation dimensions.
 * Backend implementations may compose multiple hardware operations to achieve
 * the specified logical operation.
 *
 * Format: MxNxK where:
 * - M: Output rows
 * - N: Output columns
 * - K: Reduction dimension
 */
enum class WmmaOp {
  M16N8K16,   // Single 16×8 output tile, k-dim=16
  M16N16K16,  // Logical 16×16 output tile, k-dim=16 (may be composite)
};

/*! * \brief Generic WMMA layout providing generic op-level traits
 *
 * Defines properties that are uniform across A/B/C matrices for a given
 * WMMA operation. This provides a clean way to access common constants
 * without arbitrarily choosing A, B, or C layout.
 *
 * \tparam DType Data type (half, hip_bfloat16, __nv_bfloat16)
 * \tparam Op WMMA operation type (M16N8K16, M16N16K16)
 */
template <WmmaOp Op, typename DType>
struct wmma_generic_layout;

/*!
 * \brief Base template for A-matrix (left operand) layout in WMMA operations
 *
 * Defines the fragment layout for the A-matrix
 * A-matrix has dimensions M×K, with row-major distribution for efficient
 * row-wise accumulation.
 *
 * Usage in attention kernels:
 * - Q matrix: Loaded directly, uses A-layout for Q×K^T
 * - K matrix: K is loaded using A-layout loads (without transpose). When used
 *             in Q×K^T multiplication, these K fragments serve as the transposed
 *             operand K^T. Both Q and K fragments use A-matrix layout.
 *
 * \tparam DType Data type (half, __nv_bfloat16, etc.)
 * \tparam Op WMMA operation type (M16N8K16, M16N16K16, etc.)
 */
template <WmmaOp Op, typename DType>
struct wmma_a_layout;

/*!
 * \brief Base template for B-matrix (right operand) layout in WMMA operations
 *
 * Defines the fragment layout for the B-matrix (typically V in attention).
 * B-matrix has dimensions K×N, with column-major distribution.
 *
 * Usage in attention kernels:
 * - V for S×V: V loaded with transpose to match B-layout column distribution
 *
 * \tparam DType Data type (half, __nv_bfloat16, etc.)
 * \tparam Op WMMA operation type (M16N8K16, M16N16K16, etc.)
 */
template <WmmaOp Op, typename DType>
struct wmma_b_layout;

/*!
 * \brief Base template for C-matrix (accumulator) layout in WMMA operations
 *
 * Defines the fragment layout for the C/D-matrix (accumulator/output).
 * C-matrix has dimensions M×N, typically column-major for efficient column-wise
 * operations.
 *
 * Usage in attention kernels:
 * - S (attention scores): Accumulator from Q×K^T, uses C-layout
 * - O (output): Accumulator from S×V, uses C-layout
 *
 * \tparam DType Data type (half, __nv_bfloat16, etc.)
 * \tparam Op WMMA operation type (M16N8K16, M16N16K16, etc.)
 */
template <WmmaOp Op, typename DType>
struct wmma_c_layout;

}  // namespace mma
}  // namespace gpu_iface
}  // namespace flashinfer

// Include platform-specific implementations
#if defined(PLATFORM_HIP_DEVICE)
#include "backend/hip/mma_layout_hip.h"
#elif defined(PLATFORM_CUDA_DEVICE)
#include "backend/cuda/mma_layout_cuda.h"
#endif
