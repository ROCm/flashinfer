/// 1. Allocate a 128x64 memory on CPU and init lexicographically to represent a
///    128X64 matrix.
/// 2. Copy CPU array to global memory.
//  3  Copy global memory into LDS using produce_kv function. The LDS should
//     also of be 128x64 elements
/// 4. Call transpose kernel that inplace transposes the LDS 128x64 matrix into
//     a 64x128 matrix. Each warp handles multiple blocks of 16x16 chunks
/// 5. Post transposition copy back the 128x64 LDS linear memory to global and
///    then back to CPU.
/// 6. Evaluate the output is same as the transpose of the original array.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <iostream>
#include <vector>

#include "flashinfer/attention/generic/permuted_smem.cuh"
#include "flashinfer/attention/generic/prefill.cuh"
#include "gpu_iface/backend/hip/mma_hip.h"
#include "gpu_iface/gpu_runtime_compat.hpp"

using namespace flashinfer;

namespace {

// Define matrix dimensions for the test
constexpr int MATRIX_ROWS = 128;
constexpr int MATRIX_COLS = 64;
constexpr uint32_t KV_THR_LAYOUT_ROW = 4;
constexpr uint32_t KV_THR_LAYOUT_COL = 16;
constexpr uint32_t NUM_WARPS = 4;
constexpr uint32_t NUM_MMA_KV = MATRIX_ROWS / 16;
constexpr uint32_t NUM_WARPS_Q = MATRIX_COLS / 16;
constexpr uint32_t NUM_MMA_D = 4;
constexpr uint32_t UPCAST_STRIDE = 64;
constexpr uint32_t VECTOR_BIT_WIDTH = 64;
constexpr uint32_t CTA_TILE_KV = NUM_MMA_KV * 4 * 16;

using DTypeKV = __half;

template <memory::SharedMemFillMode fill_mode>
__device__ __forceinline__ void load_matrix_global_to_smem(uint32_t warp_idx, uint32_t lane_idx,
                                                           smem_t<SwizzleMode::kLinear, uint2> smem,
                                                           uint32_t* smem_offset, DTypeKV** gptr,
                                                           const uint32_t stride_n,
                                                           const uint32_t kv_idx_base,
                                                           const uint32_t kv_len) {
  static_assert(NUM_MMA_KV * 4 % NUM_WARPS_Q == 0);

  uint32_t kv_idx = kv_idx_base + warp_idx * 4 + lane_idx / KV_THR_LAYOUT_ROW;

#pragma unroll
  for (uint32_t i = 0; i < NUM_MMA_KV * 4 / NUM_WARPS_Q; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < NUM_MMA_D / (8 / sizeof(DTypeKV)); ++j) {
      smem.template load_vector_async<fill_mode>(*smem_offset, *gptr, kv_idx < kv_len);
      *smem_offset = smem.template advance_offset_by_column<16>(*smem_offset, j);
      *gptr += 16 * upcast_size<DTypeKV, VECTOR_BIT_WIDTH>();
    }
    kv_idx += NUM_WARPS * 4;
    *smem_offset = smem.template advance_offset_by_row<NUM_WARPS * 4, UPCAST_STRIDE>(*smem_offset) -
                   (sizeof(DTypeKV) * NUM_MMA_D * 2);
    *gptr += NUM_WARPS * 4 * stride_n -
             sizeof(DTypeKV) * NUM_MMA_D * 2 * upcast_size<DTypeKV, VECTOR_BIT_WIDTH>();
  }
  *smem_offset -= CTA_TILE_KV * UPCAST_STRIDE;
}

}  // namespace

// Helper to initialize matrix with lexicographic values
void initMatrixLexicographic(half* matrix, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix[i * cols + j] = static_cast<half>(i * cols + j);
    }
  }
}

// Helper to transpose a matrix on CPU (for verification)
void transposeMatrixCPU(half* input, half* output, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      output[j * rows + i] = input[i * cols + j];
    }
  }
}

// Helper to print a matrix section (for debugging)
void printMatrixSection(half* matrix, int rows, int cols, const char* name) {
  std::cout << "Matrix " << name << " (" << rows << "x" << cols << "):" << std::endl;
  for (int i = 0; i < std::min(rows, 8); ++i) {
    for (int j = 0; j < std::min(cols, 8); ++j) {
      std::cout << static_cast<float>(matrix[i * cols + j]) << " ";
    }
    std::cout << (cols > 8 ? "..." : "") << std::endl;
  }
  if (rows > 8) std::cout << "..." << std::endl;
}

// Kernel to load the matrix from global to shared memory using produce_kv
__device__ __forceinline__ void loadGlobalToSharedKernel(__half* input,
                                                         smem_t<SwizzleMode::kLinear, uint2> v_smem,
                                                         int rows, int cols) {
  const uint32_t tid = threadIdx.x;
  const uint32_t lane_idx = tid % 64;
  const uint32_t warp_idx = tid / 64;

  uint32_t smem_offset =
      v_smem.template get_permuted_offset<64>(warp_idx * 4 + lane_idx / 16, lane_idx % 16);

  DTypeKV* input_ptr = input + (warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL) * 64 +
                       +(lane_idx % KV_THR_LAYOUT_COL) * upcast_size<DTypeKV, VECTOR_BIT_WIDTH>();

  // Load global memory to shared memory collaboratively
  load_matrix_global_to_smem<SharedMemFillMode::kNoFill>(warp_idx, lane_idx, v_smem, &smem_offset,
                                                         &input_ptr, cols, 0, rows);

  __syncthreads();

  if (tid == 0) {
    printf("\n DEBUG LDS loaded from global\n");
    auto hMem = reinterpret_cast<__half*>(v_smem.base);
    uint32_t offset_r_debug;
    // for (auto i = 0; i < rows; ++i) {
    for (auto j = 0; j < 256; ++j) {
      printf("%f ", float(hMem[j]));
    }
    printf("\n");
    //}
  }

  // TODO: Store shared memory back to global memory for verification
}

// Kernel to transpose shared memory in-place
__global__ void transposeSharedMemoryKernel(half* input, half* output, int rows, int cols) {
  // Define shared memory for the matrix
  extern __shared__ half shared_mem[];
  smem_t<SwizzleMode::kLinear, uint2> v_smem(shared_mem);

  // TODO: Load data from global to shared memory
  loadGlobalToSharedKernel(input, v_smem, rows, cols);

  __syncthreads();

  // TODO: Call transpose_4x4_half_registers to transpose in-place

  __syncthreads();

  // TODO: Copy transposed data back to global memory
}

TEST(InplaceTransposeTest, TestTransposeLDS) {
  // 1. Allocate a 128x64 memory on CPU and init lexicographically
  std::vector<half> h_input(MATRIX_ROWS * MATRIX_COLS);
  std::vector<half> h_output(MATRIX_COLS * MATRIX_ROWS);
  std::vector<half> h_expected(MATRIX_COLS * MATRIX_ROWS);

  initMatrixLexicographic(h_input.data(), MATRIX_ROWS, MATRIX_COLS);

  for (auto i = 0; i < 32; ++i) {
    std::cout << float(h_input[i]) << " ";
  }
  std::cout << std::endl;

  transposeMatrixCPU(h_input.data(), h_expected.data(), MATRIX_ROWS, MATRIX_COLS);

  // 2. Copy CPU array to global memory
  half *d_input, *d_output;
  FI_GPU_CALL(hipMalloc(&d_input, h_input.size() * sizeof(half)));
  FI_GPU_CALL(hipMalloc(&d_output, h_output.size() * sizeof(half)));
  FI_GPU_CALL(
      hipMemcpy(d_input, h_input.data(), h_input.size() * sizeof(half), hipMemcpyHostToDevice));

  // 3 & 4. Load into shared memory and transpose in-place
  const int blockSize = 256;
  const int gridSize = 1;
  size_t sharedMemSize = MATRIX_ROWS * MATRIX_COLS * sizeof(half);

  // Single wave of four wavefronts
  transposeSharedMemoryKernel<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output,
                                                                      MATRIX_ROWS, MATRIX_COLS);

  // 5. Copy back to CPU
  FI_GPU_CALL(
      hipMemcpy(h_output.data(), d_output, h_output.size() * sizeof(half), hipMemcpyDeviceToHost));

  // 6. Verify the output matches the transpose of the original array
  bool all_match = true;
  for (int i = 0; i < MATRIX_COLS * MATRIX_ROWS; ++i) {
    if (static_cast<float>(h_output[i]) != static_cast<float>(h_expected[i])) {
      std::cout << "Mismatch at index " << i << ": " << static_cast<float>(h_output[i]) << " vs "
                << static_cast<float>(h_expected[i]) << std::endl;
      all_match = false;
      if (i > 10) break;  // Limit output
    }
  }

  EXPECT_TRUE(all_match) << "Transposed matrix doesn't match expected result";

  // Clean up
  FI_GPU_CALL(hipFree(d_input));
  FI_GPU_CALL(hipFree(d_output));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
