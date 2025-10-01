
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "gpu_iface/mma_ops.hpp"

// Check HIP errors
#define HIP_CHECK(command)                                                          \
  {                                                                                 \
    hipError_t status = command;                                                    \
    if (status != hipSuccess) {                                                     \
      std::cerr << "Error: HIP reports " << hipGetErrorString(status) << std::endl; \
      std::cerr << "at " << __FILE__ << ":" << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

namespace {

__device__ void print_register(uint32_t* R) {
  auto values = reinterpret_cast<__half*>(R);
  printf("[%f %f %f %f]\n", __half2float(values[0]), __half2float(values[1]),
         __half2float(values[2]), __half2float(values[3]));
}

__device__ void print_register(float* R) { printf("[%f %f %f %f]\n", R[0], R[1], R[3], R[4]); }

}  // namespace

// Dimensions for our test matrices
constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;

// Layout dimensions for accessing matrices
constexpr int LDA = K;
constexpr int LDB = N;
constexpr int LDC = N;

// Host reference implementation for matrix multiplication
void gemm_reference(const __half* A, const __half* B, float* C, int M, int N, int K, int lda,
                    int ldb, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
      }
      C[i * N + j] = acc;
    }
  }
}

__global__ void test_mfma_kernel(const __half* A, const __half* B, float* C) {
  uint32_t a_reg[2];
  uint32_t b_reg[2];
  float c_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  int a_idx = (threadIdx.x % 16) * LDA + (threadIdx.x / 16) * 4;
  int b_idx = ((threadIdx.x % 4) + 4 * (threadIdx.x / 16)) * LDB + ((threadIdx.x % 16) / 4) * 4;

  flashinfer::gpu_iface::mma::load_fragment<__half>(a_reg, &A[a_idx]);
  flashinfer::gpu_iface::mma::load_fragment_transpose_4x4_half_registers<__half>(b_reg, &B[b_idx]);
  flashinfer::gpu_iface::mma::mma_sync_m16n16k16_row_col_f16f16f32<__half>(c_reg, a_reg, b_reg);

  for (int i = 0; i < 4; ++i) {
    int d_idx = ((threadIdx.x / 16) * 4 + i) * LDC + (threadIdx.x % 16);
    C[d_idx] = c_reg[i];
  }
}

// Test class
class MfmaTest : public ::testing::Test {
 protected:
  std::vector<__half> A_host;
  std::vector<__half> B_host;
  std::vector<float> C_host;
  std::vector<float> C_ref;

  __half* A_dev = nullptr;
  __half* B_dev = nullptr;
  float* C_dev = nullptr;

  void SetUp() override {
    // Initialize host data
    A_host.resize(M * K);
    B_host.resize(K * N);
    C_host.resize(M * N, 0.0f);
    C_ref.resize(M * N, 0.0f);

    // Fill with deterministic values
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < M * K; ++i) {
      A_host[i] = __float2half(dist(gen));
    }

    for (int i = 0; i < K * N; ++i) {
      B_host[i] = __float2half(dist(gen));
    }

    // Calculate reference result
    gemm_reference(A_host.data(), B_host.data(), C_ref.data(), M, N, K, LDA, LDB, LDC);

    // Allocate device memory
    HIP_CHECK(hipMalloc(&A_dev, M * K * sizeof(__half)));
    HIP_CHECK(hipMalloc(&B_dev, K * N * sizeof(__half)));
    HIP_CHECK(hipMalloc(&C_dev, M * N * sizeof(float)));

    // Copy input data to device
    HIP_CHECK(hipMemcpy(A_dev, A_host.data(), M * K * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_dev, B_host.data(), K * N * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(C_dev, 0, M * N * sizeof(float)));
  }

  void TearDown() override {
    // Free device memory
    HIP_CHECK(hipFree(A_dev));
    HIP_CHECK(hipFree(B_dev));
    HIP_CHECK(hipFree(C_dev));
  }
};

// Test that verifies mfma_fp32_16x16x16fp16 calculates correct results
TEST_F(MfmaTest, CorrectResults) {
  // Launch kernel with one block of 64 threads (one wavefront)
  dim3 gridDim(1);
  dim3 blockDim(64);
  test_mfma_kernel<<<gridDim, blockDim>>>(A_dev, B_dev, C_dev);

  // Copy results back to host
  HIP_CHECK(hipMemcpy(C_host.data(), C_dev, M * N * sizeof(float), hipMemcpyDeviceToHost));

  // Verify results with small tolerance for floating point differences
  const float tolerance = 1e-3f;
  bool all_pass = true;
  for (int i = 0; i < M * N; ++i) {
    float diff = std::abs(C_host[i] - C_ref[i]);
    if (diff > tolerance) {
      std::cout << "Mismatch at index " << i << ": "
                << "Actual=" << C_host[i] << ", Expected=" << C_ref[i] << ", Diff=" << diff
                << std::endl;
      all_pass = false;
    }
  }

  EXPECT_TRUE(all_pass) << "Matrix multiplication results don't match reference implementation";
}

// Main function that runs all tests
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
