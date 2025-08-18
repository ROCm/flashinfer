
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#include "gpu_iface/mma_ops.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#define HIP_ENABLE_WARP_SYNC_BUILTINS 1

using float16_t = _Float16;
using float16x4 =
    __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;
using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

// Check HIP errors
#define HIP_CHECK(command)                                                     \
    {                                                                          \
        hipError_t status = command;                                           \
        if (status != hipSuccess) {                                            \
            std::cerr << "Error: HIP reports " << hipGetErrorString(status)    \
                      << std::endl;                                            \
            std::cerr << "at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

// Dimensions for our test matrices
constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 16;

// Layout dimensions for accessing matrices
constexpr int LDA = K;
constexpr int LDB = N;
constexpr int LDC = N;

// Host reference implementation for matrix multiplication
void gemm_reference(const __half *A,
                    const __half *B,
                    float *C,
                    int M,
                    int N,
                    int K,
                    int lda,
                    int ldb,
                    int ldc)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                // Use __half_as_float to properly convert __half to float
                acc += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            }
            C[i * N + j] = acc;
        }
    }
}

__device__ __forceinline__ void transpose_4x4_half_registers(uint32_t *R,
                                                             uint32_t *out)
{
    // Assuming each thread has 4 half-precision values packed in 2 registers:
    // R[0] = [B[i][0], B[i][1]] (two halfs)
    // R[1] = [B[i][2], B[i][3]] (two halfs)

    // Calculate group mask (4 consecutive threads)
    uint32_t lane_id = threadIdx.x % 64;     // Lane ID in wavefront
    uint32_t group_id = lane_id / 4;         // Which 4-thread group
    uint32_t group_base = group_id * 4;      // Base lane ID for this group
    uint32_t group_mask = 0xF << group_base; // Mask for this 4-thread group

    // Step 1: Transpose 2x2 blocks using shuffles within 2-thread groups
    uint32_t tmp;

    // Shuffle between thread pairs (0-1, 2-3)
    tmp = __shfl_xor(group_mask, R[0], 1);
    uint32_t reg0 =
        (lane_id & 1) ? ((R[0] & 0xFFFF0000) | (tmp & 0x0000FFFF))
                      : // Thread 1: Keep high, get low from T0
            ((R[0] & 0x0000FFFF) |
             (tmp & 0xFFFF0000)); // Thread 0: Keep low, get high from T1

    tmp = __shfl_xor(group_mask, R[1], 1);
    uint32_t reg1 = (lane_id & 1) ? ((R[1] & 0xFFFF0000) | (tmp & 0x0000FFFF))
                                  : ((R[1] & 0x0000FFFF) | (tmp & 0xFFFF0000));

    // Step 2: Shuffle between thread pairs to complete the transpose
    tmp = __shfl_xor(group_mask, reg0, 2);
    out[0] = (lane_id & 2) ? // Thread 2-3 get high parts
                 ((reg0 & 0xFFFF0000) | (tmp & 0x0000FFFF))
                           : ((reg0 & 0x0000FFFF) | (tmp & 0xFFFF0000));

    tmp = __shfl_xor(group_mask, reg1, 2);
    out[1] = (lane_id & 2) ? ((reg1 & 0xFFFF0000) | (tmp & 0x0000FFFF))
                           : ((reg1 & 0x0000FFFF) | (tmp & 0xFFFF0000));
}

template <typename T>
__device__ __forceinline__ void load_fragment_transpose_v2(uint32_t *R,
                                                           const T *smem_ptr)
{
    // Each thread loads 4 elements from its row, using stride properly
    const uint16_t *v0 = reinterpret_cast<const uint16_t *>(smem_ptr) + 0;
    const uint16_t *v1 = reinterpret_cast<const uint16_t *>(++smem_ptr);
    const uint16_t *v2 = reinterpret_cast<const uint16_t *>(++smem_ptr);
    const uint16_t *v3 = reinterpret_cast<const uint16_t *>(++smem_ptr);

    R[0] = (static_cast<const uint32_t>(*v0) << 16) |
           static_cast<const uint32_t>(*v1);
    R[1] = (static_cast<const uint32_t>(*v2) << 16) |
           static_cast<const uint32_t>(*v3);

    uint32_t out[2];
    transpose_4x4_half_registers(R, out);

    float16x4 out_f16x4 = reinterpret_cast<float16x4 *>(out)[0];

    if (threadIdx.x == 0) {
        printf("out_f16x4[0] = %f\n", (float)(out_f16x4[0]));
        printf("out_f16x4[1] = %f\n", (float)(out_f16x4[1]));
        printf("out_f16x4[2] = %f\n", (float)(out_f16x4[2]));
        printf("out_f16x4[3] = %f\n", (float)(out_f16x4[3]));
    }
}

__global__ void test_mfma_kernel(const __half *A, const __half *B, float *C)
{
    uint32_t a_reg[2];
    uint32_t b_reg[2];
    float c_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // A Matrix is read row wise. Threads T0...T15 read Col 0...3 of Row 0...15
    // Threads T16...T31 read Col 4...7 of Row 0...15
    // Threads T32...T47 read Col 8...11 of Row 0...15
    // Threads T48...T63 read Col 12...15 of Row 0...15

    // B Matrix is read column wise. Threads T0...T15 read Row 0...3 of Col
    // 0...15 (Each thread reads 1 column per 4 rows) Threads T16...T31 read
    // Row 4...7 of Col 0...15 Threads T32...T47 read Row 8...11 of Col 0...15
    // Threads T48...T63 read Row 12...15 of Col 0...15
    int a_idx = (threadIdx.x / 16) * 4 + threadIdx.x % 16 * LDA;
    int b_idx = (threadIdx.x / 16) * LDB * 4 + threadIdx.x % 16;

    load_fragment_transpose_v2<__half>(a_reg, &A[a_idx]);

    // flashinfer::gpu_iface::mma::load_fragment_transpose<__half>(b_reg,
    //                                                             &B[b_idx],
    //                                                             LDB);

    // flashinfer::gpu_iface::mma::amdgcn_mfma_fp32_16x16x16fp16<__half>(
    //     c_reg, a_reg, b_reg);

    // for (int i = 0; i < 4; ++i) {
    //     const int d_idx =
    //         threadIdx.x % 16 + i * LDC + (threadIdx.x / 16) * 4 * LDC;

    //     C[d_idx] = c_reg[i];
    // }
}

// Test class
class MfmaTest : public ::testing::Test
{
protected:
    std::vector<__half> A_host;
    std::vector<__half> B_host;
    std::vector<float> C_host;
    std::vector<float> C_ref;

    __half *A_dev = nullptr;
    __half *B_dev = nullptr;
    float *C_dev = nullptr;

    void SetUp() override
    {
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

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < K; ++j) {
                std::cout << __half2float(A_host[i * K + j]) << " ";
            }
            std::cout << std::endl;
        }

        for (int i = 0; i < K * N; ++i) {
            B_host[i] = __float2half(dist(gen));
        }

        // Calculate reference result
        gemm_reference(A_host.data(), B_host.data(), C_ref.data(), M, N, K, LDA,
                       LDB, LDC);

        // Allocate device memory
        HIP_CHECK(hipMalloc(&A_dev, M * K * sizeof(__half)));
        HIP_CHECK(hipMalloc(&B_dev, K * N * sizeof(__half)));
        HIP_CHECK(hipMalloc(&C_dev, M * N * sizeof(float)));

        // Copy input data to device
        HIP_CHECK(hipMemcpy(A_dev, A_host.data(), M * K * sizeof(__half),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(B_dev, B_host.data(), K * N * sizeof(__half),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(C_dev, 0, M * N * sizeof(float)));
    }

    void TearDown() override
    {
        // Free device memory
        HIP_CHECK(hipFree(A_dev));
        HIP_CHECK(hipFree(B_dev));
        HIP_CHECK(hipFree(C_dev));
    }
};

// Test that verifies mfma_fp32_16x16x16fp16 calculates correct results
TEST_F(MfmaTest, CorrectResults)
{
    // Launch kernel with one block of 64 threads (one wavefront)
    dim3 gridDim(1);
    dim3 blockDim(64);
    test_mfma_kernel<<<gridDim, blockDim>>>(A_dev, B_dev, C_dev);

    // Copy results back to host
    HIP_CHECK(hipMemcpy(C_host.data(), C_dev, M * N * sizeof(float),
                        hipMemcpyDeviceToHost));

    // Verify results with small tolerance for floating point differences
    const float tolerance = 1e-3f;
    bool all_pass = true;
    for (int i = 0; i < M * N; ++i) {
        float diff = std::abs(C_host[i] - C_ref[i]);
        if (diff > tolerance) {
            // std::cout << "Mismatch at index " << i << ": "
            //           << "Actual=" << C_host[i] << ", Expected=" << C_ref[i]
            //           << ", Diff=" << diff << std::endl;
            all_pass = false;
        }
    }

    EXPECT_TRUE(all_pass)
        << "Matrix multiplication results don't match reference implementation";
}

// Main function that runs all tests
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
