// test_load_q_global_smem.cpp
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

// Include necessary headers
#include "flashinfer/attention/generic/default_prefill_params.cuh"
#include "flashinfer/attention/generic/prefill.cuh"
#include "flashinfer/attention/generic/variants.cuh"
#include "utils/cpu_reference_hip.h"
#include "utils/utils_hip.h"

using namespace flashinfer;

// CPU Reference Implementation for Q Loading
template <typename DTypeQ>
std::vector<DTypeQ> cpu_reference_q_smem_layout(const std::vector<DTypeQ>& q_global, size_t qo_len,
                                                size_t num_qo_heads, size_t head_dim,
                                                size_t q_stride_n, size_t q_stride_h,
                                                size_t qo_packed_idx_base, uint32_t group_size,
                                                size_t smem_height, size_t smem_width) {
  std::vector<DTypeQ> q_smem_expected(smem_height * smem_width, DTypeQ(0));

  // Simulate the loading pattern that load_q_global_smem should follow
  for (size_t smem_row = 0; smem_row < smem_height; ++smem_row) {
    uint32_t q_packed_idx = qo_packed_idx_base + smem_row;
    uint32_t q_idx = q_packed_idx / group_size;  // Sequence position
    uint32_t r = q_packed_idx % group_size;      // Head offset within group

    if (q_idx < qo_len) {
      for (size_t feat_idx = 0; feat_idx < head_dim; ++feat_idx) {
        // Calculate global memory offset
        size_t global_offset = q_idx * q_stride_n + r * q_stride_h + feat_idx;

        // Place in shared memory layout (assuming linear layout for
        // test)
        size_t smem_offset = smem_row * smem_width + feat_idx;
        if (global_offset < q_global.size()) {
          q_smem_expected[smem_offset] = q_global[global_offset];
        }
      }
    }
  }

  return q_smem_expected;
}

uint_fastdiv create_group_size_div(uint32_t group_size) { return uint_fastdiv(group_size); }

// Test kernel for Q loading
template <typename KTraits>
__global__ void test_q_loading_kernel(typename KTraits::DTypeQ* q_global,
                                      typename KTraits::DTypeQ* q_smem_output,
                                      uint32_t qo_packed_idx_base, uint32_t qo_len,
                                      uint32_t q_stride_n, uint32_t q_stride_h,
                                      uint_fastdiv group_size_div) {
  // Set up shared memory
  extern __shared__ uint8_t smem[];
  typename KTraits::SharedStorage& smem_storage =
      reinterpret_cast<typename KTraits::SharedStorage&>(smem);

  smem_t<KTraits::SWIZZLE_MODE_Q, typename KTraits::SmemBasePtrTy> q_smem(smem_storage.q_smem);

  // Call the function we're testing
  load_q_global_smem<KTraits>(qo_packed_idx_base, qo_len, q_global, q_stride_n, q_stride_h,
                              group_size_div, &q_smem, threadIdx);

  // Synchronize to ensure loading is complete
  __syncthreads();

  if (threadIdx.y == 0 && threadIdx.z == 0) {
    const uint32_t lane_idx = threadIdx.x;
    constexpr uint32_t smem_height = KTraits::CTA_TILE_Q;  // 16
    constexpr uint32_t smem_width = KTraits::HEAD_DIM_QK;  // 64
    constexpr uint32_t total_elements = smem_height * smem_width;

    // Each thread copies using proper swizzled access
    for (uint32_t linear_idx = lane_idx; linear_idx < total_elements;
         linear_idx += KTraits::NUM_THREADS) {
      if (linear_idx < total_elements) {
        uint32_t row = linear_idx / smem_width;
        uint32_t col = linear_idx % smem_width;
        uint32_t swizzled_offset = q_smem.template get_permuted_offset<
            smem_width / upcast_size<typename KTraits::DTypeQ, KTraits::VECTOR_BIT_WIDTH>()>(
            row, col / upcast_size<typename KTraits::DTypeQ, KTraits::VECTOR_BIT_WIDTH>());
        uint32_t element_idx =
            col % upcast_size<typename KTraits::DTypeQ, KTraits::VECTOR_BIT_WIDTH>();
        typename KTraits::DTypeQ* smem_ptr =
            reinterpret_cast<typename KTraits::DTypeQ*>(q_smem.base + swizzled_offset);
        q_smem_output[linear_idx] = smem_ptr[element_idx];
      }
    }
  }
}

// Main test function
template <typename DTypeQ>
bool test_q_loading_correctness() {
  std::cout << "Testing Q loading correctness with " << sizeof(DTypeQ) * 8 << "-bit precision..."
            << std::endl;

  // Test parameters - small sizes for initial validation
  constexpr size_t qo_len = 8;
  constexpr size_t num_qo_heads = 8;
  constexpr size_t num_kv_heads = 2;
  constexpr size_t head_dim = 64;
  constexpr uint32_t group_size = num_qo_heads / num_kv_heads;

  // Create test data with known pattern for easier debugging
  const size_t q_size = qo_len * num_qo_heads * head_dim;
  std::vector<DTypeQ> q_host(q_size);

  // Fill with simple pattern: row*1000 + col for easier validation
  for (size_t i = 0; i < q_size; ++i) {
    float val = float(i % 100) / 10.0f;  // Values 0.0, 0.1, 0.2, ... 9.9
    q_host[i] = fi::con::explicit_casting<float, DTypeQ>(val);
  }

  // GPU memory allocation
  DTypeQ *q_device, *q_smem_output;
  const size_t smem_elements = 16 * head_dim;  // Single MMA block
  FI_GPU_CALL(hipMalloc(&q_device, q_size * sizeof(DTypeQ)));
  FI_GPU_CALL(hipMalloc(&q_smem_output, smem_elements * sizeof(DTypeQ)));

  FI_GPU_CALL(hipMemcpy(q_device, q_host.data(), q_size * sizeof(DTypeQ), hipMemcpyHostToDevice));

  // Define kernel traits for CDNA3
  using KTraits =
      KernelTraits<MaskMode::kNone, 16, 1, 1, 4, 4, 1, 1, PosEncodingMode::kNone, DTypeQ, DTypeQ,
                   DTypeQ, float, uint32_t, DefaultAttention<false, false, false, false>>;

  // Launch parameters
  dim3 block_size(64, 1, 1);  // CDNA3: 64 threads per wavefront
  dim3 grid_size(1, 1, 1);
  size_t shared_mem_size = sizeof(typename KTraits::SharedStorage);

  // Test parameters
  const uint32_t qo_packed_idx_base = 0;  // Start from beginning
  const uint32_t q_stride_n = num_qo_heads * head_dim;
  const uint32_t q_stride_h = head_dim;

  std::cout << "Launching kernel with:" << std::endl;
  std::cout << "  Block size: " << block_size.x << "x" << block_size.y << "x" << block_size.z
            << std::endl;
  std::cout << "  Shared memory: " << shared_mem_size << " bytes" << std::endl;
  std::cout << "  Q size: " << q_size << " elements" << std::endl;

  uint_fastdiv group_size_div = create_group_size_div(group_size);

  // Launch test kernel
  test_q_loading_kernel<KTraits><<<grid_size, block_size, shared_mem_size>>>(
      q_device, q_smem_output, qo_packed_idx_base, qo_len, q_stride_n, q_stride_h, group_size_div);

  FI_GPU_CALL(hipDeviceSynchronize());

  // Get results back
  std::vector<DTypeQ> q_smem_actual(smem_elements);
  FI_GPU_CALL(hipMemcpy(q_smem_actual.data(), q_smem_output, smem_elements * sizeof(DTypeQ),
                        hipMemcpyDeviceToHost));

  // Generate CPU reference
  std::vector<DTypeQ> q_smem_expected =
      cpu_reference_q_smem_layout(q_host, qo_len, num_qo_heads, head_dim, q_stride_n, q_stride_h,
                                  qo_packed_idx_base, group_size, 16, head_dim);

  // Compare results
  bool passed = true;
  float max_diff = 0.0f;
  size_t mismatch_count = 0;

  std::cout << "\nValidation results:" << std::endl;
  std::cout << "Comparing " << q_smem_actual.size() << " elements..." << std::endl;

  for (size_t i = 0; i < std::min(q_smem_actual.size(), q_smem_expected.size()); ++i) {
    float actual = fi::con::explicit_casting<DTypeQ, float>(q_smem_actual[i]);
    float expected = fi::con::explicit_casting<DTypeQ, float>(q_smem_expected[i]);
    float diff = std::abs(actual - expected);
    max_diff = std::max(max_diff, diff);

    if (!utils::isclose(q_smem_actual[i], q_smem_expected[i], 1e-3f, 1e-4f)) {
      if (mismatch_count < 10) {  // Show first 10 mismatches
        size_t row = i / head_dim;
        size_t col = i % head_dim;
        std::cout << "Mismatch at [" << row << "][" << col << "] (index " << i << "): "
                  << "expected " << expected << ", got " << actual << ", diff " << diff
                  << std::endl;
      }
      mismatch_count++;
      passed = false;
    }
  }

  std::cout << "Max difference: " << max_diff << std::endl;
  std::cout << "Total mismatches: " << mismatch_count << " / " << q_smem_actual.size() << std::endl;
  std::cout << "Q loading test: " << (passed ? "PASSED" : "FAILED") << std::endl;

  // Show some sample values for debugging
  if (!passed) {
    std::cout << "\nFirst 10 expected vs actual values:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(10), q_smem_actual.size()); ++i) {
      float actual = fi::con::explicit_casting<DTypeQ, float>(q_smem_actual[i]);
      float expected = fi::con::explicit_casting<DTypeQ, float>(q_smem_expected[i]);
      std::cout << "[" << i << "] expected: " << expected << ", actual: " << actual << std::endl;
    }
  }

  // Cleanup
  FI_GPU_CALL(hipFree(q_device));
  FI_GPU_CALL(hipFree(q_smem_output));

  return passed;
}

// Main function
int main() {
  std::cout << "=== FlashInfer Q Loading Component Test ===" << std::endl;
  std::cout << "Testing load_q_global_smem function for CDNA3 architecture" << std::endl;

  // Initialize HIP
  hipError_t err = hipSetDevice(0);
  if (err != hipSuccess) {
    std::cout << "Failed to set HIP device: " << hipGetErrorString(err) << std::endl;
    return 1;
  }

  hipDeviceProp_t prop;
  FI_GPU_CALL(hipGetDeviceProperties(&prop, 0));
  std::cout << "Running on: " << prop.name << std::endl;

  bool all_passed = true;

  // Test with half precision
  std::cout << "\n--- Testing with FP16 ---" << std::endl;
  all_passed &= test_q_loading_correctness<__half>();

  if (all_passed) {
    std::cout << "\n✅ All Q loading tests PASSED!" << std::endl;
    return 0;
  } else {
    std::cout << "\n❌ Some Q loading tests FAILED!" << std::endl;
    return 1;
  }
}
