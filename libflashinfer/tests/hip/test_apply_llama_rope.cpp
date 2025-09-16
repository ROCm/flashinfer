// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#include <gtest/gtest.h>

#include <tuple>
#include <vector>

#include "../../utils/cpu_reference_hip.h"
#include "../../utils/utils_hip.h"
#include "flashinfer/attention/generic/prefill.cuh"
#include "gpu_iface/fastdiv.cuh"
#include "gpu_iface/gpu_runtime_compat.hpp"

namespace {
using QParamType = std::tuple<uint32_t, uint32_t, uint32_t>;

template <uint32_t HEAD_DIM>
struct TestKernelTraits {
  static constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM / 16;
  static constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM / 16;
};

template <uint32_t HEAD_DIM>
__global__ void test_init_rope_freq_kernel(float* output_freq, float rope_rcp_scale,
                                           float rope_rcp_theta) {
  using KTraits = TestKernelTraits<HEAD_DIM>;

  // Allocate local frequency array
  float rope_freq[KTraits::NUM_MMA_D_VO / 2][4];  // [2][4] for HEAD_DIM=64

  // Call the init_rope_freq function from prefill.cuh
  flashinfer::init_rope_freq<KTraits>(rope_freq, rope_rcp_scale, rope_rcp_theta, threadIdx.x);

  // Write frequencies to their correct feature indices
  const uint32_t lane_idx = threadIdx.x;
  if (lane_idx < 64) {  // Only write for valid threads
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO / 2; ++mma_d) {
      for (uint32_t j = 0; j < 4; ++j) {
        // Calculate the actual feature index this frequency corresponds
        // to
        uint32_t feature_idx = flashinfer::get_feature_index<HEAD_DIM>(mma_d, lane_idx, j);

        // Write frequency to the correct feature index in global array
        if (feature_idx < HEAD_DIM) {
          output_freq[feature_idx] = rope_freq[mma_d][j];
          if (feature_idx + HEAD_DIM / 2 < HEAD_DIM) {
            output_freq[feature_idx + HEAD_DIM / 2] = rope_freq[mma_d][j];
          }
        }
      }
    }
  }
}

template <uint32_t HEAD_DIM>
__global__ void test_q_frag_apply_llama_rope_kernel(__half* q_input, __half* q_output,
                                                    uint32_t qo_len, uint32_t num_qo_heads,
                                                    uint32_t kv_len, float rope_rcp_scale,
                                                    float rope_rcp_theta,
                                                    flashinfer::uint_fastdiv group_size_fastdiv) {
  using KTraits = TestKernelTraits<HEAD_DIM>;
  constexpr uint32_t HALF_ELEMS_PER_THREAD = 4;
  constexpr uint32_t INT32_ELEMS_PER_THREAD = 2;
  constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM / 16;

  float rope_freq[KTraits::NUM_MMA_D_VO / 2][4];
  flashinfer::init_rope_freq<KTraits>(rope_freq, rope_rcp_scale, rope_rcp_theta, threadIdx.x);

  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warp_idx = blockIdx.x;

  // TODO: Need to check that qo_len is evenly divisible by 16.
  for (uint32_t qo_head_idx = 0; qo_head_idx < num_qo_heads; ++qo_head_idx) {
    for (uint32_t seq_chunk = 0; seq_chunk < qo_len; seq_chunk += 16) {
      uint32_t seq_idx = seq_chunk + (lane_idx % 16);
      if (seq_idx >= qo_len) continue;

      uint32_t abs_position = seq_idx + kv_len - qo_len;
      // Each iteration processes 16*2=32 features (first_half +
      // second_half)
      for (uint32_t feat_chunk = 0; feat_chunk < NUM_MMA_D_QK / 2; ++feat_chunk) {
        uint32_t feat_offset_first = feat_chunk * 32;
        uint32_t feat_offset_second = feat_offset_first + HEAD_DIM / 2;

        // Load fragments from global memory
        __half q_frag_first[HALF_ELEMS_PER_THREAD];
        __half q_frag_second[HALF_ELEMS_PER_THREAD];

        // Calculate base address for this sequence and head
        uint32_t base_offset = qo_head_idx * HEAD_DIM + seq_idx * (num_qo_heads * HEAD_DIM);

        // Load first half (4 consecutive features per thread)
        for (uint32_t i = 0; i < HALF_ELEMS_PER_THREAD; ++i) {
          uint32_t feat_idx1 = flashinfer::get_feature_index<HEAD_DIM>(feat_chunk, lane_idx, i);
          uint32_t feat_idx2 = feat_idx1 + HEAD_DIM / 2;
          q_frag_first[i] = *(q_input + base_offset + feat_idx1);
          q_frag_second[i] = *(q_input + base_offset + feat_idx2);
        }

        // Apply RoPE using the validated function
        uint32_t mma_di = feat_chunk;
        flashinfer::q_frag_apply_llama_rope<__half, HALF_ELEMS_PER_THREAD>(
            q_frag_first, q_frag_second, rope_freq[mma_di % (KTraits::NUM_MMA_D_VO / 2)],
            abs_position, group_size_fastdiv);

        // Store results back to global memory
        for (uint32_t i = 0; i < HALF_ELEMS_PER_THREAD; ++i) {
          uint32_t feat_idx1 = flashinfer::get_feature_index<HEAD_DIM>(feat_chunk, lane_idx, i);
          uint32_t feat_idx2 = feat_idx1 + HEAD_DIM / 2;
          *(q_output + base_offset + feat_idx1) = q_frag_first[i];
          *(q_output + base_offset + feat_idx2) = q_frag_second[i];
        }
      }
    }
  }
}

template <typename DTypeQ>
class LLamaRopeTestFixture : public ::testing::TestWithParam<QParamType> {
 protected:
  uint32_t qo_len, num_qo_heads, head_dim;
  std::vector<DTypeQ> q;

  LLamaRopeTestFixture() {
    const auto& params = GetParam();
    qo_len = std::get<0>(params);
    num_qo_heads = std::get<1>(params);
    head_dim = std::get<2>(params);
    q.resize(qo_len * num_qo_heads * head_dim);
  }

  void SetUp() override { utils::vec_normal_(q); }

  void TearDown() override {}

  std::vector<float> apply_cpu_rope(size_t offset, float rope_scale = 1.0f,
                                    float rope_theta = 10000.0f) {
    return cpu_reference::apply_llama_rope(q.data(), head_dim, offset, rope_scale, rope_theta);
  }

  std::vector<float> get_cpu_rope_frequencies(float rope_scale = 1.0f,
                                              float rope_theta = 10000.0f) {
    std::vector<float> frequencies(head_dim);

    for (size_t k = 0; k < head_dim; ++k) {
      // Extract ONLY the frequency calculation (without position/offset)
      float freq_base = float(2 * (k % (head_dim / 2))) / float(head_dim);
      float frequency = (1.0f / rope_scale) / std::pow(rope_theta, freq_base);
      frequencies[k] = frequency;
    }

    return frequencies;
  }

  std::vector<float> get_gpu_rope_frequencies(float rope_scale = 1.0f,
                                              float rope_theta = 10000.0f) {
    // Convert to reciprocal values as expected by GPU kernel
    float rope_rcp_scale = 1.0f / rope_scale;
    float rope_rcp_theta = 1.0f / rope_theta;

    // Allocate GPU memory for output (one frequency per feature)
    float* d_output_freq;
    size_t output_size = head_dim * sizeof(float);
    FI_GPU_CALL(hipMalloc(&d_output_freq, output_size));
    FI_GPU_CALL(hipMemset(d_output_freq, 0, output_size));

    // Launch kernel with 64 threads
    dim3 grid(1);
    dim3 block(64);

    if (head_dim == 64) {
      test_init_rope_freq_kernel<64>
          <<<grid, block>>>(d_output_freq, rope_rcp_scale, rope_rcp_theta);
    }

    FI_GPU_CALL(hipDeviceSynchronize());

    // Copy all frequencies back
    std::vector<float> gpu_frequencies(head_dim);
    FI_GPU_CALL(
        hipMemcpy(gpu_frequencies.data(), d_output_freq, output_size, hipMemcpyDeviceToHost));

    FI_GPU_CALL(hipFree(d_output_freq));
    return gpu_frequencies;
  }

  std::vector<std::vector<float>> apply_cpu_rope_all_sequences(size_t kv_len = 1000,
                                                               float rope_scale = 1.0f,
                                                               float rope_theta = 10000.0f) {
    std::vector<std::vector<float>> results;

    DISPATCH_head_dim(head_dim, HEAD_DIM, {
      using namespace flashinfer;
      tensor_info_t info(qo_len, kv_len, num_qo_heads, num_qo_heads, QKVLayout::kHND, HEAD_DIM);

      // Apply RoPE to all sequences and heads
      for (size_t qo_head_idx = 0; qo_head_idx < num_qo_heads; ++qo_head_idx) {
        for (size_t q_idx = 0; q_idx < qo_len; ++q_idx) {
          size_t offset = q_idx + kv_len - qo_len;

          // Apply RoPE to this specific Q sequence/head
          auto q_rotary_local = cpu_reference::apply_llama_rope_debug(
              q.data() + info.get_q_elem_offset(q_idx, qo_head_idx, 0), head_dim, offset,
              rope_scale, rope_theta);

          results.push_back(std::move(q_rotary_local));
        }
      }
    });

    return results;
  }

  std::vector<float> test_gpu_q_frag_apply_rope(size_t kv_len = 1000, float rope_scale = 1.0f,
                                                float rope_theta = 10000.0f) {
    // Convert to reciprocal values
    float rope_rcp_scale = 1.0f / rope_scale;
    float rope_rcp_theta = 1.0f / rope_theta;
    uint32_t group_size = 1;  // Simple case for now

    // Allocate GPU memory for input and output
    __half *d_q_input, *d_q_output;
    size_t q_size = q.size() * sizeof(__half);

    FI_GPU_CALL(hipMalloc(&d_q_input, q_size));
    FI_GPU_CALL(hipMalloc(&d_q_output, q_size));

    // Copy input Q to GPU
    FI_GPU_CALL(hipMemcpy(d_q_input, q.data(), q_size, hipMemcpyHostToDevice));
    FI_GPU_CALL(hipMemset(d_q_output, 0, q_size));

    // Launch kernel - one block with 64 threads
    dim3 grid(1);    // Single block for simplicity
    dim3 block(64);  // CDNA3 wavefront size

    if (head_dim == 64) {
      test_q_frag_apply_llama_rope_kernel<64><<<grid, block>>>(d_q_input, d_q_output, qo_len,
                                                               num_qo_heads, kv_len, rope_rcp_scale,
                                                               rope_rcp_theta, group_size);
    }

    FI_GPU_CALL(hipDeviceSynchronize());

    // Copy results back to CPU
    std::vector<__half> gpu_output(q.size());
    FI_GPU_CALL(hipMemcpy(gpu_output.data(), d_q_output, q_size, hipMemcpyDeviceToHost));

    // Convert to float for comparison
    std::vector<float> result(head_dim);
    for (size_t i = 0; i < head_dim; ++i) {
      result[i] = float(gpu_output[i]);  // First sequence, first head
    }

    FI_GPU_CALL(hipFree(d_q_input));
    FI_GPU_CALL(hipFree(d_q_output));

    return result;
  }
};

using LLamaRopeTestWithFP16 = LLamaRopeTestFixture<__half>;
}  // namespace

// Wrapper to validate freq application
// call q_smem_inplace_apply_rotary and copy back results to CPU.

// Test 1. Copy CPU Q matrix to GPU call freq init validator
// launch kernel

// Test 2. Copy CPU Q matrix to GPU call freq apply validator
// launch kernel

TEST_P(LLamaRopeTestWithFP16, TestInitRopeFreq) {
  constexpr float RELATIVE_EPSILON = 1e-6f;
  size_t num_mismatches = 0;
  auto cpu_frequencies = this->get_cpu_rope_frequencies();
  auto gpu_frequencies = this->get_gpu_rope_frequencies();

  // Print side-by-side comparison for easier visual inspection
  std::cout << "\nSide-by-side comparison:\n";
  std::cout << "Index\tCPU\t\tGPU\t\tDifference\n";
  std::cout << "-----\t---\t\t---\t\t----------\n";

  for (size_t i = 0; i < std::min(16u, this->head_dim); ++i) {
    float diff = std::abs(cpu_frequencies[i] - gpu_frequencies[i]);
    std::cout << i << "\t" << cpu_frequencies[i] << "\t\t" << gpu_frequencies[i] << "\t\t" << diff
              << std::endl;
  }

  ASSERT_EQ(cpu_frequencies.size(), this->head_dim);
  ASSERT_EQ(gpu_frequencies.size(), this->head_dim);

  for (auto i = 0ul; i < cpu_frequencies.size(); ++i) {
    auto diff = std::abs(cpu_frequencies[i] - gpu_frequencies[i]);
    if (diff >= RELATIVE_EPSILON) {
      std::cout << "Diff : " << diff << " at feature index " << i << " "
                << "cpu_frequencies[i]: " << cpu_frequencies[i] << " "
                << "gpu_frequencies[i]: " << gpu_frequencies[i] << '\n';
      ++num_mismatches;
    }
  }

  ASSERT_EQ(num_mismatches, 0);
}

TEST_P(LLamaRopeTestWithFP16, VectorSizeIsCorrect) {
  const auto& params = GetParam();
  size_t expected_size = std::get<0>(params) * std::get<1>(params) * std::get<2>(params);
  ASSERT_EQ(this->q.size(), expected_size);
}

TEST_P(LLamaRopeTestWithFP16, TestQFragApplyRopeComparison) {
  constexpr float RELATIVE_EPSILON = 1e-2f;

  auto cpu_result = this->apply_cpu_rope(744);
  auto gpu_result = this->test_gpu_q_frag_apply_rope();

  std::cout << "\n=== CPU vs GPU RoPE Application Comparison ===\n";
  std::cout << "CPU result (offset=1000, first 8 features): ";
  for (size_t i = 0; i < std::min(8u, this->head_dim); ++i) {
    std::cout << cpu_result[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "GPU result (offset=1000, first 8 features): ";
  for (size_t i = 0; i < std::min(8u, this->head_dim); ++i) {
    std::cout << gpu_result[i] << " ";
  }
  std::cout << std::endl;

  // Compare element by element
  size_t num_mismatches = 0;
  for (size_t i = 0; i < std::min(cpu_result.size(), gpu_result.size()); ++i) {
    float diff = std::abs(cpu_result[i] - gpu_result[i]);
    float rel_diff = (std::abs(cpu_result[i]) > 1e-6f) ? diff / std::abs(cpu_result[i]) : diff;

    if (rel_diff > RELATIVE_EPSILON) {
      std::cout << "Mismatch at feature " << i << ": CPU=" << cpu_result[i]
                << " GPU=" << gpu_result[i] << " diff=" << diff << " rel_diff=" << rel_diff
                << std::endl;
      ++num_mismatches;
    }
  }

  std::cout << "Total mismatches: " << num_mismatches << " out of " << head_dim << std::endl;

  EXPECT_EQ(num_mismatches, 0) << "Found mismatches between CPU and GPU RoPE application";
}

INSTANTIATE_TEST_SUITE_P(
    LLamaRopeTestWithFP16, LLamaRopeTestWithFP16,
    ::testing::Values(std::make_tuple(256, 1, 64)  // qo_len=256, num_qo_heads=1, head_dim=64
                      ));
