// SPDX-FileCopyrightText: 2023-2025 Flashinfer team
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <type_traits>

#include "../../utils/cpu_reference_hip.h"
#include "../../utils/flashinfer_prefill_ops_hip.h"
#include "../../utils/utils_hip.h"
#include "flashinfer/attention/generic/prefill.cuh"
#include "gpu_iface/gpu_runtime_compat.hpp"

#define HIP_ENABLE_WARP_SYNC_BUILTINS 1

using namespace flashinfer;

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
void _TestSinglePrefillKernelCorrectness(size_t qo_len, size_t kv_len, size_t num_qo_heads,
                                         size_t num_kv_heads, size_t head_dim, bool causal,
                                         QKVLayout kv_layout, PosEncodingMode pos_encoding_mode,
                                         bool use_fp16_qk_reduction, float rtol = 1e-3,
                                         float atol = 1e-3) {
  std::vector<DTypeQ> q(qo_len * num_qo_heads * head_dim);
  std::vector<DTypeKV> k(kv_len * num_kv_heads * head_dim);
  std::vector<DTypeKV> v(kv_len * num_kv_heads * head_dim);
  std::vector<DTypeO> o(qo_len * num_qo_heads * head_dim);

  utils::vec_normal_(q);
  utils::vec_normal_(k);
  utils::vec_normal_(v);
  utils::vec_zero_(o);

  DTypeQ* q_d;
  FI_GPU_CALL(hipMalloc(&q_d, q.size() * sizeof(DTypeQ)));
  FI_GPU_CALL(hipMemcpy(q_d, q.data(), q.size() * sizeof(DTypeQ), hipMemcpyHostToDevice));

  DTypeKV* k_d;
  FI_GPU_CALL(hipMalloc(&k_d, k.size() * sizeof(DTypeKV)));
  FI_GPU_CALL(hipMemcpy(k_d, k.data(), k.size() * sizeof(DTypeKV), hipMemcpyHostToDevice));

  DTypeKV* v_d;
  FI_GPU_CALL(hipMalloc(&v_d, v.size() * sizeof(DTypeKV)));
  FI_GPU_CALL(hipMemcpy(v_d, v.data(), v.size() * sizeof(DTypeKV), hipMemcpyHostToDevice));

  DTypeO* o_d;
  FI_GPU_CALL(hipMalloc(&o_d, o.size() * sizeof(DTypeO)));
  FI_GPU_CALL(hipMemcpy(o_d, o.data(), o.size() * sizeof(DTypeO), hipMemcpyHostToDevice));

  DTypeO* tmp_d;
  FI_GPU_CALL(hipMalloc(&tmp_d, 16 * 1024 * 1024 * sizeof(DTypeO)));

  hipError_t status = flashinfer::SinglePrefillWithKVCache<DTypeQ, DTypeKV, DTypeO>(
      q_d, k_d, v_d, o_d, tmp_d,
      /*lse=*/nullptr, num_qo_heads, num_kv_heads, qo_len, kv_len, head_dim, causal, kv_layout,
      pos_encoding_mode, use_fp16_qk_reduction);

  EXPECT_EQ(status, hipSuccess) << "SinglePrefillWithKVCache kernel launch failed, error message: "
                                << hipGetErrorString(status);

  std::vector<DTypeO> o_h(o.size());
  FI_GPU_CALL(hipMemcpy(o_h.data(), o_d, o_h.size() * sizeof(DTypeO), hipMemcpyDeviceToHost));

  bool isEmpty = o_h.empty();
  EXPECT_EQ(isEmpty, false) << "Output vector is empty";

  std::vector<float> att_out;
  std::vector<DTypeO> o_ref = cpu_reference::single_mha<DTypeQ, DTypeKV, DTypeO>(
      q, k, v, qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim, causal, kv_layout,
      pos_encoding_mode, /*logits_soft_cap=*/8.0f, /*rope_scale=*/1.f, /*rope_theta=*/1e4,
      /*use_soft_cap=*/true);
  size_t num_results_error_atol = 0;
  bool nan_detected = false;

  for (size_t i = 0; i < o_ref.size(); ++i) {
    float o_h_val = fi::con::explicit_casting<DTypeO, float>(o_h[i]);
    float o_ref_val = fi::con::explicit_casting<DTypeO, float>(o_ref[i]);

    if (isnan(o_h_val)) {
      nan_detected = true;
    }

    num_results_error_atol += (!utils::isclose(o_ref_val, o_h_val, rtol, atol));
  }
  float result_accuracy = 1. - float(num_results_error_atol) / float(o_ref.size());
  std::cout << "num_qo_heads=" << num_qo_heads << ", num_kv_heads=" << num_kv_heads
            << ", qo_len=" << qo_len << ", kv_len=" << kv_len << ", head_dim=" << head_dim
            << ", causal=" << causal << ", kv_layout=" << QKVLayoutToString(kv_layout)
            << ", pos_encoding_mode=" << PosEncodingModeToString(pos_encoding_mode)
            << ", result_accuracy=" << result_accuracy << std::endl;

  EXPECT_GT(result_accuracy, 0.90) << "Result correctness test failed.";
  EXPECT_FALSE(nan_detected) << "Nan detected in the result.";
  FI_GPU_CALL(hipFree(q_d));
  FI_GPU_CALL(hipFree(k_d));
  FI_GPU_CALL(hipFree(v_d));
  FI_GPU_CALL(hipFree(o_d));
  FI_GPU_CALL(hipFree(tmp_d));
}

int main(int argc, char** argv) {
  // ::testing::InitGoogleTest(&argc, argv);
  // return RUN_ALL_TESTS();
  using DTypeIn = __half;
  using DTypeO = __half;
  bool use_fp16_qk_reduction = false;
  size_t qo_len = 128;
  size_t kv_len = 128;
  size_t num_heads = 1;
  size_t head_dim = 64;
  bool causal = false;
  size_t pos_encoding_mode = 0;  // 1 == kRopeLLama
  size_t kv_layout = 0;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--qo_len" && i + 1 < argc) {
      qo_len = std::stoi(argv[++i]);
    } else if (arg == "--kv_len" && i + 1 < argc) {
      kv_len = std::stoi(argv[++i]);
    } else if (arg == "--heads" && i + 1 < argc) {
      num_heads = std::stoi(argv[++i]);
    } else if (arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  --qo_len <len>   Query/Output length (default: 128)\n"
                << "  --kv_len <len>   Key/Value length (default: 128)\n"
                << "  --heads <num>    Number of heads (default: 1)\n"
                << "  --help           Show this help message\n";
      return 0;
    }
  }

  _TestSinglePrefillKernelCorrectness<DTypeIn, DTypeIn, DTypeO>(
      qo_len, kv_len, num_heads, num_heads, head_dim, causal, QKVLayout(kv_layout),
      PosEncodingMode(pos_encoding_mode), use_fp16_qk_reduction);
}
