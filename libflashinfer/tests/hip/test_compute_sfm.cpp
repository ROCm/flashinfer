// SPDX - FileCopyrightText : 2023 - 2025 Flashinfer team
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#include <gtest/gtest.h>

#include <type_traits>

#include "../../utils/flashinfer_prefill_ops.hip.h"
#include "../../utils/utils_hip.h"
#include "flashinfer/attention/generic/prefill.cuh"
#include "gpu_iface/gpu_runtime_compat.hpp"

#define HIP_ENABLE_WARP_SYNC_BUILTINS 1

using namespace flashinfer;

namespace {
template <typename dtype_q, typename dtype_kv, typename dtype_out>
std::vector<dtype_out> test_compute_qk_and_softmax_cpu(
    const std::vector<dtype_q>& q, const std::vector<dtype_kv>& k, const std::vector<dtype_kv>& v,
    size_t qo_len, size_t kv_len, size_t num_qo_heads, size_t num_kv_heads, size_t head_dim,
    bool causal = true, QKVLayout kv_layout = QKVLayout::kHND,
    PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone, float rope_scale = 1.f,
    float rope_theta = 1e4) {
  assert(qo_len <= kv_len);
  assert(num_qo_heads % num_kv_heads == 0);
  float sm_scale = 1.f / std::sqrt(float(head_dim));
  std::vector<dtype_out> o(qo_len * num_qo_heads * head_dim);
  std::vector<float> att(kv_len);
  std::vector<float> q_rotary_local(head_dim);
  std::vector<float> k_rotary_local(head_dim);
  DISPATCH_head_dim(head_dim, HEAD_DIM, {
    tensor_info_t info(qo_len, kv_len, num_qo_heads, num_kv_heads, kv_layout, HEAD_DIM);

    for (size_t qo_head_idx = 0; qo_head_idx < num_qo_heads; ++qo_head_idx) {
      const size_t kv_head_idx = qo_head_idx / info.get_group_size();
      for (size_t q_idx = 0; q_idx < qo_len; ++q_idx) {
        float max_val = -5e4;

        for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
          att[kv_idx] = 0.;
          switch (pos_encoding_mode) {
            case PosEncodingMode::kNone: {
              for (size_t feat_idx = 0; feat_idx < head_dim; ++feat_idx) {
                att[kv_idx] += fi::con::explicit_casting<dtype_q, float>(
                                   q[info.get_q_elem_offset(q_idx, qo_head_idx, feat_idx)]) *
                               fi::con::explicit_casting<dtype_kv, float>(
                                   k[info.get_kv_elem_offset(kv_idx, kv_head_idx, feat_idx)]) *
                               sm_scale;
              }
              break;
            }
            default: {
              std::ostringstream err_msg;
              err_msg << "Unsupported rotary mode.";
              FLASHINFER_ERROR(err_msg.str());
            }
          }
          max_val = std::max(max_val, att[kv_idx]);
        }
        // exp minus max
        float denom = 0;
        for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
          att[kv_idx] = std::exp(att[kv_idx] - max_val);
          denom += att[kv_idx];
        }

        // divide by denom
        for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
          att[kv_idx] /= denom;
        }
      }
    }
  });
  return std::move(att);
}
}  // namespace

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
void _TestComputeSFMCorrectness(size_t qo_len, size_t kv_len, size_t num_qo_heads,
                                size_t num_kv_heads, size_t head_dim, bool causal,
                                QKVLayout kv_layout, PosEncodingMode pos_encoding_mode,
                                bool use_fp16_qk_reduction, float rtol = 1e-3, float atol = 1e-3) {
  std::vector<DTypeQ> q(qo_len * num_qo_heads * head_dim);
  std::vector<DTypeKV> k(kv_len * num_kv_heads * head_dim);
  std::vector<DTypeKV> v(kv_len * num_kv_heads * head_dim);
  std::vector<DTypeO> o(qo_len * num_qo_heads * head_dim);

  utils::generate_data<DTypeQ, utils::Predicate::Linear>(q);
  utils::generate_data<DTypeQ, utils::Predicate::Ones>(k);
  utils::generate_data<DTypeQ, utils::Predicate::Ones>(v);
  utils::generate_data<DTypeQ, utils::Predicate::Zeros>(o);

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

  // Print the first 10 elements of the output vector for debugging
  // std::cout << "Output vector (first 10 elements):";
  // std::cout << "[" << std::endl;
  // for (int i = 0; i < 10; ++i) {
  //     std::cout << fi::con::explicit_casting<DTypeO, float>(o_h[i]) << " ";
  // }
  // std::cout << "]" << std::endl;

  bool isEmpty = o_h.empty();
  EXPECT_EQ(isEmpty, false) << "Output vector is empty";

  std::vector<DTypeO> o_ref = test_compute_qk_and_softmax_cpu<DTypeQ, DTypeKV, DTypeO>(
      q, k, v, qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim, causal, kv_layout,
      pos_encoding_mode);
  size_t num_results_error_atol = 0;
  bool nan_detected = false;

  for (size_t i = 0; i < o_ref.size(); ++i) {
    float o_h_val = fi::con::explicit_casting<DTypeO, float>(o_h[i]);
    float o_ref_val = fi::con::explicit_casting<DTypeO, float>(o_ref[i]);

    if (isnan(o_h_val)) {
      nan_detected = true;
    }

    num_results_error_atol += (!utils::isclose(o_ref_val, o_h_val, rtol, atol));
    if (!utils::isclose(o_ref_val, o_h_val, rtol, atol)) {
      std::cout << "i=" << i << ", o_ref[i]=" << o_ref_val << ", o_h[i]=" << o_h_val << std::endl;
    }
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
  using DTypeIn = __half;
  using DTypeO = __half;
  bool use_fp16_qk_reduction = false;
  size_t qo_len = 399;
  size_t kv_len = 533;
  size_t num_heads = 1;
  size_t head_dim = 64;
  bool causal = false;
  size_t pos_encoding_mode = 0;
  size_t kv_layout = 0;

  _TestComputeSFMCorrectness<DTypeIn, DTypeIn, DTypeO>(
      qo_len, kv_len, num_heads, num_heads, head_dim, causal, QKVLayout(kv_layout),
      PosEncodingMode(pos_encoding_mode), use_fp16_qk_reduction);
}
