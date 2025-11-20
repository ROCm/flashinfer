// SPDX-FileCopyrightText: 2023-2025 Flashinfer team
// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache 2.0

#include <gtest/gtest.h>

#include <cstdint>

#include "../../utils/cpu_reference_hip.h"
#include "../../utils/flashinfer_prefill_ops_hip.h"
#include "../../utils/utils_hip.h"
#include "flashinfer/attention/generic/pos_enc.cuh"

using namespace flashinfer;
constexpr QKVLayout kv_layout = QKVLayout::kNHD;

// template <typename DTypeQO, typename DTypeKV>
// void _TestBatchPagedPrefillKernelOneHotCorrectness(size_t num_kv_heads, size_t num_qo_heads,
//                                                    size_t page_size, size_t head_dim, bool
//                                                    causal, PosEncodingMode pos_encoding_mode,
//                                                    bool use_fp16_qk_reduction) {
//   uint32_t batch_size = 9;
//   std::vector<int32_t> q_lens(batch_size), kv_lens(batch_size);
//   utils::vec_randint_(q_lens, 1, 15);
//   utils::vec_randint_(kv_lens, 15, 257);
//   std::vector<int32_t> append_indptr{0};
//   for (size_t request_idx = 0; request_idx < batch_size; ++request_idx) {
//     append_indptr.push_back(append_indptr.back() + kv_lens[request_idx]);
//   }
//   std::vector<DTypeKV> k_data;
//   std::vector<DTypeKV> v_data;
//   std::vector<int32_t> kv_indptr{0};
//   std::vector<int32_t> kv_indices;
//   std::vector<int32_t> kv_last_page_len;
//   size_t page_counter = 0;

//   std::vector<std::vector<DTypeKV>> key, value;
//   for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
//     size_t kv_len = kv_lens[request_idx];
//     size_t num_pages = (kv_len + page_size - 1) / page_size;
//     size_t last_page_len = (kv_len - 1) % page_size + 1;
//     std::vector<DTypeKV> k(kv_len * num_kv_heads * head_dim), v(kv_len * num_kv_heads *
//     head_dim); utils::vec_normal_(k); utils::vec_normal_(v); key.push_back(k);
//     value.push_back(v);
//     kv_last_page_len.push_back(last_page_len);
//     kv_indptr.push_back(kv_indptr.back() + num_pages);
//     for (size_t j = 0; j < num_pages; ++j) {
//       kv_indices.push_back(page_counter++);
//     }
//   }

//   k_data.resize(page_counter * num_kv_heads * page_size * head_dim);
//   v_data.resize(page_counter * num_kv_heads * page_size * head_dim);
//   flashinfer::paged_kv_t<DTypeKV, int32_t> paged_kv_cpu(
//       num_kv_heads, page_size, head_dim, batch_size, kv_layout, k_data.data(), v_data.data(),
//       kv_indices.data(), kv_indptr.data(), kv_last_page_len.data());
//   cpu_reference::append_paged_kv_cache<DTypeKV, int32_t>(paged_kv_cpu, key, value,
//   append_indptr);

//   // copy data to device
//   DTypeKV* k_data_device = k_data.data();
//   FI_GPU_CALL(hipMalloc(&k_data_device, k_data.size() * sizeof(DTypeKV)));
//   FI_GPU_CALL(hipMemcpy(k_data_device, k_data.data(), k_data.size() * sizeof(DTypeKV),
//                         hipMemcpyHostToDevice));

//   DTypeKV* v_data_device = v_data.data();
//   FI_GPU_CALL(hipMalloc(&v_data_device, v_data.size() * sizeof(DTypeKV)));
//   FI_GPU_CALL(hipMemcpy(v_data_device, v_data.data(), v_data.size() * sizeof(DTypeKV),
//                         hipMemcpyHostToDevice));

//   int32_t* kv_indptr_device = kv_indptr.data();
//   FI_GPU_CALL(hipMalloc(&kv_indptr_device, kv_indptr.size() * sizeof(int32_t)));
//   FI_GPU_CALL(hipMemcpy(kv_indptr_device, kv_indptr.data(), kv_indptr.size() * sizeof(int32_t),
//                         hipMemcpyHostToDevice));

//   int32_t* kv_indices_device = kv_indices.data();
//   FI_GPU_CALL(hipMalloc(&kv_indices_device, kv_indices.size() * sizeof(int32_t)));
//   FI_GPU_CALL(hipMemcpy(kv_indices_device, kv_indices.data(), kv_indices.size() *
//   sizeof(int32_t),
//                         hipMemcpyHostToDevice));

//   int32_t* kv_last_page_len_device = kv_last_page_len.data();
//   FI_GPU_CALL(hipMalloc(&kv_last_page_len_device, kv_last_page_len.size() * sizeof(int32_t)));
//   FI_GPU_CALL(hipMemcpy(kv_last_page_len_device, kv_last_page_len.data(),
//                         kv_last_page_len.size() * sizeof(int32_t), hipMemcpyHostToDevice));

//   // create paged_kv object
//   flashinfer::paged_kv_t<DTypeKV, int32_t> paged_kv = paged_kv_cpu;
//   paged_kv.k_data = k_data_device;
//   paged_kv.v_data = v_data_device;
//   paged_kv.indices = kv_indices_device;
//   paged_kv.indptr = kv_indptr_device;
//   paged_kv.last_page_len = kv_last_page_len_device;

//   BatchPrefillHandler handler;
//   size_t float_workspace_size_in_bytes = 128 * 1024 * 1024;
//   char* float_buffer;
//   FI_GPU_CALL(hipMalloc(&float_buffer, float_workspace_size_in_bytes));

//   size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;
//   char* int_buffer;
//   FI_GPU_CALL(hipMalloc(&int_buffer, int_workspace_size_in_bytes));

//   for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
//     // create one-hot queries
//     int32_t q_len = q_lens[request_idx], kv_len = kv_lens[request_idx];
//     std::vector<int32_t> q_indptr{0};
//     for (uint32_t i = 0; i < batch_size; ++i) {
//       q_indptr.push_back(i >= request_idx ? q_len : 0);
//     }
//     std::vector<DTypeQO> q(q_len * num_qo_heads * head_dim);
//     utils::vec_normal_(q);

//     std::vector<DTypeQO> o_ref = cpu_reference::single_mha<DTypeQO, DTypeKV, DTypeQO>(
//         q, key[request_idx], value[request_idx], q_len, kv_len, num_qo_heads, num_kv_heads,
//         head_dim, causal, QKVLayout::kNHD, pos_encoding_mode);

//     int32_t* q_indptr_device;
//     FI_GPU_CALL(hipMalloc(&q_indptr_device, q_indptr.size() * sizeof(int32_t)));
//     FI_GPU_CALL(hipMemcpy(q_indptr_device, q_indptr.data(), q_indptr.size() * sizeof(int32_t),
//                           hipMemcpyHostToDevice));

//     DTypeQO* q_device;
//     FI_GPU_CALL(hipMalloc(&q_device, q.size() * sizeof(DTypeQO)));
//     FI_GPU_CALL(hipMemcpy(q_device, q.data(), q.size() * sizeof(DTypeQO),
//     hipMemcpyHostToDevice));

//     DTypeQO* o_device;
//     FI_GPU_CALL(hipMalloc(&o_device, q_len * num_qo_heads * head_dim * sizeof(DTypeQO)));

//     handler.Plan<DTypeQO, int32_t>((void*)float_buffer, float_workspace_size_in_bytes,
//                                    (void*)int_buffer, int_workspace_size_in_bytes,
//                                    q_indptr.data(), kv_indptr.data(),
//                                    /*total_num_rows=*/q_indptr.back(), batch_size, num_qo_heads,
//                                    num_kv_heads, head_dim, page_size);

//     for (uint32_t num_runs = 0; num_runs < 10; ++num_runs) {
//       auto status =
//           flashinfer::BatchPrefillWithPagedKVCacheWrapper<DTypeQO, DTypeKV, DTypeQO, int32_t>(
//               &handler, q_device, q_indptr_device, /*q_rope_offset=*/nullptr, paged_kv, o_device,
//               /*lse=*/nullptr, num_qo_heads, causal, pos_encoding_mode, use_fp16_qk_reduction);
//       EXPECT_EQ(status, hipSuccess) << "HIP error: " + std::string(hipGetErrorString(status));
//     }

//     std::vector<DTypeQO> o_host(q_len * num_qo_heads * head_dim);
//     FI_GPU_CALL(hipMemcpy(o_host.data(), o_device,
//                           q_len * num_qo_heads * head_dim * sizeof(DTypeQO),
//                           hipMemcpyDeviceToHost));

//     size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
//     bool nan_detected = false;
//     for (size_t i = 0; i < q_len * num_qo_heads * head_dim; ++i) {
//       float gpu_value = fi::con::explicit_casting<float, DTypeQO>(o_host[i]);
//       float cpu_value = fi::con::explicit_casting<float, DTypeQO>(o_ref[i]);
//       if (std::isnan(gpu_value)) {
//         nan_detected = true;
//       }
//       num_result_errors_atol_1e_3_rtol_1e_3 += (!utils::isclose(gpu_value, cpu_value, 1e-3,
//       1e-3));
//     }
//     float result_accuracy = 1. - float(num_result_errors_atol_1e_3_rtol_1e_3) /
//                                      max(float(q_len * num_qo_heads * head_dim), 1.f);
//     std::cout << "request_idx=" << request_idx << ", page_size=" << page_size
//               << ", num_qo_heads=" << num_qo_heads << ", num_kv_heads=" << num_kv_heads
//               << ", q_len=" << q_len << ", kv_len=" << kv_len << ", head_dim=" << head_dim
//               << ", causal=" << causal
//               << ", pos_encoding_mode=" << PosEncodingModeToString(pos_encoding_mode)
//               << ", result_accuracy=" << result_accuracy << std::endl;
//     EXPECT_GT(result_accuracy, 0.99) << "Result correctness test failed.";
//     EXPECT_EQ(nan_detected, false) << "NaN detected in output.";

//     FI_GPU_CALL(hipFree(q_indptr_device));
//     FI_GPU_CALL(hipFree(q_device));
//     FI_GPU_CALL(hipFree(o_device));
//   }

//   FI_GPU_CALL(hipFree(k_data_device));
//   FI_GPU_CALL(hipFree(v_data_device));
//   FI_GPU_CALL(hipFree(kv_indptr_device));
//   FI_GPU_CALL(hipFree(kv_indices_device));
//   FI_GPU_CALL(hipFree(kv_last_page_len_device));
//   FI_GPU_CALL(hipFree(float_buffer));
//   FI_GPU_CALL(hipFree(int_buffer));
// }

// template <typename DTypeQO, typename DTypeKV>
// void _TestBatchRaggedPrefillKernelCorrectness(size_t num_kv_heads, size_t num_qo_heads,
//                                               size_t head_dim, bool causal,
//                                               PosEncodingMode pos_encoding_mode,
//                                               bool use_fp16_qk_reduction) {
//   uint32_t batch_size = 9;
//   std::vector<int32_t> q_lens(batch_size), kv_lens(batch_size);
//   utils::vec_randint_(q_lens, 10, 15);
//   utils::vec_randint_(kv_lens, 128, 2048);
//   std::vector<int32_t> append_indptr{0}, kv_indptr{0};

//   for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
//     append_indptr.push_back(append_indptr.back() + q_lens[request_idx]);
//     kv_indptr.push_back(kv_indptr.back() + kv_lens[request_idx]);
//   }

//   std::vector<DTypeQO> queries;
//   std::vector<DTypeKV> keys;
//   std::vector<DTypeKV> values;
//   std::vector<DTypeKV> output_refs;

//   BatchPrefillHandler handler;
//   size_t float_workspace_size_in_bytes = 128 * 1024 * 1024;
//   char* float_buffer;
//   FI_GPU_CALL(hipMalloc(&float_buffer, float_workspace_size_in_bytes));

//   size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;
//   char* int_buffer;
//   FI_GPU_CALL(hipMalloc(&int_buffer, int_workspace_size_in_bytes));

//   for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
//     std::vector<DTypeQO> q(q_lens[request_idx] * num_qo_heads * head_dim);
//     std::vector<DTypeKV> k(kv_lens[request_idx] * num_kv_heads * head_dim),
//         v(kv_lens[request_idx] * num_kv_heads * head_dim);
//     uint32_t q_len = q_lens[request_idx], kv_len = kv_lens[request_idx];
//     utils::vec_normal_(q);
//     utils::vec_normal_(k);
//     utils::vec_normal_(v);
//     std::vector<DTypeQO> o_ref = cpu_reference::single_mha<DTypeQO, DTypeKV, DTypeQO>(
//         q, k, v, q_len, kv_len, num_qo_heads, num_kv_heads, head_dim, causal, QKVLayout::kNHD,
//         pos_encoding_mode);
//     // NOTE(Zihao): The following code is only compatible with kv_layout = QKVLayout::kNHD
//     std::copy(q.begin(), q.end(), std::back_inserter(queries));
//     std::copy(k.begin(), k.end(), std::back_inserter(keys));
//     std::copy(v.begin(), v.end(), std::back_inserter(values));
//     std::copy(o_ref.begin(), o_ref.end(), std::back_inserter(output_refs));
//   }

//   DTypeQO* queries_device;
//   FI_GPU_CALL(hipMalloc(&queries_device, queries.size() * sizeof(DTypeQO)));
//   FI_GPU_CALL(hipMemcpy(queries_device, queries.data(), queries.size() * sizeof(DTypeQO),
//                         hipMemcpyHostToDevice));

//   DTypeKV* keys_device;
//   FI_GPU_CALL(hipMalloc(&keys_device, keys.size() * sizeof(DTypeKV)));
//   FI_GPU_CALL(
//       hipMemcpy(keys_device, keys.data(), keys.size() * sizeof(DTypeKV), hipMemcpyHostToDevice));

//   DTypeKV* values_device;
//   FI_GPU_CALL(hipMalloc(&values_device, values.size() * sizeof(DTypeKV)));
//   FI_GPU_CALL(hipMemcpy(values_device, values.data(), values.size() * sizeof(DTypeKV),
//                         hipMemcpyHostToDevice));

//   DTypeQO* output_device;
//   FI_GPU_CALL(hipMalloc(&output_device, queries.size() * sizeof(DTypeQO)));

//   int32_t* append_indptr_device;
//   FI_GPU_CALL(hipMalloc(&append_indptr_device, append_indptr.size() * sizeof(int32_t)));
//   FI_GPU_CALL(hipMemcpy(append_indptr_device, append_indptr.data(),
//                         append_indptr.size() * sizeof(int32_t), hipMemcpyHostToDevice));

//   int32_t* kv_indptr_device;
//   FI_GPU_CALL(hipMalloc(&kv_indptr_device, kv_indptr.size() * sizeof(int32_t)));
//   FI_GPU_CALL(hipMemcpy(kv_indptr_device, kv_indptr.data(), kv_indptr.size() * sizeof(int32_t),
//                         hipMemcpyHostToDevice));

//   handler.Plan<DTypeQO, int32_t>((void*)float_buffer, float_workspace_size_in_bytes,
//                                  (void*)int_buffer, int_workspace_size_in_bytes,
//                                  append_indptr.data(), kv_indptr.data(),
//                                  /*total_num_rows=*/append_indptr.back(), batch_size,
//                                  num_qo_heads, num_kv_heads, head_dim, /*page_size=*/1);

//   auto status = BatchPrefillWithRaggedKVCacheWrapper<DTypeQO, DTypeKV, DTypeQO, int32_t>(
//       &handler, queries_device, append_indptr_device, keys_device, values_device,
//       kv_indptr_device,
//       /*q_rope_offset=*/nullptr, /*k_rope_offset=*/nullptr, output_device, /*lse=*/nullptr,
//       batch_size, num_qo_heads, num_kv_heads, head_dim, causal, kv_layout, pos_encoding_mode,
//       use_fp16_qk_reduction);

//   EXPECT_EQ(status, hipSuccess) << "HIP error: " + std::string(hipGetErrorString(status));

//   std::vector<DTypeQO> output_host(queries.size());
//   FI_GPU_CALL(hipMemcpy(output_host.data(), output_device, queries.size() * sizeof(DTypeQO),
//                         hipMemcpyDeviceToHost));

//   size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
//   bool nan_detected = false;
//   for (size_t i = 0; i < output_refs.size(); ++i) {
//     float cpu_value = fi::con::explicit_casting<float, DTypeQO>(output_refs[i]);
//     float gpu_value = fi::con::explicit_casting<float, DTypeQO>(output_host[i]);

//     if (std::isnan(gpu_value)) {
//       nan_detected = true;
//     }
//     num_result_errors_atol_1e_3_rtol_1e_3 += (!utils::isclose(gpu_value, cpu_value, 1e-3, 1e-3));
//   }

//   float result_accuracy =
//       1. - float(num_result_errors_atol_1e_3_rtol_1e_3) / max(float(output_refs.size()), 1.f);
//   std::cout << "num_qo_heads=" << num_qo_heads << ", num_kv_heads=" << num_kv_heads
//             << ", head_dim=" << head_dim << ", causal=" << causal
//             << ", pos_encoding_mode=" << PosEncodingModeToString(pos_encoding_mode)
//             << ", result_accuracy=" << result_accuracy << std::endl;

//   EXPECT_GT(result_accuracy, 0.99) << "Result correctness test failed.";
//   EXPECT_EQ(nan_detected, false) << "NaN detected in output.";

//   FI_GPU_CALL(hipFree(float_buffer));
//   FI_GPU_CALL(hipFree(int_buffer));
//   FI_GPU_CALL(hipFree(queries_device));
//   FI_GPU_CALL(hipFree(keys_device));
//   FI_GPU_CALL(hipFree(values_device));
//   FI_GPU_CALL(hipFree(output_device));
//   FI_GPU_CALL(hipFree(append_indptr_device));
//   FI_GPU_CALL(hipFree(kv_indptr_device));
// }

template <typename DTypeQO, typename DTypeKV>
void _TestBatchPagedPrefillKernelShortContextCorrectness(size_t num_kv_heads, size_t num_qo_heads,
                                                         size_t page_size, size_t head_dim,
                                                         bool causal,
                                                         PosEncodingMode pos_encoding_mode,
                                                         bool use_fp16_qk_reduction) {
  const uint32_t batch_size = 7;
  std::vector<int32_t> q_lens(batch_size);
  utils::vec_randint_(q_lens, 1, 64);
  std::vector<int32_t> kv_lens(q_lens);

  std::vector<int32_t> q_indptr{0};
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    q_indptr.push_back(q_indptr.back() + q_lens[request_idx]);
  }
  std::vector<int32_t> append_indptr{0};
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    append_indptr.push_back(append_indptr.back() + kv_lens[request_idx]);
  }
  std::vector<DTypeKV> k_data;
  std::vector<DTypeKV> v_data;
  std::vector<int32_t> kv_indptr{0};
  std::vector<int32_t> kv_indices;
  std::vector<int32_t> kv_last_page_len;
  size_t page_counter = 0;
  std::vector<std::vector<DTypeKV>> key, value;
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    size_t kv_len = kv_lens[request_idx];
    size_t num_pages = (kv_len + page_size - 1) / page_size;
    size_t last_page_len = (kv_len - 1) % page_size + 1;
    std::vector<DTypeKV> k(kv_len * num_kv_heads * head_dim), v(kv_len * num_kv_heads * head_dim);
    utils::vec_normal_(k);
    utils::vec_normal_(v);
    key.push_back(k);
    value.push_back(v);
    kv_last_page_len.push_back(last_page_len);
    kv_indptr.push_back(kv_indptr.back() + num_pages);
    for (size_t j = 0; j < num_pages; ++j) {
      kv_indices.push_back(page_counter++);
    }
  }

  k_data.resize(page_counter * num_kv_heads * page_size * head_dim);
  v_data.resize(page_counter * num_kv_heads * page_size * head_dim);
  flashinfer::paged_kv_t<DTypeKV, int32_t> paged_kv_cpu(
      num_kv_heads, page_size, head_dim, batch_size, kv_layout, k_data.data(), v_data.data(),
      kv_indices.data(), kv_indptr.data(), kv_last_page_len.data());
  cpu_reference::append_paged_kv_cache<DTypeKV, int32_t>(paged_kv_cpu, key, value, append_indptr);

  // copy data to device
  DTypeKV* k_data_device;
  FI_GPU_CALL(hipMalloc(&k_data_device, k_data.size() * sizeof(DTypeKV)));
  FI_GPU_CALL(hipMemcpy(k_data_device, k_data.data(), k_data.size() * sizeof(DTypeKV),
                        hipMemcpyHostToDevice));

  DTypeKV* v_data_device;
  FI_GPU_CALL(hipMalloc(&v_data_device, v_data.size() * sizeof(DTypeKV)));
  FI_GPU_CALL(hipMemcpy(v_data_device, v_data.data(), v_data.size() * sizeof(DTypeKV),
                        hipMemcpyHostToDevice));

  int32_t* kv_indptr_device;
  FI_GPU_CALL(hipMalloc(&kv_indptr_device, kv_indptr.size() * sizeof(int32_t)));
  FI_GPU_CALL(hipMemcpy(kv_indptr_device, kv_indptr.data(), kv_indptr.size() * sizeof(int32_t),
                        hipMemcpyHostToDevice));

  int32_t* kv_indices_device;
  FI_GPU_CALL(hipMalloc(&kv_indices_device, kv_indices.size() * sizeof(int32_t)));
  FI_GPU_CALL(hipMemcpy(kv_indices_device, kv_indices.data(), kv_indices.size() * sizeof(int32_t),
                        hipMemcpyHostToDevice));

  int32_t* kv_last_page_len_device;
  FI_GPU_CALL(hipMalloc(&kv_last_page_len_device, kv_last_page_len.size() * sizeof(int32_t)));
  FI_GPU_CALL(hipMemcpy(kv_last_page_len_device, kv_last_page_len.data(),
                        kv_last_page_len.size() * sizeof(int32_t), hipMemcpyHostToDevice));

  // create paged_kv object
  flashinfer::paged_kv_t<DTypeKV, int32_t> paged_kv = paged_kv_cpu;
  paged_kv.k_data = k_data_device;
  paged_kv.v_data = v_data_device;
  paged_kv.indices = kv_indices_device;
  paged_kv.indptr = kv_indptr_device;
  paged_kv.last_page_len = kv_last_page_len_device;

  std::vector<std::vector<DTypeQO>> q, o_ref;
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    int32_t q_len = q_lens[request_idx];
    std::vector<DTypeQO> qi(q_len * num_qo_heads * head_dim);
    utils::vec_normal_(qi);
    q.push_back(qi);
  }
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    int32_t q_len = q_lens[request_idx], kv_len = kv_lens[request_idx];
    std::vector<DTypeQO> o_ref_i = cpu_reference::single_mha<DTypeQO, DTypeKV, DTypeQO>(
        q[request_idx], key[request_idx], value[request_idx], q_len, kv_len, num_qo_heads,
        num_kv_heads, head_dim, causal, QKVLayout::kNHD, pos_encoding_mode);
    o_ref.push_back(o_ref_i);
  }

  std::vector<DTypeQO> q_concat, o_concat_ref;
  for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    q_concat.insert(q_concat.end(), q[request_idx].begin(), q[request_idx].end());
    o_concat_ref.insert(o_concat_ref.end(), o_ref[request_idx].begin(), o_ref[request_idx].end());
  }

  DTypeQO* q_device;
  FI_GPU_CALL(hipMalloc(&q_device, q_concat.size() * sizeof(DTypeQO)));
  FI_GPU_CALL(hipMemcpy(q_device, q_concat.data(), q_concat.size() * sizeof(DTypeQO),
                        hipMemcpyHostToDevice));

  int32_t* q_indptr_device;
  FI_GPU_CALL(hipMalloc(&q_indptr_device, q_indptr.size() * sizeof(int32_t)));
  FI_GPU_CALL(hipMemcpy(q_indptr_device, q_indptr.data(), q_indptr.size() * sizeof(int32_t),
                        hipMemcpyHostToDevice));

  DTypeQO* o_device;
  FI_GPU_CALL(hipMalloc(&o_device, o_concat_ref.size() * sizeof(DTypeQO)));

  BatchPrefillHandler handler;
  size_t float_workspace_size_in_bytes = 32 * 1024 * 1024;
  char* float_buffer;
  FI_GPU_CALL(hipMalloc(&float_buffer, float_workspace_size_in_bytes));

  size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;
  char* int_buffer;
  FI_GPU_CALL(hipMalloc(&int_buffer, int_workspace_size_in_bytes));

  handler.Plan<DTypeQO, int32_t>((void*)float_buffer, float_workspace_size_in_bytes,
                                 (void*)int_buffer, int_workspace_size_in_bytes, q_indptr.data(),
                                 kv_indptr.data(), /*total_num_rows=*/q_indptr.back(), batch_size,
                                 num_qo_heads, num_kv_heads, head_dim, page_size);

  auto status = BatchPrefillWithPagedKVCacheWrapper<DTypeQO, DTypeKV, DTypeQO, int32_t>(
      &handler, q_device, q_indptr_device, /*q_rope_offset=*/nullptr, paged_kv, o_device,
      /*lse=*/nullptr, num_qo_heads, causal, pos_encoding_mode, use_fp16_qk_reduction);
  EXPECT_EQ(status, hipSuccess) << "HIP error: " + std::string(hipGetErrorString(status));

  std::vector<DTypeQO> o_host(o_concat_ref.size());
  FI_GPU_CALL(hipMemcpy(o_host.data(), o_device, o_concat_ref.size() * sizeof(DTypeQO),
                        hipMemcpyDeviceToHost));

  size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
  bool nan_detected = false;
  for (size_t i = 0; i < o_concat_ref.size(); ++i) {
    float cpu_value = fi::con::explicit_casting<float, DTypeQO>(o_concat_ref[i]);
    float gpu_value = fi::con::explicit_casting<float, DTypeQO>(o_host[i]);

    if (std::isnan(gpu_value)) {
      nan_detected = true;
    }
    num_result_errors_atol_1e_3_rtol_1e_3 += (!utils::isclose(gpu_value, cpu_value, 1e-3, 1e-3));
  }
  float result_accuracy =
      1. - float(num_result_errors_atol_1e_3_rtol_1e_3) / max(float(o_concat_ref.size()), 1.f);
  std::cout << "page_size=" << page_size << ", num_qo_heads=" << num_qo_heads
            << ", num_kv_heads=" << num_kv_heads << ", head_dim=" << head_dim
            << ", causal=" << causal
            << ", pos_encoding_mode=" << PosEncodingModeToString(pos_encoding_mode)
            << ", result_accuracy=" << result_accuracy << std::endl;
  EXPECT_GT(result_accuracy, 0.99) << "Result correctness test failed.";
  EXPECT_EQ(nan_detected, false) << "NaN detected in output.";

  FI_GPU_CALL(hipFree(k_data_device));
  FI_GPU_CALL(hipFree(v_data_device));
  FI_GPU_CALL(hipFree(kv_indptr_device));
  FI_GPU_CALL(hipFree(kv_indices_device));
  FI_GPU_CALL(hipFree(kv_last_page_len_device));
  FI_GPU_CALL(hipFree(float_buffer));
  FI_GPU_CALL(hipFree(int_buffer));
  FI_GPU_CALL(hipFree(q_device));
  FI_GPU_CALL(hipFree(q_indptr_device));
  FI_GPU_CALL(hipFree(o_device));
}

// template <typename DTypeQO, typename DTypeKV>
// void _TestBatchPagedPrefillKernelQMinMaxKVMinMaxCorrectness(
//     size_t batch_size, size_t num_kv_heads, size_t num_qo_heads, size_t page_size, size_t
//     head_dim, bool use_fp16_qk_reduction, uint32_t q_len_min, uint32_t q_len_max, uint32_t
//     kv_len_min, uint32_t kv_len_max) {
//   std::vector<int32_t> q_lens(batch_size);
//   utils::vec_randint_(q_lens, q_len_min, q_len_max);
//   std::vector<int32_t> kv_lens(batch_size);
//   utils::vec_randint_(kv_lens, kv_len_min, kv_len_max);

//   std::vector<int32_t> q_indptr{0};
//   for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
//     q_indptr.push_back(q_indptr.back() + q_lens[request_idx]);
//   }
//   std::vector<int32_t> append_indptr{0};
//   for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
//     append_indptr.push_back(append_indptr.back() + kv_lens[request_idx]);
//   }
//   std::vector<DTypeKV> k_data;
//   std::vector<DTypeKV> v_data;
//   std::vector<int32_t> kv_indptr{0};
//   std::vector<int32_t> kv_indices;
//   std::vector<int32_t> kv_last_page_len;
//   size_t page_counter = 0;
//   std::vector<std::vector<DTypeKV>> key, value;
//   for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
//     size_t kv_len = kv_lens[request_idx];
//     size_t num_pages = (kv_len + page_size - 1) / page_size;
//     size_t last_page_len = num_pages == 0 ? 0 : (kv_len - 1) % page_size + 1;
//     std::vector<DTypeKV> k(kv_len * num_kv_heads * head_dim), v(kv_len * num_kv_heads *
//     head_dim); utils::vec_normal_(k); utils::vec_normal_(v); key.push_back(k);
//     value.push_back(v);
//     kv_last_page_len.push_back(last_page_len);
//     kv_indptr.push_back(kv_indptr.back() + num_pages);
//     for (size_t j = 0; j < num_pages; ++j) {
//       kv_indices.push_back(page_counter++);
//     }
//   }

//   k_data.resize(page_counter * num_kv_heads * page_size * head_dim);
//   v_data.resize(page_counter * num_kv_heads * page_size * head_dim);
//   flashinfer::paged_kv_t<DTypeKV, int32_t> paged_kv_cpu(
//       num_kv_heads, page_size, head_dim, batch_size, kv_layout, k_data.data(), v_data.data(),
//       kv_indices.data(), kv_indptr.data(), kv_last_page_len.data());
//   cpu_reference::append_paged_kv_cache<DTypeKV, int32_t>(paged_kv_cpu, key, value,
//   append_indptr);

//   // copy data to device
//   DTypeKV* k_data_device;
//   FI_GPU_CALL(hipMalloc(&k_data_device, k_data.size() * sizeof(DTypeKV)));
//   FI_GPU_CALL(hipMemcpy(k_data_device, k_data.data(), k_data.size() * sizeof(DTypeKV),
//                         hipMemcpyHostToDevice));

//   DTypeKV* v_data_device;
//   FI_GPU_CALL(hipMalloc(&v_data_device, v_data.size() * sizeof(DTypeKV)));
//   FI_GPU_CALL(hipMemcpy(v_data_device, v_data.data(), v_data.size() * sizeof(DTypeKV),
//                         hipMemcpyHostToDevice));

//   int32_t* kv_indptr_device;
//   FI_GPU_CALL(hipMalloc(&kv_indptr_device, kv_indptr.size() * sizeof(int32_t)));
//   FI_GPU_CALL(hipMemcpy(kv_indptr_device, kv_indptr.data(), kv_indptr.size() * sizeof(int32_t),
//                         hipMemcpyHostToDevice));

//   int32_t* kv_indices_device;
//   FI_GPU_CALL(hipMalloc(&kv_indices_device, kv_indices.size() * sizeof(int32_t)));
//   FI_GPU_CALL(hipMemcpy(kv_indices_device, kv_indices.data(), kv_indices.size() *
//   sizeof(int32_t),
//                         hipMemcpyHostToDevice));

//   int32_t* kv_last_page_len_device;
//   FI_GPU_CALL(hipMalloc(&kv_last_page_len_device, kv_last_page_len.size() * sizeof(int32_t)));
//   FI_GPU_CALL(hipMemcpy(kv_last_page_len_device, kv_last_page_len.data(),
//                         kv_last_page_len.size() * sizeof(int32_t), hipMemcpyHostToDevice));

//   // create paged_kv object
//   flashinfer::paged_kv_t<DTypeKV, int32_t> paged_kv = paged_kv_cpu;
//   paged_kv.k_data = k_data_device;
//   paged_kv.v_data = v_data_device;
//   paged_kv.indices = kv_indices_device;
//   paged_kv.indptr = kv_indptr_device;
//   paged_kv.last_page_len = kv_last_page_len_device;

//   std::vector<std::vector<DTypeQO>> q, o_ref;
//   for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
//     int32_t q_len = q_lens[request_idx];
//     std::vector<DTypeQO> qi(q_len * num_qo_heads * head_dim);
//     utils::vec_normal_(qi);
//     q.push_back(qi);
//   }
//   for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
//     int32_t q_len = q_lens[request_idx], kv_len = kv_lens[request_idx];
//     std::vector<DTypeQO> o_ref_i = cpu_reference::single_mha<DTypeQO, DTypeKV, DTypeQO>(
//         q[request_idx], key[request_idx], value[request_idx], q_len, kv_len, num_qo_heads,
//         num_kv_heads, head_dim, /*causal=*/false, QKVLayout::kNHD,
//         /*pos_encoding_mode*/ PosEncodingMode::kNone);
//     o_ref.push_back(o_ref_i);
//   }

//   std::vector<DTypeQO> q_concat, o_concat_ref;
//   for (uint32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
//     q_concat.insert(q_concat.end(), q[request_idx].begin(), q[request_idx].end());
//     o_concat_ref.insert(o_concat_ref.end(), o_ref[request_idx].begin(),
//     o_ref[request_idx].end());
//   }

//   DTypeQO* q_device;
//   FI_GPU_CALL(hipMalloc(&q_device, q_concat.size() * sizeof(DTypeQO)));
//   FI_GPU_CALL(hipMemcpy(q_device, q_concat.data(), q_concat.size() * sizeof(DTypeQO),
//                         hipMemcpyHostToDevice));

//   int32_t* q_indptr_device;
//   FI_GPU_CALL(hipMalloc(&q_indptr_device, q_indptr.size() * sizeof(int32_t)));
//   FI_GPU_CALL(hipMemcpy(q_indptr_device, q_indptr.data(), q_indptr.size() * sizeof(int32_t),
//                         hipMemcpyHostToDevice));

//   DTypeQO* o_device;
//   FI_GPU_CALL(hipMalloc(&o_device, o_concat_ref.size() * sizeof(DTypeQO)));

//   BatchPrefillHandler handler;
//   size_t float_workspace_size_in_bytes = 32 * 1024 * 1024;
//   char* float_buffer;
//   FI_GPU_CALL(hipMalloc(&float_buffer, float_workspace_size_in_bytes));

//   size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;
//   char* int_buffer;
//   FI_GPU_CALL(hipMalloc(&int_buffer, int_workspace_size_in_bytes));

//   handler.Plan<DTypeQO, int32_t>((void*)float_buffer, float_workspace_size_in_bytes,
//                                  (void*)int_buffer, int_workspace_size_in_bytes, q_indptr.data(),
//                                  kv_indptr.data(), /*total_num_rows=*/q_indptr.back(),
//                                  batch_size, num_qo_heads, num_kv_heads, head_dim, page_size);

//   auto status = BatchPrefillWithPagedKVCacheWrapper<DTypeQO, DTypeKV, DTypeQO, int32_t>(
//       &handler, q_device, q_indptr_device, /*q_rope_offset=*/nullptr, paged_kv, o_device,
//       /*lse=*/nullptr, num_qo_heads, /*causal=*/false,
//       /*pos_encoding_mode*/ PosEncodingMode::kNone);
//   EXPECT_EQ(status, hipSuccess) << "HIP error: " + std::string(hipGetErrorString(status));

//   std::vector<DTypeQO> o_host(o_concat_ref.size());
//   FI_GPU_CALL(hipMemcpy(o_host.data(), o_device, o_concat_ref.size() * sizeof(DTypeQO),
//                         hipMemcpyDeviceToHost));

//   size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
//   bool nan_detected = false;
//   for (size_t i = 0; i < o_concat_ref.size(); ++i) {
//     float cpu_value = fi::con::explicit_casting<float, DTypeQO>(o_concat_ref[i]);
//     float gpu_value = fi::con::explicit_casting<float, DTypeQO>(o_host[i]);

//     if (std::isnan(gpu_value)) {
//       nan_detected = true;
//     }
//     num_result_errors_atol_1e_3_rtol_1e_3 += (!utils::isclose(gpu_value, cpu_value, 1e-3, 1e-3));
//   }
//   float result_accuracy =
//       1. - float(num_result_errors_atol_1e_3_rtol_1e_3) / max(float(o_concat_ref.size()), 1.f);
//   std::cout << "batch_size=" << batch_size << ", page_size=" << page_size
//             << ", num_qo_heads=" << num_qo_heads << ", num_kv_heads=" << num_kv_heads
//             << ", head_dim=" << head_dim << ", result_accuracy=" << result_accuracy << std::endl;
//   EXPECT_GT(result_accuracy, 0.99) << "Result correctness test failed.";
//   EXPECT_EQ(nan_detected, false) << "NaN detected in output.";

//   FI_GPU_CALL(hipFree(k_data_device));
//   FI_GPU_CALL(hipFree(v_data_device));
//   FI_GPU_CALL(hipFree(kv_indptr_device));
//   FI_GPU_CALL(hipFree(kv_indices_device));
//   FI_GPU_CALL(hipFree(kv_last_page_len_device));
//   FI_GPU_CALL(hipFree(q_device));
//   FI_GPU_CALL(hipFree(q_indptr_device));
//   FI_GPU_CALL(hipFree(o_device));
//   FI_GPU_CALL(hipFree(float_buffer));
//   FI_GPU_CALL(hipFree(int_buffer));
// }

// template <typename DTypeQO, typename DTypeKV>
// void _TestBatchPagedPrefillKernelLongContextCorrectness(size_t num_kv_heads, size_t num_qo_heads,
//                                                         size_t page_size, size_t head_dim,
//                                                         bool causal,
//                                                         PosEncodingMode pos_encoding_mode,
//                                                         bool use_fp16_qk_reduction) {
//   std::vector<std::vector<std::vector<DTypeKV>>> keys, values;
//   std::vector<int32_t> q_lens{33}, kv_lens{32768};
//   std::vector<int32_t> q_indptr{0, 33};
//   std::vector<int32_t> append_indptr{0, 32768};
//   std::vector<DTypeKV> k_data;
//   std::vector<DTypeKV> v_data;
//   std::vector<int32_t> kv_indptr{0};
//   std::vector<int32_t> kv_indices;
//   std::vector<int32_t> kv_last_page_len;
//   size_t page_counter = 0;

//   size_t num_pages = (kv_lens[0] + page_size - 1) / page_size;
//   size_t last_page_len = (kv_lens[0] - 1) % page_size + 1;
//   std::vector<DTypeKV> k(kv_lens[0] * num_kv_heads * head_dim),
//       v(kv_lens[0] * num_kv_heads * head_dim);
//   utils::vec_normal_(k);
//   utils::vec_normal_(v);
//   kv_last_page_len.push_back(last_page_len);
//   kv_indptr.push_back(kv_indptr.back() + num_pages);
//   for (size_t j = 0; j < num_pages; ++j) {
//     kv_indices.push_back(page_counter++);
//   }

//   k_data.resize(page_counter * 1 * num_kv_heads * page_size * head_dim);
//   v_data.resize(page_counter * 1 * num_kv_heads * page_size * head_dim);
//   flashinfer::paged_kv_t<DTypeKV, int32_t> paged_kv_cpu(
//       num_kv_heads, page_size, head_dim, 1, kv_layout, k_data.data(), v_data.data(),
//       kv_indices.data(), kv_indptr.data(), kv_last_page_len.data());
//   cpu_reference::append_paged_kv_cache<DTypeKV, int32_t>(paged_kv_cpu, {k}, {v}, append_indptr);

//   // copy data to device
//   DTypeKV* k_data_device;
//   FI_GPU_CALL(hipMalloc(&k_data_device, k_data.size() * sizeof(DTypeKV)));
//   FI_GPU_CALL(hipMemcpy(k_data_device, k_data.data(), k_data.size() * sizeof(DTypeKV),
//                         hipMemcpyHostToDevice));

//   DTypeKV* v_data_device;
//   FI_GPU_CALL(hipMalloc(&v_data_device, v_data.size() * sizeof(DTypeKV)));
//   FI_GPU_CALL(hipMemcpy(v_data_device, v_data.data(), v_data.size() * sizeof(DTypeKV),
//                         hipMemcpyHostToDevice));

//   int32_t* kv_indptr_device;
//   FI_GPU_CALL(hipMalloc(&kv_indptr_device, kv_indptr.size() * sizeof(int32_t)));
//   FI_GPU_CALL(hipMemcpy(kv_indptr_device, kv_indptr.data(), kv_indptr.size() * sizeof(int32_t),
//                         hipMemcpyHostToDevice));

//   int32_t* kv_indices_device;
//   FI_GPU_CALL(hipMalloc(&kv_indices_device, kv_indices.size() * sizeof(int32_t)));
//   FI_GPU_CALL(hipMemcpy(kv_indices_device, kv_indices.data(), kv_indices.size() *
//   sizeof(int32_t),
//                         hipMemcpyHostToDevice));

//   int32_t* kv_last_page_len_device;
//   FI_GPU_CALL(hipMalloc(&kv_last_page_len_device, kv_last_page_len.size() * sizeof(int32_t)));
//   FI_GPU_CALL(hipMemcpy(kv_last_page_len_device, kv_last_page_len.data(),
//                         kv_last_page_len.size() * sizeof(int32_t), hipMemcpyHostToDevice));

//   // create paged_kv object
//   flashinfer::paged_kv_t<DTypeKV, int32_t> paged_kv = paged_kv_cpu;
//   paged_kv.k_data = k_data_device;
//   paged_kv.v_data = v_data_device;
//   paged_kv.indices = kv_indices_device;
//   paged_kv.indptr = kv_indptr_device;
//   paged_kv.last_page_len = kv_last_page_len_device;

//   // create one-hot queries
//   std::vector<DTypeQO> q(q_lens[0] * num_qo_heads * head_dim);
//   utils::vec_normal_(q);

//   std::vector<DTypeQO> o_ref = cpu_reference::single_mha<DTypeQO, DTypeKV, DTypeQO>(
//       q, k, v, q_lens[0], kv_lens[0], num_qo_heads, num_kv_heads, head_dim, causal,
//       QKVLayout::kNHD, pos_encoding_mode);

//   int32_t* q_indptr_device;
//   FI_GPU_CALL(hipMalloc(&q_indptr_device, q_indptr.size() * sizeof(int32_t)));
//   FI_GPU_CALL(hipMemcpy(q_indptr_device, q_indptr.data(), q_indptr.size() * sizeof(int32_t),
//                         hipMemcpyHostToDevice));

//   DTypeQO* q_device;
//   FI_GPU_CALL(hipMalloc(&q_device, q.size() * sizeof(DTypeQO)));
//   FI_GPU_CALL(hipMemcpy(q_device, q.data(), q.size() * sizeof(DTypeQO), hipMemcpyHostToDevice));

//   DTypeQO* o_device;
//   FI_GPU_CALL(hipMalloc(&o_device, q_lens[0] * num_qo_heads * head_dim * sizeof(DTypeQO)));

//   BatchPrefillHandler handler;
//   size_t float_workspace_size_in_bytes = 32 * 1024 * 1024;
//   char* float_buffer;
//   FI_GPU_CALL(hipMalloc(&float_buffer, float_workspace_size_in_bytes));

//   size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;
//   char* int_buffer;
//   FI_GPU_CALL(hipMalloc(&int_buffer, int_workspace_size_in_bytes));

//   handler.Plan<DTypeQO, int32_t>((void*)float_buffer, float_workspace_size_in_bytes,
//                                  (void*)int_buffer, int_workspace_size_in_bytes,
//                                  append_indptr.data(), kv_indptr.data(),
//                                  /*total_num_rows=*/append_indptr.back(),
//                                  /*batch_size=*/1, num_qo_heads, num_kv_heads, head_dim,
//                                  page_size);

//   auto status = BatchPrefillWithPagedKVCacheWrapper<DTypeQO, DTypeKV, DTypeQO, int32_t>(
//       &handler, q_device, q_indptr_device,
//       /*q_rope_offset=*/nullptr, paged_kv, o_device,
//       /*lse=*/nullptr, num_qo_heads, causal, pos_encoding_mode, use_fp16_qk_reduction);
//   EXPECT_EQ(status, hipSuccess) << "HIP error: " + std::string(hipGetErrorString(status));

//   std::vector<DTypeQO> o_host(q_lens[0] * num_qo_heads * head_dim);
//   FI_GPU_CALL(hipMemcpy(o_host.data(), o_device,
//                         q_lens[0] * num_qo_heads * head_dim * sizeof(DTypeQO),
//                         hipMemcpyDeviceToHost));

//   size_t num_result_errors_atol_1e_3_rtol_1e_3 = 0;
//   bool nan_detected = false;
//   for (size_t i = 0; i < q_lens[0] * num_qo_heads * head_dim; ++i) {
//     float gpu_value = fi::con::explicit_casting<float, DTypeQO>(o_host[i]);
//     float cpu_value = fi::con::explicit_casting<float, DTypeQO>(o_ref[i]);

//     if (std::isnan(gpu_value)) {
//       nan_detected = true;
//     }
//     num_result_errors_atol_1e_3_rtol_1e_3 += (!utils::isclose(gpu_value, cpu_value, 1e-3, 1e-3));
//   }
//   float result_accuracy = 1. - float(num_result_errors_atol_1e_3_rtol_1e_3) /
//                                    max(float(q_lens[0] * num_qo_heads * head_dim), 1.f);
//   std::cout << "page_size=" << page_size << ", num_qo_heads=" << num_qo_heads
//             << ", num_kv_heads=" << num_kv_heads << ", q_len=" << q_lens[0]
//             << ", kv_len=" << kv_lens[0] << ", head_dim=" << head_dim << ", causal=" << causal
//             << ", pos_encoding_mode=" << PosEncodingModeToString(pos_encoding_mode)
//             << ", result_accuracy=" << result_accuracy << std::endl;
//   EXPECT_GT(result_accuracy, 0.99) << "Result correctness test failed.";
//   EXPECT_EQ(nan_detected, false) << "NaN detected in output.";

//   FI_GPU_CALL(hipFree(k_data_device));
//   FI_GPU_CALL(hipFree(v_data_device));
//   FI_GPU_CALL(hipFree(kv_indptr_device));
//   FI_GPU_CALL(hipFree(kv_indices_device));
//   FI_GPU_CALL(hipFree(kv_last_page_len_device));
//   FI_GPU_CALL(hipFree(float_buffer));
//   FI_GPU_CALL(hipFree(int_buffer));
//   FI_GPU_CALL(hipFree(q_device));
//   FI_GPU_CALL(hipFree(q_indptr_device));
//   FI_GPU_CALL(hipFree(o_device));
// }

// template <typename T>
// void TestBatchPagedPrefillKernelOneHotCorrectness(bool use_fp16_qk_reduction) {
//   for (size_t num_kv_heads : {4, 8, 32}) {
//     for (size_t num_qo_heads : {32}) {
//       for (size_t page_size : {1, 16}) {
//         for (size_t head_dim : {64, 128, 256}) {
//           for (size_t causal : {false, true}) {
//             for (size_t pos_encoding_mode : {0, 1}) {
//               _TestBatchPagedPrefillKernelOneHotCorrectness<T, T>(
//                   num_kv_heads, num_qo_heads, page_size, head_dim, causal,
//                   PosEncodingMode(pos_encoding_mode), use_fp16_qk_reduction);
//             }
//           }
//         }
//       }
//     }
//   }
// }

template <typename T>
void TestBatchPagedPrefillKernelShortContextCorrectness(bool use_fp16_qk_reduction) {
  for (size_t num_kv_heads : {4, 8, 32}) {
    for (size_t num_qo_heads : {32}) {
      for (size_t page_size : {1, 16}) {
        for (size_t head_dim : {64, 128, 256}) {
          for (size_t causal : {false, true}) {
            for (size_t pos_encoding_mode : {0, 1}) {
              _TestBatchPagedPrefillKernelShortContextCorrectness<T, T>(
                  num_kv_heads, num_qo_heads, page_size, head_dim, causal,
                  PosEncodingMode(pos_encoding_mode), use_fp16_qk_reduction);
            }
          }
        }
      }
    }
  }
}

// template <typename DTypeKV>
// void TestBatchPagedPrefillFP8KernelShortContextCorrectness(bool use_fp16_qk_reduction) {
//   for (size_t num_kv_heads : {4, 8, 32}) {
//     for (size_t num_qo_heads : {32}) {
//       for (size_t page_size : {1, 16}) {
//         for (size_t head_dim : {64, 128, 256}) {
//           for (size_t causal : {false, true}) {
//             for (size_t pos_encoding_mode : {0}) {
//               _TestBatchPagedPrefillKernelShortContextCorrectness<__half, DTypeKV>(
//                   num_kv_heads, num_qo_heads, page_size, head_dim, causal,
//                   PosEncodingMode(pos_encoding_mode), use_fp16_qk_reduction);
//             }
//           }
//         }
//       }
//     }
//   }
// }

// template <typename T>
// void TestBatchPagedPrefillKernelLongContextCorrectness(bool use_fp16_qk_reduction) {
//   for (size_t num_kv_heads : {1, 2, 8}) {
//     for (size_t group_size : {1, 3, 4, 5, 6, 7, 8}) {
//       size_t num_qo_heads = num_kv_heads * group_size;
//       for (size_t page_size : {1, 16}) {
//         for (size_t head_dim : {64, 128, 256}) {
//           for (size_t causal : {false, true}) {
//             for (size_t pos_encoding_mode : {0, 1}) {
//               _TestBatchPagedPrefillKernelLongContextCorrectness<T, T>(
//                   num_kv_heads, num_qo_heads, page_size, head_dim, causal,
//                   PosEncodingMode(pos_encoding_mode), use_fp16_qk_reduction);
//             }
//           }
//         }
//       }
//     }
//   }
// }

// template <typename DTypeKV>
// void TestBatchPagedPrefillFP8KernelLongContextCorrectness(bool use_fp16_qk_reduction) {
//   for (size_t num_kv_heads : {1, 2, 8}) {
//     for (size_t group_size : {1, 3, 4, 5, 6, 7, 8}) {
//       size_t num_qo_heads = num_kv_heads * group_size;
//       for (size_t page_size : {1, 16}) {
//         for (size_t head_dim : {64, 128, 256}) {
//           for (size_t causal : {false, true}) {
//             for (size_t pos_encoding_mode : {0}) {
//               _TestBatchPagedPrefillKernelLongContextCorrectness<__half, DTypeKV>(
//                   num_kv_heads, num_qo_heads, page_size, head_dim, causal,
//                   PosEncodingMode(pos_encoding_mode), use_fp16_qk_reduction);
//             }
//           }
//         }
//       }
//     }
//   }
// }

// template <typename T>
// void TestBatchPagedPrefillKernelZeroContextCorrectness(bool use_fp16_qk_reduction) {
//   for (size_t batch_size : {1, 4, 7, 11, 19, 37, 99}) {
//     for (size_t num_kv_heads : {1, 4}) {
//       for (size_t group_size : {1, 8}) {
//         size_t num_qo_heads = num_kv_heads * group_size;
//         for (size_t page_size : {1, 16}) {
//           for (size_t head_dim : {64, 128, 256}) {
//             for (size_t kv_len_max : {0, 3}) {
//               _TestBatchPagedPrefillKernelQMinMaxKVMinMaxCorrectness<T, T>(
//                   batch_size, num_kv_heads, num_qo_heads, page_size, head_dim,
//                   use_fp16_qk_reduction,
//                   /*q_len_min=*/1, /*q_len_max=*/3, /*kv_len_min=*/0, kv_len_max);
//             }
//           }
//         }
//       }
//     }
//   }
// }

// template <typename T>
// void TestBatchRaggedPrefillKernelCorrectness(bool use_fp16_qk_reduction) {
//   for (size_t num_kv_heads : {4, 8, 32}) {
//     for (size_t num_qo_heads : {32}) {
//       for (size_t head_dim : {64, 128, 256}) {
//         for (size_t causal : {false, true}) {
//           for (size_t pos_encoding_mode : {0, 1}) {
//             _TestBatchRaggedPrefillKernelCorrectness<T, T>(
//                 num_kv_heads, num_qo_heads, head_dim, causal, PosEncodingMode(pos_encoding_mode),
//                 use_fp16_qk_reduction);
//           }
//         }
//       }
//     }
//   }
// }

// template <typename DTypeKV>
// void TestBatchRaggedPrefillFP8KernelCorrectness(bool use_fp16_qk_reduction) {
//   for (size_t num_kv_heads : {4, 8, 32}) {
//     for (size_t num_qo_heads : {32}) {
//       for (size_t head_dim : {64, 128, 256}) {
//         for (size_t causal : {false, true}) {
//           for (size_t pos_encoding_mode : {0}) {
//             _TestBatchRaggedPrefillKernelCorrectness<__half, DTypeKV>(
//                 num_kv_heads, num_qo_heads, head_dim, causal, PosEncodingMode(pos_encoding_mode),
//                 use_fp16_qk_reduction);
//           }
//         }
//       }
//     }
//   }
// }

TEST(FlashInferCorrectnessTest, BatchPagedPrefillShortContextTestFP16) {
  TestBatchPagedPrefillKernelShortContextCorrectness<__half>(false);
}

// TEST(FlashInferCorrectnessTest, BatchPagedPrefillShortContextTestFP16QKHalfAccum) {
//   TestBatchPagedPrefillKernelShortContextCorrectness<__half>(false);
// }

// TEST(FlashInferCorrectnessTest, BatchPagedPrefillLongContextTestFP16) {
//   TestBatchPagedPrefillKernelLongContextCorrectness<__half>(false);
// }

// TEST(FlashInferCorrectnessTest, BatchPagedPrefillZeroContextTestFP16) {
//   TestBatchPagedPrefillKernelZeroContextCorrectness<__half>(false);
// }

// TEST(FlashInferCorrectnessTest, BatchPagedPrefillLongContextTestFP16QKHalfAccum) {
//   TestBatchPagedPrefillKernelLongContextCorrectness<__half>(true);
// }

// TEST(FlashInferCorrectnessTest, BatchPagedPrefillKernelCorrectnessTestOneHotFP16) {
//   TestBatchPagedPrefillKernelOneHotCorrectness<__half>(false);
// }

// TEST(FlashInferCorrectnessTest, BatchPagedPrefillKernelCorrectnessTestOneHotFP16QKHalfAccum) {
//   TestBatchPagedPrefillKernelOneHotCorrectness<__half>(true);
// }

// TEST(FlashInferCorrectnessTest, BatchRaggedPrefillTestFP16) {
//   TestBatchRaggedPrefillKernelCorrectness<__half>(false);
// }

// TEST(FlashInferCorrectnessTest, BatchRaggedPrefillTestFP16QKHalfAccum) {
//   TestBatchRaggedPrefillKernelCorrectness<__half>(true);
// }

// #ifdef FLASHINFER_ENABLE_FP8_E4M3

// TEST(FlashInferCorrectnessTest, BatchPagedPrefillShortContextTestE4M3) {
//   TestBatchPagedPrefillFP8KernelShortContextCorrectness<__hip_fp8_e4m3fnuz>(false);
// }

// TEST(FlashInferCorrectnessTest, BatchPagedPrefillLongContextTestE4M3) {
//   TestBatchPagedPrefillFP8KernelLongContextCorrectness<__hip_fp8_e4m3fnuz>(false);
// }

// #endif

// #ifdef FLASHINFER_ENABLE_FP8_E5M2

// TEST(FlashInferCorrectnessTest, BatchPagedPrefillShortContextTestE5M2) {
//   TestBatchPagedPrefillFP8KernelShortContextCorrectness<__hip_fp8_e5m2fnuz>(false);
// }

// TEST(FlashInferCorrectnessTest, BatchPagedPrefillLongContextTestE5M2) {
//   TestBatchPagedPrefillFP8KernelLongContextCorrectness<__hip_fp8_e5m2fnuz>(false);
// }
// #endif

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
