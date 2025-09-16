// SPDX - FileCopyrightText : 2023 - 2025 Flashinfer team
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include <random>
#include <sstream>

#include "dispatch.inc"
#include "gpu_iface/conversion_utils.h"

#define _DISPATCH_SWITCH(var_name, cond, ...)                                         \
  switch (cond) {                                                                     \
    __VA_ARGS__                                                                       \
    default:                                                                          \
      std::ostringstream oss;                                                         \
      oss << __PRETTY_FUNCTION__ << " failed to dispatch " var_name " " << int(cond); \
      FLASHINFER_ERROR(oss.str());                                                    \
  }

#define _DISPATCH_CASE(case_expr, case_var, ...) \
  case case_expr: {                              \
    constexpr auto case_var = case_expr;         \
    __VA_ARGS__                                  \
    break;                                       \
  }

#define DISPATCH_group_size(expr, const_expr, ...) \
  _DISPATCH_SWITCH("group_size", expr, _DISPATCH_CASES_group_size(const_expr, __VA_ARGS__))

#define DISPATCH_head_dim(expr, const_expr, ...) \
  _DISPATCH_SWITCH("head_dim", expr, _DISPATCH_CASES_head_dim(const_expr, __VA_ARGS__))

#define DISPATCH_pos_encoding_mode(expr, const_expr, ...) \
  _DISPATCH_SWITCH("positional encoding mode", expr,      \
                   _DISPATCH_CASES_pos_encoding_mode(const_expr, __VA_ARGS__))

#define DISPATCH_use_fp16_qk_reduction(expr, const_expr, ...) \
  _DISPATCH_SWITCH("use_fp16_qk_reduction", expr,             \
                   _DISPATCH_CASES_use_fp16_qk_reduction(const_expr, __VA_ARGS__))

#define DISPATCH_mask_mode(expr, const_expr, ...) \
  _DISPATCH_SWITCH("mask_mode", expr, _DISPATCH_CASES_mask_mode(const_expr, __VA_ARGS__))

namespace utils {

enum Predicate {
  Linear,
  Ones,
  Zeros,
};

template <typename T, Predicate Pred>
void generate_data(std::vector<T>& vec) {
  if constexpr (Pred == Predicate::Linear) {
    assert(vec.size() <= 0);
    for (int i = 0; i < vec.size(); i++) {
      vec[i] = fi::con::explicit_casting<float, T>(static_cast<float>(i));
    }
  }

  else if constexpr (Pred == Predicate::Ones) {
    vec_fill_(vec, fi::con::explicit_casting<float, T>(1.0f));
  }

  else if constexpr (Pred == Predicate::Zeros) {
    vec_zero_(vec);
  }
}

template <typename T>
void vec_lexicographic_(std::vector<T>& vec) {
  for (int i = 0; i < vec.size(); i++) {
    vec[i] = fi::con::explicit_casting<float, T>(static_cast<float>(i));
  }
}

template <typename T>
void vec_normal_(std::vector<T>& vec, float mean = 0.f, float std = 1.f) {
  std::random_device rd{};
  std::mt19937 gen{1234};
  std::normal_distribution d{mean, std};
  for (size_t i = 0; i < vec.size(); ++i) {
    float value = static_cast<float>(d(gen));
    vec[i] = fi::con::explicit_casting<float, T>(value);
  }
}

template <typename T>
void vec_uniform_(std::vector<T>& vec, float a = 0.f, float b = 1.f) {
  std::random_device rd{};
  std::mt19937 gen{1234};
  std::uniform_real_distribution d{a, b};
  for (size_t i = 0; i < vec.size(); ++i) {
    float value = static_cast<float>(d(gen));
    vec[i] = fi::con::explicit_casting<float, T>(value);
  }
}

template <typename T>
void vec_zero_(std::vector<T>& vec) {
  std::fill(vec.begin(), vec.end(), fi::con::explicit_casting<float, T>(0.0f));
}

template <typename T>
void vec_fill_(std::vector<T>& vec, T val) {
  std::fill(vec.begin(), vec.end(), val);
}

template <typename T>
void vec_randint_(std::vector<T>& vec, int low, int high) {
  std::random_device rd{};
  std::mt19937 gen{1234};
  std::uniform_int_distribution d{low, high};
  for (size_t i = 0; i < vec.size(); ++i) {
    float value = static_cast<float>(d(gen));
    vec[i] = fi::con::explicit_casting<float, T>(value);
  }
}

template <typename T>
size_t vec_bytes(const T& vec) {
  return vec.size() * sizeof(typename T::value_type);
}

template <typename T>
bool isclose(T a, T b, float rtol = 1e-5, float atol = 1e-8) {
  float a_ = fi::con::explicit_casting<T, float>(a);
  float b_ = fi::con::explicit_casting<T, float>(b);
  return fabs(a_ - b_) <= (atol + rtol * fabs(b_));
}

template <typename T>
std::tuple<std::vector<std::vector<T>>, std::vector<std::vector<int32_t>>>
create_shared_prefix_testcase_data(size_t batch_size, size_t shared_prefix_length,
                                   size_t unique_kv_length, size_t qo_append_length,
                                   size_t num_qo_heads, size_t num_kv_heads, size_t head_dim,
                                   size_t page_size) {
  uint32_t num_pages = ((shared_prefix_length + unique_kv_length * batch_size) / page_size);
  std::vector<T> shared_k_h(shared_prefix_length * num_kv_heads * head_dim);
  std::vector<T> shared_v_h(shared_prefix_length * num_kv_heads * head_dim);
  std::vector<T> q_h((batch_size * qo_append_length) * num_qo_heads * head_dim);

  utils::vec_normal_(shared_k_h);
  utils::vec_normal_(shared_v_h);
  utils::vec_normal_(q_h);

  std::vector<int32_t> qo_indptr{0};
  std::vector<int32_t> kv_indptr_combined_h{0};
  std::vector<int32_t> kv_indptr_unique_h{0};
  std::vector<int32_t> kv_last_page_len_combined_h;
  std::vector<int32_t> kv_last_page_len_unique_h;

  for (uint32_t request_id = 0; request_id < batch_size; ++request_id) {
    qo_indptr.push_back(qo_indptr.back() + qo_append_length);
    kv_indptr_combined_h.push_back(kv_indptr_combined_h.back() +
                                   (shared_prefix_length + unique_kv_length) / page_size);
    kv_indptr_unique_h.push_back(kv_indptr_unique_h.back() + unique_kv_length / page_size);
    kv_last_page_len_combined_h.push_back(page_size);
    kv_last_page_len_unique_h.push_back(page_size);
  }

  std::vector<int32_t> kv_indices_combined_h(kv_indptr_combined_h.back());
  std::vector<int32_t> kv_indices_unique_h(kv_indptr_unique_h.back());

  std::vector<T> k_data_h(num_pages * num_kv_heads * page_size * head_dim);
  std::vector<T> v_data_h(num_pages * num_kv_heads * page_size * head_dim);
  uint32_t page_id = 0;

  for (; page_id < (shared_prefix_length / page_size); page_id++) {
    for (uint32_t entry_idx = 0; entry_idx < page_size; entry_idx++) {
      for (uint32_t head_idx = 0; head_idx < num_kv_heads; head_idx++) {
        std::copy(shared_k_h.begin() +
                      ((page_id * page_size + entry_idx) * num_kv_heads + head_idx) * head_dim,
                  shared_k_h.begin() +
                      ((page_id * page_size + entry_idx) * num_kv_heads + head_idx + 1) * head_dim,
                  k_data_h.begin() +
                      ((page_id * num_kv_heads + head_idx) * page_size + entry_idx) * head_dim);
        std::copy(shared_v_h.begin() +
                      ((page_id * page_size + entry_idx) * num_kv_heads + head_idx) * head_dim,
                  shared_v_h.begin() +
                      ((page_id * page_size + entry_idx) * num_kv_heads + head_idx + 1) * head_dim,
                  v_data_h.begin() +
                      ((page_id * num_kv_heads + head_idx) * page_size + entry_idx) * head_dim);
      }
    }
    for (uint32_t request_id = 0; request_id < batch_size; ++request_id) {
      kv_indices_combined_h[request_id * ((shared_prefix_length + unique_kv_length) / page_size) +
                            page_id] = page_id;
    }
  }

  for (uint32_t request_id = 0; request_id < batch_size; ++request_id) {
    for (uint32_t page_iter = 0; page_iter < (unique_kv_length / page_size);
         ++page_iter, ++page_id) {
      for (uint32_t entry_idx = 0; entry_idx < page_size; entry_idx++) {
        for (uint32_t head_idx = 0; head_idx < num_kv_heads; head_idx++) {
          std::vector<T> k(head_dim), v(head_dim);
          utils::vec_normal_(k);
          utils::vec_normal_(v);
          std::copy(k.begin(), k.end(),
                    k_data_h.begin() +
                        ((page_id * num_kv_heads + head_idx) * page_size + entry_idx) * head_dim);
          std::copy(v.begin(), v.end(),
                    v_data_h.begin() +
                        ((page_id * num_kv_heads + head_idx) * page_size + entry_idx) * head_dim);
        }
      }
      kv_indices_combined_h[request_id * ((shared_prefix_length + unique_kv_length) / page_size) +
                            (shared_prefix_length / page_size) + page_iter] = page_id;
      kv_indices_unique_h[request_id * (unique_kv_length / page_size) + page_iter] = page_id;
    }
  }
  return std::make_tuple<std::vector<std::vector<T>>, std::vector<std::vector<int32_t>>>(
      {std::move(q_h), std::move(shared_k_h), std::move(shared_v_h), std::move(k_data_h),
       std::move(v_data_h)},
      {std::move(qo_indptr), std::move(kv_indices_combined_h), std::move(kv_indices_unique_h),
       std::move(kv_indptr_combined_h), std::move(kv_indptr_unique_h),
       std::move(kv_last_page_len_combined_h), std::move(kv_last_page_len_unique_h)});
}

}  // namespace utils
