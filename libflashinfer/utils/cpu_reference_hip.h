// SPDX - FileCopyrightText : 2023 - 2025 Flashinfer team
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#pragma once

#include "flashinfer/exception.h"

#include "flashinfer/attention/generic/page.cuh"
#include "flashinfer/attention/generic/pos_enc.cuh"

#include "utils_hip.h"

#include <cmath>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <iostream>

namespace cpu_reference
{

using namespace flashinfer;

template <typename T>
inline std::vector<T> rms_norm(const T *input,
                               const T *weight,
                               size_t batch_size,
                               size_t d,
                               float eps = 1e-5)
{
    std::vector<T> output(batch_size * d);
    for (size_t i = 0; i < batch_size; ++i) {
        float sum = 0;
        for (size_t j = 0; j < d; ++j) {
            sum += float(input[i * d + j]) * float(input[i * d + j]);
        }
        float rms_rcp = 1.f / (std::sqrt(sum / float(d)) + eps);
        for (size_t j = 0; j < d; ++j) {
            output[i * d + j] =
                (float(input[i * d + j]) * rms_rcp) * float(weight[j]);
        }
    }
    return std::move(output);
}

template <typename T>
inline std::vector<T>
exclusive_prefix_sum(const T *input, size_t batch_size, size_t d)
{
    std::vector<T> output(batch_size * d);
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < d; ++j) {
            output[i * d + j] =
                (j == 0) ? 0 : output[i * d + j - 1] + input[i * d + j - 1];
        }
    }
    return std::move(output);
}

template <typename T>
inline std::vector<float> apply_llama_rope_debug(const T *input,
                                                 size_t D,
                                                 size_t offset,
                                                 float rope_scale,
                                                 float rope_theta)
{
    std::vector<float> rst(D);
    std::vector<float> permuted_input(D);
    // Print the input parameters
    // Only print for first position to avoid flood
    if (offset == 134) { // First position in your log
        std::cout << "=== CPU ROPE DEBUG ===\n";
        std::cout << "D: " << D << ", offset: " << offset
                  << ", rope_scale: " << rope_scale
                  << ", rope_theta: " << rope_theta << std::endl;

        std::cout << "CPU Frequencies vs GPU comparison:\n";
        for (size_t k = 0; k < min(4ul, D); ++k) {
            float freq_base = float(2 * (k % (D / 2))) / float(D);
            float frequency =
                1.0f / std::pow(rope_theta, freq_base); // This should match GPU
            float angle =
                (offset / rope_scale) / std::pow(rope_theta, freq_base);

            std::cout << "CPU: feature[" << k << "] freq_base=" << freq_base
                      << " frequency=" << frequency << " angle=" << angle
                      << std::endl;
        }
    }

    for (size_t k = 0; k < D; ++k) {
        permuted_input[k] =
            (k < D / 2) ? -fi::con::explicit_casting<T, float>(input[k + D / 2])
                        : fi::con::explicit_casting<T, float>(input[k - D / 2]);
    }

    for (size_t k = 0; k < D; ++k) {
        float inv_freq =
            (offset / rope_scale) /
            (std::pow(rope_theta, float(2 * (k % (D / 2))) / float(D)));
        float cos = std::cos(inv_freq);
        float sin = std::sin(inv_freq);

        if (std::is_same_v<T, half>)
            rst[k] = cos * fi::con::explicit_casting<T, float>(input[k]) +
                     sin * permuted_input[k];
    }
    return rst;
}

template <typename T>
inline std::vector<float> apply_llama_rope(const T *input,
                                           size_t D,
                                           size_t offset,
                                           float rope_scale,
                                           float rope_theta)
{
    std::vector<float> rst(D);
    std::vector<float> permuted_input(D);
    for (size_t k = 0; k < D; ++k) {

        permuted_input[k] =
            (k < D / 2) ? -fi::con::explicit_casting<T, float>(input[k + D / 2])
                        : fi::con::explicit_casting<T, float>(input[k - D / 2]);
    }

    for (size_t k = 0; k < D; ++k) {
        float inv_freq =
            (offset / rope_scale) /
            (std::pow(rope_theta, float(2 * (k % (D / 2))) / float(D)));
        float cos = std::cos(inv_freq);
        float sin = std::sin(inv_freq);

        if (std::is_same_v<T, half>)
            rst[k] = cos * fi::con::explicit_casting<T, float>(input[k]) +
                     sin * permuted_input[k];
    }
    return rst;
}

template <typename dtype_q, typename dtype_kv>
std::vector<float> compute_qk(const std::vector<dtype_q> &q,
                              const std::vector<dtype_kv> &k,
                              size_t qo_len,
                              size_t kv_len,
                              size_t num_qo_heads,
                              size_t num_kv_heads,
                              size_t head_dim,
                              QKVLayout kv_layout = QKVLayout::kHND)
{

    assert(num_qo_heads % num_kv_heads == 0);
    assert(q.size() == qo_len * num_qo_heads * head_dim);
    assert(k.size() == kv_len * num_kv_heads * head_dim);

    std::vector<float> qk_scores(qo_len * num_qo_heads * kv_len);

    DISPATCH_head_dim(head_dim, HEAD_DIM, {
        tensor_info_t info(qo_len, kv_len, num_qo_heads, num_kv_heads,
                           kv_layout, HEAD_DIM);

        for (size_t qo_head_idx = 0; qo_head_idx < num_qo_heads; ++qo_head_idx)
        {
            const size_t kv_head_idx = qo_head_idx / info.get_group_size();

            for (size_t q_idx = 0; q_idx < qo_len; ++q_idx) {
                for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
                    float qk_score = 0.0f;

                    // Pure Q*K^T - NO scaling (matching HIP compute_qk)
                    for (size_t feat_idx = 0; feat_idx < head_dim; ++feat_idx) {
                        qk_score += fi::con::explicit_casting<dtype_q, float>(
                                        q[info.get_q_elem_offset(
                                            q_idx, qo_head_idx, feat_idx)]) *
                                    fi::con::explicit_casting<dtype_kv, float>(
                                        k[info.get_kv_elem_offset(
                                            kv_idx, kv_head_idx, feat_idx)]);
                    }

                    size_t output_idx =
                        qo_head_idx * qo_len * kv_len + q_idx * kv_len + kv_idx;
                    qk_scores[output_idx] = qk_score;
                }
            }
        }
    });

    return qk_scores;
}

template <typename dtype_q, typename dtype_kv, typename dtype_out>
std::vector<dtype_out>
single_mha(const std::vector<dtype_q> &q,
           const std::vector<dtype_kv> &k,
           const std::vector<dtype_kv> &v,
           size_t qo_len,
           size_t kv_len,
           size_t num_qo_heads,
           size_t num_kv_heads,
           size_t head_dim,
           bool causal = true,
           QKVLayout kv_layout = QKVLayout::kHND,
           PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone,
           float logits_soft_cap = 8.0f,
           float rope_scale = 1.f,
           float rope_theta = 1e4,
           bool use_soft_cap = false)
{
    assert(qo_len <= kv_len);
    assert(num_qo_heads % num_kv_heads == 0);
    float sm_scale = 1.f / std::sqrt(float(head_dim));
    // float sm_scale = 1.0;
    std::vector<dtype_out> o(qo_len * num_qo_heads * head_dim);
    std::vector<float> att(kv_len);
    std::vector<float> q_rotary_local(head_dim);
    std::vector<float> k_rotary_local(head_dim);

    float soft_cap_pre_tanh_scale = sm_scale / logits_soft_cap;

    DISPATCH_head_dim(head_dim, HEAD_DIM, {
        tensor_info_t info(qo_len, kv_len, num_qo_heads, num_kv_heads,
                           kv_layout, HEAD_DIM);
#if Debug1
        std::cout << "DEBUG: Original Q (CPU): " << '\n';
        for (auto i = 0ul; i < 16; ++i) {
            for (int j = 0; j < 64; ++j) {
                std::cout << (float)q[info.get_q_elem_offset(i, 0, j)] << " ";
            }
            std::cout << std::endl;
            //  q[info.get_q_elem_offset(q_idx, qo_head_idx, feat_idx)
            // std::cout << (float)q[info.get_q_elem_offset(0, 0, i)] << " ";
        }
        std::cout << std::endl;

        std::cout << "DEBUG: Original K (CPU): " << '\n';
        for (auto i = 0ul; i < 64; ++i) {
            for (int j = 0ul; j < 64; ++j) {
                std::cout << (float)k[info.get_kv_elem_offset(i, 0, j)] << " ";
            }
            std::cout << std::endl;
            //  q[info.get_q_elem_offset(q_idx, qo_head_idx, feat_idx)
        }
        std::cout << std::endl;
#endif
        for (size_t qo_head_idx = 0; qo_head_idx < num_qo_heads; ++qo_head_idx)
        {
            const size_t kv_head_idx = qo_head_idx / info.get_group_size();
            for (size_t q_idx = 0; q_idx < qo_len; ++q_idx) {
                float max_val = -5e4;
                if (pos_encoding_mode == PosEncodingMode::kRoPELlama) {
                    q_rotary_local = std::move(cpu_reference::apply_llama_rope(
                        q.data() +
                            info.get_q_elem_offset(q_idx, qo_head_idx, 0),
                        head_dim, q_idx + kv_len - qo_len, rope_scale,
                        rope_theta));
                }
                for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
                    att[kv_idx] = 0.;
                    switch (pos_encoding_mode) {
                    case PosEncodingMode::kNone:
                    {
                        for (size_t feat_idx = 0; feat_idx < head_dim;
                             ++feat_idx)
                        {
                            att[kv_idx] +=
                                fi::con::explicit_casting<dtype_q, float>(
                                    q[info.get_q_elem_offset(q_idx, qo_head_idx,
                                                             feat_idx)]) *
                                fi::con::explicit_casting<dtype_kv, float>(
                                    k[info.get_kv_elem_offset(
                                        kv_idx, kv_head_idx, feat_idx)]) *
                                sm_scale;
                        }
                        break;
                    }
                    case PosEncodingMode::kRoPELlama:
                    {
                        k_rotary_local =
                            std::move(cpu_reference::apply_llama_rope(
                                k.data() + info.get_kv_elem_offset(
                                               kv_idx, kv_head_idx, 0),
                                head_dim, kv_idx, rope_scale, rope_theta));
                        for (size_t feat_idx = 0; feat_idx < head_dim;
                             ++feat_idx)
                        {
                            att[kv_idx] += q_rotary_local[feat_idx] *
                                           k_rotary_local[feat_idx] * sm_scale;
                        }
                        break;
                    }
                    default:
                    {
                        std::ostringstream err_msg;
                        err_msg << "Unsupported rotary mode.";
                        FLASHINFER_ERROR(err_msg.str());
                    }
                    }
                    // apply mask
                    if (causal && kv_idx > kv_len + q_idx - qo_len) {
                        att[kv_idx] = -5e4;
                    }
                    max_val = std::max(max_val, att[kv_idx]);
                }

#if Debug
                if (qo_head_idx == 0) {
                    // for qo_len = 128, each warp on the GPU will store 128/4,
                    // that is, 32 attention scores. For CDNA3, these 32 scores
                    // are spread across 4 threads.
                    for (auto i = 0ul; i < 128; ++i) {
                        std::cout << att[i] << " ";
                    }
                    std::cout << std::endl;
                }
#endif
                // exp minus max
                float denom = 0;
                for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
                    att[kv_idx] = std::exp(att[kv_idx] - max_val);
                    denom += att[kv_idx];
                }

#if Debug1
                if (qo_head_idx == 0) {
                    // for qo_len = 128, each warp on the GPU will store 128/4,
                    // that is, 32 attention scores. For CDNA3, these 32 scores
                    // are spread across 4 threads.
                    for (auto i = 0ul; i < 128; ++i) {
                        std::cout << att[i] << " ";
                    }
                    std::cout << std::endl;
                }
#endif

                // divide by denom
                for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
                    att[kv_idx] /= denom;
                }

                for (size_t feat_idx = 0; feat_idx < head_dim; ++feat_idx) {
                    float o_float = 0.;
                    for (size_t kv_idx = 0; kv_idx < kv_len; ++kv_idx) {
                        o_float += att[kv_idx] *
                                   fi::con::explicit_casting<dtype_kv, float>(
                                       v[info.get_kv_elem_offset(
                                           kv_idx, kv_head_idx, feat_idx)]);
                    }
                    o[info.get_o_elem_offset(q_idx, qo_head_idx, feat_idx)] =
                        fi::con::explicit_casting<float, dtype_out>(o_float);
                }
            }
        }
    });
    return std::move(o);
}

template <typename T, typename IdxType>
void append_paged_kv_cache(paged_kv_t<T, IdxType> page_cpu,
                           const std::vector<std::vector<T>> &keys,
                           const std::vector<std::vector<T>> &values,
                           const std::vector<IdxType> &append_indptr)
{
    size_t batch_size = page_cpu.batch_size;
    size_t num_heads = page_cpu.num_heads;
    size_t head_dim = page_cpu.head_dim;
    size_t page_size = page_cpu.page_size;
    for (size_t i = 0; i < batch_size; ++i) {
        const std::vector<T> &ki = keys[i];
        const std::vector<T> &vi = values[i];
        size_t append_seq_len = append_indptr[i + 1] - append_indptr[i];
        size_t num_pages_i = page_cpu.indptr[i + 1] - page_cpu.indptr[i];
        size_t seq_len =
            (num_pages_i - 1) * page_size + page_cpu.last_page_len[i];
        assert(append_seq_len <= seq_len);
        size_t append_start = seq_len - append_seq_len;

        for (size_t j = 0; j < append_seq_len; ++j) {
            size_t page_seq_idx = j + append_start;
            size_t page_idx =
                page_cpu.indices[page_cpu.indptr[i] + page_seq_idx / page_size];
            size_t entry_idx = page_seq_idx % page_size;
            for (size_t h = 0; h < num_heads; ++h) {
                std::copy(ki.begin() + (j * num_heads + h) * head_dim,
                          ki.begin() + (j * num_heads + h + 1) * head_dim,
                          page_cpu.k_data + page_cpu.get_elem_offset(
                                                page_idx, h, entry_idx, 0));
                std::copy(vi.begin() + (j * num_heads + h) * head_dim,
                          vi.begin() + (j * num_heads + h + 1) * head_dim,
                          page_cpu.v_data + page_cpu.get_elem_offset(
                                                page_idx, h, entry_idx, 0));
            }
        }
    }
}

} // namespace cpu_reference
