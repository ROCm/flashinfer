// SPDX - FileCopyrightText : 2023 - 2025 Flashinfer team
// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#include "../../utils/cpu_reference_hip.h"
#include "../../utils/flashinfer_prefill_ops.hip.h"
#include "../../utils/utils_hip.h"
#include "flashinfer/attention/generic/prefill.cuh"
#include "gpu_iface/gpu_runtime_compat.hpp"

#include <type_traits>

#include <gtest/gtest.h>

#define HIP_ENABLE_WARP_SYNC_BUILTINS 1

using namespace flashinfer;

#if 0
template <typename DTypeQ, typename DTypeKV, typename DTypeO>
void _TestComputeQKCorrectness(size_t qo_len,
                               size_t kv_len,
                               size_t num_qo_heads,
                               size_t num_kv_heads,
                               size_t head_dim,
                               bool causal,
                               QKVLayout kv_layout,
                               PosEncodingMode pos_encoding_mode,
                               bool use_fp16_qk_reduction,
                               float rtol = 1e-3,
                               float atol = 1e-3)
{
    // std::cout << "Testing compute_qk: qo_len=" << qo_len
    //           << ", kv_len=" << kv_len << ", num_qo_heads=" << num_qo_heads
    //           << ", num_kv_heads=" << num_kv_heads << ", head_dim=" << head_dim
    //           << std::endl;

    // Generate test data (same as original test)
    std::vector<DTypeQ> q(qo_len * num_qo_heads * head_dim);
    std::vector<DTypeKV> k(kv_len * num_kv_heads * head_dim);
    std::vector<DTypeKV> v(kv_len * num_kv_heads *
                           head_dim); // Still needed for params
    std::vector<DTypeO> o(qo_len * num_qo_heads *
                          head_dim); // Still needed for params

    utils::vec_normal_(q);
    utils::vec_normal_(k);
    utils::vec_normal_(v); // Initialize even though we won't use it
    utils::vec_zero_(o);

    // GPU memory allocation (same pattern as original)
    DTypeQ *q_d;
    FI_GPU_CALL(hipMalloc(&q_d, q.size() * sizeof(DTypeQ)));
    FI_GPU_CALL(hipMemcpy(q_d, q.data(), q.size() * sizeof(DTypeQ),
                          hipMemcpyHostToDevice));

    DTypeKV *k_d;
    FI_GPU_CALL(hipMalloc(&k_d, k.size() * sizeof(DTypeKV)));
    FI_GPU_CALL(hipMemcpy(k_d, k.data(), k.size() * sizeof(DTypeKV),
                          hipMemcpyHostToDevice));

    DTypeKV *v_d;
    FI_GPU_CALL(hipMalloc(&v_d, v.size() * sizeof(DTypeKV)));
    FI_GPU_CALL(hipMemcpy(v_d, v.data(), v.size() * sizeof(DTypeKV),
                          hipMemcpyHostToDevice));

    DTypeO *o_d;
    FI_GPU_CALL(hipMalloc(&o_d, o.size() * sizeof(DTypeO)));
    FI_GPU_CALL(hipMemcpy(o_d, o.data(), o.size() * sizeof(DTypeO),
                          hipMemcpyHostToDevice));

    DTypeO *tmp_d;
    FI_GPU_CALL(hipMalloc(&tmp_d, 16 * 1024 * 1024 * sizeof(DTypeO)));

    // Allocate output buffer for QK scores
    const size_t qk_output_size = qo_len * kv_len * num_qo_heads;
    float *qk_scores_d;
    FI_GPU_CALL(hipMalloc(&qk_scores_d, qk_output_size * sizeof(float)));

    // std::cout << "Debug: Kernel launch parameters:" << std::endl;
    // std::cout << "  qo_len=" << qo_len << ", kv_len=" << kv_len << std::endl;
    // std::cout << "  num_qo_heads=" << num_qo_heads
    //           << ", num_kv_heads=" << num_kv_heads << std::endl;
    // std::cout << "  head_dim=" << head_dim << std::endl;
    // std::cout << "  qk_output_size=" << qk_output_size << std::endl;
    // std::cout << "  Launching ComputeQKStubCaller..." << std::endl;

    // Call ComputeQKStubCaller instead of SinglePrefillWithKVCache
    hipError_t status =
        flashinfer::ComputeQKStubCaller<DTypeQ, DTypeKV, DTypeO>(
            q_d, k_d, v_d, o_d, tmp_d,
            /*lse=*/nullptr, qk_scores_d, // Add qk_scores_d parameter
            num_qo_heads, num_kv_heads, qo_len, kv_len, head_dim, causal,
            kv_layout, pos_encoding_mode, use_fp16_qk_reduction);

    // std::cout << "  Kernel launch status: " << hipGetErrorString(status)
    //           << std::endl;
    EXPECT_EQ(status, hipSuccess)
        << "ComputeQKStubCaller kernel launch failed, error message: "
        << hipGetErrorString(status);

    // Get GPU QK scores
    std::vector<float> gpu_qk_scores(qk_output_size);
    FI_GPU_CALL(hipMemcpy(gpu_qk_scores.data(), qk_scores_d,
                          qk_output_size * sizeof(float),
                          hipMemcpyDeviceToHost));

    // Check if GPU output is not empty
    bool isEmpty = gpu_qk_scores.empty();
    EXPECT_EQ(isEmpty, false) << "GPU QK scores vector is empty";

    // Compute CPU reference using our cpu_reference::compute_qk
    std::vector<float> cpu_qk_scores = cpu_reference::compute_qk(
        q, k, qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim, kv_layout);

    // Validate results (same pattern as original test)
    size_t num_results_error_atol = 0;
    bool nan_detected = false;

    // Compare element-by-element
    size_t comparison_size =
        std::min(gpu_qk_scores.size(), cpu_qk_scores.size());
    for (size_t i = 0; i < comparison_size; ++i) {
        float gpu_val = gpu_qk_scores[i];
        float cpu_val = cpu_qk_scores[i];

        if (isnan(gpu_val)) {
            nan_detected = true;
        }

        if (!utils::isclose(cpu_val, gpu_val, rtol, atol)) {
            num_results_error_atol++;
            if (num_results_error_atol <= 10)
            { // Only print first 10 mismatches
                std::cout << "QK mismatch at i=" << i << ", cpu_val=" << cpu_val
                          << ", gpu_val=" << gpu_val << std::endl;
            }
        }
    }
#if 0
    // Calculate and report accuracy
    float result_accuracy =
        1.0f - float(num_results_error_atol) / float(comparison_size);

    std::cout << "compute_qk results: num_qo_heads=" << num_qo_heads
              << ", num_kv_heads=" << num_kv_heads << ", qo_len=" << qo_len
              << ", kv_len=" << kv_len << ", head_dim=" << head_dim
              << ", causal=" << causal
              << ", kv_layout=" << QKVLayoutToString(kv_layout)
              << ", pos_encoding_mode="
              << PosEncodingModeToString(pos_encoding_mode)
              << ", qk_accuracy=" << result_accuracy << " ("
              << num_results_error_atol << "/" << comparison_size
              << " mismatches)" << std::endl;

    // Print some sample values for debugging
    std::cout << "Sample QK scores (first 10): GPU vs CPU" << std::endl;
    for (size_t i = 0; i < std::min(size_t(10), comparison_size); ++i) {
        std::cout << "  [" << i << "] GPU=" << gpu_qk_scores[i]
                  << ", CPU=" << cpu_qk_scores[i] << std::endl;
    }

    // Assertions (slightly relaxed for initial testing)
    EXPECT_GT(result_accuracy, 0.80)
        << "compute_qk accuracy too low"; // Start with 80%
    EXPECT_FALSE(nan_detected) << "NaN detected in compute_qk results";
#endif
    // Cleanup
    FI_GPU_CALL(hipFree(q_d));
    FI_GPU_CALL(hipFree(k_d));
    FI_GPU_CALL(hipFree(v_d));
    FI_GPU_CALL(hipFree(o_d));
    FI_GPU_CALL(hipFree(tmp_d));
    FI_GPU_CALL(hipFree(qk_scores_d));
}

#endif

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
void _TestSinglePrefillKernelCorrectness(size_t qo_len,
                                         size_t kv_len,
                                         size_t num_qo_heads,
                                         size_t num_kv_heads,
                                         size_t head_dim,
                                         bool causal,
                                         QKVLayout kv_layout,
                                         PosEncodingMode pos_encoding_mode,
                                         bool use_fp16_qk_reduction,
                                         uint32_t debug_thread_id,
                                         float rtol = 1e-3,
                                         float atol = 1e-3)
{
    std::vector<DTypeQ> q(qo_len * num_qo_heads * head_dim);
    std::vector<DTypeKV> k(kv_len * num_kv_heads * head_dim);
    std::vector<DTypeKV> v(kv_len * num_kv_heads * head_dim);
    std::vector<DTypeO> o(qo_len * num_qo_heads * head_dim);

    utils::vec_normal_(q);
    utils::vec_normal_(k);
    utils::vec_normal_(v);
    // utils::vec_lexicographic_(q);
    // utils::vec_lexicographic_(k);
    // utils::vec_fill_(v, __float2half(1.0f));
    utils::vec_zero_(o);

    DTypeQ *q_d;
    FI_GPU_CALL(hipMalloc(&q_d, q.size() * sizeof(DTypeQ)));
    FI_GPU_CALL(hipMemcpy(q_d, q.data(), q.size() * sizeof(DTypeQ),
                          hipMemcpyHostToDevice));

    DTypeKV *k_d;
    FI_GPU_CALL(hipMalloc(&k_d, k.size() * sizeof(DTypeKV)));
    FI_GPU_CALL(hipMemcpy(k_d, k.data(), k.size() * sizeof(DTypeKV),
                          hipMemcpyHostToDevice));

    DTypeKV *v_d;
    FI_GPU_CALL(hipMalloc(&v_d, v.size() * sizeof(DTypeKV)));
    FI_GPU_CALL(hipMemcpy(v_d, v.data(), v.size() * sizeof(DTypeKV),
                          hipMemcpyHostToDevice));

    DTypeO *o_d;
    FI_GPU_CALL(hipMalloc(&o_d, o.size() * sizeof(DTypeO)));
    FI_GPU_CALL(hipMemcpy(o_d, o.data(), o.size() * sizeof(DTypeO),
                          hipMemcpyHostToDevice));

    DTypeO *tmp_d;
    FI_GPU_CALL(hipMalloc(&tmp_d, 16 * 1024 * 1024 * sizeof(DTypeO)));

    hipError_t status =
        flashinfer::SinglePrefillWithKVCache<DTypeQ, DTypeKV, DTypeO>(
            q_d, k_d, v_d, o_d, tmp_d,
            /*lse=*/nullptr, num_qo_heads, num_kv_heads, qo_len, kv_len,
            head_dim, causal, kv_layout, pos_encoding_mode,
            use_fp16_qk_reduction, debug_thread_id);

    EXPECT_EQ(status, hipSuccess)
        << "SinglePrefillWithKVCache kernel launch failed, error message: "
        << hipGetErrorString(status);

    std::vector<DTypeO> o_h(o.size());
    FI_GPU_CALL(hipMemcpy(o_h.data(), o_d, o_h.size() * sizeof(DTypeO),
                          hipMemcpyDeviceToHost));

    // Print the first 10 elements of the output vector for debugging
    //  std::cout << "Output vector (first 10 elements):";
    //  std::cout << "[" << std::endl;
    //  for (int i = 0; i < 10; ++i) {
    //      std::cout << fi::con::explicit_casting<DTypeO, float>(o_h[i]) << "
    //      ";
    //  }
    //  std::cout << "]" << std::endl;

    bool isEmpty = o_h.empty();
    EXPECT_EQ(isEmpty, false) << "Output vector is empty";

    std::vector<float> att_out;
    std::vector<DTypeO> o_ref =
        cpu_reference::single_mha<DTypeQ, DTypeKV, DTypeO>(
            q, k, v, qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim,
            causal, kv_layout, pos_encoding_mode);
    size_t num_results_error_atol = 0;
    bool nan_detected = false;

    for (size_t i = 0; i < o_ref.size(); ++i) {
        float o_h_val = fi::con::explicit_casting<DTypeO, float>(o_h[i]);
        float o_ref_val = fi::con::explicit_casting<DTypeO, float>(o_ref[i]);

        if (isnan(o_h_val)) {
            nan_detected = true;
        }

        num_results_error_atol +=
            (!utils::isclose(o_ref_val, o_h_val, rtol, atol));
        // if (!utils::isclose(o_ref_val, o_h_val, rtol, atol)) {
        //     std::cout << "i=" << i << ", o_ref[i]=" << o_ref_val
        //               << ", o_h[i]=" << o_h_val << std::endl;
        // }
    }
    // std::cout<<"Printing att_out vector:\n";
    // for(auto i: att_out) {
    //     std::cout << i << "\n";
    // }
#if 0
    float result_accuracy =
        1. - float(num_results_error_atol) / float(o_ref.size());
    std::cout << "num_qo_heads=" << num_qo_heads
              << ", num_kv_heads=" << num_kv_heads << ", qo_len=" << qo_len
              << ", kv_len=" << kv_len << ", head_dim=" << head_dim
              << ", causal=" << causal
              << ", kv_layout=" << QKVLayoutToString(kv_layout)
              << ", pos_encoding_mode="
              << PosEncodingModeToString(pos_encoding_mode)
              << ", result_accuracy=" << result_accuracy << std::endl;

    EXPECT_GT(result_accuracy, 0.90) << "Result correctness test failed.";
    EXPECT_FALSE(nan_detected) << "Nan detected in the result.";
#endif
    FI_GPU_CALL(hipFree(q_d));
    FI_GPU_CALL(hipFree(k_d));
    FI_GPU_CALL(hipFree(v_d));
    FI_GPU_CALL(hipFree(o_d));
    FI_GPU_CALL(hipFree(tmp_d));
}

// template <typename DTypeIn, typename DTypeO>
// void TestSinglePrefillKernelLongContextCorrectness(bool
// use_fp16_qk_reduction)
// {
//     for (size_t qo_len : {1, 31, 63, 127}) {
//         for (size_t kv_len : {31717}) {
//             for (size_t num_heads : {1}) {
//                 for (size_t head_dim : {64, 128, 256}) {
//                     for (bool causal : {false, true}) {
//                         for (size_t pos_encoding_mode : {0, 1}) {
//                             for (size_t kv_layout : {0, 1}) {
//                                 _TestSinglePrefillKernelCorrectness<
//                                     DTypeIn, DTypeIn, DTypeO>(
//                                     qo_len, kv_len, num_heads, num_heads,
//                                     head_dim, causal, QKVLayout(kv_layout),
//                                     PosEncodingMode(pos_encoding_mode),
//                                     use_fp16_qk_reduction);
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }
//***********************************************************************
// The following tests are disabled because we dont support fp8 <-> float
// conversions

// template <typename DTypeKV>
// void TestSinglePrefillFP8KernelLongContextCorrectness(bool
// use_fp16_qk_reduction) {
//   for (size_t qo_len : {1, 31, 63, 127}) {
//     for (size_t kv_len : {31717}) {
//       for (size_t num_heads : {1}) {
//         for (size_t head_dim : {64, 128, 256}) {
//           for (bool causal : {false, true}) {
//             for (size_t pos_encoding_mode : {0}) {
//               for (size_t kv_layout : {0, 1}) {
//                 _TestSinglePrefillKernelCorrectness<__half, DTypeKV, __half>(
//                     qo_len, kv_len, num_heads, num_heads, head_dim, causal,
//                     QKVLayout(kv_layout), PosEncodingMode(pos_encoding_mode),
//                     use_fp16_qk_reduction);
//               }
//             }
//           }
//         }
//       }
//     }
//   }
// }

// template <typename DTypeIn, typename DTypeO>
// void TestSinglePrefillKernelShortContextCorrectness(bool
// use_fp16_qk_reduction)
// {
//     float rtol = std::is_same<DTypeO, __hip_bfloat16>::value ? 1e-2 : 1e-3;
//     float atol = std::is_same<DTypeO, __hip_bfloat16>::value ? 1e-2 : 1e-3;
//     for (size_t qkv_len : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}) {
//         for (size_t num_qo_heads : {32}) {
//             for (size_t num_kv_heads : {4, 8, 32}) {
//                 for (size_t head_dim : {64, 128, 256}) {
//                     for (bool causal : {false, true}) {
//                         for (size_t pos_encoding_mode : {0, 1}) {
//                             for (size_t kv_layout : {0, 1}) {
//                                 _TestSinglePrefillKernelCorrectness<
//                                     DTypeIn, DTypeIn, DTypeO>(
//                                     qkv_len, qkv_len, num_qo_heads,
//                                     num_kv_heads, head_dim, causal,
//                                     QKVLayout(kv_layout),
//                                     PosEncodingMode(pos_encoding_mode),
//                                     use_fp16_qk_reduction, rtol, atol);
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

//***********************************************************************
// The following tests are disabled because we dont support fp8 <-> float
// conversions

// template <typename DTypeKV>
// void TestSinglePrefillFP8KernelShortContextCorrectness(bool
// use_fp16_qk_reduction) {
//   float rtol = 1e-3;
//   float atol = 1e-3;
//   for (size_t qkv_len : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}) {
//     for (size_t num_qo_heads : {32}) {
//       for (size_t num_kv_heads : {4, 8, 32}) {
//         for (size_t head_dim : {64, 128, 256}) {
//           for (bool causal : {false, true}) {
//             for (size_t pos_encoding_mode : {0}) {
//               for (size_t kv_layout : {0, 1}) {
//                 _TestSinglePrefillKernelCorrectness<__half, DTypeKV, __half>(
//                     qkv_len, qkv_len, num_qo_heads, num_kv_heads, head_dim,
//                     causal, QKVLayout(kv_layout),
//                     PosEncodingMode(pos_encoding_mode),
//                     use_fp16_qk_reduction, rtol, atol);
//               }
//             }
//           }
//         }
//       }
//     }
//   }
// }

// template <typename DTypeIn, typename DTypeO>
// void TestSinglePrefillKernelCorrectness(bool use_fp16_qk_reduction)
// {
//     for (size_t qo_len : {399, 400, 401}) {
//         for (size_t kv_len : {533, 534, 535}) {
//             for (size_t num_heads : {12}) {
//                 for (size_t head_dim : {64, 128, 256}) {
//                     for (bool causal : {false, true}) {
//                         for (size_t pos_encoding_mode : {0, 1}) {
//                             for (size_t kv_layout : {0, 1}) {
//                                 _TestSinglePrefillKernelCorrectness<
//                                     DTypeIn, DTypeIn, DTypeO>(
//                                     qo_len, kv_len, num_heads, num_heads,
//                                     head_dim, causal, QKVLayout(kv_layout),
//                                     PosEncodingMode(pos_encoding_mode),
//                                     use_fp16_qk_reduction);
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// template <typename DTypeKV>
// void TestSinglePrefillFP8KernelCorrectness(bool use_fp16_qk_reduction)
// {
//     for (size_t qo_len : {399, 400, 401}) {
//         for (size_t kv_len : {533, 534, 535}) {
//             for (size_t num_heads : {12}) {
//                 for (size_t head_dim : {64, 128, 256}) {
//                     for (bool causal : {false, true}) {
//                         for (size_t pos_encoding_mode : {0}) {
//                             for (size_t kv_layout : {0, 1}) {
//                                 _TestSinglePrefillKernelCorrectness<
//                                     __half, DTypeKV, __half>(
//                                     qo_len, kv_len, num_heads, num_heads,
//                                     head_dim, causal, QKVLayout(kv_layout),
//                                     PosEncodingMode(pos_encoding_mode),
//                                     use_fp16_qk_reduction);
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// TEST(FlashInferCorrectnessTest,
//      TestSinglePrefillKernelLongContextCorrectnessFP16)
// {
//     TestSinglePrefillKernelLongContextCorrectness<__half, __half>(false);
// }

// TEST(FlashInferCorrectnessTest,
//      TestSinglePrefillKernelLongContextCorrectnessFP16QKHalfAccum)
// {
//     TestSinglePrefillKernelLongContextCorrectness<__half, __half>(true);
// }

// TEST(FlashInferCorrectnessTest,
//      TestSinglePrefillKernelShortContextCorrectnessFP16)
// {
//     TestSinglePrefillKernelShortContextCorrectness<__half, __half>(false);
// }

// TEST(FlashInferCorrectnessTest,
//      TestSinglePrefillKernelShortContextCorrectnessFP16QKHalfAccum)
// {
//     TestSinglePrefillKernelShortContextCorrectness<__half, __half>(true);
// }

// TEST(FlashInferCorrectnessTest, TestSinglePrefillKernelCorrectnessTestFP16)
// {
//     TestSinglePrefillKernelCorrectness<__half, __half>(false);
// }

// TEST(FlashInferCorrectnessTest,
//      TestSinglePrefillKernelCorrectnessTestFP16QKHalfAccum)
// {
//     TestSinglePrefillKernelCorrectness<__half, __half>(true);
// }

// #ifdef FLASHINFER_ENABLE_BF16
// TEST(FlashInferCorrectnessTest,
//      TestSinglePrefillKernelLongContextCorrectnessBF16)
// {
//     TestSinglePrefillKernelLongContextCorrectness<__hip_bfloat16,
//     __hip_bfloat16>(
//         false);
// }
// TEST(FlashInferCorrectnessTest,
//      TestSinglePrefillKernelShortContextCorrectnessBF16)
// {
//     TestSinglePrefillKernelShortContextCorrectness<__hip_bfloat16,
//     __hip_bfloat16>(
//         false);
// }
// TEST(FlashInferCorrectnessTest, TestSinglePrefillKernelCorrectnessTestBF16)
// {
//     TestSinglePrefillKernelCorrectness<__hip_bfloat16,
//     __hip_bfloat16>(false);
// }
// #endif

//***********************************************************************
// The following tests are disabled because we dont support fp8 <-> float
// conversions

// #ifdef FLASHINFER_ENABLE_FP8_E4M3
// TEST(FlashInferCorrectnessTest,
// TestSinglePrefillKernelShortContextCorrectnessE4M3) {
//   TestSinglePrefillFP8KernelShortContextCorrectness<__nv_fp8_e4m3>(false);
// }
// TEST(FlashInferCorrectnessTest, TestSinglePrefillKernelCorrectnessTestE4M3) {
//   TestSinglePrefillFP8KernelCorrectness<__nv_fp8_e4m3>(false);
// }
// TEST(FlashInferCorrectnessTest,
// TestSinglePrefillKernelLongContextCorrectnessE4M3) {
//   TestSinglePrefillFP8KernelLongContextCorrectness<__nv_fp8_e4m3>(false);
// }
// #endif

// #ifdef FLASHINFER_ENABLE_FP8_E5M2
// TEST(FlashInferCorrectnessTest,
// TestSinglePrefillKernelShortContextCorrectnessE5M2) {
//   TestSinglePrefillFP8KernelShortContextCorrectness<__nv_fp8_e5m2>(false);
// }
// TEST(FlashInferCorrectnessTest, TestSinglePrefillKernelCorrectnessTestE5M2) {
//   TestSinglePrefillFP8KernelCorrectness<__nv_fp8_e5m2>(false);
// }
// TEST(FlashInferCorrectnessTest,
// TestSinglePrefillKernelLongContextCorrectnessE5M2) {
//   TestSinglePrefillFP8KernelLongContextCorrectness<__nv_fp8_e5m2>(false);
// }
// #endif

int main(int argc, char **argv)
{
    // ::testing::InitGoogleTest(&argc, argv);
    // return RUN_ALL_TESTS();
    using DTypeIn = __half;
    using DTypeO = __half;
    uint32_t debug_thread_id = 0;
    bool use_fp16_qk_reduction = false;
    size_t qo_len = 128;
    size_t kv_len = 128;
    size_t num_heads = 1;
    size_t head_dim = 64;
    bool causal = false;
    size_t pos_encoding_mode = 0; // 1 == kRopeLLama
    size_t kv_layout = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--thread" && i + 1 < argc) {
            debug_thread_id = std::stoi(argv[++i]);
            std::cout << "Debug thread ID set to: " << debug_thread_id
                      << std::endl;
        }
        else if (arg == "--qo_len" && i + 1 < argc) {
            qo_len = std::stoi(argv[++i]);
        }
        else if (arg == "--kv_len" && i + 1 < argc) {
            kv_len = std::stoi(argv[++i]);
        }
        else if (arg == "--heads" && i + 1 < argc) {
            num_heads = std::stoi(argv[++i]);
        }
        else if (arg == "--help") {
            std::cout
                << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  --thread <id>    Debug thread ID (0-255 for 4 warps)\n"
                << "  --qo_len <len>   Query/Output length (default: 128)\n"
                << "  --kv_len <len>   Key/Value length (default: 128)\n"
                << "  --heads <num>    Number of heads (default: 1)\n"
                << "  --help           Show this help message\n";
            return 0;
        }
    }

    _TestSinglePrefillKernelCorrectness<DTypeIn, DTypeIn, DTypeO>(
        qo_len, kv_len, num_heads, num_heads, head_dim, causal,
        QKVLayout(kv_layout), PosEncodingMode(pos_encoding_mode),
        use_fp16_qk_reduction, debug_thread_id);
}

// int main(int argc, char **argv)
// {
//     // Test compute_qk first with simple parameters
//     std::cout << "=== Testing compute_qk function ===" << std::endl;
//     using DTypeIn = __half;
//     using DTypeO = __half;
//     bool use_fp16_qk_reduction = false;
//     bool causal = false;
//     size_t pos_encoding_mode = 0;
//     size_t kv_layout = 0;

//     // Start with small dimensions for easier debugging
//     _TestComputeQKCorrectness<DTypeIn, DTypeIn, DTypeO>(
//         16, // qo_len - small for debugging
//         32, // kv_len
//         1,  // num_qo_heads - single head
//         1,  // num_kv_heads - single head
//         64, // head_dim
//         causal, QKVLayout(kv_layout), PosEncodingMode(pos_encoding_mode),
//         use_fp16_qk_reduction);

//     std::cout << "\n=== Testing full single prefill ===" << std::endl;
//     // Your existing test...
//     size_t qo_len = 399;
//     size_t kv_len = 533;
//     size_t num_heads = 1;
//     size_t head_dim = 64;

//     _TestSinglePrefillKernelCorrectness<DTypeIn, DTypeIn, DTypeO>(
//         qo_len, kv_len, num_heads, num_heads, head_dim, causal,
//         QKVLayout(kv_layout), PosEncodingMode(pos_encoding_mode),
//         use_fp16_qk_reduction);

//     return 0;
// }
