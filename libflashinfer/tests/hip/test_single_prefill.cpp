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
            use_fp16_qk_reduction);

    EXPECT_EQ(status, hipSuccess)
        << "SinglePrefillWithKVCache kernel launch failed, error message: "
        << hipGetErrorString(status);

    std::vector<DTypeO> o_h(o.size());
    FI_GPU_CALL(hipMemcpy(o_h.data(), o_d, o_h.size() * sizeof(DTypeO),
                          hipMemcpyDeviceToHost));

    // Print the first 10 elements of the output vector for debugging
    // std::cout << "Output vector (first 10 elements):";
    // std::cout << "[" << std::endl;
    // for (int i = 0; i < 10; ++i) {
    //     std::cout << fi::con::explicit_casting<DTypeO, float>(o_h[i]) << " ";
    // }
    // std::cout << "]" << std::endl;

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
        if (!utils::isclose(o_ref_val, o_h_val, rtol, atol)) {
            std::cout << "i=" << i << ", o_ref[i]=" << o_ref_val
                      << ", o_h[i]=" << o_h_val << std::endl;
        }
    }
    // std::cout<<"Printing att_out vector:\n";
    // for(auto i: att_out) {
    //     std::cout << i << "\n";
    // }
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
    bool use_fp16_qk_reduction = false;
    size_t qo_len = 399;
    size_t kv_len = 533;
    size_t num_heads = 1;
    size_t head_dim = 64;
    bool causal = false;
    size_t pos_encoding_mode = 0;
    size_t kv_layout = 0;

    _TestSinglePrefillKernelCorrectness<DTypeIn, DTypeIn, DTypeO>(
        qo_len, kv_len, num_heads, num_heads, head_dim, causal,
        QKVLayout(kv_layout), PosEncodingMode(pos_encoding_mode),
        use_fp16_qk_reduction);
}
