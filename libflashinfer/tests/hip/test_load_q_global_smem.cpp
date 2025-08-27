// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
//
// SPDX - License - Identifier : Apache 2.0

#include <cassert>
#include <cmath>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#include "flashinfer/attention/generic/default_prefill_params.cuh"
#include "flashinfer/attention/generic/prefill.cuh"
#include "flashinfer/attention/generic/variants.cuh"
#include "utils/cpu_reference_hip.h"
#include "utils/utils_hip.h" // vec_normal_

namespace
{
constexpr uint32_t qo_len = 64;
constexpr uint32_t num_qo_heads = 1;
constexpr uint32_t head_dim = 64;
} // namespace

// CPU reference implementation that creates a Q matrix with a kNHD layout and
// initializes.
void initialize_cpu_q()
{
    std::vector<DTypeQ> q(qo_len * num_qo_heads * head_dim);
    utils::vec_normal_(q);
}

// Validates the original Q matrix on CPU with the copied over data from GPU.
// Ensures that the copied over data matches both the CDNA3 A-matrix layout and
// also validates with the original Q matrix.

// GPU kernel that launches exactly one warp and calls prefill.cuh's
// load_q_global_smem to populate a LDS array from a global array. Then copies
// back the shared memory array to another output global array.

// Laucher of GPU kernel.
// Copies the Q array from the CPU reference to GPU and then calls the kernel
// to copy from global to shared memory.
