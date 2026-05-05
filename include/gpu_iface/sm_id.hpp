// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "macros.hpp"

namespace flashinfer {
namespace gpu_iface {

__device__ __forceinline__ uint32_t get_processor_id() {
#if defined(PLATFORM_CUDA_DEVICE)
  uint32_t smid;
  asm volatile("mov.u32 %0, %smid;" : "=r"(smid));
  return smid;
#elif defined(PLATFORM_HIP_DEVICE)
  // HW_ID register bits [15:8]: SE_ID[15:14] | SH_ID[13:12] | CU_ID[11:8].
  // Together these 8 bits uniquely identify a CU across the chip.
  // The caller passes num_SMs and we modulo for safety.
  uint32_t hw_id = __builtin_amdgcn_s_getreg((4 | (0 << 6) | (31 << 11)));
  return (hw_id >> 8) & 0xFF;
#else
  return 0;
#endif
}

}  // namespace gpu_iface
}  // namespace flashinfer
