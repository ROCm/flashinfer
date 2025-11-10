// SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.

#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

namespace {

__host__ __device__ __inline__ __hip_fp8_e5m2fnuz convert_float_to_fp8(
    float in, __hip_fp8_interpretation_t interpret, __hip_saturation_t sat) {
  return __hip_cvt_float_to_fp8(in, sat, interpret);
}

__host__ __device__ __inline__ __hip_fp8_e4m3fnuz convert_float_to_fp8(
    float in, __hip_fp8_interpretation_t interpret, __hip_saturation_t sat) {
  return __hip_cvt_float_to_fp8(in, sat, interpret);
}

__host__ __device__ __inline__ float convert_fp8_to_float(float in,
                                                          __hip_fp8_interpretation_t interpret) {
  float hf = __hip_cvt_fp8_to_float(in, interpret);
  return hf;
}

}  // namespace
namespace fi::con {
template <typename DTypeIn, typename DTypeOut>
__host__ __device__ __inline__ DTypeOut explicit_casting(DTypeIn value) {
  return DTypeOut(value);
}

template <>
__host__ __device__ __inline__ float explicit_casting<__half, float>(__half value) {
  return __half2float(value);
}

template <>
__host__ __device__ __inline__ float explicit_casting<__hip_bfloat16, float>(__hip_bfloat16 value) {
  return __bfloat162float(value);
}

template <>
__host__ __device__ __inline__ __half explicit_casting<float, __half>(float value) {
  return __float2half(value);
}

template <>
__host__ __device__ __inline__ __hip_bfloat16 explicit_casting<__half, __hip_bfloat16>(
    __half value) {
  return __float2bfloat16(__half2float(value));
}

template <>
__host__ __device__ __inline__ float explicit_casting<float, float>(float value) {
  return value;
}

template <>
__host__ __device__ __inline__ __half explicit_casting<__half, __half>(__half value) {
  return value;
}

template <>
__host__ __device__ __inline__ __hip_bfloat16 explicit_casting<__hip_bfloat16, __hip_bfloat16>(
    __hip_bfloat16 value) {
  return value;
}

template <>
__host__ __device__ __inline__ __hip_fp8_e4m3fnuz explicit_casting<float, __hip_fp8_e4m3fnuz>(
    float value) {
  return convert_float_to_fp8(value, __HIP_E4M3_FNUZ, __HIP_SATURATE);
}

template <>
__host__ __device__ __inline__ float explicit_casting<__hip_fp8_e4m3fnuz, float>(
    __hip_fp8_e4m3fnuz value) {
  return convert_fp8_to_float(value, __HIP_E4M3_FNUZ);
}

template <>
__host__ __device__ __inline__ __hip_fp8_e4m3fnuz explicit_casting<__half, __hip_fp8_e4m3fnuz>(
    __half value) {
  float temp = __half2float(value);
  return convert_float_to_fp8(temp, __HIP_E4M3_FNUZ, __HIP_SATURATE);
}

template <>
__host__ __device__ __inline__ __half explicit_casting<__hip_fp8_e4m3fnuz, __half>(
    __hip_fp8_e4m3fnuz value) {
  float temp = convert_fp8_to_float(value, __HIP_E4M3_FNUZ);
  return __float2half(temp);
}

// *************

template <>
__host__ __device__ __inline__ __hip_fp8_e5m2fnuz explicit_casting<float, __hip_fp8_e5m2fnuz>(
    float value) {
  return convert_float_to_fp8(value, __HIP_E5M2_FNUZ, __HIP_SATURATE);
}

template <>
__host__ __device__ __inline__ float explicit_casting<__hip_fp8_e5m2fnuz, float>(
    __hip_fp8_e5m2fnuz value) {
  return convert_fp8_to_float(value, __HIP_E5M2_FNUZ);
}

template <>
__host__ __device__ __inline__ __hip_fp8_e5m2fnuz explicit_casting<__half, __hip_fp8_e5m2fnuz>(
    __half value) {
  float temp = __half2float(value);
  return convert_float_to_fp8(temp, __HIP_E5M2_FNUZ, __HIP_SATURATE);
}

template <>
__host__ __device__ __inline__ __half explicit_casting<__hip_fp8_e5m2fnuz, __half>(
    __hip_fp8_e5m2fnuz value) {
  float temp = convert_fp8_to_float(value, __HIP_E5M2_FNUZ);
  return __float2half(temp);
}
}  // namespace fi::con
