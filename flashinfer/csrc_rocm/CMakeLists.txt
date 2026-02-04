# Include the custom torch extension helper module
include(ConfigureTorchExtension)
include(${CMAKE_SOURCE_DIR}/cmake/utils/ConfigurePrebuitUris.cmake)

message(STATUS "Building FlashInfer with AOT Torch ROCm Extensions")

# Source files for PyTorch extensions
file(
  GLOB
  KERNEL_HIP_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/activation.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/batch_decode.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/cascade.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/flashinfer_ops.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/norm.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/page.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/rope.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/single_decode.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/batch_prefill.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/single_prefill.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/sampling.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/renorm.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/quantization.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/flashinfer_quantization_ops.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/flashinfer_sampling_ops.cu")

set(COMMON_HIP_INCLUDE_DIRS
    ${FLASHINFER_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS}
    ${FLASHINFER_GENERATED_SOURCE_DIR} ${FLASHINFER_GENERATED_SOURCE_DIR_ROOT})
# Common compile definitions
set(COMMON_COMPILE_DEFS
    $<$<BOOL:${FLASHINFER_ENABLE_FP8_E4M3}>:FLASHINFER_ENABLE_FP8_E4M3>
    $<$<BOOL:${FLASHINFER_ENABLE_FP8_E5M2}>:FLASHINFER_ENABLE_FP8_E5M2>
    $<$<BOOL:${FLASHINFER_ENABLE_BF16}>:FLASHINFER_ENABLE_BF16>
    $<$<BOOL:${FLASHINFER_ENABLE_F16}>:FLASHINFER_ENABLE_F16>)

# cmake-format: off
# Build a torch extension for the HIP kernels
add_hip_torch_extension(
  EXT_NAME flashinfer_hip_kernels
  SOURCES
    ${KERNEL_HIP_SOURCES}
  INCLUDE_DIRS
    ${COMMON_HIP_INCLUDE_DIRS}
  LINK_LIBS
    $<$<TARGET_EXISTS:flashinfer::decode_kernels>:flashinfer::decode_kernels>
    $<$<TARGET_EXISTS:flashinfer::prefill_kernels>:flashinfer::prefill_kernels>
  COMPILE_FLAGS
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-switch-bool>
  PY_LIMITED_API ${FLASHINFER_PY_LIMITED_API}
  MIN_PYTHON_ABI "${FLASHINFER_MIN_PYTHON_ABI}"
)
target_compile_definitions(flashinfer_hip_kernels
                          PRIVATE FLASHINFER_ENABLE_HIP)
set_target_properties(
  flashinfer_hip_kernels
  PROPERTIES INSTALL_RPATH
              "${FLASHINFER_RPATH_BASE};${FLASHINFER_RPATH_BASE}/../lib"
              HIP_SOURCES_PROPERTY_FORMAT 1
            HIP_SEPARABLE_COMPILATION ON
            LINKER_LANGUAGE HIP)
# Install the extension to flashinfer directory
install(TARGETS flashinfer_hip_kernels DESTINATION flashinfer)
# cmake-format: on

flashinfer_configure_prebuilt_uris()
message(STATUS "AMD-FlashInfer AOT PyTorch extensions configured")
