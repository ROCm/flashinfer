# cmake/AOTKernels.cmake AOT kernel generation and compilation for
# FlashInfer+ROCm This module handles ahead-of-time compilation of kernels for
# HIP/ROCm only Only supports wheel builds (Python packages)

# Apply data type definitions based on enabled options
if(FLASHINFER_ENABLE_FP8_E4M3)
  message(STATUS "Compile fp8_e4m3 kernels.")
  add_definitions(-DFLASHINFER_ENABLE_FP8_E4M3)
endif()

if(FLASHINFER_ENABLE_FP8_E5M2)
  message(STATUS "Compile fp8_e5m2 kernels.")
  add_definitions(-DFLASHINFER_ENABLE_FP8_E5M2)
endif()

if(FLASHINFER_ENABLE_BF16)
  message(STATUS "Compile bf16 kernels.")
  add_definitions(-DFLASHINFER_ENABLE_BF16)
endif()

# FP16 QK Reductions support
if(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS)
  add_definitions(-DFP16_QK_REDUCTION_SUPPORTED)
endif()

# Set kernel generation parameters
set(HEAD_DIMS ${FLASHINFER_GEN_HEAD_DIMS})
set(POS_ENCODING_MODES ${FLASHINFER_GEN_POS_ENCODING_MODES})
set(MASK_MODES ${FLASHINFER_GEN_MASK_MODES})
set(USE_FP16_QK_REDUCTIONS ${FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS})

# Log kernel generation options
message(STATUS "FLASHINFER_HEAD_DIMS=${HEAD_DIMS}")
message(STATUS "FLASHINFER_POS_ENCODING_MODES=${POS_ENCODING_MODES}")
message(
  STATUS "FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS=${USE_FP16_QK_REDUCTIONS}")
message(STATUS "FLASHINFER_MASK_MODES=${MASK_MODES}")

# ----------------------- SM90 head dims computation ------------------------#
# Include the logic to caclulate the head dims for SM90
include(CalculateSM90HeadDims)
set(HEAD_DIMS_SM90 "")
flashinfer_compute_sm90_head_dims(RESULT HEAD_DIMS_SM90)
# Log SM90_ALLOWED_HEAD_DIMS and HEAD_DIMS_SM90
message(STATUS "SM90_ALLOWED_HEAD_DIMS=${FLASHINFER_SM90_ALLOWED_HEAD_DIMS}")
message(STATUS "HEAD_DIMS_SM90=${HEAD_DIMS_SM90}")
# ---------------------------------------------------------------------------#

# ---------------------- Kernels and dispatch.inc generation ----------------#
# Generate kernel source files and dispatch.inc
include(ConfigureKernelGeneration)
if(FLASHINFER_BUILD_KERNELS)
  flashinfer_configure_kernel_generation()
endif()
# ---------------------------------------------------------------------------#

# ---------------------- Generate configure.h -------------------------------#
include(${CMAKE_SOURCE_DIR}/cmake/utils/GenerateConfigHeader.cmake)

# Wheel builds install headers inside the Python package
set(_FLASHINFER_INCLUDE_INSTALL_DIR "flashinfer/include")

# cmake-format: off
flashinfer_generate_config_header(
  SOURCE_DIR ${CMAKE_SOURCE_DIR}
  BINARY_DIR ${CMAKE_BINARY_DIR}
  INSTALL_DIR ${_FLASHINFER_INCLUDE_INSTALL_DIR}/flashinfer
  COMPONENT Headers
  EXCLUDE_PATTERNS
    ".*_DIR$"
    ".*_PATH$"
    ".*_FOUND$"
    ".*_FILE$"
    ".*_VERSION$"
)
# cmake-format: on
# ---------------------------------------------------------------------------#

# Build decode_kernels and prefill_kernels static libraries
if(FLASHINFER_BUILD_KERNELS)
  # Create decode kernels library
  add_library(decode_kernels STATIC ${DECODE_KERNELS_SRCS})
  add_library(flashinfer::decode_kernels ALIAS decode_kernels)
  target_include_directories(decode_kernels PRIVATE ${FLASHINFER_INCLUDE_DIR})

  if(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS)
    target_link_libraries(decode_kernels PRIVATE Boost::math)
  endif()

  # Create prefill kernels library
  add_library(prefill_kernels STATIC ${PREFILL_KERNELS_SRCS})
  add_library(flashinfer::prefill_kernels ALIAS prefill_kernels)
  target_include_directories(prefill_kernels PRIVATE ${FLASHINFER_INCLUDE_DIR})

  if(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS)
    target_link_libraries(prefill_kernels PRIVATE Boost::math)
  endif()

  # Configure HIP properties for all kernel sources and targets
  if(FLASHINFER_ENABLE_HIP)
    set_source_files_properties(${DECODE_KERNELS_SRCS} PROPERTIES LANGUAGE HIP)
    set_source_files_properties(${PREFILL_KERNELS_SRCS} PROPERTIES LANGUAGE HIP)

    foreach(flag ${FLASHINFER_HIPCC_FLAGS})
      target_compile_options(decode_kernels
                             PRIVATE $<$<COMPILE_LANGUAGE:HIP>:${flag}>)
      target_compile_options(prefill_kernels
                             PRIVATE $<$<COMPILE_LANGUAGE:HIP>:${flag}>)
    endforeach()

    set_target_properties(
      decode_kernels prefill_kernels
      PROPERTIES HIP_SOURCES_PROPERTY_FORMAT 1
                 HIP_SEPARABLE_COMPILATION ON
                 LINKER_LANGUAGE HIP)
  endif()
endif()

# Install Boost::math headers if FP16 QK reductions are enabled
if(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS)
  install(
    DIRECTORY ${boost_math_SOURCE_DIR}/include/
    DESTINATION ${_FLASHINFER_INCLUDE_INSTALL_DIR}/
    COMPONENT Headers
    FILES_MATCHING
    REGEX "\\.(h|hpp)$")
endif()
