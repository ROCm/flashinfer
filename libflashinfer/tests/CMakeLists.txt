# Set global paths and initialize test list
set(ALL_TEST_TARGETS "")

# Include centralized config utilities
include(ConfigureTargets)

# Include GoogleTest to enable test discovery
include(GoogleTest)
find_package(Threads REQUIRED)

# cmake-format: off
# === HIP C++ Unit Tests Configuration ===
if(FLASHINFER_ENABLE_HIP)

  configure_flashinfer_target(
    TARGET_NAME test_batch_decode_hip
    SOURCES "hip/test_batch_decode.cpp"
    IS_GTEST ON
    IS_HIP ON
  )

  configure_flashinfer_target(
    TARGET_NAME test_cascade_hip
    SOURCES "hip/test_cascade.cpp"
    IS_GTEST ON
    IS_HIP ON
  )

  configure_flashinfer_target(
    TARGET_NAME test_layout_transform_hip
    SOURCES "hip/test_layout_transform.cpp"
    IS_GTEST ON
    IS_HIP ON
  )

  configure_flashinfer_target(
    TARGET_NAME test_math_hip
    SOURCES "hip/test_math.cpp"
    IS_GTEST ON
    IS_HIP ON
  )

  configure_flashinfer_target(
    TARGET_NAME test_mfma_fp32_16x16x16fp16_hip
    SOURCES "hip/test_mfma_fp32_16x16x16fp16.cpp"
    IS_GTEST ON
    IS_HIP ON
  )

  configure_flashinfer_target(
    TARGET_NAME test_permuted_smem_hip
    SOURCES "hip/test_permuted_smem.cpp"
    IS_GTEST ON
    IS_HIP ON
  )

  configure_flashinfer_target(
    TARGET_NAME test_pos_enc_hip
    SOURCES "hip/test_pos_enc.cpp"
    IS_GTEST ON
    IS_HIP ON
  )

  configure_flashinfer_target(
    TARGET_NAME test_rowsum_hip
    SOURCES "hip/test_rowsum.cpp"
    IS_GTEST ON
    IS_HIP ON
  )

  configure_flashinfer_target(
    TARGET_NAME test_single_decode_hip
    SOURCES "hip/test_single_decode.cpp"
    IS_GTEST ON
    IS_HIP ON
  )

endif(FLASHINFER_ENABLE_HIP)
# cmake-format: on

# === Test Discovery and Targets ===
message(STATUS "All unit test targets: ${ALL_TEST_TARGETS}")

# Use GoogleTest's discover_tests to find all test cases
foreach(test_target IN LISTS ALL_TEST_TARGETS)
  if(TARGET ${test_target})
    gtest_discover_tests(${test_target})
  endif()
endforeach()

# Create target to build all tests
add_custom_target(build_tests)
add_dependencies(build_tests ${ALL_TEST_TARGETS})
