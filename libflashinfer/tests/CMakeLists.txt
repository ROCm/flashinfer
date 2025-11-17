# Set global paths and initialize test list
set(ALL_TEST_TARGETS "")

# List of tests to skip. These tests are currently disabled due to known issues
# or pending fixes.
set(SKIP_TESTS
    test_batch_prefill # This is being skipped because
                       # BatchPrefillWithPagedKVCache is not yet implemented for
                       # HIP.
    test_layout_transform # This test is being skipped due to incorrect test
                          # cases
    test_mfma_fp32_16x16x16fp16 # This test will be fixed in later update. Look
                                # at PR #52
    test_cascade)

# Include centralized config utilities
include(ConfigureTargets)

find_package(Threads REQUIRED)

# cmake-format: off
# === HIP C++ Unit Tests Configuration ===
if(FLASHINFER_ENABLE_HIP)
  file(GLOB HIP_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/hip/*.cpp")

  foreach(test_source IN LISTS HIP_TEST_SOURCES)
    get_filename_component(test_name ${test_source} NAME_WE)

    if(test_name IN_LIST SKIP_TESTS)
      message(STATUS "Skipping HIP test: ${test_name}")
      continue()
    endif()

    set(target_name "${test_name}_hip")
    configure_flashinfer_target(
      TARGET_NAME ${target_name}
      SOURCES ${test_source}
      IS_GTEST ON
      IS_HIP ON
    )
  endforeach()

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
