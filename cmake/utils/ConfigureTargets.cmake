# cmake-format: off
# Common configuration function for tests and benchmarks
function(configure_flashinfer_target)
  set(options IS_GTEST IS_BENCHMARK IS_HIP)
  set(oneValueArgs TARGET_NAME)
  set(multiValueArgs SOURCES LINK_LIBS COMPILE_FLAGS INCLUDE_DIRS)

  cmake_parse_arguments(PARSE_ARGV 0 arg
    "${options}" "${oneValueArgs}" "${multiValueArgs}")

  # Validate required arguments
  if(NOT DEFINED arg_TARGET_NAME)
    message(FATAL_ERROR "TARGET_NAME is required")
  endif()

  if(NOT DEFINED arg_SOURCES)
    message(FATAL_ERROR "SOURCES is required")
  endif()

  # Set appropriate message based on target type
  if(arg_IS_BENCHMARK)
    message(STATUS "Configure ${arg_TARGET_NAME} benchmark")
  else()
    message(STATUS "Configure ${arg_TARGET_NAME} test")
  endif()

  # Create executable target
  add_executable(${arg_TARGET_NAME} EXCLUDE_FROM_ALL ${arg_SOURCES})

  if(arg_IS_HIP)
    set_source_files_properties(${arg_SOURCES} PROPERTIES LANGUAGE HIP)
  endif()

  # Add all include directories
  target_include_directories(
    ${arg_TARGET_NAME}
    PRIVATE ${FLASHINFER_INCLUDE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${FLASHINFER_UTILS_INCLUDE_DIR}
            ${GENERATED_SOURCE_DIR_ROOT})

  # Add any extra include directories
  foreach(extra_include_dir IN LISTS arg_INCLUDE_DIRS)
    target_include_directories(${arg_TARGET_NAME} PRIVATE ${extra_include_dir})
  endforeach()

  # Add benchmark-specific library for benchmarks
  if(arg_IS_BENCHMARK)
    target_link_libraries(${arg_TARGET_NAME} PRIVATE nvbench::main)
  endif()

  # Add standard link libraries and compile flags
  foreach(lib IN LISTS arg_LINK_LIBS)
    target_link_libraries(${arg_TARGET_NAME} PRIVATE ${lib})
  endforeach()

  foreach(flag IN LISTS arg_COMPILE_FLAGS)
    target_compile_options(${arg_TARGET_NAME} PRIVATE ${flag})
  endforeach()

  # Add Google Test libraries if required
  if(arg_IS_GTEST)
    target_link_libraries(${arg_TARGET_NAME} PRIVATE GTest::gtest GTest::gtest_main Threads::Threads)
  endif()

  # Register with CTest if it's a test
  if(NOT arg_IS_BENCHMARK)
    add_test(NAME ${arg_TARGET_NAME} COMMAND ${arg_TARGET_NAME})
  endif()

  # Add to appropriate global list
  if(arg_IS_BENCHMARK)
    list(APPEND ALL_BENCHMARK_TARGETS "${arg_TARGET_NAME}")
    set(ALL_BENCHMARK_TARGETS "${ALL_BENCHMARK_TARGETS}" PARENT_SCOPE)
  else()
    list(APPEND ALL_TEST_TARGETS "${arg_TARGET_NAME}")
    set(ALL_TEST_TARGETS "${ALL_TEST_TARGETS}" PARENT_SCOPE)
  endif()

  # Final configuration message
  if(arg_IS_BENCHMARK)
    message(STATUS "Configured ${arg_TARGET_NAME} benchmark")
  else()
    message(STATUS "Configured ${arg_TARGET_NAME} test")
  endif()
endfunction()
# cmake-format: on
