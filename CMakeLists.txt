cmake_minimum_required(VERSION 3.26.1)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_INSTALL_MESSAGE LAZY)

project(flashinfer DESCRIPTION "Fast Attention Algorithms for LLM Inference")

include(GNUInstallDirs)

message(STATUS "Building Flashinfer PyTorch extension")
add_subdirectory(flashinfer)
set(_FLASHINFER_INCLUDE_INSTALL_DIR "flashinfer/include")

install(
  DIRECTORY ${CMAKE_SOURCE_DIR}/include/
  DESTINATION ${_FLASHINFER_INCLUDE_INSTALL_DIR}
  FILES_MATCHING
  REGEX "\\.(cuh|h|hpp)$"
  PATTERN "distributed" EXCLUDE)
