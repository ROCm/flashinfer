# === Required Dependencies for Core Functionality ===
if(FLASHINFER_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  find_package(Thrust REQUIRED)
endif()

# === HIP Dependencies ===
if(FLASHINFER_ENABLE_HIP)
  # Check for HIP
  include(ConfigureRocmPath)
  find_package(HIP REQUIRED)
  message(STATUS "Found HIP: ${HIP_VERSION}")
endif()

find_package(Python3 REQUIRED)
if(NOT Python3_FOUND)
  message(
    FATAL_ERROR
      "Python3 not found it is required to generate the kernel sources.")
endif()

# === Boost Dependency for FP16 QK Reductions ===
if(FLASHINFER_GEN_USE_FP16_QK_REDUCTIONS)
  include(FetchContent)
  set(BOOST_ENABLE_CMAKE ON)
  FetchContent_Declare(boost_math
                       GIT_REPOSITORY https://github.com/boostorg/math.git)
  FetchContent_MakeAvailable(boost_math)

  set(USE_FP16_QK_REDUCTIONS "true")
  message(STATUS "USE_FP16_QK_REDUCTIONS=${USE_FP16_QK_REDUCTIONS}")
else()
  set(USE_FP16_QK_REDUCTIONS "false")
  message(STATUS "USE_FP16_QK_REDUCTIONS=${USE_FP16_QK_REDUCTIONS}")
endif()

# === Distributed component dependencies ===
if(FLASHINFER_DISTRIBUTED)
  include(FetchContent)
  FetchContent_Declare(
    mscclpp
    GIT_REPOSITORY https://github.com/microsoft/mscclpp.git
    GIT_TAG 11e62024d3eb190e005b4689f8c8443d91a6c82e)
  FetchContent_MakeAvailable(mscclpp)

  # Create alias for distributed component
  if(NOT TARGET flashinfer::mscclpp)
    add_library(flashinfer::mscclpp ALIAS mscclpp)
  endif()

  # Fetch spdlog for distributed tests (header-only usage)
  FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG f355b3d58f7067eee1706ff3c801c2361011f3d5 # release-1.15.1
    FIND_PACKAGE_ARGS NAMES spdlog)

  # Use Populate instead of MakeAvailable since we only need the headers
  FetchContent_Populate(spdlog)

  # Set the include directory for later use
  set(SPDLOG_INCLUDE_DIR "${spdlog_SOURCE_DIR}/include")
  message(STATUS "Using spdlog from ${SPDLOG_INCLUDE_DIR}")

  find_package(MPI REQUIRED)
endif()

# === Path definitions ===
# Define all include paths centrally - don't use global include_directories

# FlashInfer internal paths
set(FLASHINFER_INCLUDE_DIR
    "${CMAKE_SOURCE_DIR}/include"
    CACHE INTERNAL "FlashInfer include directory")

# Generated code paths
set(FLASHINFER_GENERATED_SOURCE_DIR
    "${CMAKE_BINARY_DIR}/src/generated"
    CACHE INTERNAL "FlashInfer generated source directory")

set(FLASHINFER_GENERATED_SOURCE_DIR_ROOT
    "${CMAKE_BINARY_DIR}/src"
    CACHE INTERNAL "FlashInfer generated source root directory")

# === CUTLASS Configuration ===
if(FLASHINFER_ENABLE_CUDA)
  if(FLASHINFER_CUTLASS_DIR)
    if(IS_ABSOLUTE ${FLASHINFER_CUTLASS_DIR})
      set(CUTLASS_DIR ${FLASHINFER_CUTLASS_DIR})
    else()
      set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/${FLASHINFER_CUTLASS_DIR}")
    endif()

    list(APPEND CMAKE_PREFIX_PATH ${CUTLASS_DIR})
    set(CUTLASS_INCLUDE_DIRS
        "${CUTLASS_DIR}/include" "${CUTLASS_DIR}/tools/util/include"
        CACHE INTERNAL "CUTLASS include directories")

    message(STATUS "CUTLASS include directories: ${CUTLASS_INCLUDE_DIRS}")
  else()
    message(
      FATAL_ERROR "FLASHINFER_CUTLASS_DIR must be set to the path of CUTLASS")
  endif()
endif()

# === Python dependencies for PyTorch extensions ===
if(FLASHINFER_AOT_TORCH_EXTS)
  find_package(
    Python
    COMPONENTS Interpreter Development.Module
    REQUIRED)

  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "-c"
            "import torch;print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX COMMAND_ECHO STDOUT
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};${TORCH_CMAKE_PREFIX})

  if(FLASHINFER_ENABLE_CUDA)
    find_package(CUDA)
  endif()

  # Find PyTorch
  find_package(Torch REQUIRED)

  # Report found versions
  message(STATUS "Found Python: ${Python_VERSION}")
  message(STATUS "Found PyTorch: ${TORCH_VERSION}")

  # pybind11 for core module
  if(NOT TARGET pybind11::module)
    include(FetchContent)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.11.1)
    FetchContent_MakeAvailable(pybind11)
  endif()
endif()
