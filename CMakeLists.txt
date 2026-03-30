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

# In editable installs, create flashinfer/include as a symlink to the source
# include/ directory so that get_include_paths.py (Path(__file__).parent /
# "include") resolves to the live headers immediately without any cmake rebuild.
# Guarded by SKBUILD_STATE so the symlink is only created for editable builds;
# wheel and sdist builds continue to use the install(DIRECTORY …) copy above.
if(SKBUILD_STATE MATCHES "^editable")
  install(
    CODE "
    set(_inc_src \"${CMAKE_SOURCE_DIR}/include\")
    set(_inc_dst \"${CMAKE_SOURCE_DIR}/flashinfer/include\")
    set(_needs_link TRUE)
    if(IS_SYMLINK \"\${_inc_dst}\")
      file(READ_SYMLINK \"\${_inc_dst}\" _existing)
      if(\"\${_existing}\" STREQUAL \"\${_inc_src}\")
        set(_needs_link FALSE)
      else()
        file(REMOVE \"\${_inc_dst}\")
      endif()
    elseif(IS_DIRECTORY \"\${_inc_dst}\")
      file(REMOVE_RECURSE \"\${_inc_dst}\")
    elseif(EXISTS \"\${_inc_dst}\")
      file(REMOVE \"\${_inc_dst}\")
    endif()
    if(_needs_link)
      file(CREATE_LINK \"\${_inc_src}\" \"\${_inc_dst}\" SYMBOLIC)
      message(STATUS \"flashinfer: symlinked flashinfer/include -> include/\")
    endif()
  ")
endif()
