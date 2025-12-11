# Define the component structure
set(FLASHINFER_COMPONENTS "Headers")

if(FLASHINFER_BUILD_KERNELS)
  list(APPEND FLASHINFER_COMPONENTS "Kernels")
endif()

if(FLASHINFER_DISTRIBUTED)
  list(APPEND FLASHINFER_COMPONENTS "Distributed")
endif()
