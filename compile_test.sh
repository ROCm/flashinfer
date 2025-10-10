amdclang++ -x hip \
           -std=c++17 \
           -I/home/AMD/diptodeb/devel/flashinfer/libflashinfer/include \
           -I/home/AMD/diptodeb/devel/flashinfer/libflashinfer \
           -I${CONDA_PREFIX}/include \
           -Wall \
           -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 \
           -L${CONDA_PREFIX}/lib \
           -lgtest \
           -DDebug \
           -Wl,-rpath=${CONDA_PREFIX}/lib \
           libflashinfer/tests/hip/test_single_prefill.cpp  \
           --offload-arch=gfx942
