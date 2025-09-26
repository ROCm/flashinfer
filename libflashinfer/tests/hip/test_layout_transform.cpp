#include <hip/hip_runtime.h>
#include <stdio.h>

#include "gpu_iface/backend/hip/mma_hip.h"

// Define test dimensions
constexpr int MATRIX_SIZE_X = 16;
constexpr int MATRIX_SIZE_Y = 16;

namespace {
// Print register values for debugging
__device__ void print_register(uint32_t* R) {
  auto values = reinterpret_cast<__half*>(R);
  printf("[%5.1f %5.1f %5.1f %5.1f]\n", __half2float(values[0]), __half2float(values[1]),
         __half2float(values[2]), __half2float(values[3]));
}

// Initialize LDS array with lexicographic values
__device__ void init_lds_array(half* lds_array) {
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int y = 0; y < MATRIX_SIZE_Y; ++y) {
      for (int x = 0; x < MATRIX_SIZE_X; ++x) {
        lds_array[y * MATRIX_SIZE_X + x] = __half(y * MATRIX_SIZE_X + x);
      }
    }
  }
  __syncthreads();
}

// Each thread loads 4 elements in A-matrix layout
__device__ void load_amatrix_layout(half* lds_array, uint32_t* R) {
  const int tid = threadIdx.x;
  const int lane_id = tid % 64;
  const int row = lane_id % 16;
  const int col_start = (lane_id / 16) * 4;

  auto offset = lds_array + row * MATRIX_SIZE_X + col_start;

  if (tid == 0) {
    printf("DEBUG:::: %f %f %f %f\n", __half2float(*offset), __half2float(*(offset + 1)),
           __half2float(*(offset + 2)), __half2float(*(offset + 3)));
  }

  flashinfer::gpu_iface::mma_impl::hip::load_fragment(R, offset);

  if (tid == 0) {
    print_register(R);
  }
}

// Print LDS array using one thread
__device__ void print_lds_array(half* lds_array) {
  if (threadIdx.x == 0) {
    printf("LDS Array (%dx%d):\n", MATRIX_SIZE_X, MATRIX_SIZE_Y);
    for (int y = 0; y < MATRIX_SIZE_Y; ++y) {
      for (int x = 0; x < MATRIX_SIZE_X; ++x) {
        printf("%5.1f ", __half2float(lds_array[y * MATRIX_SIZE_X + x]));
      }
      printf("\n");
    }
    printf("\n");
  }
  __syncthreads();
}

}  // namespace

__global__ void test_mini_tile_transpose_kernel() {
  // Allocate shared memory for the 16x16 matrix
  __shared__ half lds_array[MATRIX_SIZE_X * MATRIX_SIZE_Y];

  // Step 1: Initialize the LDS array with lexicographic values
  init_lds_array(lds_array);

  // Step 2: Print the LDS array (for debugging)
  print_lds_array(lds_array);

  // Step 3: Load data from LDS to registers in A-matrix layout
  uint32_t registers[2];
  load_amatrix_layout(lds_array, registers);

  // Step 4: Print initial register values for verification
  __syncthreads();

  // Step 5: Apply transpose to convert from A-matrix to B/C-matrix layout
  flashinfer::gpu_iface::mma_impl::hip::transpose_4x4_half_registers(registers);

  // Step 6: Print transposed register values
  __syncthreads();
  if (threadIdx.x == 0) {
    printf("After Transpose\n");
    print_lds_array(lds_array);
  }
}

// Host code to launch the kernel
void test_mini_tile_transpose() {
  // Launch with 1 block of 64 threads (full warp for CDNA3)
  test_mini_tile_transpose_kernel<<<1, 64>>>();
  hipDeviceSynchronize();
}

int main() {
  test_mini_tile_transpose();
  return 0;
}
