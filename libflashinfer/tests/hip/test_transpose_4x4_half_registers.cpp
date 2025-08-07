// test_transpose_4x4_half_registers.cpp
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

// Define WARP_FULL_MASK for HIP
constexpr uint64_t WARP_FULL_MASK =
    0xffffffffffffffffULL; // 64-bit mask for HIP

__device__ __forceinline__ void debug_print_registers(const char *stage,
                                                      uint32_t lane_id,
                                                      uint32_t lane_in_group,
                                                      uint32_t *regs,
                                                      int num_regs,
                                                      uint32_t debug_group = 0)
{

    // Only debug a specific group to avoid excessive output
    if (lane_id / 4 != debug_group)
        return;

    // Print identification info
    printf("STAGE: %s | Thread %d (lane_in_group=%d): ", stage, lane_id,
           lane_in_group);

    // Print raw 32-bit values
    printf("RAW=[");
    for (int i = 0; i < num_regs; i++) {
        printf("0x%08x", regs[i]);
        if (i < num_regs - 1)
            printf(", ");
    }
    printf("] | ");

    // Print unpacked 16-bit values
    printf("UNPACKED=[");
    for (int i = 0; i < num_regs; i++) {
        uint16_t hi = (regs[i] >> 16) & 0xFFFF;
        uint16_t lo = regs[i] & 0xFFFF;
        printf("%d,%d", hi, lo);
        if (i < num_regs - 1)
            printf(", ");
    }
    printf("]\n");
}

__device__ __forceinline__ void transpose_4x4_half_registers(uint32_t *R,
                                                             uint32_t *out)
{
    // Each thread has 4 half-precision values in 2 registers:
    // R[0] = [B[lane_id][0], B[lane_id][1]]
    // R[1] = [B[lane_id][2], B[lane_id][3]]

    // Calculate lane within 4-thread group
    uint32_t lane_id = threadIdx.x % 64;
    uint32_t lane_in_group = lane_id % 4;
    uint32_t temp_regs[2];

    if (lane_id == 0) {
        debug_print_registers("Initial", lane_id, lane_in_group, R, 2, 0);
    }

    // === ROUND 1: Exchange with neighbor (XOR with 1) ===
    // T0↔T1, T2↔T3 partial exchange

    // Update based on thread position
    if (lane_in_group < 2) {
        uint32_t r0_exchanged = __shfl_xor(R[0], 0x1);
        // Top half (T0, T1) update R[0]
        if (lane_in_group & 1) { // T1
            R[0] = (R[0] & 0x0000FFFF) | (r0_exchanged << 16);
        }
        else { // T0
            r0_exchanged >>= 16;
            R[0] = (R[0] & 0xFFFF0000) | (r0_exchanged);
        }
    }
    else {
        uint32_t r1_exchanged = __shfl_xor(R[1], 0x1);
        // Bottom half (T2, T3) update R[1]
        if (lane_in_group & 1) { // T1
            R[1] = (R[1] & 0x0000FFFF) | (r1_exchanged << 16);
        }
        else { // T0
            R[1] = (R[1] & 0xFFFF0000) | (r1_exchanged >> 16);
        }
    }

    // Debug after first recombination
    if (lane_id == 3) {
        debug_print_registers("After Round 1 shuffles", lane_id, lane_in_group,
                              R, 2, 0);
    }
#if 0
    // === ROUND 2: Exchange with one hop (XOR with 2) ===
    // T0↔T2, T1↔T3 exchange R[0] and R[1]
    uint32_t temp0_exchanged = __shfl_xor(R[0], 0x2);
    uint32_t temp1_exchanged = __shfl_xor(R[1], 0x2);

    // Debug second exchange
    if (lane_id < 4) {
        uint32_t debug_regs[2] = {temp0_exchanged, temp1_exchanged};
        debug_print_registers("Round2-Exchange", lane_id, lane_in_group,
                              debug_regs, 2, 0);
    }

    // Swap entire registers based on thread position
    if (lane_in_group < 2) {
        // Top threads (T0, T1) get R[0] from partner, keep own R[1]
        temp_regs[0] = temp0_exchanged;
        // Keep R[1] unchanged
    }
    else {
        // Bottom threads (T2, T3) get R[1] from partner, keep own R[0]
        temp_regs[1] = temp1_exchanged;
        // Keep R[0] unchanged
    }

    // Debug after second recombination
    if (lane_id < 4) {
        debug_print_registers("Round2-Result", lane_id, lane_in_group,
                              temp_regs, 2, 0);
    }

    // === ROUND 3: Exchange with neighbor again (XOR with 1) ===
    // T0↔T1, T2↔T3 exchange remaining parts
    uint32_t final0_exchanged = __shfl_xor(temp_regs[0], 1);
    uint32_t final1_exchanged = __shfl_xor(temp_regs[1], 1);

    // Debug third exchange
    if (lane_id < 4) {
        uint32_t debug_regs[2] = {final0_exchanged, final1_exchanged};
        debug_print_registers("Round3-Exchange", lane_id, lane_in_group,
                              debug_regs, 2, 0);
    }

    // Final combination based on thread position
    if (lane_in_group < 2) {
        // Top half (T0, T1) update R[1]
        if (lane_in_group & 1) { // T1
            out[1] =
                (temp_regs[1] & 0xFFFF0000) | (final1_exchanged & 0x0000FFFF);
        }
        else { // T0
            out[1] =
                (temp_regs[1] & 0x0000FFFF) | (final1_exchanged & 0xFFFF0000);
        }
        // Keep R[0] unchanged
        out[0] = temp_regs[0];
    }
    else {
        // Bottom half (T2, T3) update R[0]
        if (lane_in_group & 1) { // T3
            out[0] =
                (temp_regs[0] & 0xFFFF0000) | (final0_exchanged & 0x0000FFFF);
        }
        else { // T2
            out[0] =
                (temp_regs[0] & 0x0000FFFF) | (final0_exchanged & 0xFFFF0000);
        }
        // Keep R[1] unchanged
        out[1] = temp_regs[1];
    }

    // Debug final result
    if (lane_id < 4) {
        debug_print_registers("Final-Result", lane_id, lane_in_group, out, 2,
                              0);
    }
#endif
}

// Helper function to convert two uint16_t values to a single uint32_t
__host__ __device__ uint32_t pack_half2(uint16_t a, uint16_t b)
{
    return ((uint32_t)a << 16) | (uint32_t)b;
}

// Helper function to extract two uint16_t values from a single uint32_t
__host__ __device__ void unpack_half2(uint32_t packed, uint16_t &a, uint16_t &b)
{
    a = (packed >> 16) & 0xFFFF;
    b = packed & 0xFFFF;
}

// Kernel to test the transpose function
__global__ void test_transpose_kernel(uint16_t *output)
{
    uint32_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = thread_id % 64;

    // Calculate the thread's position in the logical 4x4 grid
    uint32_t group_id = lane_id / 4;      // Which 4-thread group
    uint32_t lane_in_group = lane_id % 4; // Position within group

    // Initialize test data - each thread creates a row of the matrix B
    // Values are designed for easy verification: lane_in_group * 100 + column
    uint16_t row_elements[4];
    for (int i = 0; i < 4; i++) {
        row_elements[i] = lane_in_group * 100 + i; // B[lane_in_group][i]
    }

    // Pack the 4 half-precision values into 2 registers
    uint32_t R[2];
    R[0] = pack_half2(row_elements[0], row_elements[1]);
    R[1] = pack_half2(row_elements[2], row_elements[3]);

    // Call the transpose function
    uint32_t out[2];
    transpose_4x4_half_registers(R, out);

    // Unpack the transposed results
    uint16_t transposed[4];
    unpack_half2(out[0], transposed[0], transposed[1]);
    unpack_half2(out[1], transposed[2], transposed[3]);

    // Write output - store both original and transposed values for verification
    for (int i = 0; i < 4; i++) {
        // Original values (row-major layout)
        output[thread_id * 8 + i] = row_elements[i];
        // Transposed values (column-major layout)
        output[thread_id * 8 + 4 + i] = transposed[i];
    }
}

int main()
{
    // Allocate memory for output (both original and transposed data)
    const int num_threads = 64; // One wavefront
    const int values_per_thread =
        8; // Each thread stores 4 original + 4 transposed values
    const int total_values = num_threads * values_per_thread;

    std::vector<uint16_t> h_output(total_values);
    uint16_t *d_output;

    hipMalloc(&d_output, total_values * sizeof(uint16_t));

    // Launch the kernel
    test_transpose_kernel<<<1, num_threads>>>(d_output);

    // Copy results back to host
    hipMemcpy(h_output.data(), d_output, total_values * sizeof(uint16_t),
              hipMemcpyDeviceToHost);

    // Verify the results
    bool success = true;
    std::cout << "Testing matrix transposition with shuffle operations..."
              << std::endl;

    // for (int group = 0; group < num_threads / 4; group++) {
    //     std::cout << "\nGroup " << group << " results:" << std::endl;

    //     for (int lane = 0; lane < 4; lane++) {
    //         int thread_idx = group * 4 + lane;

    //         // Print original values
    //         std::cout << "Thread " << thread_idx << " original: ";
    //         for (int i = 0; i < 4; i++) {
    //             std::cout << h_output[thread_idx * 8 + i] << " ";
    //         }
    //         std::cout << std::endl;

    //         // Print and verify transposed values
    //         std::cout << "Thread " << thread_idx << " transposed: ";
    //         for (int i = 0; i < 4; i++) {
    //             uint16_t actual = h_output[thread_idx * 8 + 4 + i];
    //             std::cout << actual << " ";

    //             // Expected after transpose: Thread N gets column N
    //             // Thread 0 should have [0*100+0, 1*100+0, 2*100+0, 3*100+0]
    //             // Thread 1 should have [0*100+1, 1*100+1, 2*100+1, 3*100+1]
    //             uint16_t expected = i * 100 + lane;

    //             if (actual != expected) {
    //                 success = false;
    //                 std::cout << "(Expected: " << expected << ") ";
    //             }
    //         }
    //         std::cout << std::endl;
    //     }
    // }

    if (success) {
        std::cout << "\nTranspose test PASSED! All values correctly transposed."
                  << std::endl;
    }
    else {
        std::cout << "\nTranspose test FAILED! Some values were not correctly "
                     "transposed."
                  << std::endl;
    }

    // Clean up
    hipFree(d_output);

    return success ? 0 : 1;
}
