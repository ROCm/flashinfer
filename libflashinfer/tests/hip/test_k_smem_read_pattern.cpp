#include <algorithm>
#include <cstdio>
#include <gtest/gtest.h>
#include <iomanip>
#include <vector>

// Constants for MI300
constexpr uint32_t WARP_SIZE = 64; // 64 threads per wavefront
constexpr uint32_t HALF_ELEMS_PER_THREAD =
    4; // Each thread processes 4 half elements
constexpr uint32_t INT32_ELEMS_PER_THREAD = 2; // 2 int32 registers per thread

// Simplified linear shared memory operations (CPU implementation)
template <uint32_t stride>
uint32_t get_permuted_offset_linear(uint32_t row, uint32_t col)
{
    return row * stride + col;
}

template <uint32_t step_size>
uint32_t advance_offset_by_column_linear(uint32_t offset, uint32_t step_idx)
{
    return offset + step_size;
}

template <uint32_t step_size, uint32_t row_stride>
uint32_t advance_offset_by_row_linear(uint32_t offset)
{
    return offset + step_size * row_stride;
}

// CPU-based simulation of k-matrix access pattern in compute_qk
template <uint32_t HEAD_DIM, uint32_t NUM_MMA_KV = 1>
void SimulateKReadPattern(std::vector<int> &thread_ids_reading_offsets)
{
    // Constants derived from HEAD_DIM
    constexpr uint32_t UPCAST_STRIDE_K = HEAD_DIM / HALF_ELEMS_PER_THREAD;
    constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM / 16;
    constexpr uint32_t grid_width = HEAD_DIM / HALF_ELEMS_PER_THREAD;
    constexpr uint32_t grid_height = 16 * NUM_MMA_KV;

    constexpr uint32_t K_SMEM_COLUMN_ADVANCE =
        16 / HALF_ELEMS_PER_THREAD; // = 4 for MI300

    // Initialize with -1 (unread)
    thread_ids_reading_offsets.assign(grid_height * grid_width, -1);

    // Simulate each thread's read pattern
    for (uint32_t tid = 0; tid < WARP_SIZE; tid++) {
        // Map tid to kernel's lane_idx
        uint32_t lane_idx = tid;
        uint32_t warp_idx_kv = 0; // For simplicity, assuming one warp group

        // Exactly match the kernel's initial offset calculation - MI300 version
        uint32_t k_smem_offset_r = get_permuted_offset_linear<UPCAST_STRIDE_K>(
            warp_idx_kv * NUM_MMA_KV * 16 + 4 * (lane_idx / 16) + lane_idx % 4,
            (lane_idx % 16) / 4);

        // uint32_t k_smem_offset_r =
        // get_permuted_offset_linear<UPCAST_STRIDE_K>(
        //     warp_idx_kv * NUM_MMA_KV * 16 +
        //     4 * (lane_idx / 16),
        //     (lane_idx % 16));

        // Follow the same loop structure as in compute_qk
        for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_QK; ++mma_d) {
            for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
                // Mark grid positions accessed by ldmatrix_m8n8x4 /
                // load_fragment
                uint32_t read_row = k_smem_offset_r / UPCAST_STRIDE_K;
                uint32_t read_col = k_smem_offset_r % UPCAST_STRIDE_K;

                if (tid == 0) {
                    std::cout << "Thread " << tid << " k_smem_offset_r "
                              << k_smem_offset_r << '\n';
                }

                // Simulate loading a matrix fragment
                for (uint32_t reg_id = 0; reg_id < INT32_ELEMS_PER_THREAD;
                     reg_id++)
                {
                    if (read_row < grid_height && read_col < grid_width) {
                        thread_ids_reading_offsets[read_row * grid_width +
                                                   read_col] = tid;
                    }

                    // Each INT32_ELEMS_PER_THREAD register holds 2 half
                    // elements For simplicity, we're just recording the base
                    // offset
                }

                // Advance to next row, exactly as in compute_qk
                k_smem_offset_r =
                    advance_offset_by_row_linear<16, UPCAST_STRIDE_K>(
                        k_smem_offset_r);
            }

            // Reset row position and advance to next column section, exactly as
            // in compute_qk For MI300, advance by 4 columns (vs 2 for NVIDIA)
            k_smem_offset_r =
                advance_offset_by_column_linear<K_SMEM_COLUMN_ADVANCE>(
                    k_smem_offset_r, mma_d) -
                NUM_MMA_KV * 16 * UPCAST_STRIDE_K;
        }
    }
}

// Helper function to run the test with configurable parameters
template <uint32_t HEAD_DIM, uint32_t NUM_MMA_KV = 1> void RunKReadPatternTest()
{
    constexpr uint32_t grid_width = HEAD_DIM / HALF_ELEMS_PER_THREAD;
    constexpr uint32_t grid_height = 16 * NUM_MMA_KV;

    printf("\n=== Testing key read pattern with HEAD_DIM = %u, NUM_MMA_KV = %u "
           "===\n",
           HEAD_DIM, NUM_MMA_KV);

    // Host array to store thread IDs at each offset
    std::vector<int> thread_ids(grid_height * grid_width, -1);

    // Run CPU simulation of read pattern
    SimulateKReadPattern<HEAD_DIM, NUM_MMA_KV>(thread_ids);

    // Print the grid of thread IDs
    printf("Thread IDs reading from each offset (%dx%d grid):\n", grid_height,
           grid_width);

    // Column headers
    printf("    ");
    for (int c = 0; c < grid_width; c++) {
        printf("%3d ", c);
        if (c == 15 && grid_width > 16)
            printf("| "); // Divider for HEAD_DIM=128
    }
    printf("\n   +");
    for (int c = 0; c < grid_width; c++) {
        printf("----");
        if (c == 15 && grid_width > 16)
            printf("+");
    }
    printf("\n");

    // Print the grid
    for (int r = 0; r < grid_height; r++) {
        printf("%2d | ", r);
        for (int c = 0; c < grid_width; c++) {
            int thread_id = thread_ids[r * grid_width + c];
            if (thread_id >= 0) {
                printf("%3d ", thread_id);
            }
            else {
                printf("  . "); // Dot for unread positions
            }
            if (c == 15 && grid_width > 16)
                printf("| "); // Divider for HEAD_DIM=128
        }
        printf("\n");
    }

    // Check for unread positions
    int unread = 0;
    for (int i = 0; i < grid_height * grid_width; i++) {
        if (thread_ids[i] == -1) {
            unread++;
        }
    }

    // Print statistics
    printf("\nStatistics:\n");
    printf("- Positions read: %d/%d (%.1f%%)\n",
           grid_height * grid_width - unread, grid_height * grid_width,
           100.0f * (grid_height * grid_width - unread) /
               (grid_height * grid_width));
    printf("- Unread positions: %d/%d (%.1f%%)\n", unread,
           grid_height * grid_width,
           100.0f * unread / (grid_height * grid_width));

    // Validate full coverage
    EXPECT_EQ(unread, 0) << "Not all positions were read";
}

// Tests for different configurations
TEST(MI300KReadPatternTest, HeadDim64_NumMmaKV1)
{
    RunKReadPatternTest<64, 1>();
}

// TEST(MI300KReadPatternTest, HeadDim128_NumMmaKV1) {
//     RunKReadPatternTest<128, 1>();
// }

// TEST(MI300KReadPatternTest, HeadDim64_NumMmaKV2) {
//     RunKReadPatternTest<64, 2>();
// }

// TEST(MI300KReadPatternTest, HeadDim128_NumMmaKV2) {
//     RunKReadPatternTest<128, 2>();
// }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
