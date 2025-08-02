#include <algorithm>
#include <cstdio>
#include <gtest/gtest.h>
#include <iomanip>
#include <vector>

// Constants for MI300
constexpr uint32_t WARP_STEP_SIZE = 16; // 16 threads per warp row
constexpr uint32_t QUERY_ELEMS_PER_THREAD = 4;
constexpr uint32_t WARP_THREAD_ROWS = 4; // 4 rows of threads in a warp

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

// CPU-based simulation of the read pattern in compute_qk
template <uint32_t HEAD_DIM, uint32_t NUM_MMA_Q = 1>
void SimulateReadPattern(std::vector<int> &thread_ids_reading_offsets)
{
    // Constants derived from HEAD_DIM
    constexpr uint32_t UPCAST_STRIDE_Q = HEAD_DIM / QUERY_ELEMS_PER_THREAD;
    constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM / 16;
    constexpr uint32_t grid_width =
        (HEAD_DIM / QUERY_ELEMS_PER_THREAD); // 16 for 64, 32 for 128
    constexpr uint32_t grid_height =
        16 * NUM_MMA_Q; // 16 for NUM_MMA_Q=1, 32 for NUM_MMA_Q=2

    // Initialize with -1 (unread)
    thread_ids_reading_offsets.assign(grid_height * grid_width, -1);

    // Simulate each thread's read pattern
    for (uint32_t tid = 0; tid < 64; tid++) {
        // Map tid to kernel's lane_idx (same for a single warp)
        uint32_t lane_idx = tid;

        // Get warp_idx_q (this is 0 for our single warp simulation)
        uint32_t warp_idx_q = 0;

        // Exactly match the kernel's initial offset calculation
        uint32_t q_smem_offset_r = get_permuted_offset_linear<UPCAST_STRIDE_Q>(
            warp_idx_q * NUM_MMA_Q * 16 + lane_idx % 16, lane_idx / 16);

        // Follow exactly the same loop structure as in compute_qk
        for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_QK; ++mma_d) {
            for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
                // This would be a ldmatrix_m8n8x4 call in the actual code
                uint32_t read_row = q_smem_offset_r / UPCAST_STRIDE_Q;
                uint32_t read_col = q_smem_offset_r % UPCAST_STRIDE_Q;

                if (read_row < grid_height && read_col < grid_width) {
                    thread_ids_reading_offsets[read_row * grid_width +
                                               read_col] = tid;
                }

                // Advance to next row, exactly as in compute_qk
                q_smem_offset_r =
                    advance_offset_by_row_linear<16, UPCAST_STRIDE_Q>(
                        q_smem_offset_r);
            }

            // Reset row position and advance to next column, exactly as in
            // compute_qk
            q_smem_offset_r =
                advance_offset_by_column_linear<4>(q_smem_offset_r, mma_d) -
                NUM_MMA_Q * 16 * UPCAST_STRIDE_Q;
        }
    }
}

// Helper function to run the test with configurable NUM_MMA_Q
template <uint32_t HEAD_DIM, uint32_t NUM_MMA_Q = 1> void RunReadPatternTest()
{
    constexpr uint32_t grid_width =
        (HEAD_DIM / QUERY_ELEMS_PER_THREAD); // 16 for 64, 32 for 128
    constexpr uint32_t grid_height =
        16 * NUM_MMA_Q; // 16 for NUM_MMA_Q=1, 32 for NUM_MMA_Q=2

    printf("\n=== Testing query read pattern with HEAD_DIM = %u, NUM_MMA_Q = "
           "%u ===\n",
           HEAD_DIM, NUM_MMA_Q);

    // Host array to store thread IDs at each offset
    std::vector<int> thread_ids(grid_height * grid_width, -1);

    // Run CPU simulation of read pattern
    SimulateReadPattern<HEAD_DIM, NUM_MMA_Q>(thread_ids);

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
TEST(MI300ReadPatternTest, HeadDim64_NumMmaQ1) { RunReadPatternTest<64, 1>(); }

TEST(MI300ReadPatternTest, HeadDim128_NumMmaQ1)
{
    RunReadPatternTest<128, 1>();
}

TEST(MI300ReadPatternTest, HeadDim64_NumMmaQ2) { RunReadPatternTest<64, 2>(); }

TEST(MI300ReadPatternTest, HeadDim128_NumMmaQ2)
{
    RunReadPatternTest<128, 2>();
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
