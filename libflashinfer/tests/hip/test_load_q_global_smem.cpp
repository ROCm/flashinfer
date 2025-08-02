#include <algorithm>
#include <cstdio>
#include <gtest/gtest.h>
#include <iomanip>
#include <vector>

// Constants for MI300
constexpr uint32_t WARP_STEP_SIZE = 16; // 16 threads per warp row
constexpr uint32_t QUERY_ELEMS_PER_THREAD =
    4;                                   // Each thread loads 4 fp16 elements
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

// CPU-based offset pattern verification with configurable NUM_MMA_Q
template <uint32_t HEAD_DIM, uint32_t NUM_MMA_Q = 1>
void SimulateOffsetPattern(std::vector<int> &thread_ids_at_offsets)
{
    // Constants derived from HEAD_DIM
    constexpr uint32_t UPCAST_STRIDE_Q = HEAD_DIM / QUERY_ELEMS_PER_THREAD;
    constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM / 16;
    constexpr uint32_t COLUMN_RESET_OFFSET =
        (NUM_MMA_D_QK / 4) * WARP_STEP_SIZE;
    constexpr uint32_t grid_width =
        (HEAD_DIM / QUERY_ELEMS_PER_THREAD); // 16 for 64, 32 for 128
    constexpr uint32_t grid_height =
        16 * NUM_MMA_Q; // 16 for NUM_MMA_Q=1, 32 for NUM_MMA_Q=2

    // Initialize with -1 (unwritten)
    thread_ids_at_offsets.assign(grid_height * grid_width, -1);

    // Simulate each thread
    for (uint32_t tid = 0; tid < 64; tid++) {
        uint32_t row = tid / WARP_STEP_SIZE; // 0-3 for 64 threads
        uint32_t col = tid % WARP_STEP_SIZE; // 0-15

        // Calculate initial offset using linear addressing
        uint32_t q_smem_offset_w =
            get_permuted_offset_linear<UPCAST_STRIDE_Q>(row, col);

        // Main loop structure from load_q_global_smem
        for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
            for (uint32_t j = 0; j < 4; ++j) {
                // Calculate sequence index
                const uint32_t seq_idx = row + mma_q * 16 + j;

                for (uint32_t mma_do = 0; mma_do < NUM_MMA_D_QK / 4; ++mma_do) {
                    // Record which thread wrote to this offset
                    if (q_smem_offset_w < grid_height * grid_width)
                    { // Safety check
                        thread_ids_at_offsets[q_smem_offset_w] = tid;
                    }
                    else {
                        printf("ERROR by tid: %d, offset: %d\n", tid,
                               q_smem_offset_w);
                    }

                    // Advance to next column within same row
                    q_smem_offset_w =
                        advance_offset_by_column_linear<WARP_STEP_SIZE>(
                            q_smem_offset_w, mma_do);
                }

                // Advance to next sequence (row) with adjustment back to first
                // column
                q_smem_offset_w = advance_offset_by_row_linear<WARP_THREAD_ROWS,
                                                               UPCAST_STRIDE_Q>(
                                      q_smem_offset_w) -
                                  COLUMN_RESET_OFFSET;
            }
        }
    }
}

// Helper function to run the test with configurable NUM_MMA_Q
template <uint32_t HEAD_DIM, uint32_t NUM_MMA_Q = 1> void RunOffsetTest()
{
    constexpr uint32_t grid_width =
        (HEAD_DIM / QUERY_ELEMS_PER_THREAD); // 16 for 64, 32 for 128
    constexpr uint32_t grid_height =
        16 * NUM_MMA_Q; // 16 for NUM_MMA_Q=1, 32 for NUM_MMA_Q=2

    printf("\n=== Testing offset calculations with HEAD_DIM = %u, NUM_MMA_Q = "
           "%u ===\n",
           HEAD_DIM, NUM_MMA_Q);

    // Host array to store thread IDs at each offset
    std::vector<int> thread_ids(grid_height * grid_width, -1);

    // Run CPU simulation of offset pattern
    SimulateOffsetPattern<HEAD_DIM, NUM_MMA_Q>(thread_ids);

    // Print the grid of thread IDs (potentially truncated for readability)
    printf("Thread IDs writing to each offset (%dx%d grid):\n", grid_height,
           grid_width);

    // Column headers
    printf("    ");
    for (int c = 0; c < grid_width; c++) {
        printf("%3d ", c);
        if (c == 15 && grid_width > 16)
            printf("| "); // Divider between first and second half
    }
    printf("\n   +");
    for (int c = 0; c < grid_width; c++) {
        printf("----");
        if (c == 15 && grid_width > 16)
            printf("+"); // Divider between first and second half
    }
    printf("\n");

    // Print quadrants with clear separation
    for (int r = 0; r < grid_height; r++) {
        printf("%2d | ", r);
        for (int c = 0; c < grid_width; c++) {
            int thread_id = thread_ids[r * grid_width + c];
            if (thread_id >= 0) {
                printf("%3d ", thread_id);
            }
            else {
                printf("  . "); // Dot for unwritten positions
            }
            if (c == 15 && grid_width > 16)
                printf("| "); // Divider between first and second half
        }
        printf("\n");

        // Add horizontal divider between first and second block of sequences
        if (r == 15 && NUM_MMA_Q > 1) {
            printf("   +");
            for (int c = 0; c < grid_width; c++) {
                printf("----");
                if (c == 15 && grid_width > 16)
                    printf("+"); // Intersection divider
            }
            printf("\n");
        }
    }

    // Check for unwritten positions
    int unwritten = 0;
    for (int i = 0; i < grid_height * grid_width; i++) {
        if (thread_ids[i] == -1) {
            unwritten++;
        }
    }

    // Print statistics
    printf("\nStatistics:\n");
    printf("- Positions written: %d/%d (%.1f%%)\n",
           grid_height * grid_width - unwritten, grid_height * grid_width,
           100.0f * (grid_height * grid_width - unwritten) /
               (grid_height * grid_width));
    printf("- Unwritten positions: %d/%d (%.1f%%)\n", unwritten,
           grid_height * grid_width,
           100.0f * unwritten / (grid_height * grid_width));

    // Validate full coverage
    EXPECT_EQ(unwritten, 0) << "Not all positions were written";
}

// Original tests with NUM_MMA_Q = 1
TEST(MI300OffsetTest, HeadDim64_NumMmaQ1) { RunOffsetTest<64, 1>(); }

TEST(MI300OffsetTest, HeadDim128_NumMmaQ1) { RunOffsetTest<128, 1>(); }

// New tests with NUM_MMA_Q = 2
TEST(MI300OffsetTest, HeadDim64_NumMmaQ2) { RunOffsetTest<64, 2>(); }

TEST(MI300OffsetTest, HeadDim128_NumMmaQ2) { RunOffsetTest<128, 2>(); }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
