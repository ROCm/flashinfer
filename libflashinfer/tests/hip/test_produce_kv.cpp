#include <algorithm>
#include <cstdio>
#include <gtest/gtest.h>
#include <iomanip>
#include <vector>

// Constants
constexpr uint32_t WARP_SIZE_NV = 32;
constexpr uint32_t WARP_SIZE_AMD = 64;
constexpr uint32_t WARP_STEP_SIZE = 16;  // 16 threads per warp row for AMD
constexpr uint32_t WARP_THREAD_ROWS = 4; // 4 rows of threads in a warp for AMD

// SwizzleMode enum to match the original code
enum class SwizzleMode
{
    k64B = 0U,   // Original NVIDIA mode (32 threads, 8 rows x 4 columns)
    k128B = 1U,  // Original pseudo-128B mode (32 threads, 4 rows x 8 columns)
    kLinear = 2U // New AMD-specific mode (64 threads, 4 rows x 16 columns)
};

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

// CPU-based simulation of produce_kv for AMD MI300 with linear offset
// addressing
template <uint32_t HEAD_DIM, uint32_t NUM_MMA_KV = 1>
void SimulateProduceKV(std::vector<int> &thread_ids_at_offsets)
{
    // Constants for MI300 (64-thread warp, 4×16 thread layout)
    constexpr uint32_t WARP_SIZE = 64;
    constexpr uint32_t WARP_THREAD_ROWS = 4; // 4 rows of threads
    constexpr uint32_t WARP_STEP_SIZE = 16;  // 16 threads per row
    constexpr uint32_t ELEMS_PER_THREAD =
        4; // Each thread loads 4 fp16 elements

    // Derived constants
    constexpr uint32_t UPCAST_STRIDE = HEAD_DIM / ELEMS_PER_THREAD;
    constexpr uint32_t NUM_MMA_D = HEAD_DIM / 16;
    constexpr uint32_t grid_width = HEAD_DIM / ELEMS_PER_THREAD;
    constexpr uint32_t grid_height = 16 * NUM_MMA_KV;
    constexpr uint32_t NUM_WARPS = 1;
    constexpr uint32_t NUM_WARPS_Q = 1;
    constexpr uint32_t COLUMN_RESET_OFFSET = (NUM_MMA_D / 4) * WARP_STEP_SIZE;
    //(NUM_MMA_D / (4 / sizeof(uint16_t))) * WARP_STEP_SIZE;

    // Initialize with -1 (unwritten)
    thread_ids_at_offsets.assign(grid_height * grid_width, -1);

    // Simulate each thread's write pattern
    for (uint32_t tid = 0; tid < WARP_SIZE; tid++) {
        uint32_t warp_idx = 0; // Always 0 for single warp
        uint32_t lane_idx = tid;

        // Calculate thread's row and column
        uint32_t row = lane_idx / WARP_STEP_SIZE;
        uint32_t col = lane_idx % WARP_STEP_SIZE;

        // Calculate initial offset
        uint32_t kv_smem_offset_w = get_permuted_offset_linear<UPCAST_STRIDE>(
            warp_idx * WARP_THREAD_ROWS + row, col);

        // Initial kv_idx points to the first row this thread handles
        uint32_t kv_idx = warp_idx * WARP_THREAD_ROWS + row;

        // Handle all blocks of rows
        for (uint32_t i = 0; i < NUM_MMA_KV * 4 / NUM_WARPS_Q; ++i) {
            // Process columns within a row (each thread loads 4 elements per
            // iteration)
            // for (uint32_t j = 0; j < NUM_MMA_D / (4 / sizeof(uint16_t)); ++j)
            // {
            for (uint32_t j = 0; j < NUM_MMA_D / 4; ++j) {
                // Record which thread writes to this offset
                // if(tid == 0) {
                //     std::cout << "tid : " << tid << " kv_smem_offset_w at
                //     start " << kv_smem_offset_w << '\n';
                // }
                if (kv_smem_offset_w < grid_height * grid_width &&
                    kv_idx < grid_height)
                {
                    thread_ids_at_offsets[kv_smem_offset_w] = tid;
                }
                else {
                    std::cerr << "ERROR: Out of bound offset ("
                              << kv_smem_offset_w << ") at " << tid << '\n';
                }

                // Advance to next column by 16 (number of threads per row)
                kv_smem_offset_w =
                    advance_offset_by_column_linear<WARP_STEP_SIZE>(
                        kv_smem_offset_w, j);
                // if(tid == 0) {
                //     std::cout << "tid : " << tid << " kv_smem_offset_w after
                //     column inc: " << kv_smem_offset_w << '\n';
                // }
            }

            // Move to next set of rows
            kv_idx += WARP_THREAD_ROWS;

            // if(tid == 0) {
            //     std::cout << "tid : " << tid << " kv_smem_offset_w before row
            //     inc " << kv_smem_offset_w << '\n';
            // }
            // Reset column position and advance rows
            kv_smem_offset_w =
                advance_offset_by_row_linear<NUM_WARPS * WARP_THREAD_ROWS,
                                             UPCAST_STRIDE>(kv_smem_offset_w) -
                COLUMN_RESET_OFFSET;

            // if(tid == 0) {
            //     std::cout << "tid : " << tid << " kv_smem_offset_w after row
            //     inc " << kv_smem_offset_w << '\n';
            // }
        }
        // FIXME: Verify with original in prefill.cuh
        kv_smem_offset_w -= 16 * NUM_MMA_KV * UPCAST_STRIDE;
    }
}

// Helper function to run the test
template <uint32_t HEAD_DIM, uint32_t NUM_MMA_KV = 1> void RunProduceKVTest()
{
    constexpr uint32_t grid_width = HEAD_DIM / 4; // 16 for 64, 32 for 128
    constexpr uint32_t grid_height =
        16 * NUM_MMA_KV; // 16 for NUM_MMA_KV=1, 32 for NUM_MMA_KV=2

    printf("\n=== Testing produce_kv with HEAD_DIM = %u, NUM_MMA_KV = %u ===\n",
           HEAD_DIM, NUM_MMA_KV);

    // Host array to store thread IDs at each offset
    std::vector<int> thread_ids(grid_height * grid_width, -1);

    // Run CPU simulation of produce_kv
    SimulateProduceKV<HEAD_DIM, NUM_MMA_KV>(thread_ids);

    // Print the grid of thread IDs
    printf("Thread IDs writing to each offset (%dx%d grid):\n", grid_height,
           grid_width);

    // Column headers
    printf("    ");
    for (int c = 0; c < std::min(32, (int)grid_width); c++) {
        printf("%3d ", c);
        if (c == 15 && grid_width > 16)
            printf("| ");
    }
    printf("\n   +");
    for (int c = 0; c < std::min(32, (int)grid_width); c++) {
        printf("----");
        if (c == 15 && grid_width > 16)
            printf("+");
    }
    printf("\n");

    // Print grid with clear separation
    for (int r = 0; r < grid_height; r++) {
        printf("%2d | ", r);
        for (int c = 0; c < std::min(32, (int)grid_width); c++) {
            int thread_id = thread_ids[r * grid_width + c];
            if (thread_id >= 0) {
                printf("%3d ", thread_id);
            }
            else {
                printf("  . ");
            }
            if (c == 15 && grid_width > 16)
                printf("| ");
        }
        printf("\n");

        // Add horizontal divider between blocks
        if (r == 15 && NUM_MMA_KV > 1) {
            printf("   +");
            for (int c = 0; c < std::min(32, (int)grid_width); c++) {
                printf("----");
                if (c == 15 && grid_width > 16)
                    printf("+");
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
}

TEST(KVCacheWritePatternTest, HeadDim64_AMD_kLinear)
{
    RunProduceKVTest<64, 1>();
}

TEST(KVCacheWritePatternTest, HeadDim128_AMD_kLinear)
{
    RunProduceKVTest<128, 1>();
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
