#include <algorithm>
#include <cstdio>
#include <gtest/gtest.h>
#include <iomanip>
#include <vector>

// Constants
constexpr uint32_t WARP_SIZE_NV = 32;
constexpr uint32_t WARP_SIZE_AMD = 64;

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

// CPU-based simulation of produce_kv for different SwizzleMode values
template <uint32_t HEAD_DIM,
          uint32_t NUM_MMA_KV = 1,
          SwizzleMode SWIZZLE_MODE = SwizzleMode::kLinear>
void SimulateProduceKV(std::vector<int> &thread_ids_at_offsets,
                       uint32_t warp_size = WARP_SIZE_AMD)
{
    // Constants derived from HEAD_DIM and SwizzleMode
    constexpr uint32_t ELEMS_PER_THREAD = 4;
    constexpr uint32_t UPCAST_STRIDE = HEAD_DIM / ELEMS_PER_THREAD;
    constexpr uint32_t NUM_MMA_D = HEAD_DIM / 16;
    constexpr uint32_t grid_width =
        HEAD_DIM / ELEMS_PER_THREAD; // 16 for 64, 32 for 128
    constexpr uint32_t grid_height =
        16 * NUM_MMA_KV; // 16 for NUM_MMA_KV=1, 32 for NUM_MMA_KV=2

    // Thread layout constants based on SwizzleMode
    constexpr uint32_t KV_THR_LAYOUT_ROW =
        SWIZZLE_MODE == SwizzleMode::k128B  ? 4
        : SWIZZLE_MODE == SwizzleMode::k64B ? 8
                                            : 4; // 4 for kLinear (AMD)

    constexpr uint32_t KV_THR_LAYOUT_COL =
        SWIZZLE_MODE == SwizzleMode::k128B  ? 8
        : SWIZZLE_MODE == SwizzleMode::k64B ? 4
                                            : 16; // 16 for kLinear (AMD)

    constexpr uint32_t NUM_WARPS = 1;
    constexpr uint32_t NUM_WARPS_Q = 1;

    // Initialize with -1 (unwritten)
    thread_ids_at_offsets.assign(grid_height * grid_width, -1);

    // Simulate each thread's write pattern
    for (uint32_t tid = 0; tid < warp_size; tid++) {
        uint32_t warp_idx = tid / warp_size; // Always 0 for single warp
        uint32_t lane_idx = tid;

        // Calculate the initial shared memory offset and global memory index
        uint32_t kv_smem_offset_w = get_permuted_offset_linear<UPCAST_STRIDE>(
            warp_idx * KV_THR_LAYOUT_ROW + lane_idx / KV_THR_LAYOUT_COL,
            lane_idx % KV_THR_LAYOUT_COL);

        if constexpr (SWIZZLE_MODE == SwizzleMode::k128B) {
            // k128B mode (original pseudo-128B mode)
            uint32_t kv_idx = warp_idx * 4 + lane_idx / 8;

            static_assert(NUM_MMA_KV * 4 % NUM_WARPS_Q == 0);
            for (uint32_t i = 0; i < NUM_MMA_KV * 4 / NUM_WARPS_Q; ++i) {
                for (uint32_t j = 0; j < NUM_MMA_D / (8 / sizeof(uint16_t));
                     ++j)
                {
                    // Record which thread writes to this offset
                    if (kv_smem_offset_w < grid_height * grid_width &&
                        kv_idx < grid_height)
                    {
                        thread_ids_at_offsets[kv_smem_offset_w] = tid;
                    }

                    // Advance to next column within same row
                    kv_smem_offset_w =
                        advance_offset_by_column_linear<8>(kv_smem_offset_w, j);
                }

                kv_idx += NUM_WARPS * 4;
                kv_smem_offset_w =
                    advance_offset_by_row_linear<NUM_WARPS * 4, UPCAST_STRIDE>(
                        kv_smem_offset_w) -
                    sizeof(uint16_t) * NUM_MMA_D;
            }
        }
        else if constexpr (SWIZZLE_MODE == SwizzleMode::k64B) {
            // k64B mode (original NVIDIA mode)
            uint32_t kv_idx = warp_idx * 8 + lane_idx / 4;

            static_assert(NUM_MMA_KV * 2 % NUM_WARPS_Q == 0);
            for (uint32_t i = 0; i < NUM_MMA_KV * 2 / NUM_WARPS_Q; ++i) {
                // Record which thread writes to this offset
                if (kv_smem_offset_w < grid_height * grid_width &&
                    kv_idx < grid_height)
                {
                    thread_ids_at_offsets[kv_smem_offset_w] = tid;
                }

                kv_smem_offset_w =
                    advance_offset_by_row_linear<NUM_WARPS * 8, UPCAST_STRIDE>(
                        kv_smem_offset_w);
                kv_idx += NUM_WARPS * 8;
            }
        }
        else if constexpr (SWIZZLE_MODE == SwizzleMode::kLinear) {
            // kLinear mode (AMD-specific, using all 64 threads)
            uint32_t kv_idx = warp_idx * 4 + lane_idx / 16;

            // For AMD's 64-thread warp, we need to process 4 rows with 16
            // threads per row
            for (uint32_t i = 0; i < NUM_MMA_KV; ++i) {
                for (uint32_t j = 0; j < NUM_MMA_D; ++j) {
                    // Record which thread writes to this offset
                    if (kv_smem_offset_w < grid_height * grid_width &&
                        kv_idx < grid_height)
                    {
                        thread_ids_at_offsets[kv_smem_offset_w] = tid;
                    }

                    // Advance to next column within same row
                    kv_smem_offset_w =
                        advance_offset_by_column_linear<ELEMS_PER_THREAD>(
                            kv_smem_offset_w, j);
                }

                kv_idx += 4; // Advance by 4 rows
                kv_smem_offset_w =
                    advance_offset_by_row_linear<4, UPCAST_STRIDE>(
                        kv_smem_offset_w) -
                    NUM_MMA_D * ELEMS_PER_THREAD;
            }
        }
    }
}

// Helper function to run the test for different SwizzleModes
template <uint32_t HEAD_DIM,
          uint32_t NUM_MMA_KV = 1,
          SwizzleMode SWIZZLE_MODE = SwizzleMode::kLinear>
void RunProduceKVTest(uint32_t warp_size = WARP_SIZE_AMD)
{
    constexpr uint32_t grid_width = HEAD_DIM / 4; // 16 for 64, 32 for 128
    constexpr uint32_t grid_height =
        16 * NUM_MMA_KV; // 16 for NUM_MMA_KV=1, 32 for NUM_MMA_KV=2

    std::string swizzle_mode_str;
    switch (SWIZZLE_MODE) {
    case SwizzleMode::k64B:
        swizzle_mode_str = "k64B (NVIDIA)";
        break;
    case SwizzleMode::k128B:
        swizzle_mode_str = "k128B (NVIDIA pseudo-128B)";
        break;
    case SwizzleMode::kLinear:
        swizzle_mode_str = "kLinear (AMD)";
        break;
    }

    printf("\n=== Testing produce_kv with HEAD_DIM = %u, NUM_MMA_KV = %u, "
           "SwizzleMode = %s ===\n",
           HEAD_DIM, NUM_MMA_KV, swizzle_mode_str.c_str());

    // Host array to store thread IDs at each offset
    std::vector<int> thread_ids(grid_height * grid_width, -1);

    // Run CPU simulation of produce_kv
    SimulateProduceKV<HEAD_DIM, NUM_MMA_KV, SWIZZLE_MODE>(thread_ids,
                                                          warp_size);

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

// Tests for different SwizzleModes
// TEST(KVCacheWritePatternTest, HeadDim64_NVIDIA_k64B) {
//     RunProduceKVTest<64, 1, SwizzleMode::k64B>(WARP_SIZE_NV);
// }

// TEST(KVCacheWritePatternTest, HeadDim64_NVIDIA_k128B) {
//     RunProduceKVTest<64, 1, SwizzleMode::k128B>(WARP_SIZE_NV);
// }

TEST(KVCacheWritePatternTest, HeadDim64_AMD_kLinear)
{
    RunProduceKVTest<64, 1, SwizzleMode::kLinear>(WARP_SIZE_AMD);
}

TEST(KVCacheWritePatternTest, HeadDim128_AMD_kLinear)
{
    RunProduceKVTest<128, 1, SwizzleMode::kLinear>(WARP_SIZE_AMD);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
