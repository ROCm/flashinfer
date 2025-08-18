// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "flashinfer/attention/generic/prefill.cuh"
#include "gpu_iface/gpu_runtime_compat.hpp"

using namespace flashinfer;

template <typename KTraits>
__global__ void ComputeQKStubKernel(typename KTraits::DTypeQ *q,
                                    typename KTraits::DTypeKV *k,
                                    float *qk_scores_output,
                                    uint32_t qo_len,
                                    uint32_t kv_len,
                                    uint32_t num_qo_heads,
                                    uint32_t num_kv_heads,
                                    uint32_t q_stride_n,
                                    uint32_t q_stride_h,
                                    uint32_t k_stride_n,
                                    uint32_t k_stride_h,
                                    uint_fastdiv group_size)
{
    using DTypeQ = typename KTraits::DTypeQ;
    using DTypeKV = typename KTraits::DTypeKV;
    using DTypeQKAccum = typename KTraits::DTypeQKAccum;

    extern __shared__ uint8_t smem[];
    typename KTraits::SharedStorage &smem_storage =
        reinterpret_cast<typename KTraits::SharedStorage &>(smem);

    // Initialize shared memory objects
    smem_t<KTraits::SWIZZLE_MODE_Q, typename KTraits::SmemBasePtrTy> q_smem(
        smem_storage.q_smem);
    smem_t<KTraits::SWIZZLE_MODE_KV, typename KTraits::SmemBasePtrTy> k_smem(
        smem_storage.k_smem);

    const uint32_t lane_idx = threadIdx.x;
    const uint32_t warp_idx = get_warp_idx<KTraits>(threadIdx.y, threadIdx.z);
    const uint32_t kv_head_idx = blockIdx.z;

    // 1. Load Q into shared memory (same as SinglePrefillWithKVCacheDevice)
    const uint32_t qo_packed_idx_base = (blockIdx.x * KTraits::NUM_WARPS_Q +
                                         get_warp_idx_q<KTraits>(threadIdx.y)) *
                                        KTraits::NUM_MMA_Q * 16;

    DTypeQ *q_ptr_base = q + (kv_head_idx * group_size) * q_stride_h;

    load_q_global_smem<KTraits>(qo_packed_idx_base, qo_len, q_ptr_base,
                                q_stride_n, q_stride_h, group_size, &q_smem,
                                threadIdx);

    // 2. Load K into shared memory (same as SinglePrefillWithKVCacheDevice)
    DTypeKV *k_ptr = k +
                     (warp_idx * KTraits::KV_THR_LAYOUT_ROW +
                      lane_idx / KTraits::KV_THR_LAYOUT_COL) *
                         k_stride_n +
                     kv_head_idx * k_stride_h +
                     (lane_idx % KTraits::KV_THR_LAYOUT_COL) *
                         upcast_size<DTypeKV, KTraits::VECTOR_BIT_WIDTH>();

    uint32_t k_smem_offset_w =
        k_smem.template get_permuted_offset<KTraits::UPCAST_STRIDE_K>(
            warp_idx * KTraits::KV_THR_LAYOUT_ROW +
                lane_idx / KTraits::KV_THR_LAYOUT_COL,
            lane_idx % KTraits::KV_THR_LAYOUT_COL);

    produce_kv<false, SharedMemFillMode::kNoFill, KTraits>(
        k_smem, &k_smem_offset_w, &k_ptr, k_stride_n, 0, kv_len, threadIdx);

    __syncthreads();

    // 3. Set up fragment offsets for compute_qk (same as
    // SinglePrefillWithKVCacheDevice)
    uint32_t q_smem_offset_r =
        q_smem.template get_permuted_offset<KTraits::UPCAST_STRIDE_Q>(
            get_warp_idx_q<KTraits>(threadIdx.y) * KTraits::NUM_MMA_Q * 16 +
                lane_idx % 16,
            lane_idx / 16);

    uint32_t k_smem_offset_r =
        k_smem.template get_permuted_offset<KTraits::UPCAST_STRIDE_K>(
            get_warp_idx_kv<KTraits>(threadIdx.z) * KTraits::NUM_MMA_KV * 16 +
                KTraits::HALF_ELEMS_PER_THREAD * (lane_idx / 16) +
                lane_idx % KTraits::HALF_ELEMS_PER_THREAD,
            (lane_idx % 16) / KTraits::HALF_ELEMS_PER_THREAD);

    // 4. Call compute_qk (the function we want to test)
    DTypeQKAccum s_frag[KTraits::NUM_MMA_Q][KTraits::NUM_MMA_KV]
                       [KTraits::HALF_ELEMS_PER_THREAD];

    compute_qk<KTraits>(&q_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r,
                        s_frag);

    // 5. Extract attention scores from s_frag to global memory
    // Simple extraction for validation - use first warp only
    if (get_warp_idx_q<KTraits>(threadIdx.y) == 0 &&
        get_warp_idx_kv<KTraits>(threadIdx.z) == 0)
    {
        // Map from MMA fragment layout to sequence indices
        for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
            for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
                for (uint32_t reg_id = 0;
                     reg_id < KTraits::HALF_ELEMS_PER_THREAD; ++reg_id)
                {
                    // Calculate global Q,K indices from fragment indices
#if defined(PLATFORM_HIP_DEVICE)
                    uint32_t q_idx =
                        mma_q * 16 +
                        reg_id % KTraits::NUM_ACCUM_ROWS_PER_THREAD;
                    uint32_t kv_idx =
                        mma_kv * 16 +
                        2 * (lane_idx % KTraits::THREADS_PER_MATRIX_ROW_SET) +
                        8 * (reg_id / 2) + reg_id % 2;
#else
                    uint32_t q_idx = mma_q * 16 + (reg_id % 4) / 2;
                    uint32_t kv_idx =
                        mma_kv * 16 + 8 * (reg_id / 4) + reg_id % 2;
#endif

                    if (q_idx < qo_len && kv_idx < kv_len) {
                        uint32_t output_idx = q_idx * kv_len + kv_idx;
                        qk_scores_output[output_idx] =
                            float(s_frag[mma_q][mma_kv][reg_id]);
                    }
                }
            }
        }
    }
}

template <typename DTypeQ, typename DTypeKV>
hipError_t ComputeQKStub(DTypeQ *q,
                         DTypeKV *k,
                         float *qk_scores_output,
                         uint32_t qo_len,
                         uint32_t kv_len,
                         uint32_t num_qo_heads,
                         uint32_t num_kv_heads,
                         uint32_t head_dim,
                         hipStream_t stream = nullptr)
{
    // Use same KernelTraits selection as SinglePrefillWithKVCache
    constexpr uint32_t NUM_MMA_D_QK = 4; // head_dim=64 -> 64/16=4
    constexpr uint32_t NUM_MMA_D_VO = 4;
    constexpr uint32_t CTA_TILE_Q = 16;
    constexpr uint32_t NUM_MMA_Q = 1;
    constexpr uint32_t NUM_MMA_KV = 1;
    constexpr uint32_t NUM_WARPS_Q = 1;
    constexpr uint32_t NUM_WARPS_KV = 1;

    using KTraits =
        KernelTraits<MaskMode::kNone, CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV,
                     NUM_MMA_D_QK, NUM_MMA_D_VO, NUM_WARPS_Q, NUM_WARPS_KV,
                     PosEncodingMode::kNone, DTypeQ, DTypeKV, DTypeQ, float,
                     uint32_t, DefaultAttention<false, false, false, false>>;

    // Launch configuration (same pattern as SinglePrefillWithKVCache)
    const uint32_t group_size = num_qo_heads / num_kv_heads;
    const uint_fastdiv group_size_fastdiv(group_size);

    dim3 block_size(KTraits::NUM_THREADS, 1, 1);
    dim3 grid_size(1, 1, num_kv_heads);
    size_t shared_mem_size = sizeof(typename KTraits::SharedStorage);

    const uint32_t q_stride_n = num_qo_heads * head_dim;
    const uint32_t q_stride_h = head_dim;
    const uint32_t k_stride_n = num_kv_heads * head_dim;
    const uint32_t k_stride_h = head_dim;

    ComputeQKStubKernel<KTraits>
        <<<grid_size, block_size, shared_mem_size, stream>>>(
            q, k, qk_scores_output, qo_len, kv_len, num_qo_heads, num_kv_heads,
            q_stride_n, q_stride_h, k_stride_n, k_stride_h, group_size_fastdiv);

    return hipGetLastError();
}

// Explicit instantiation for common types
template hipError_t ComputeQKStub<__half, __half>(__half *,
                                                  __half *,
                                                  float *,
                                                  uint32_t,
                                                  uint32_t,
                                                  uint32_t,
                                                  uint32_t,
                                                  uint32_t,
                                                  hipStream_t);
