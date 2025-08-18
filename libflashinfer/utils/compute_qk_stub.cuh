// SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "flashinfer/attention/generic/default_prefill_params.cuh"
#include "flashinfer/attention/generic/prefill.cuh"
#include "gpu_iface/gpu_runtime_compat.hpp"

using namespace flashinfer;

template <typename KTraits, typename Params>
__device__ __forceinline__ void
ComputeQKStubKernelDevice(const Params params,
                          typename KTraits::SharedStorage &smem_storage,
                          float *qk_scores_output,
                          const dim3 tid = threadIdx,
                          const uint32_t bx = blockIdx.x,
                          const uint32_t chunk_idx = blockIdx.y,
                          const uint32_t kv_head_idx = blockIdx.z,
                          const uint32_t num_chunks = gridDim.y,
                          const uint32_t num_kv_heads = gridDim.z)
{
    using DTypeKV = typename Params::DTypeKV;
    using DTypeQ = typename Params::DTypeQ;
    using DTypeQKAccum = typename KTraits::DTypeQKAccum;

    [[maybe_unused]] constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
    [[maybe_unused]] constexpr uint32_t NUM_MMA_D_QK = KTraits::NUM_MMA_D_QK;
    [[maybe_unused]] constexpr uint32_t HEAD_DIM_QK = KTraits::HEAD_DIM_QK;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_Q =
        KTraits::UPCAST_STRIDE_Q;
    [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_K =
        KTraits::UPCAST_STRIDE_K;
    [[maybe_unused]] constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
    [[maybe_unused]] constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
    [[maybe_unused]] constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
    [[maybe_unused]] constexpr uint32_t NUM_WARPS_KV = KTraits::NUM_WARPS_KV;
    [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q =
        KTraits::SWIZZLE_MODE_Q;
    [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_KV =
        KTraits::SWIZZLE_MODE_KV;
    [[maybe_unused]] constexpr uint32_t KV_THR_LAYOUT_ROW =
        KTraits::KV_THR_LAYOUT_ROW;
    [[maybe_unused]] constexpr uint32_t KV_THR_LAYOUT_COL =
        KTraits::KV_THR_LAYOUT_COL;
    [[maybe_unused]] constexpr uint32_t HALF_ELEMS_PER_THREAD =
        KTraits::HALF_ELEMS_PER_THREAD;
    [[maybe_unused]] constexpr uint32_t VECTOR_BIT_WIDTH =
        KTraits::VECTOR_BIT_WIDTH;

    DTypeQ *q = params.q;
    DTypeKV *k = params.k;

    const uint32_t qo_len = params.qo_len;
    const uint32_t kv_len = params.kv_len;

    const uint32_t q_stride_n = params.q_stride_n;
    const uint32_t q_stride_h = params.q_stride_h;
    const uint32_t k_stride_n = params.k_stride_n;
    const uint32_t k_stride_h = params.k_stride_h;
    const uint_fastdiv &group_size = params.group_size;

    static_assert(sizeof(DTypeQ) == 2);
    const uint32_t lane_idx = tid.x,
                   warp_idx = get_warp_idx<KTraits>(tid.y, tid.z);
    const uint32_t chunk_start = 0;
    const uint32_t chunk_size = kv_len;

    auto block = cg::this_thread_block();
    DTypeQKAccum s_frag[NUM_MMA_Q][NUM_MMA_KV][HALF_ELEMS_PER_THREAD];

    // cooperative fetch q fragment from gmem to reg
    const uint32_t qo_packed_idx_base =
        (bx * NUM_WARPS_Q + get_warp_idx_q<KTraits>(tid.y)) * NUM_MMA_Q * 16;
    smem_t<SWIZZLE_MODE_Q, typename KTraits::SmemBasePtrTy> qo_smem(
        smem_storage.q_smem);
    DTypeQ *q_ptr_base = q + (kv_head_idx * group_size) * q_stride_h;

    uint32_t q_smem_offset_r =
        qo_smem.template get_permuted_offset<UPCAST_STRIDE_Q>(
            get_warp_idx_q<KTraits>(tid.y) * NUM_MMA_Q * 16 + lane_idx % 16,
            lane_idx / 16);

    load_q_global_smem<KTraits>(qo_packed_idx_base, qo_len, q_ptr_base,
                                q_stride_n, q_stride_h, group_size, &qo_smem,
                                tid);

    memory::commit_group();
    smem_t<SWIZZLE_MODE_KV, typename KTraits::SmemBasePtrTy> k_smem(
        smem_storage.k_smem);
    DTypeKV *k_ptr = k +
                     (chunk_start + warp_idx * KV_THR_LAYOUT_ROW +
                      lane_idx / KV_THR_LAYOUT_COL) *
                         k_stride_n +
                     kv_head_idx * k_stride_h +
                     (lane_idx % KV_THR_LAYOUT_COL) *
                         upcast_size<DTypeKV, VECTOR_BIT_WIDTH>();

    uint32_t k_smem_offset_r =
                 k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
                     get_warp_idx_kv<KTraits>(tid.z) * NUM_MMA_KV * 16 +
                         HALF_ELEMS_PER_THREAD * (lane_idx / 16) +
                         lane_idx % HALF_ELEMS_PER_THREAD,
                     (lane_idx % 16) / HALF_ELEMS_PER_THREAD),
             k_smem_offset_w =
                 k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
                     warp_idx * KV_THR_LAYOUT_ROW +
                         lane_idx / KV_THR_LAYOUT_COL,
                     lane_idx % KV_THR_LAYOUT_COL);
    produce_kv<false, SharedMemFillMode::kNoFill, KTraits>(
        k_smem, &k_smem_offset_w, &k_ptr, k_stride_n, 0, chunk_size, tid);
    memory::commit_group();

    memory::wait_group<1>();
    block.sync();
    // compute attention score
    compute_qk<KTraits>(&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r,
                        s_frag);
    memory::wait_group<0>();
    block.sync();

    // Extract QK scores from s_frag to global memory
    if (get_warp_idx_q<KTraits>(tid.y) == 0 &&
        get_warp_idx_kv<KTraits>(tid.z) == 0)
    {
        for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
            for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
                for (uint32_t reg_id = 0; reg_id < HALF_ELEMS_PER_THREAD;
                     ++reg_id)
                {
                    // Map from MMA fragment layout to sequence indices

                    // CDNA3 mapping
                    uint32_t q_idx =
                        mma_q * 16 +
                        reg_id % KTraits::NUM_ACCUM_ROWS_PER_THREAD;
                    uint32_t kv_idx =
                        mma_kv * 16 +
                        2 * (lane_idx % KTraits::THREADS_PER_MATRIX_ROW_SET) +
                        8 * (reg_id / 2) + reg_id % 2;

                    if (q_idx < qo_len && kv_idx < kv_len) {
                        // Match CPU layout: [qo_head_idx][q_idx][kv_idx]
                        uint32_t qo_head_idx =
                            kv_head_idx *
                            group_size; // Simple for single head case
                        uint32_t output_idx = qo_head_idx * qo_len * kv_len +
                                              q_idx * kv_len + kv_idx;
                        qk_scores_output[output_idx] =
                            float(s_frag[mma_q][mma_kv][reg_id]);
                    }
                }
            }
        }
    }
}

template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void ComputeQKStubKernel(
    const __grid_constant__ Params params,
    float *qk_scores_output)
{
    extern __shared__ uint8_t smem[];
    auto &smem_storage =
        reinterpret_cast<typename KTraits::SharedStorage &>(smem);
    ComputeQKStubKernelDevice<KTraits>(params, smem_storage, qk_scores_output);
}

template <uint32_t HEAD_DIM_QK,
          uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE,
          bool USE_FP16_QK_REDUCTION,
          MaskMode MASK_MODE,
          typename AttentionVariant,
          typename Params>
gpuError_t ComputeQKStubDispatched(Params params,
                                   typename Params::DTypeO *tmp,
                                   float *qk_scores_output,
                                   gpuStream_t stream)
{
    using DTypeQ = typename Params::DTypeQ;
    using DTypeKV = typename Params::DTypeKV;
    using DTypeO = typename Params::DTypeO;
    const uint32_t num_qo_heads = params.num_qo_heads;
    const uint32_t num_kv_heads = params.num_kv_heads;
    const uint32_t qo_len = params.qo_len;
    const uint32_t kv_len = params.kv_len;
    if (kv_len < qo_len && MASK_MODE == MaskMode::kCausal) {
        std::ostringstream err_msg;
        err_msg << "When mask_mode is set to MaskMode::kCausal, kv_len must be "
                   "greater than or equal to qo_len, got kv_len"
                << kv_len << " and qo_len " << qo_len;
        FLASHINFER_ERROR(err_msg.str());
    }

    const uint32_t group_size = num_qo_heads / num_kv_heads;
    constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
    constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;
    int64_t packed_qo_len = qo_len * group_size;
    uint32_t cta_tile_q = FA2DetermineCtaTileQ(packed_qo_len, HEAD_DIM_VO);

    DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, {
        constexpr uint32_t NUM_WARPS_Q = get_num_warps_q(CTA_TILE_Q);
        constexpr uint32_t NUM_WARPS_KV = get_num_warps_kv(CTA_TILE_Q);
        constexpr uint32_t NUM_MMA_Q = get_num_mma_q(CTA_TILE_Q);

        using DTypeQKAccum =
            typename std::conditional<USE_FP16_QK_REDUCTION &&
                                          std::is_same_v<DTypeQ, half>,
                                      half, float>::type;

        int dev_id = 0;
        FI_GPU_CALL(gpuGetDevice(&dev_id));
        int max_smem_per_sm = getMaxSharedMemPerMultiprocessor(dev_id);
        // we expect each sm execute two threadblocks
        const int num_ctas_per_sm =
            max_smem_per_sm >= 2 * (CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ) +
                                    (HEAD_DIM_QK + HEAD_DIM_VO) * 16 *
                                        NUM_WARPS_KV * sizeof(DTypeKV))
                ? 2
                : 1;
        const int max_smem_per_threadblock = max_smem_per_sm / num_ctas_per_sm;

        const uint32_t max_num_mma_kv_reg =
            (HEAD_DIM_VO >= 128 && NUM_MMA_Q == 2 &&
             POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
             !USE_FP16_QK_REDUCTION)
                ? 2
                : (8 / NUM_MMA_Q);
        const uint32_t max_num_mma_kv_smem =
            (max_smem_per_threadblock -
             CTA_TILE_Q * HEAD_DIM_QK * sizeof(DTypeQ)) /
            ((HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV * sizeof(DTypeKV));

        // control NUM_MMA_KV for maximum warp occupancy
        DISPATCH_NUM_MMA_KV(
            min(max_num_mma_kv_smem, max_num_mma_kv_reg), NUM_MMA_KV, {
                using KTraits =
                    KernelTraits<MASK_MODE, CTA_TILE_Q, NUM_MMA_Q, NUM_MMA_KV,
                                 NUM_MMA_D_QK, NUM_MMA_D_VO, NUM_WARPS_Q,
                                 NUM_WARPS_KV, POS_ENCODING_MODE, DTypeQ,
                                 DTypeKV, DTypeO, DTypeQKAccum,
                                 typename Params::IdType, AttentionVariant>;
                if constexpr (KTraits::IsInvalid()) {
                    // Invalid configuration, skip
                    std::ostringstream err_msg;
                    err_msg << "FlashInfer Internal Error: Invalid "
                               "configuration : NUM_MMA_Q="
                            << NUM_MMA_Q << " NUM_MMA_D_QK=" << NUM_MMA_D_QK
                            << " NUM_MMA_D_VO=" << NUM_MMA_D_VO
                            << " NUM_MMA_KV=" << NUM_MMA_KV
                            << " NUM_WARPS_Q=" << NUM_WARPS_Q
                            << " NUM_WARPS_KV=" << NUM_WARPS_KV
                            << " please create an issue "
                               "(https://github.com/flashinfer-ai/flashinfer/"
                               "issues)"
                               " and report the issue to the developers.";
                    FLASHINFER_ERROR(err_msg.str());
                }
                else {
                    constexpr uint32_t num_threads =
                        (NUM_WARPS_Q * NUM_WARPS_KV) * WARP_SIZE;
                    auto kernel = ComputeQKStubKernel<KTraits, Params>;
                    size_t smem_size = sizeof(typename KTraits::SharedStorage);
                    FI_GPU_CALL(gpuFuncSetAttribute(
                        kernel, gpuFuncAttributeMaxDynamicSharedMemorySize,
                        smem_size));
                    int num_blocks_per_sm = 0;
                    int num_sm = 0;
                    FI_GPU_CALL(gpuDeviceGetAttribute(
                        &num_sm, gpuDevAttrMultiProcessorCount, dev_id));
                    FI_GPU_CALL(gpuOccupancyMaxActiveBlocksPerMultiprocessor(
                        &num_blocks_per_sm, kernel, num_threads, smem_size));
                    uint32_t max_num_kv_chunks =
                        (num_blocks_per_sm * num_sm) /
                        (num_kv_heads *
                         ceil_div(qo_len * group_size, CTA_TILE_Q));
                    uint32_t num_chunks;
                    if (max_num_kv_chunks > 0) {
                        uint32_t chunk_size =
                            max(ceil_div(kv_len, max_num_kv_chunks), 256);
                        num_chunks = ceil_div(kv_len, chunk_size);
                    }
                    else {
                        num_chunks = 0;
                    }

                    if (num_chunks <= 1 || tmp == nullptr) {
                        // Enough parallelism, do not split-kv
                        params.partition_kv = false;
                        void *args[] = {(void *)&params,
                                        (void *)&qk_scores_output};
                        dim3 nblks(ceil_div(qo_len * group_size, CTA_TILE_Q), 1,
                                   num_kv_heads);
                        dim3 nthrs(WARP_SIZE, NUM_WARPS_Q, NUM_WARPS_KV);
                        FI_GPU_CALL(gpuLaunchKernel((void *)kernel, nblks,
                                                    nthrs, args, smem_size,
                                                    stream));
                    }
                    else {
                        // Use cooperative groups to increase occupancy
                        params.partition_kv = true;
                        float *tmp_lse =
                            (float *)(tmp + num_chunks * qo_len * num_qo_heads *
                                                HEAD_DIM_VO);
                        auto o = params.o;
                        auto lse = params.lse;
                        params.o = tmp;
                        params.lse = tmp_lse;
                        void *args[] = {(void *)&params};
                        dim3 nblks(ceil_div(qo_len * group_size, CTA_TILE_Q),
                                   num_chunks, num_kv_heads);
                        dim3 nthrs(WARP_SIZE, NUM_WARPS_Q, NUM_WARPS_KV);
                        FI_GPU_CALL(gpuLaunchKernel((void *)kernel, nblks,
                                                    nthrs, args, smem_size,
                                                    stream));
                        if constexpr (AttentionVariant::use_softmax) {
                            FI_GPU_CALL(MergeStates(
                                tmp, tmp_lse, o, lse, num_chunks, qo_len,
                                num_qo_heads, HEAD_DIM_VO, stream));
                        }
                        else {
                            FI_GPU_CALL(AttentionSum(tmp, o, num_chunks, qo_len,
                                                     num_qo_heads, HEAD_DIM_VO,
                                                     stream));
                        }
                    }
                }
            })
    });
    return gpuSuccess;
}
