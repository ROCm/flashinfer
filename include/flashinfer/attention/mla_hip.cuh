// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// HIP MLA kernel for CDNA3 (MI300X).
//
// CTA config: dim3(64,4,1) — 4 wavefronts, 256 threads.
// Wave 0 computes QK (q_pe·kpe^T + q_nope·ckv^T) and softmax, then broadcasts
// the softmax weights (p_frag) and updated m/d scalars via LDS to waves 1..3.
// All 4 waves then compute their PV head_dim shard in parallel.
// Wave w owns head_dim shard [w*(HEAD_DIM_CKV/4) .. (w+1)*(HEAD_DIM_CKV/4)).
// Q is loaded once into registers per work item (loop-invariant across KV tiles).
//
// MFMA layout note: CDNA3 mfma_f32_16x16x16f16 produces D in column-major
// per-thread layout (each thread holds D[m=(t/16)*4+r][n=t%16] for r=0..3).
// To get a more natural row-major s_frag/o_frag (each thread holds 1 row, 4
// cols), we swap the A and B operands in both MFMAs:
//   QK: A=K, B=Q  → D[m=kv][n=q] = S[q][kv]; thread t's r-th reg = S[q=t%16][kv=col_group*4+r]
//   PV: A=V, B=P  → D[m=d][n=q] = O[q][d];   thread t's r-th reg =
//   O[q=t%16][d=mma_d_abs*16+col_group*4+r]
// This avoids per-iteration in-register transposes of the s_frag tile.
#ifndef FLASHINFER_MLA_HIP_CUH_
#define FLASHINFER_MLA_HIP_CUH_

#include <stdint.h>

#include "../fastdiv.cuh"
#include "gpu_iface/cooperative_groups.h"
#include "gpu_iface/enums.hpp"
#include "gpu_iface/math_ops.hpp"
#include "gpu_iface/mma_ops.hpp"
#include "gpu_iface/mma_types.hpp"
#include "gpu_iface/utils.cuh"
#include "mla_params.cuh"

namespace flashinfer {
namespace mla {

constexpr uint32_t MLA_HIP_WAVE_SIZE = 64;
constexpr uint32_t MLA_HIP_NUM_WAVES = 4;
constexpr uint32_t MLA_HIP_NUM_THREADS = MLA_HIP_WAVE_SIZE * MLA_HIP_NUM_WAVES;
constexpr uint32_t MLA_HIP_CTA_TILE_Q = 16;
constexpr uint32_t MLA_HIP_CTA_TILE_KV = 16;
// Sentinel for the running max in online softmax. Matches the CUDA path
// (state_t::init uses -math::inf, which has the same numeric value 5e4f).
constexpr float MLA_HIP_NEG_INF = -5e4f;

// Two KV tiles in LDS to support double-buffered load/compute pipelining.
// CKV rows are padded by 8 fp16 (one uint4) so that the stride-row reads in
// compute_pv_hip don't collide on the same LDS bank — without padding all 4
// strided reads from one thread land on bank 0 (HEAD_DIM_CKV*2/4 % 32 == 0),
// giving a 4-way bank conflict per thread. With +8 fp16 padding the per-thread
// reads spread across banks {b, b+8, b+16, b+24}.
constexpr uint32_t MLA_HIP_CKV_LDS_PAD = 8;
constexpr uint32_t MLA_HIP_NUM_STAGES = 2;

// Wave-specialization LDS stage: wave 0 writes softmax weights and online-softmax
// scalars here after QK; waves 1..3 read them to skip QK MFMAs entirely.
// p_frag row stride is 17 (= CTA_TILE_KV + 1) to reduce LDS bank conflicts:
// thread (q_row=a, col_group=b) maps to float offset a*17 + b*4, giving a more
// uniform bank distribution than stride=16 which produces 8-way conflicts.
constexpr uint32_t MLA_HIP_P_FRAG_STRIDE = MLA_HIP_CTA_TILE_KV + 1;
struct WaveSpecStageMLAHIP {
  float p_frag[MLA_HIP_CTA_TILE_Q][MLA_HIP_P_FRAG_STRIDE];
  float m_stage[MLA_HIP_CTA_TILE_Q];
  float o_scale_stage[MLA_HIP_CTA_TILE_Q];
  float d_stage[MLA_HIP_CTA_TILE_Q];
};

template <uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, typename DTypeKV>
struct SharedStorageMLAHIP {
  DTypeKV ckv_smem[MLA_HIP_NUM_STAGES][MLA_HIP_CTA_TILE_KV][HEAD_DIM_CKV + MLA_HIP_CKV_LDS_PAD];
  DTypeKV kpe_smem[MLA_HIP_NUM_STAGES][MLA_HIP_CTA_TILE_KV][HEAD_DIM_KPE];
  WaveSpecStageMLAHIP wave_spec;
};

// ---------------------------------------------------------------------------
// load_kv_hip: cooperative KV tile load, all 256 threads participate.
// packed_kv_tile_base = kv_indptr * block_size + kv_tile_abs_start
// packed_kv_bound     = kv_indptr * block_size + kv_len
// ---------------------------------------------------------------------------
template <uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, typename DTypeKV, typename IdType>
__device__ __forceinline__ void load_kv_hip(
    DTypeKV* __restrict__ ckv_smem, DTypeKV* __restrict__ kpe_smem,
    const DTypeKV* __restrict__ ckv_global, const DTypeKV* __restrict__ kpe_global,
    const IdType* __restrict__ kv_indices, uint32_t ckv_stride_page, uint32_t ckv_stride_n,
    uint32_t kpe_stride_page, uint32_t kpe_stride_n, uint32_t packed_kv_tile_base,
    uint32_t packed_kv_bound, const uint_fastdiv& block_size, uint32_t tid) {
  // 128-bit (uint4 = 8 fp16) loads to maximize per-instruction DRAM throughput.
  // ckv_smem rows are padded by MLA_HIP_CKV_LDS_PAD fp16 (= 1 uint4); load by
  // (kv_row, col_u4) to handle the LDS stride correctly.
  constexpr uint32_t CKV_U4_PER_ROW = HEAD_DIM_CKV / 8;
  constexpr uint32_t CKV_LDS_U4_STRIDE = (HEAD_DIM_CKV + MLA_HIP_CKV_LDS_PAD) / 8;
  constexpr uint32_t CKV_TOTAL_U4 = MLA_HIP_CTA_TILE_KV * CKV_U4_PER_ROW;
  constexpr uint32_t CKV_U4_PER_THREAD = CKV_TOTAL_U4 / MLA_HIP_NUM_THREADS;
  static_assert(CKV_TOTAL_U4 % MLA_HIP_NUM_THREADS == 0, "CKV load not evenly divisible");

  uint4* smem_ckv = reinterpret_cast<uint4*>(ckv_smem);
#pragma unroll
  for (uint32_t i = 0; i < CKV_U4_PER_THREAD; ++i) {
    uint32_t u4_idx = tid * CKV_U4_PER_THREAD + i;
    uint32_t kv_row = u4_idx / CKV_U4_PER_ROW;
    uint32_t col_u4 = u4_idx % CKV_U4_PER_ROW;
    uint32_t packed = packed_kv_tile_base + kv_row;
    uint32_t page_idx, row_in_page;
    block_size.divmod(packed, page_idx, row_in_page);
    bool valid = (packed < packed_kv_bound);
    const DTypeKV* gptr = ckv_global +
                          (valid ? kv_indices[page_idx] : IdType(0)) * ckv_stride_page +
                          row_in_page * ckv_stride_n + col_u4 * 8;
    smem_ckv[kv_row * CKV_LDS_U4_STRIDE + col_u4] =
        valid ? *reinterpret_cast<const uint4*>(gptr) : uint4{0u, 0u, 0u, 0u};
  }

  constexpr uint32_t KPE_U4_PER_ROW = HEAD_DIM_KPE / 8;
  constexpr uint32_t KPE_TOTAL_U4 = MLA_HIP_CTA_TILE_KV * KPE_U4_PER_ROW;
  constexpr uint32_t KPE_U4_PER_THREAD = KPE_TOTAL_U4 / MLA_HIP_NUM_THREADS;
  // KPE may be too small for one uint4 per thread; fall back to a per-thread loop only if work
  // exists. For HEAD_DIM_KPE=64, CTA_TILE_KV=16: total=128 uint4, per-thread=0 (only 128 of 256
  // threads load). Handle as "first 128 threads do 1 load each".
  uint4* smem_kpe = reinterpret_cast<uint4*>(kpe_smem);
  if constexpr (KPE_U4_PER_THREAD >= 1) {
#pragma unroll
    for (uint32_t i = 0; i < KPE_U4_PER_THREAD; ++i) {
      uint32_t u4_idx = tid * KPE_U4_PER_THREAD + i;
      uint32_t kv_row = u4_idx / KPE_U4_PER_ROW;
      uint32_t col_u4 = u4_idx % KPE_U4_PER_ROW;
      uint32_t packed = packed_kv_tile_base + kv_row;
      uint32_t page_idx, row_in_page;
      block_size.divmod(packed, page_idx, row_in_page);
      bool valid = (packed < packed_kv_bound);
      const DTypeKV* gptr = kpe_global +
                            (valid ? kv_indices[page_idx] : IdType(0)) * kpe_stride_page +
                            row_in_page * kpe_stride_n + col_u4 * 8;
      smem_kpe[u4_idx] = valid ? *reinterpret_cast<const uint4*>(gptr) : uint4{0u, 0u, 0u, 0u};
    }
  } else {
    // Less than 1 uint4 per thread: only the first KPE_TOTAL_U4 threads do a load.
    if (tid < KPE_TOTAL_U4) {
      uint32_t kv_row = tid / KPE_U4_PER_ROW;
      uint32_t col_u4 = tid % KPE_U4_PER_ROW;
      uint32_t packed = packed_kv_tile_base + kv_row;
      uint32_t page_idx, row_in_page;
      block_size.divmod(packed, page_idx, row_in_page);
      bool valid = (packed < packed_kv_bound);
      const DTypeKV* gptr = kpe_global +
                            (valid ? kv_indices[page_idx] : IdType(0)) * kpe_stride_page +
                            row_in_page * kpe_stride_n + col_u4 * 8;
      smem_kpe[tid] = valid ? *reinterpret_cast<const uint4*>(gptr) : uint4{0u, 0u, 0u, 0u};
    }
  }
}

// ---------------------------------------------------------------------------
// load_q_frags_hip: read this thread's Q fragments once from global memory.
// Q is loop-invariant across the kv_tile loop, so caching saves repeated
// global loads (per work_idx, save (num_kv_tiles - 1) reloads of all Q).
//
// q_pe_frag[mma_d][2] holds Q_pe[q=t%16][d=mma_d*16 + col_group*4 + 0..3] as 4 fp16.
// q_nope_frag is the same for Q_nope across NUM_MMA_D_CKV head_dim tiles.
// Out-of-range threads (batch_idx >= q_len) get zero-filled fragments.
// ---------------------------------------------------------------------------
template <uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, typename DTypeQ>
__device__ __forceinline__ void load_q_frags_hip(uint32_t (*q_pe_frag)[2],
                                                 uint32_t (*q_nope_frag)[2], const DTypeQ* q_nope,
                                                 const DTypeQ* q_pe, uint32_t q_nope_stride_n,
                                                 uint32_t q_nope_stride_h, uint32_t q_pe_stride_n,
                                                 uint32_t q_pe_stride_h, uint32_t batch_idx,
                                                 uint32_t head_idx, bool q_valid,
                                                 uint32_t col_group) {
  constexpr uint32_t NUM_MMA_D_CKV = HEAD_DIM_CKV / 16;
  constexpr uint32_t NUM_MMA_D_KPE = HEAD_DIM_KPE / 16;

#pragma unroll
  for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_KPE; ++mma_d) {
    q_pe_frag[mma_d][0] = 0u;
    q_pe_frag[mma_d][1] = 0u;
    if (q_valid) {
      const DTypeQ* ptr =
          q_pe + batch_idx * q_pe_stride_n + head_idx * q_pe_stride_h + mma_d * 16 + col_group * 4;
      *reinterpret_cast<uint2*>(q_pe_frag[mma_d]) = *reinterpret_cast<const uint2*>(ptr);
    }
  }
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV; ++mma_d) {
    q_nope_frag[mma_d][0] = 0u;
    q_nope_frag[mma_d][1] = 0u;
    if (q_valid) {
      const DTypeQ* ptr = q_nope + batch_idx * q_nope_stride_n + head_idx * q_nope_stride_h +
                          mma_d * 16 + col_group * 4;
      *reinterpret_cast<uint2*>(q_nope_frag[mma_d]) = *reinterpret_cast<const uint2*>(ptr);
    }
  }
}

// ---------------------------------------------------------------------------
// compute_qk_hip: accumulate s_frag = Q_pe*KPE^T + Q_nope*CKV^T
//
// CDNA3 MFMA computes D = A*B where the D output is column-major per-thread:
// thread t holds D[m=(t/16)*4+r][n=t%16]. To get the natural row-major s_frag
// layout (thread t holds s_frag[r] = S[q=t%16][kv=col_group*4+r]) we swap the
// A and B operands: A=K, B=Q. The math:
//   D[m=kv][n=q] = sum_d K[kv=m][d=k] * Q[q=n][d=k] = (K@Q^T)[kv][q] = S[q][kv].
// Combined with the column-major D layout, thread t's r-th register =
// S[q=t%16][kv=(t/16)*4+r].
// ---------------------------------------------------------------------------
template <uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, typename DTypeKV>
__device__ __forceinline__ void compute_qk_hip(float* s_frag, const uint32_t (*q_pe_frag)[2],
                                               const uint32_t (*q_nope_frag)[2],
                                               const DTypeKV* ckv_smem, const DTypeKV* kpe_smem,
                                               uint32_t q_row, uint32_t col_group) {
  constexpr uint32_t NUM_MMA_D_CKV = HEAD_DIM_CKV / 16;
  constexpr uint32_t NUM_MMA_D_KPE = HEAD_DIM_KPE / 16;
  constexpr uint32_t CKV_LDS_U2_STRIDE = (HEAD_DIM_CKV + MLA_HIP_CKV_LDS_PAD) / 4;
  constexpr uint32_t KPE_U2_PER_ROW = HEAD_DIM_KPE / 4;

  const uint2* smem_ckv = reinterpret_cast<const uint2*>(ckv_smem);
  const uint2* smem_kpe = reinterpret_cast<const uint2*>(kpe_smem);

  // KPE tiles run first — kInit on tile 0 zeros the accumulator before CKV accumulates.
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_KPE; ++mma_d) {
    uint32_t k_frag[2];
    *reinterpret_cast<uint2*>(k_frag) = smem_kpe[q_row * KPE_U2_PER_ROW + mma_d * 4 + col_group];
    if (mma_d == 0) {
      gpu_iface::mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeKV, gpu_iface::mma::MMAMode::kInit>(
          s_frag, k_frag, const_cast<uint32_t*>(q_pe_frag[mma_d]));
    } else {
      gpu_iface::mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeKV>(
          s_frag, k_frag, const_cast<uint32_t*>(q_pe_frag[mma_d]));
    }
  }

#pragma unroll
  for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV; ++mma_d) {
    uint32_t k_frag[2];
    *reinterpret_cast<uint2*>(k_frag) = smem_ckv[q_row * CKV_LDS_U2_STRIDE + mma_d * 4 + col_group];
    gpu_iface::mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeKV>(
        s_frag, k_frag, const_cast<uint32_t*>(q_nope_frag[mma_d]));
  }
}

// ---------------------------------------------------------------------------
// logits_mask_hip: zero out scores for out-of-bounds or causally-masked positions.
// s_frag[r] = S[q_row=lane%16][kv_col = kv_idx_base + (lane/16)*4 + r]
// ---------------------------------------------------------------------------
template <bool CAUSAL>
__device__ __forceinline__ void logits_mask_hip(float* s_frag, uint32_t qo_packed_idx_base,
                                                uint32_t kv_idx_base, uint32_t q_len,
                                                uint32_t kv_len, uint32_t kv_end,
                                                const uint_fastdiv& num_heads, uint32_t lane_idx) {
  uint32_t q_idx = (qo_packed_idx_base + lane_idx % 16) / num_heads;
  uint32_t col_group = lane_idx / 16;

#pragma unroll
  for (uint32_t r = 0; r < 4; ++r) {
    uint32_t kv_idx = kv_idx_base + col_group * 4 + r;
    bool out_of_bound = (kv_idx >= kv_end);
    bool causal_masked = CAUSAL && (kv_idx + q_len > kv_len + q_idx);
    if (out_of_bound || causal_masked) s_frag[r] = MLA_HIP_NEG_INF;
  }
}

// ---------------------------------------------------------------------------
// broadcast_softmax_hip: called by wave 0 after compute_qk + logits_mask.
// Inlines update_mdo_states, captures o_scale, and writes p_frag + scalars to
// the wave_spec LDS stage for waves 1..3 to consume via read_softmax_hip.
// ---------------------------------------------------------------------------
template <uint32_t NUM_MMA_D_PER_WAVE>
__device__ __forceinline__ void broadcast_softmax_hip(float* s_frag, float (*o_frag)[4],
                                                      float& m_val, float& d_scalar,
                                                      WaveSpecStageMLAHIP& stage, uint32_t q_row,
                                                      uint32_t col_group, float sm_scale_log2) {
  float m_local = fmaxf(fmaxf(s_frag[0], s_frag[1]), fmaxf(s_frag[2], s_frag[3]));
  m_local = fmaxf(m_local, math::shfl_xor_sync(m_local, 0x10));
  m_local = fmaxf(m_local, math::shfl_xor_sync(m_local, 0x20));

  float m_prev = m_val;
  m_val = fmaxf(m_prev, m_local);
  float o_scale = math::ptx_exp2((m_prev - m_val) * sm_scale_log2);
  d_scalar *= o_scale;

#pragma unroll
  for (uint32_t d = 0; d < NUM_MMA_D_PER_WAVE; ++d)
#pragma unroll
    for (uint32_t r = 0; r < 4; ++r) o_frag[d][r] *= o_scale;

  const float m_scaled = m_val * sm_scale_log2;
  float partial_d = 0.f;
#pragma unroll
  for (uint32_t r = 0; r < 4; ++r) {
    float p = math::ptx_exp2(s_frag[r] * sm_scale_log2 - m_scaled);
    s_frag[r] = p;
    partial_d += p;
  }
  partial_d += math::shfl_xor_sync(partial_d, 0x10);
  partial_d += math::shfl_xor_sync(partial_d, 0x20);
  d_scalar += partial_d;

  // Broadcast p_frag and per-q_row scalars to LDS for waves 1..3.
#pragma unroll
  for (uint32_t r = 0; r < 4; ++r) stage.p_frag[q_row][col_group * 4 + r] = s_frag[r];
  if (col_group == 0) {
    stage.m_stage[q_row] = m_val;
    stage.o_scale_stage[q_row] = o_scale;
    stage.d_stage[q_row] = d_scalar;
  }
}

// ---------------------------------------------------------------------------
// read_softmax_hip: called by waves 1..3 after the __syncthreads() following
// broadcast_softmax_hip. Reads p_frag and applies the o_scale / state update
// that wave 0 already applied to its own o_frag.
// ---------------------------------------------------------------------------
template <uint32_t NUM_MMA_D_PER_WAVE>
__device__ __forceinline__ void read_softmax_hip(float* s_frag, float (*o_frag)[4], float& m_val,
                                                 float& d_scalar, const WaveSpecStageMLAHIP& stage,
                                                 uint32_t q_row, uint32_t col_group) {
#pragma unroll
  for (uint32_t r = 0; r < 4; ++r) s_frag[r] = stage.p_frag[q_row][col_group * 4 + r];
  float o_scale = stage.o_scale_stage[q_row];
#pragma unroll
  for (uint32_t d = 0; d < NUM_MMA_D_PER_WAVE; ++d)
#pragma unroll
    for (uint32_t r = 0; r < 4; ++r) o_frag[d][r] *= o_scale;
  m_val = stage.m_stage[q_row];
  d_scalar = stage.d_stage[q_row];
}

// ---------------------------------------------------------------------------
// compute_pv_hip: accumulate P*V into o_frag for this wave's head_dim shard.
// s_frag must already hold exp-scaled softmax weights (output of broadcast/read_softmax_hip).
//
// As with compute_qk_hip, we swap A/B so the column-major D output gives a
// row-major o_frag: thread t's r-th register = O[q=t%16][d=mma_d_abs*16+col_group*4+r].
// We pass A=V (loaded strided so f16x4[j]=V[kv=col_group*4+j][d=mma_d_abs*16+t%16])
// and B=P (s_frag, with f16x4[r]=P[q=t%16][kv=col_group*4+r]).
// The math: D[m=d_local][n=q] = sum_kv V[kv][d] * P[q][kv] = O[q][d_global].
// ---------------------------------------------------------------------------
template <uint32_t HEAD_DIM_CKV, uint32_t NUM_MMA_D_PER_WAVE, typename DTypeKV>
__device__ __forceinline__ void compute_pv_hip(float (*o_frag)[4], const float* s_frag,
                                               const DTypeKV* ckv_smem, uint32_t wave_idx,
                                               uint32_t lane_idx) {
  uint32_t q_row = lane_idx % 16;
  uint32_t col_group = lane_idx / 16;
  uint32_t d_start = wave_idx * NUM_MMA_D_PER_WAVE;

  DTypeKV p_f16[4];
#pragma unroll
  for (uint32_t r = 0; r < 4; ++r) p_f16[r] = static_cast<DTypeKV>(s_frag[r]);
  uint32_t p_frag[2];
  *reinterpret_cast<uint2*>(p_frag) = *reinterpret_cast<const uint2*>(p_f16);

#pragma unroll
  for (uint32_t wave_mma_d = 0; wave_mma_d < NUM_MMA_D_PER_WAVE; ++wave_mma_d) {
    uint32_t mma_d_abs = d_start + wave_mma_d;
    // Load V via 4 strided scalar reads: V[kv=col_group*4+j][d=mma_d_abs*16+q_row].
    // Used as A operand: A[m=t%16][k=col_group*4+j] = V[kv=k][d=mma_d_abs*16+m].
    uint32_t d_col = mma_d_abs * 16 + q_row;
    DTypeKV v_vals[4];
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      v_vals[j] = ckv_smem[(col_group * 4 + j) * (HEAD_DIM_CKV + MLA_HIP_CKV_LDS_PAD) + d_col];
    }
    uint32_t v_frag[2];
    *reinterpret_cast<uint2*>(v_frag) = *reinterpret_cast<const uint2*>(v_vals);
    gpu_iface::mma::mma_sync_m16n16k16_row_col_f16f16f32<DTypeKV>(o_frag[wave_mma_d], v_frag,
                                                                  p_frag);
  }
}

template <uint32_t NUM_MMA_D_PER_WAVE>
__device__ __forceinline__ void normalize_d_hip(float (*o_frag)[4], float m_val, float d_scalar) {
  float d_rcp = (m_val != MLA_HIP_NEG_INF) ? math::ptx_rcp(d_scalar) : 0.f;
#pragma unroll
  for (uint32_t d = 0; d < NUM_MMA_D_PER_WAVE; ++d) {
#pragma unroll
    for (uint32_t r = 0; r < 4; ++r) o_frag[d][r] *= d_rcp;
  }
}

// ---------------------------------------------------------------------------
// write_o_hip: store results to final_o or partial_o.
//
// Each thread writes for q_row = lane_idx%16 and its wave's head_dim shard.
// Only wave_idx==0 && col_group==0 (lane_idx<16) writes the LSE scalar.
//
// partial_o / partial_lse are the GLOBAL arrays; partial_indptr is the start
// row index into these arrays for this CTA's work item. Pass -1 if no partial.
// ---------------------------------------------------------------------------
template <uint32_t HEAD_DIM_CKV, uint32_t NUM_MMA_D_PER_WAVE, typename DTypeO, typename IdType>
__device__ __forceinline__ void write_o_hip(
    float (*o_frag)[4], float m_val, float d_scalar, DTypeO* final_o, float* final_lse,
    DTypeO* partial_o, float* partial_lse, uint32_t o_stride_n, uint32_t o_stride_h, uint32_t q_len,
    uint32_t qo_packed_idx_base, int32_t partial_indptr, const uint_fastdiv& num_heads,
    bool return_lse_base_on_e, uint32_t lane_idx, uint32_t wave_idx) {
  uint32_t q_row = lane_idx % 16;
  uint32_t col_group = lane_idx / 16;
  uint32_t q_packed = qo_packed_idx_base + q_row;
  uint32_t batch_idx, head_idx;
  num_heads.divmod(q_packed, batch_idx, head_idx);
  bool q_valid = (batch_idx < q_len);
  uint32_t d_start = wave_idx * NUM_MMA_D_PER_WAVE;

  if (partial_indptr >= 0) {
    uint32_t partial_row = static_cast<uint32_t>(partial_indptr) + q_row;
    if (q_valid) {
      // One thread per q_row writes the LSE (lane_idx < 16 means col_group == 0)
      if (wave_idx == 0 && col_group == 0) {
        partial_lse[partial_row] = math::ptx_log2(d_scalar) + m_val;
      }
#pragma unroll
      for (uint32_t wave_mma_d = 0; wave_mma_d < NUM_MMA_D_PER_WAVE; ++wave_mma_d) {
        uint32_t head_dim_col = (d_start + wave_mma_d) * 16 + col_group * 4;
        DTypeO* ptr = partial_o + partial_row * HEAD_DIM_CKV + head_dim_col;
#pragma unroll
        for (uint32_t r = 0; r < 4; ++r) ptr[r] = static_cast<DTypeO>(o_frag[wave_mma_d][r]);
      }
    }
  } else {
    if (q_valid) {
      if (final_lse && wave_idx == 0 && col_group == 0) {
        float lse = math::ptx_log2(d_scalar) + m_val;
        if (return_lse_base_on_e) lse *= math::loge2;
        final_lse[batch_idx * (uint32_t)num_heads + head_idx] = lse;
      }
#pragma unroll
      for (uint32_t wave_mma_d = 0; wave_mma_d < NUM_MMA_D_PER_WAVE; ++wave_mma_d) {
        uint32_t head_dim_col = (d_start + wave_mma_d) * 16 + col_group * 4;
        DTypeO* ptr = final_o + batch_idx * o_stride_n + head_idx * o_stride_h + head_dim_col;
#pragma unroll
        for (uint32_t r = 0; r < 4; ++r) ptr[r] = static_cast<DTypeO>(o_frag[wave_mma_d][r]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// DevicePersistentMergeStatesHIP: reduce partial outputs → final output.
//
// partial_o[row * HEAD_DIM_CKV + col] stores DTypeO (normalized by each partial's d).
// partial_lse[row] = log2(d) + m for each partial block.
// Merge treats partial_lse as the combined log-sum-exp weight for each partial.
// ---------------------------------------------------------------------------
template <uint32_t HEAD_DIM_CKV, typename DTypeO, typename IdType>
__device__ void DevicePersistentMergeStatesHIP(
    const IdType* merge_packed_offset_start, const IdType* merge_packed_offset_end,
    const IdType* merge_partial_packed_offset_start, const IdType* merge_partial_packed_offset_end,
    const IdType* merge_partial_stride, const DTypeO* partial_o, const float* partial_lse,
    DTypeO* final_o, float* final_lse, uint32_t o_stride_n, uint32_t o_stride_h,
    const uint_fastdiv& num_heads, bool return_lse_base_on_e) {
  constexpr uint32_t VEC_SIZE = 8;
  constexpr uint32_t NUM_THRS_PER_ROW = HEAD_DIM_CKV / VEC_SIZE;
  constexpr uint32_t ROWS_PER_ITER = MLA_HIP_NUM_THREADS / NUM_THRS_PER_ROW;

  uint32_t cta_idx = gridDim.x * blockIdx.y + blockIdx.x;
  uint32_t thread_id = threadIdx.y * MLA_HIP_WAVE_SIZE + threadIdx.x;

  uint32_t offset_start = merge_packed_offset_start[cta_idx];
  uint32_t len = merge_packed_offset_end[cta_idx] - offset_start;
  uint32_t partial_start = merge_partial_packed_offset_start[cta_idx];
  uint32_t partial_end = merge_partial_packed_offset_end[cta_idx];
  uint32_t stride = merge_partial_stride[cta_idx];

  for (uint32_t local_row = thread_id / NUM_THRS_PER_ROW; local_row < len;
       local_row += ROWS_PER_ITER) {
    uint32_t global_packed = offset_start + local_row;
    uint32_t q, r;
    num_heads.divmod(global_packed, q, r);
    uint32_t thr_col = thread_id % NUM_THRS_PER_ROW;

    // Accumulate state over partial blocks
    // State representation: (st_m, st_d, st_o) where st_m tracks running max of partial_lse,
    // st_d accumulates exp-weighted partial sums, st_o is the weighted output sum.
    float st_o[VEC_SIZE];
#pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; ++i) st_o[i] = 0.f;
    float st_m = MLA_HIP_NEG_INF;
    float st_d = 1.f;

    for (uint32_t pp = partial_start + local_row; pp < partial_end; pp += stride) {
      float other_lse = partial_lse[pp];  // = log2(d_partial) + m_partial
      const DTypeO* src = partial_o + (uint64_t)pp * HEAD_DIM_CKV + thr_col * VEC_SIZE;

      float new_m = fmaxf(st_m, other_lse);
      float scale_st = math::ptx_exp2(st_m - new_m);
      float scale_other = math::ptx_exp2(other_lse - new_m);
      st_d = st_d * scale_st + scale_other;
#pragma unroll
      for (uint32_t i = 0; i < VEC_SIZE; ++i)
        st_o[i] = st_o[i] * scale_st + static_cast<float>(src[i]) * scale_other;
      st_m = new_m;
    }

    float d_rcp = (st_m != MLA_HIP_NEG_INF) ? math::ptx_rcp(st_d) : 0.f;
    DTypeO* dst = final_o + q * o_stride_n + r * o_stride_h + thr_col * VEC_SIZE;
#pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; ++i) dst[i] = static_cast<DTypeO>(st_o[i] * d_rcp);

    if (final_lse && thr_col == 0) {
      float lse = st_m + math::ptx_log2(st_d);
      if (return_lse_base_on_e) lse *= math::loge2;
      final_lse[q * (uint32_t)num_heads + r] = lse;
    }
  }
}

template <bool CAUSAL, uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, typename Params>
__global__ __launch_bounds__(MLA_HIP_NUM_THREADS) void BatchMLAPagedAttentionKernelHIP(
    Params params) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;

  constexpr uint32_t NUM_MMA_D_CKV = HEAD_DIM_CKV / 16;
  constexpr uint32_t NUM_MMA_D_KPE = HEAD_DIM_KPE / 16;
  constexpr uint32_t NUM_MMA_D_PER_WAVE = NUM_MMA_D_CKV / MLA_HIP_NUM_WAVES;
  static_assert(NUM_MMA_D_CKV % MLA_HIP_NUM_WAVES == 0,
                "HEAD_DIM_CKV must be divisible by 4 * 16 = 64");

  uint32_t lane_idx = threadIdx.x;                         // 0..63
  uint32_t wave_idx = threadIdx.y;                         // 0..3
  uint32_t tid = wave_idx * MLA_HIP_WAVE_SIZE + lane_idx;  // 0..255

  extern __shared__ uint8_t smem[];
  using SharedStorage = SharedStorageMLAHIP<HEAD_DIM_CKV, HEAD_DIM_KPE, DTypeKV>;
  SharedStorage& smem_storage = *reinterpret_cast<SharedStorage*>(smem);

  const float sm_scale_log2 = params.sm_scale * math::log2e;
  const uint_fastdiv& num_heads = params.num_heads;
  const uint_fastdiv& block_size = params.block_size;
  const uint32_t block_size_val = static_cast<uint32_t>(block_size);

  const uint32_t q_row = lane_idx % 16;
  const uint32_t col_group = lane_idx / 16;

  float s_frag[4];
  float o_frag[NUM_MMA_D_PER_WAVE][4];
  float m_val, d_scalar;
  uint32_t q_pe_frag[NUM_MMA_D_KPE][2];
  uint32_t q_nope_frag[NUM_MMA_D_CKV][2];

  for (IdType work_idx = params.work_indptr[blockIdx.y];
       work_idx < params.work_indptr[blockIdx.y + 1]; ++work_idx) {
    const uint32_t q_indptr = params.q_indptr[work_idx];
    const uint32_t kv_indptr = params.kv_indptr[work_idx];
    const int32_t partial_indptr = params.partial_indptr[work_idx];
    const uint32_t q_len = params.q_len[work_idx];
    const uint32_t kv_len = params.kv_len[work_idx];
    const uint32_t kv_start = params.kv_start[work_idx];
    const uint32_t kv_end = params.kv_end[work_idx];
    const uint32_t packed_qo_start = params.q_start[work_idx];

    const uint32_t qo_packed_idx_base = packed_qo_start + blockIdx.x * MLA_HIP_CTA_TILE_Q;

    const uint32_t q_packed = qo_packed_idx_base + q_row;
    uint32_t batch_idx, head_idx;
    num_heads.divmod(q_packed, batch_idx, head_idx);
    const bool q_valid = (batch_idx < q_len);

    load_q_frags_hip<HEAD_DIM_CKV, HEAD_DIM_KPE, DTypeQ>(
        q_pe_frag, q_nope_frag, params.q_nope + q_indptr * params.q_nope_stride_n,
        params.q_pe + q_indptr * params.q_pe_stride_n, params.q_nope_stride_n,
        params.q_nope_stride_h, params.q_pe_stride_n, params.q_pe_stride_h, batch_idx, head_idx,
        q_valid, col_group);

    m_val = MLA_HIP_NEG_INF;
    d_scalar = 1.f;
#pragma unroll
    for (uint32_t d = 0; d < NUM_MMA_D_PER_WAVE; ++d)
#pragma unroll
      for (uint32_t r = 0; r < 4; ++r) o_frag[d][r] = 0.f;

    const uint32_t packed_kv_bound = kv_indptr * block_size_val + kv_len;
    const int32_t num_kv_tiles =
        static_cast<int32_t>(ceil_div(kv_end - kv_start, MLA_HIP_CTA_TILE_KV));

    // Double-buffered pipeline: stage 0 loaded outside the loop, then each
    // iteration N loads stage[(N+1)%2] while computing on stage[N%2]. The hope
    // is the AMD compiler interleaves the global→VGPR→LDS load chain with the
    // MFMAs on the previous tile's data, hiding global-load latency.
    auto kv_tile_to_packed_base = [&](int32_t kv_tile_idx) {
      return kv_indptr * block_size_val + kv_start +
             static_cast<uint32_t>(kv_tile_idx) * MLA_HIP_CTA_TILE_KV;
    };
    auto kv_tile_to_abs_start = [&](int32_t kv_tile_idx) {
      return kv_start + static_cast<uint32_t>(kv_tile_idx) * MLA_HIP_CTA_TILE_KV;
    };

    // Prologue: load tile (num_kv_tiles - 1) into stage 0.
    {
      const int32_t first_tile = num_kv_tiles - 1;
      load_kv_hip<HEAD_DIM_CKV, HEAD_DIM_KPE, DTypeKV, IdType>(
          smem_storage.ckv_smem[0][0], smem_storage.kpe_smem[0][0], params.ckv, params.kpe,
          params.kv_indices, params.ckv_stride_page, params.ckv_stride_n, params.kpe_stride_page,
          params.kpe_stride_n, kv_tile_to_packed_base(first_tile), packed_kv_bound, block_size,
          tid);
    }

    uint32_t cur_stage = 0;
    for (int32_t kv_tile_idx = num_kv_tiles - 1; kv_tile_idx >= 0; --kv_tile_idx) {
      const uint32_t next_stage = 1 - cur_stage;
      const bool has_next = (kv_tile_idx > 0);

      // Issue load for tile (kv_tile_idx - 1) into next_stage. Without
      // cp.async, this still blocks the wave but lets the compiler schedule
      // the global VGPR loads ahead of compute.
      if (has_next) {
        load_kv_hip<HEAD_DIM_CKV, HEAD_DIM_KPE, DTypeKV, IdType>(
            smem_storage.ckv_smem[next_stage][0], smem_storage.kpe_smem[next_stage][0], params.ckv,
            params.kpe, params.kv_indices, params.ckv_stride_page, params.ckv_stride_n,
            params.kpe_stride_page, params.kpe_stride_n, kv_tile_to_packed_base(kv_tile_idx - 1),
            packed_kv_bound, block_size, tid);
      }

      __syncthreads();

      // Wave 0 computes QK and softmax, then broadcasts via LDS.
      // Waves 1..3 skip QK (saves 75% of MFMA work) and read p_frag from LDS.
      if (wave_idx == 0) {
        compute_qk_hip<HEAD_DIM_CKV, HEAD_DIM_KPE, DTypeKV>(
            s_frag, q_pe_frag, q_nope_frag, smem_storage.ckv_smem[cur_stage][0],
            smem_storage.kpe_smem[cur_stage][0], q_row, col_group);
        logits_mask_hip<CAUSAL>(s_frag, qo_packed_idx_base, kv_tile_to_abs_start(kv_tile_idx),
                                q_len, kv_len, kv_end, num_heads, lane_idx);
        broadcast_softmax_hip<NUM_MMA_D_PER_WAVE>(s_frag, o_frag, m_val, d_scalar,
                                                  smem_storage.wave_spec, q_row, col_group,
                                                  sm_scale_log2);
      }
      __syncthreads();  // wave_spec visible to all waves
      if (wave_idx > 0) {
        read_softmax_hip<NUM_MMA_D_PER_WAVE>(s_frag, o_frag, m_val, d_scalar,
                                             smem_storage.wave_spec, q_row, col_group);
      }

      compute_pv_hip<HEAD_DIM_CKV, NUM_MMA_D_PER_WAVE, DTypeKV>(
          o_frag, s_frag, smem_storage.ckv_smem[cur_stage][0], wave_idx, lane_idx);

      cur_stage = next_stage;
    }

    normalize_d_hip<NUM_MMA_D_PER_WAVE>(o_frag, m_val, d_scalar);

    // finalize_m: scale the running max into log2 space so that
    //   lse = log2(d_scaled) + m_scaled = log2(sum exp(s * sm_scale))
    // is a true log-sum-exp in log2 base. The downstream merge relies on this.
    float m_scaled = (m_val != MLA_HIP_NEG_INF) ? m_val * sm_scale_log2 : m_val;

    write_o_hip<HEAD_DIM_CKV, NUM_MMA_D_PER_WAVE, DTypeO, IdType>(
        o_frag, m_scaled, d_scalar, params.final_o + q_indptr * params.o_stride_n,
        params.final_lse ? params.final_lse + q_indptr * (uint32_t)num_heads : nullptr,
        params.partial_o, params.partial_lse, params.o_stride_n, params.o_stride_h, q_len,
        qo_packed_idx_base, partial_indptr, num_heads, params.return_lse_base_on_e, lane_idx,
        wave_idx);
  }

  gpu_iface::cg::this_grid().sync();

  DevicePersistentMergeStatesHIP<HEAD_DIM_CKV, DTypeO, IdType>(
      params.merge_packed_offset_start, params.merge_packed_offset_end,
      params.merge_partial_packed_offset_start, params.merge_partial_packed_offset_end,
      params.merge_partial_stride, params.partial_o, params.partial_lse, params.final_o,
      params.final_lse, params.o_stride_n, params.o_stride_h, num_heads,
      params.return_lse_base_on_e);
}

template <MaskMode MASK_MODE, uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, typename Params>
hipError_t BatchMLAPagedAttentionHIP(Params params, uint32_t num_blks_x, uint32_t num_blks_y,
                                     hipStream_t stream) {
  if (MASK_MODE == MaskMode::kCustom) return hipErrorNotSupported;
  constexpr bool CAUSAL = (MASK_MODE == MaskMode::kCausal);

  using SharedStorage = SharedStorageMLAHIP<HEAD_DIM_CKV, HEAD_DIM_KPE, typename Params::DTypeKV>;
  size_t smem_size = sizeof(SharedStorage);

  dim3 nblks(num_blks_x, num_blks_y);
  dim3 nthrs(MLA_HIP_WAVE_SIZE, MLA_HIP_NUM_WAVES, 1);

  auto kernel = BatchMLAPagedAttentionKernelHIP<CAUSAL, HEAD_DIM_CKV, HEAD_DIM_KPE, Params>;
  hipError_t err =
      hipFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                          hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_size));
  if (err != hipSuccess) return err;

  void* args[] = {reinterpret_cast<void*>(&params)};
  return hipLaunchCooperativeKernel(reinterpret_cast<const void*>(kernel), nblks, nthrs, args,
                                    smem_size, stream);
}

}  // namespace mla
}  // namespace flashinfer

#endif  // FLASHINFER_MLA_HIP_CUH_
