# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# MLA decode benchmark for HIP/ROCm (DeepSeek shapes).
# Mirrors benchmarks/bench_deepseek_mla.py with backend="hip".
#
# Run:
#   python benchmarks/rocm_benchmarks/bench_mla_hip.py
#   python benchmarks/rocm_benchmarks/bench_mla_hip.py --batch 64 --seq 8192 --heads 128

import argparse
import math

import numpy as np
import torch

import flashinfer
import flashinfer.mla
from flashinfer.jit import build_jit_specs, gen_batch_mla_module
from flashinfer.testing.utils import bench_gpu_time

HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64


def _warmup_jit(dtype: torch.dtype) -> None:
    build_jit_specs(
        [
            gen_batch_mla_module(
                "hip",
                dtype,
                dtype,
                dtype,
                torch.int32,
                HEAD_DIM_CKV,
                HEAD_DIM_KPE,
                False,
            )
        ],
        verbose=False,
    )


@torch.inference_mode()
def bench(batch_size: int, seq_len: int, num_heads: int, dtype: torch.dtype) -> None:
    page_size = 1
    sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV + HEAD_DIM_KPE)

    q_nope = torch.randn(
        batch_size, num_heads, HEAD_DIM_CKV, dtype=dtype, device="cuda"
    )
    q_pe = torch.randn(batch_size, num_heads, HEAD_DIM_KPE, dtype=dtype, device="cuda")
    ckv = torch.randn(
        batch_size * seq_len, page_size, HEAD_DIM_CKV, dtype=dtype, device="cuda"
    )
    kpe = torch.randn(
        batch_size * seq_len, page_size, HEAD_DIM_KPE, dtype=dtype, device="cuda"
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="auto")

    q_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
    kv_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * seq_len
    )
    kv_indices = torch.arange(0, batch_size * seq_len, dtype=torch.int32, device="cuda")
    kv_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")

    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        HEAD_DIM_CKV,
        HEAD_DIM_KPE,
        page_size,
        False,
        sm_scale,
        dtype,
        dtype,
    )

    # Correctness smoke-check before timing.
    o = wrapper.run(q_nope, q_pe, ckv, kpe)
    assert o.shape == (batch_size, num_heads, HEAD_DIM_CKV)

    measurements = bench_gpu_time(
        lambda: wrapper.run(q_nope, q_pe, ckv, kpe),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
    ms = np.median(measurements)

    io_bytes = sum(t.numel() * t.element_size() for t in [q_nope, q_pe, ckv, kpe, o])
    # Two GEMMs: Q·KVᵀ (ckv + kpe dim) and P·V (ckv dim).
    flops = 2 * batch_size * num_heads * (2 * HEAD_DIM_CKV + HEAD_DIM_KPE) * seq_len

    dtype_str = "fp16" if dtype == torch.float16 else "bf16"
    print(
        f"[{dtype_str}] batch={batch_size:>4d} seq={seq_len:>6d} heads={num_heads:>4d} | "
        f"lat={ms * 1e3:.1f} µs | "
        f"BW={io_bytes * 1e-6 / ms:.1f} GB/s | "
        f"TFLOPS={flops * 1e-9 / ms:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MLA HIP decode benchmark")
    parser.add_argument(
        "--batch", type=int, default=0, help="single batch size (0 = sweep)"
    )
    parser.add_argument("--seq", type=int, default=0, help="single seq len (0 = sweep)")
    parser.add_argument(
        "--heads", type=int, default=16, help="number of attention heads"
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "both"],
        default="fp16",
        help="data type (default: fp16)",
    )
    args = parser.parse_args()

    dtypes = []
    if args.dtype in ("fp16", "both"):
        dtypes.append(torch.float16)
    if args.dtype in ("bf16", "both"):
        dtypes.append(torch.bfloat16)

    batch_sizes = [args.batch] if args.batch > 0 else [64, 128, 768]
    seq_lens = [args.seq] if args.seq > 0 else [1024, 2048, 8192]

    for dtype in dtypes:
        print(
            f"\n=== Warming up JIT ({('fp16' if dtype == torch.float16 else 'bf16')}) ==="
        )
        _warmup_jit(dtype)
        print("=== Benchmarking ===")
        for seq_len in seq_lens:
            for batch_size in batch_sizes:
                bench(batch_size, seq_len, args.heads, dtype)


if __name__ == "__main__":
    main()
