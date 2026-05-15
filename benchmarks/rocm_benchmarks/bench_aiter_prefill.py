"""
Copyright (c) 2026 Advanced Micro Devices, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

AITER prefill benchmark: single-prefill and batch-paged-prefill via backend="aiter".

Shapes: Sample production profile — q_len=256, GQA 16/4, HD=64, causal,
        kv ∈ {512, 1024, 2048, 3072, 4096, 8192}

Batch-paged tested over two page-size regimes:
  page_size=256  — native-paged path (mha_batch_prefill)
  page_size=16   — flat-gather path  (mha_fwd via index_select + varlen)

Run:

    # Full roofline pipeline (timing + counter collection + roofline PNG):
    python benchmarks/rocm_benchmarks/bench_aiter_prefill.py

    # Select a different counter preset:
    python benchmarks/rocm_benchmarks/bench_aiter_prefill.py --counters occupancy
    python benchmarks/rocm_benchmarks/bench_aiter_prefill.py --counters stall
    python benchmarks/rocm_benchmarks/bench_aiter_prefill.py --counters compute

    # Override the output file label prefix:
    python benchmarks/rocm_benchmarks/bench_aiter_prefill.py --label aiter_bench

    # Timing only (no rocprofv3):
    python benchmarks/rocm_benchmarks/bench_aiter_prefill.py --timing-only

    # Regenerate plot from existing CSVs (no GPU required):
    python benchmarks/rocm_benchmarks/bench_aiter_prefill.py --replot

    # List all available counter presets:
    python benchmarks/rocm_benchmarks/bench_aiter_prefill.py --list-presets

Output files (all gitignored):
    benchmarks/rocm_benchmarks/<label>_timing.csv
    benchmarks/rocm_benchmarks/<label>_counters.yml
    benchmarks/rocm_benchmarks/<label>_counter_collection.csv
    benchmarks/rocm_benchmarks/<label>_roofline.png   (roofline preset only)

Counter presets available out of the box:
    roofline  — FetchSize, WriteSize, MFMA ops, TCC DRAM requests (default)
    compute   — MFMA ops and cycle counters
    memory    — L2 and DRAM bandwidth breakdown
    basic     — minimal: FetchSize / WriteSize only
    occupancy — SQ_WAVES, SQ_BUSY_CYCLES, SQ_VALU_MFMA_BUSY_CYCLES,
                SQ_WAIT_INST_ANY, SQ_INSTS_LDS
    stall     — SQ_INSTS_MFMA, SQ_WAIT_INST_VMEM, SQ_VALU_MFMA_BUSY_CYCLES,
                SQ_WAIT_INST_LDS, SQ_BUSY_CYCLES

    Or pass a path to a YAML file in rocprofv3 native job format.

Design note — why --counters is parsed at module level
-------------------------------------------------------
rocprofv3 re-executes this script as a subprocess (passing the same sys.argv)
to collect hardware counters. The RocmProfiler object must therefore be
constructed at module import time with the correct `counters=` value, so that
both the outer driver and the inner rocprofv3 subprocess use the same preset.
We extract --counters / --label here (using parse_known_args so we don't
conflict with the profiler's own argparse), strip them from sys.argv, and then
pass the values to the RocmProfiler constructor.
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import torch

import flashinfer
from flashinfer.jit.core import logger

logger.setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "rocm_profiler"))
from rocm_profiler import KernelConfig, RocmProfiler

# ---------------------------------------------------------------------------
# Bench-script-level argument parsing
# ---------------------------------------------------------------------------
_bench_parser = argparse.ArgumentParser(add_help=False)
_bench_parser.add_argument(
    "--counters",
    default="roofline",
    metavar="PRESET_OR_FILE",
    help=(
        "Counter preset name ('roofline', 'occupancy', 'stall', 'compute', "
        "'memory', 'basic') or path to a rocprofv3 YAML file. "
        "Default: roofline."
    ),
)
_bench_parser.add_argument(
    "--label",
    default=None,
    metavar="PREFIX",
    help="Output-file label prefix (default: 'aiter' for roofline, 'aiter_<preset>' otherwise).",
)
_bench_args, _remaining = _bench_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

_counters = _bench_args.counters
_label = (
    _bench_args.label
    if _bench_args.label is not None
    else ("aiter" if _counters == "roofline" else f"aiter_{_counters}")
)

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
_Q_LEN = 256
_KV_LENS = [512, 1024, 2048, 3072, 4096, 8192]
_NUM_QO_HEADS = 16
_NUM_KV_HEADS = 4
_HEAD_DIM = 64
_DTYPE = torch.float16
_CAUSAL = True
_BATCH = 4

_OUTPUT_DIR = str(Path(__file__).parent)


def _flops(
    q_len: int, kv_len: int, num_qo_heads: int, head_dim: int, causal: bool
) -> int:
    return q_len * kv_len * num_qo_heads * head_dim * (2 if causal else 4)


def _bytes(
    q_len: int, kv_len: int, num_qo_heads: int, num_kv_heads: int, head_dim: int
) -> int:
    return (
        2
        * head_dim
        * (
            q_len * num_qo_heads
            + kv_len * num_kv_heads
            + kv_len * num_kv_heads
            + q_len * num_qo_heads
        )
    )


def _build_paged_kv(batch, kv_len, page_size, num_kv_heads, head_dim, dtype, device):
    """Build batch-paged KV tensors for one (batch, kv_len, page_size) combination."""
    num_full_pages, last_tokens = divmod(kv_len, page_size)
    if last_tokens == 0:
        last_tokens = page_size
    else:
        num_full_pages += 1
    total_pages = num_full_pages * batch

    # kv_data layout: [total_pages, 2, page_size, num_kv_heads, head_dim]  (NHD paged)
    kv_data = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device) * _Q_LEN
    kv_indptr = (
        torch.arange(batch + 1, dtype=torch.int32, device=device) * num_full_pages
    )
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full(
        (batch,), last_tokens, dtype=torch.int32, device=device
    )
    q = torch.randn(batch * _Q_LEN, _NUM_QO_HEADS, head_dim, dtype=dtype, device=device)

    return q, kv_data, qo_indptr, kv_indptr, kv_indices, kv_last_page_len


@torch.inference_mode()
def _make_configs() -> list[KernelConfig]:
    from flashinfer.prefill_rocm import _aiter_native_page_sizes

    native_pages = _aiter_native_page_sizes()
    # Pick the largest available native page size ≤ 256 (prefer 256 for the sweep)
    native_page = max(p for p in native_pages if p <= 256) if native_pages else 128
    flat_gather_page = 16  # deliberately not in native_pages

    configs: list[KernelConfig] = []

    # ── Single-prefill ───────────────────────────────────────────────────────
    for kv_len in _KV_LENS:
        q = torch.randn(_Q_LEN, _NUM_QO_HEADS, _HEAD_DIM, dtype=_DTYPE, device="cuda")
        k = torch.randn(kv_len, _NUM_KV_HEADS, _HEAD_DIM, dtype=_DTYPE, device="cuda")
        v = torch.randn(kv_len, _NUM_KV_HEADS, _HEAD_DIM, dtype=_DTYPE, device="cuda")

        flops = _flops(_Q_LEN, kv_len, _NUM_QO_HEADS, _HEAD_DIM, _CAUSAL)
        theo_bytes = _bytes(_Q_LEN, kv_len, _NUM_QO_HEADS, _NUM_KV_HEADS, _HEAD_DIM)

        configs.append(
            KernelConfig(
                name=f"single_kv{kv_len}",
                run_fn=torch.inference_mode()(
                    lambda q=q, k=k, v=v: flashinfer.single_prefill_with_kv_cache_return_lse(
                        q, k, v, kv_layout="NHD", causal=_CAUSAL, backend="aiter"
                    )
                ),
                theoretical_flops=flops,
                theoretical_bytes=theo_bytes,
                label=f"single  kv={kv_len:>5d}",
            )
        )

    # ── Batch-paged prefill — native-paged path ──────────────────────────────
    for kv_len in _KV_LENS:
        q, kv_data, qo_indptr, kv_indptr, kv_indices, kv_last_page_len = (
            _build_paged_kv(
                _BATCH,
                kv_len,
                native_page,
                _NUM_KV_HEADS,
                _HEAD_DIM,
                _DTYPE,
                torch.device("cuda"),
            )
        )
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", backend="aiter"
        )
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            _NUM_QO_HEADS,
            _NUM_KV_HEADS,
            _HEAD_DIM,
            native_page,
            causal=_CAUSAL,
            q_data_type=_DTYPE,
        )

        flops = _BATCH * _flops(_Q_LEN, kv_len, _NUM_QO_HEADS, _HEAD_DIM, _CAUSAL)
        theo_bytes = _BATCH * _bytes(
            _Q_LEN, kv_len, _NUM_QO_HEADS, _NUM_KV_HEADS, _HEAD_DIM
        )

        configs.append(
            KernelConfig(
                name=f"batch_native_kv{kv_len}",
                run_fn=torch.inference_mode()(
                    lambda q=q, kv=kv_data, w=wrapper: w.run(q, kv, return_lse=True)
                ),
                theoretical_flops=flops,
                theoretical_bytes=theo_bytes,
                label=f"batch(native,p={native_page:>3d})  kv={kv_len:>5d}",
            )
        )

    # ── Batch-paged prefill — flat-gather path ───────────────────────────────
    for kv_len in _KV_LENS:
        q, kv_data, qo_indptr, kv_indptr, kv_indices, kv_last_page_len = (
            _build_paged_kv(
                _BATCH,
                kv_len,
                flat_gather_page,
                _NUM_KV_HEADS,
                _HEAD_DIM,
                _DTYPE,
                torch.device("cuda"),
            )
        )
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", backend="aiter"
        )
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            _NUM_QO_HEADS,
            _NUM_KV_HEADS,
            _HEAD_DIM,
            flat_gather_page,
            causal=_CAUSAL,
            q_data_type=_DTYPE,
        )

        flops = _BATCH * _flops(_Q_LEN, kv_len, _NUM_QO_HEADS, _HEAD_DIM, _CAUSAL)
        theo_bytes = _BATCH * _bytes(
            _Q_LEN, kv_len, _NUM_QO_HEADS, _NUM_KV_HEADS, _HEAD_DIM
        )

        configs.append(
            KernelConfig(
                name=f"batch_flat_kv{kv_len}",
                run_fn=torch.inference_mode()(
                    lambda q=q, kv=kv_data, w=wrapper: w.run(q, kv, return_lse=True)
                ),
                theoretical_flops=flops,
                theoretical_bytes=theo_bytes,
                label=f"batch(flat,  p={flat_gather_page:>3d})  kv={kv_len:>5d}",
            )
        )

    return configs


if __name__ == "__main__":
    if not flashinfer.aiter_utils.is_aiter_supported(torch.device("cuda:0")):
        print("AITER backend not supported on this device. Exiting.")
        sys.exit(0)

    _skip_gpu = "--replot" in sys.argv or "--list-presets" in sys.argv
    profiler = RocmProfiler(
        configs=[] if _skip_gpu else _make_configs(),
        num_warmup=3,
        dry_run_ms=100,
        repeat_ms=1000,
        counters=_counters,
        kernel_name_regex="",
        output_dir=_OUTPUT_DIR,
        label=_label,
        roofline=(_counters == "roofline"),
    )
    profiler.run()
