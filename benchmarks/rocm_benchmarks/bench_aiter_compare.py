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

Three-way latency comparison for single-prefill attention on ROCm:

  fa2       — FlashInfer native FA2 kernel (baseline)
  aiter_py  — AITER Python passthrough (aiter.ops.mha.flash_attn_varlen_func)
  aiter_cpp — AITER routed via FlashInfer C++ JIT harness (backend="aiter")

Shapes: Alibaba production profile — q=256, GQA 16/4, HD=64, causal,
        kv ∈ {512, 1024, 2048, 3072, 4096, 8192}

Run:

    # Timing comparison table (default):
    python benchmarks/rocm_benchmarks/bench_aiter_compare.py --timing-only

    # Full roofline (fa2 group only; aiter groups use timing-only internally):
    python benchmarks/rocm_benchmarks/bench_aiter_compare.py

Output files (all gitignored):
    benchmarks/rocm_benchmarks/cmp_timing.csv
    benchmarks/rocm_benchmarks/cmp_comparison.txt  — formatted table
"""

import argparse
import csv
import math
import sys
import logging
from pathlib import Path

import torch
import flashinfer
from flashinfer.jit.core import logger

logger.setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "rocm_profiler"))
from rocm_profiler import KernelConfig, RocmProfiler

# ---------------------------------------------------------------------------
# Bench-script-level argument parsing (strip before RocmProfiler sees argv)
# ---------------------------------------------------------------------------
_bench_parser = argparse.ArgumentParser(add_help=False)
_bench_parser.add_argument(
    "--counters",
    default="roofline",
    metavar="PRESET_OR_FILE",
)
_bench_parser.add_argument("--label", default="cmp", metavar="PREFIX")
_bench_args, _remaining = _bench_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

_counters = _bench_args.counters
_label = _bench_args.label

# ---------------------------------------------------------------------------
# Sweep: (kv_len, num_qo_heads, num_kv_heads, head_dim, causal)
# Alibaba production profile — q_len fixed at 256
# ---------------------------------------------------------------------------
_Q_LEN = 256
_CONFIGS = [
    (512, 16, 4, 64, True),
    (1024, 16, 4, 64, True),
    (2048, 16, 4, 64, True),
    (3072, 16, 4, 64, True),
    (4096, 16, 4, 64, True),
    (8192, 16, 4, 64, True),
]

_OUTPUT_DIR = str(Path(__file__).parent)
_BACKENDS = ("fa2", "aiter_py", "aiter_cpp")


def _flops(q_len: int, kv_len: int, num_qo_heads: int, head_dim: int, causal: bool) -> int:
    factor = 2 if causal else 4
    return q_len * kv_len * num_qo_heads * head_dim * factor


def _bytes(q_len: int, kv_len: int, num_qo_heads: int, num_kv_heads: int, head_dim: int) -> int:
    # Q + K + V (reads) + O (write), fp16 = 2 bytes each
    return 2 * head_dim * (q_len * num_qo_heads + kv_len * num_kv_heads + kv_len * num_kv_heads + q_len * num_qo_heads)


@torch.inference_mode()
def _make_configs() -> list[KernelConfig]:
    # Import AITER Python API once
    from aiter.ops.mha import flash_attn_varlen_func as _aiter_fwd

    configs: list[KernelConfig] = []

    for kv_len, num_qo_heads, num_kv_heads, head_dim, causal in _CONFIGS:
        q_len = _Q_LEN
        sm_scale = 1.0 / math.sqrt(head_dim)

        q = torch.randn(q_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda")
        k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")
        v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")

        flops = _flops(q_len, kv_len, num_qo_heads, head_dim, causal)
        theo_bytes = _bytes(q_len, kv_len, num_qo_heads, num_kv_heads, head_dim)

        # Pre-allocate cu_seqlens for aiter_py (avoid allocation inside timed loop)
        cu_q = torch.tensor([0, q_len], dtype=torch.int32, device="cuda")
        cu_k = torch.tensor([0, kv_len], dtype=torch.int32, device="cuda")

        causal_str = "causal" if causal else "nc"
        shape_tag = f"kv{kv_len}_{causal_str}"

        # ── FA2 backend ──────────────────────────────────────────────────────
        configs.append(
            KernelConfig(
                name=f"fa2_{shape_tag}",
                run_fn=torch.inference_mode()(
                    lambda q=q, k=k, v=v, c=causal: flashinfer.single_prefill_with_kv_cache_return_lse(
                        q, k, v, causal=c, backend="fa2"
                    )
                ),
                theoretical_flops=flops,
                theoretical_bytes=theo_bytes,
                label=f"FA2      kv={kv_len:>5d}  {'causal' if causal else 'non-causal'}",
            )
        )

        # ── AITER Python passthrough ─────────────────────────────────────────
        # Replicates the old amd-integration flash_attn_varlen_func approach.
        configs.append(
            KernelConfig(
                name=f"aiter_py_{shape_tag}",
                run_fn=torch.inference_mode()(
                    lambda q=q, k=k, v=v, c=causal, cq=cu_q, ck=cu_k,
                           ql=q_len, kl=kv_len, sc=sm_scale: _aiter_fwd(
                        q=q, k=k, v=v,
                        cu_seqlens_q=cq,
                        cu_seqlens_k=ck,
                        max_seqlen_q=ql,
                        max_seqlen_k=kl,
                        dropout_p=0.0,
                        softmax_scale=sc,
                        logits_soft_cap=0.0,
                        causal=c,
                        window_size=(-1, -1, 0),
                        return_lse=True,
                        return_attn_probs=False,
                        out=None,
                    )
                ),
                theoretical_flops=flops,
                theoretical_bytes=theo_bytes,
                label=f"AITER-Py kv={kv_len:>5d}  {'causal' if causal else 'non-causal'}",
            )
        )

        # ── AITER C++ harness ────────────────────────────────────────────────
        configs.append(
            KernelConfig(
                name=f"aiter_cpp_{shape_tag}",
                run_fn=torch.inference_mode()(
                    lambda q=q, k=k, v=v, c=causal: flashinfer.single_prefill_with_kv_cache_return_lse(
                        q, k, v, causal=c, backend="aiter"
                    )
                ),
                theoretical_flops=flops,
                theoretical_bytes=theo_bytes,
                label=f"AITER-C++ kv={kv_len:>5d}  {'causal' if causal else 'non-causal'}",
            )
        )

    return configs


def _print_comparison_table(timing_csv: Path) -> None:
    """Read the timing CSV and print a side-by-side comparison table."""
    rows: dict[str, dict[str, float]] = {}  # shape_tag -> {backend: median_ms}
    with open(timing_csv) as f:
        for r in csv.DictReader(f):
            name: str = r["name"]
            ms = float(r["median_ms"])
            for backend in _BACKENDS:
                if name.startswith(f"{backend}_"):
                    tag = name[len(backend) + 1:]
                    rows.setdefault(tag, {})[backend] = ms
                    break

    def _fmt(v: float) -> str:
        return f"{v * 1000:.1f} µs" if not math.isnan(v) else "    N/A  "

    # Header
    w = 11
    sep = "-" * (20 + w * 3 + 16)
    hdr = f"{'Shape':<20}{'FA2':>{w}}{'AITER-Py':>{w}}{'AITER-C++':>{w}}  {'C++ overhead':>14}"
    print()
    print("=" * len(hdr))
    print("  Single-Prefill Latency Comparison  (q=256, GQA 16/4, HD=64, causal)")
    print("=" * len(hdr))
    print(hdr)
    print(sep)

    geom_overhead = 1.0
    n = 0
    for kv_len, num_qo_heads, num_kv_heads, head_dim, causal in _CONFIGS:
        causal_str = "causal" if causal else "nc"
        tag = f"kv{kv_len}_{causal_str}"
        if tag not in rows:
            continue
        d = rows[tag]
        fa2 = d.get("fa2", float("nan"))
        py = d.get("aiter_py", float("nan"))
        cpp = d.get("aiter_cpp", float("nan"))
        overhead = (cpp - py) / py * 100 if py > 0 else float("nan")
        geom_overhead += overhead
        n += 1

        shape_label = f"kv={kv_len:>5d}"
        print(
            f"  {shape_label:<18}{_fmt(fa2):>{w}}{_fmt(py):>{w}}{_fmt(cpp):>{w}}"
            f"  {overhead:>+11.1f}%"
        )

    mean_overhead = geom_overhead / n if n > 0 else float("nan")
    print(sep)
    print(
        f"  {'Mean overhead':18}{'':>{w}}{'':>{w}}{'':>{w}}  {mean_overhead:>+11.1f}%"
    )
    print()
    print("  C++ overhead: positive = C++ harness is slower than Python passthrough.")
    print("  FA2 shown for absolute reference; both AITER paths are 4-7× faster than FA2.")
    print()


if __name__ == "__main__":
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

    if not _skip_gpu and "--replot" not in sys.argv:
        timing_csv = Path(_OUTPUT_DIR) / f"{_label}_timing.csv"
        if timing_csv.exists():
            _print_comparison_table(timing_csv)
