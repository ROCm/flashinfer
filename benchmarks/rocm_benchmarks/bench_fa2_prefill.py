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

Single-prefill roofline benchmark using rocm_profiler.

Run:

    # Full roofline pipeline (timing + counter collection + roofline PNG):
    python benchmarks/rocm_benchmarks/bench_fa2_prefill.py

    # Production shapes, multiple backends (timing compared side-by-side):
    python benchmarks/rocm_benchmarks/bench_fa2_prefill.py --backends fa2,fa3,aiter

    # Arithmetic-intensity sweep (old symmetric sweep):
    python benchmarks/rocm_benchmarks/bench_fa2_prefill.py --shape-set intensity

    # Select a different counter preset:
    python benchmarks/rocm_benchmarks/bench_fa2_prefill.py --counters occupancy
    python benchmarks/rocm_benchmarks/bench_fa2_prefill.py --counters stall
    python benchmarks/rocm_benchmarks/bench_fa2_prefill.py --counters compute

    # Override the output file label prefix:
    python benchmarks/rocm_benchmarks/bench_fa2_prefill.py --counters occupancy --label my_label

    # Timing only (no rocprofv3):
    python benchmarks/rocm_benchmarks/bench_fa2_prefill.py --timing-only

    # Skip roofline plot after profiling:
    python benchmarks/rocm_benchmarks/bench_fa2_prefill.py --skip-roofline

    # Regenerate plot from existing CSVs (no GPU required):
    python benchmarks/rocm_benchmarks/bench_fa2_prefill.py --replot

    # List all available counter presets:
    python benchmarks/rocm_benchmarks/bench_fa2_prefill.py --list-presets

Output files (all gitignored):
    benchmarks/rocm_benchmarks/<label>_timing.csv
    benchmarks/rocm_benchmarks/<label>_meta.json
    benchmarks/rocm_benchmarks/<label>_counters.yml
    benchmarks/rocm_benchmarks/<label>_counter_collection.csv
    benchmarks/rocm_benchmarks/<label>_roofline.png   (roofline preset only)

Shape sets:
    production  — Alibaba production sweep: q=256, GQA 16/4, hd=64,
                  kv ∈ {512,1024,2048,3072,4096,8192}, causal (default)
    intensity   — Arithmetic-intensity sweep: q=kv, MHA 32/32, hd=128,
                  causal + non-causal, crosses MI300X ridge at seq≈1024

Counter presets available out of the box:
    roofline  — FetchSize, WriteSize, MFMA ops, TCC DRAM requests,
                SQ_VALU_MFMA_BUSY_CYCLES (default)
    compute   — MFMA ops and cycle counters
    memory    — L2 and DRAM bandwidth breakdown
    basic     — minimal: FetchSize / WriteSize only
    occupancy — SQ_WAVES, SQ_BUSY_CYCLES, SQ_VALU_MFMA_BUSY_CYCLES,
                SQ_WAIT_INST_ANY, SQ_INSTS_LDS
    stall     — SQ_INSTS_MFMA, SQ_WAIT_INST_VMEM, SQ_VALU_MFMA_BUSY_CYCLES,
                SQ_WAIT_INST_LDS, SQ_BUSY_CYCLES

    Or pass a path to a YAML file in rocprofv3 native job format.

Design note — why --counters/--shape-set/--backends are parsed at module level
-------------------------------------------------------------------------------
rocprofv3 re-executes this script as a subprocess (passing the same sys.argv)
to collect hardware counters.  The RocmProfiler object must therefore be
constructed at module import time with the correct parameter values, so that
both the outer driver and the inner rocprofv3 subprocess use the same preset.
We extract these flags here (using parse_known_args so we don't conflict with
the profiler's own argparse), strip them from sys.argv, and then pass the
values to the RocmProfiler constructor.
"""

import argparse
import math
import sys
import logging
from pathlib import Path

import torch
import flashinfer
from flashinfer.jit.core import logger as _jit_logger
from flashinfer.aiter_utils import HAS_AITER

# Suppress routine JIT INFO/DEBUG output; WARNING still surfaces compile errors.
_jit_logger.setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "rocm_profiler"))
from rocm_profiler import KernelConfig, RocmProfiler

# ---------------------------------------------------------------------------
# Bench-script-level argument parsing
#
# parse_known_args() extracts only these flags and leaves all others (
# --timing-only, --skip-roofline, --replot, --list-presets, …) in sys.argv
# for RocmProfiler._parse_args() to consume.
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
    help="Output-file label prefix. Auto-derived from backends+shape-set if omitted.",
)
_bench_parser.add_argument(
    "--shape-set",
    default="production",
    choices=["production", "intensity"],
    help=(
        "'production' = Alibaba production shapes (q=256, GQA 16/4, hd=64, causal). "
        "'intensity' = arithmetic-intensity sweep (q=kv, MHA 32/32, hd=128). "
        "Default: production."
    ),
)
_bench_parser.add_argument(
    "--backends",
    default="fa2",
    metavar="BACKEND[,BACKEND...]",
    help=(
        "Comma-separated backends to benchmark: fa2, fa3, aiter. "
        "Counter collection applies to fa2 and fa3 (SinglePrefillWithKVCacheKernel); "
        "aiter configs are timed but not counter-collected by default. "
        "Default: fa2."
    ),
)
_bench_args, _remaining = _bench_parser.parse_known_args()
# Strip these flags so RocmProfiler's own argparse doesn't error on them.
sys.argv = [sys.argv[0]] + _remaining

_counters = _bench_args.counters
_shape_set = _bench_args.shape_set
_backends: list[str] = [b.strip() for b in _bench_args.backends.split(",")]

if _bench_args.label is not None:
    _label = _bench_args.label
else:
    parts = ["-".join(_backends)]
    if _shape_set != "production":
        parts.append(_shape_set)
    if _counters != "roofline":
        parts.append(_counters)
    _label = "_".join(parts)

# ---------------------------------------------------------------------------
# Shape sweep definitions
# (q_len, kv_len, num_qo_heads, num_kv_heads, head_dim, causal)
# ---------------------------------------------------------------------------

# Production target: Alibaba inference distribution.
# q_len=256 (new tokens), GQA 16/4, hd=64, causal-only.
_PRODUCTION_SHAPES = [
    (256, 512, 16, 4, 64, True),
    (256, 1024, 16, 4, 64, True),
    (256, 2048, 16, 4, 64, True),
    (256, 3072, 16, 4, 64, True),
    (256, 4096, 16, 4, 64, True),
    (256, 8192, 16, 4, 64, True),
]

# Arithmetic-intensity sweep that crosses the MI300X ridge (~247 FLOPs/B)
# near seq_len=1024.  Useful for roofline analysis across the memory/compute
# boundary; not a realistic inference workload.
_INTENSITY_SHAPES = [
    # Causal
    (512, 512, 32, 32, 128, True),
    (1024, 1024, 32, 32, 128, True),
    (2048, 2048, 32, 32, 128, True),
    (4096, 4096, 32, 32, 128, True),
    (8192, 8192, 32, 32, 128, True),
    # Non-causal — 2× FLOPs at same bytes → 2× arithmetic intensity
    (512, 512, 32, 32, 128, False),
    (1024, 1024, 32, 32, 128, False),
    (2048, 2048, 32, 32, 128, False),
    (4096, 4096, 32, 32, 128, False),
]

_SHAPE_SETS: dict[str, list] = {
    "production": _PRODUCTION_SHAPES,
    "intensity": _INTENSITY_SHAPES,
}

_OUTPUT_DIR = str(Path(__file__).parent)


@torch.inference_mode()
def _make_configs(
    shapes: list,
    backends: list[str],
) -> list[KernelConfig]:
    """
    Build a KernelConfig list for a cross-product of shapes × backends.

    For each (shape, backend) pair:
      - Tensors are allocated at module import time (so rocprofv3 subprocess
        reuse the same shapes without GPU reallocation).
      - Q/K/V are scaled by 1/sqrt(head_dim) to keep dot-product magnitudes
        in a safe fp16 range and avoid misleading NaN output during debug.
    """
    skipped: list[str] = []
    configs: list[KernelConfig] = []
    multi = len(backends) > 1

    for backend in backends:
        if backend == "aiter" and not HAS_AITER:
            skipped.append("aiter")
            continue

        for q_len, kv_len, num_qo_heads, num_kv_heads, head_dim, causal in shapes:
            scale = 1.0 / math.sqrt(head_dim)
            q = torch.randn(q_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda") * scale
            k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda") * scale
            v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda") * scale

            # FLOPs: two GEMMs (QK^T and AV), both of shape attended×head_dim.
            # For causal with q_len ≤ kv_len, each query token i attends to
            # kv_len-q_len+i+1 key positions → total attended pairs =
            #   q_len*(kv_len-q_len) + q_len*(q_len+1)//2
            #   = q_len*kv_len - q_len*(q_len-1)//2
            # For non-causal every pair is attended.
            if causal:
                attended = q_len * kv_len - q_len * (q_len - 1) // 2
            else:
                attended = q_len * kv_len
            flops = attended * num_qo_heads * head_dim * 4  # ×2 matmuls × ×2 mul-add

            # Bytes (cold-cache lower bound, fp16):
            #   Q: (q_len, num_qo_heads, head_dim) — read
            #   K: (kv_len, num_kv_heads, head_dim) — read
            #   V: (kv_len, num_kv_heads, head_dim) — read
            #   O: (q_len, num_qo_heads, head_dim) — write
            #   LSE: (q_len, num_qo_heads) × fp32 — write (return_lse variant)
            theo_bytes = (
                (2 * q_len * num_qo_heads + 2 * kv_len * num_kv_heads) * head_dim * 2
                + q_len * num_qo_heads * 4
            )

            causal_str = "causal" if causal else "nc"
            backend_tag = f"_{backend}" if multi else ""
            name = f"kv{kv_len}h{head_dim}_{causal_str}{backend_tag}"

            if q_len == kv_len:
                shape_label = f"seq={q_len:>5d}"
            else:
                shape_label = f"q={q_len} kv={kv_len:>5d}"
            label = f"{shape_label} {'causal':6s} {backend}" if causal else f"{shape_label} {'nc':6s} {backend}"

            configs.append(
                KernelConfig(
                    name=name,
                    run_fn=torch.inference_mode()(
                        lambda q=q, k=k, v=v, c=causal, b=backend: (
                            flashinfer.single_prefill_with_kv_cache(
                                q, k, v, causal=c, backend=b, return_lse=True
                            )
                        )
                    ),
                    theoretical_flops=flops,
                    theoretical_bytes=theo_bytes,
                    label=label,
                )
            )

    if skipped:
        print(
            f"[bench] WARNING: skipping backends {skipped} — "
            "not installed (install from https://github.com/ROCm/aiter).",
            file=sys.stderr,
        )
    return configs


if __name__ == "__main__":
    # Defer GPU tensor allocation: --replot and --list-presets don't need a CUDA device.
    _skip_gpu = "--replot" in sys.argv or "--list-presets" in sys.argv
    shapes = _SHAPE_SETS[_shape_set]

    # Counter collection via kernel_name_regex matches the FA2/FA3 kernel.
    # AITER dispatches a different kernel; its dispatches won't appear in the
    # counter CSV (timing is still accurate for all backends).
    profiler = RocmProfiler(
        configs=[] if _skip_gpu else _make_configs(shapes, _backends),
        num_warmup=3,
        dry_run_ms=100,
        repeat_ms=1000,
        counters=_counters,
        kernel_name_regex="SinglePrefillWithKVCacheKernel",
        output_dir=_OUTPUT_DIR,
        label=_label,
        roofline=(_counters == "roofline"),
    )
    profiler.run()
