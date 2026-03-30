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

FA2 single prefill roofline benchmark using rocm_profiler.

Run:

    # Full roofline pipeline (timing + counter collection + roofline PNG):
    python benchmarks/rocm_benchmarks/bench_fa2_prefill.py

    # Select a different counter preset:
    python benchmarks/rocm_benchmarks/bench_fa2_prefill.py --counters occupancy
    python benchmarks/rocm_benchmarks/bench_fa2_prefill.py --counters stall
    python benchmarks/rocm_benchmarks/bench_fa2_prefill.py --counters compute

    # Override the output file label prefix:
    python benchmarks/rocm_benchmarks/bench_fa2_prefill.py --counters occupancy --label fa2_occ

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
to collect hardware counters.  The RocmProfiler object must therefore be
constructed at module import time with the correct `counters=` value, so that
both the outer driver and the inner rocprofv3 subprocess use the same preset.
We extract --counters / --label here (using parse_known_args so we don't
conflict with the profiler's own argparse), strip them from sys.argv, and then
pass the values to the RocmProfiler constructor.
"""

import argparse
import sys
import logging
from pathlib import Path

import torch
import flashinfer
from flashinfer.jit.core import logger

logger.setLevel(logging.ERROR)

# rocm_profiler lives at flashinfer/rocm_profiler/rocm_profiler.py
# (two levels up from benchmarks/rocm_benchmarks/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "rocm_profiler"))
from rocm_profiler import KernelConfig, RocmProfiler

# ---------------------------------------------------------------------------
# Bench-script-level argument parsing
#
# parse_known_args() extracts only --counters / --label and leaves all other
# flags (--timing-only, --skip-roofline, --replot, --list-presets, …) in
# sys.argv for RocmProfiler._parse_args() to consume.
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
    help=(
        "Output-file label prefix (default: 'fa2' for the roofline preset, "
        "'fa2_<preset>' for all others)."
    ),
)
_bench_args, _remaining = _bench_parser.parse_known_args()
# Strip --counters / --label so RocmProfiler's own argparse doesn't error on them.
sys.argv = [sys.argv[0]] + _remaining

_counters = _bench_args.counters
_label = (
    _bench_args.label
    if _bench_args.label is not None
    else ("fa2" if _counters == "roofline" else f"fa2_{_counters}")
)

# ---------------------------------------------------------------------------
# Sweep configuration
# (seq_len, num_qo_heads, num_kv_heads, head_dim, causal)
# ---------------------------------------------------------------------------
_CONFIGS = [
    # Causal sweep — crosses the MI300X ridge point (~247 FLOPs/B) near seq_len=1024
    (512, 32, 32, 128, True),
    (1024, 32, 32, 128, True),
    (2048, 32, 32, 128, True),
    (4096, 32, 32, 128, True),
    (8192, 32, 32, 128, True),
    # Non-causal sweep — 2× FLOPs, same bytes → 2× arithmetic intensity
    (512, 32, 32, 128, False),
    (1024, 32, 32, 128, False),
    (2048, 32, 32, 128, False),
    (4096, 32, 32, 128, False),
]

_OUTPUT_DIR = str(Path(__file__).parent)


def _make_configs() -> list[KernelConfig]:
    configs = []
    for seq_len, num_qo_heads, num_kv_heads, head_dim, causal in _CONFIGS:
        q = torch.randn(
            seq_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda"
        )
        k = torch.randn(
            seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
        )
        v = torch.randn(
            seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
        )

        # FLOPs: Q·Kᵀ + softmax(S)·V, causal mask halves each matmul
        factor = 2 if causal else 4
        flops = seq_len * seq_len * num_qo_heads * head_dim * factor

        # Bytes: Q + K + V + O  (4 tensors × fp16 × elements), cold-cache lower bound
        theo_bytes = 8 * seq_len * num_qo_heads * head_dim * 2

        causal_str = "causal" if causal else "nc"
        configs.append(
            KernelConfig(
                name=f"s{seq_len}_{causal_str}",
                run_fn=lambda q=q, k=k, v=v, c=causal: (
                    flashinfer.single_prefill_with_kv_cache_return_lse(
                        q, k, v, causal=c, backend="fa2"
                    )
                ),
                theoretical_flops=flops,
                theoretical_bytes=theo_bytes,
                label=f"seq={seq_len:>5d}  {'causal' if causal else 'non-causal'}",
            )
        )
    return configs


# Constructed at module level so rocprofv3 subprocess replay (which re-imports
# this module with the same sys.argv) picks up the correct counters preset.
profiler = RocmProfiler(
    configs=_make_configs(),
    num_warmup=3,
    dry_run_ms=100,
    repeat_ms=1000,
    counters=_counters,
    kernel_name_regex="SinglePrefillWithKVCacheKernel",
    output_dir=_OUTPUT_DIR,
    label=_label,
    roofline=(_counters == "roofline"),
)

if __name__ == "__main__":
    profiler.run()
