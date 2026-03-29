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

    # Full pipeline: timing + rocprofv3 counter collection + roofline PNG
    python benchmarks/rocm_benchmarks/bench_fa2_prefill.py

    # Timing only (no rocprofv3):
    python benchmarks/rocm_benchmarks/bench_fa2_prefill.py --timing-only

    # Regenerate plot from existing CSVs (no GPU required):
    python benchmarks/rocm_benchmarks/bench_fa2_prefill.py --replot

Output files (all gitignored):
    benchmarks/rocm_benchmarks/fa2_timing.csv
    benchmarks/rocm_benchmarks/fa2_counters.yml
    benchmarks/rocm_benchmarks/fa2_counter_collection.csv
    benchmarks/rocm_benchmarks/fa2_roofline.png

Custom counters:

    Instead of a built-in preset ("roofline", "compute", "memory", "basic") you can
    point `counters=` at a YAML file in rocprofv3's native job format:

        profiler = RocmProfiler(..., counters="my_counters.yml", ...)

    The YAML groups hardware counters into passes. On gfx942 (MI300X) the hardware
    cannot collect all counters in a single pass; counters that share the same
    internal resource must be placed in separate `pmc:` entries.  Key constraints:

        • FetchSize and WriteSize must be in different passes.
        • SQ_INSTS_VALU cannot be combined with FetchSize or WriteSize.
        • MemUnitBusy does not exist on gfx942.

    Example — collect occupancy + wave stall breakdown in two passes:

        # my_counters.yml
        jobs:
          # Pass 1: wave issue rate + MFMA activity
          - pmc: [SQ_WAVES, SQ_INSTS_MFMA, SQ_INSTS_VALU_MFMA_MOPS_F16, FetchSize]
          # Pass 2: stall reasons (must be separate — share SQ resource with Pass 1)
          - pmc: [SQ_WAIT_INST_ANY, SQ_ACTIVE_INST_VALU, WriteSize]

    To discover available counters for your GPU run:

        rocprofv3 --list-counters
"""

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


profiler = RocmProfiler(
    configs=_make_configs(),
    num_warmup=3,
    dry_run_ms=100,
    repeat_ms=1000,
    counters="roofline",
    kernel_name_regex="SinglePrefillWithKVCacheKernel",
    output_dir=_OUTPUT_DIR,
    label="fa2",
    roofline=True,
)

if __name__ == "__main__":
    profiler.run()
