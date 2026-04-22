---
name: benchmark-kernel
description: Guide for benchmarking FlashInfer+ROCm kernels on AMD Instinct (CDNA3/CDNA4)
---

# Tutorial: Benchmarking FlashInfer+ROCm Kernels

This guide shows how to accurately benchmark kernels on the ROCm port of FlashInfer (the
`amd-flashinfer` package), targeting AMD Instinct CDNA3 (`gfx942`) and CDNA4 (`gfx950`).

## Goal

Measure the performance of FlashInfer+ROCm kernels:

- Get accurate GPU kernel execution time on MI300X / MI325X / MI350X / MI355X.
- Compare HIP-native and AITER (Composable-Kernel) prefill backends.
- Generate reproducible benchmark results for regression tracking.
- Save results to CSV / PNG rooflines for later analysis.

## Timing methods on ROCm

FlashInfer+ROCm supports three practical timing paths. **CUPTI is NVIDIA-only — do not try to
install `cupti-python` on a ROCm host.**

| Method | When to use | Source |
| --- | --- | --- |
| **CUDA events (HIP-backed via PyTorch)** | Default. Quick in-loop timing from Python. Good accuracy for kernels ≳ 50 µs. | `flashinfer.testing.bench_gpu_time` (the "CUDA event" path) |
| **`rocprofv3` + [`rocm_profiler/rocm_profiler.py`](../../../rocm_profiler/rocm_profiler.py)** | Preferred for authoring or optimizing a kernel. Gives per-kernel time, hardware counters, and a two-panel roofline plot. | Wrapper spawns `rocprofv3` as a subprocess. |
| **`omnitrace`** | Whole-process timeline with host + device events. Use when interaction with dataloaders / Python overhead is suspect. | Installed separately from ROCm. |

Internally, `bench_gpu_time` on ROCm uses PyTorch's `torch.cuda.Event`, which maps to HIP events
under the ROCm build. The `bench_gpu_time_with_cupti` code path in
[`flashinfer/testing/utils.py`](../../../flashinfer/testing/utils.py) is never selected on a ROCm
install because `cupti-python` will not import.

## Pre-flight: what you can actually benchmark

On a ROCm install of `amd-flashinfer`, only the APIs exposed in the `IS_HIP` branch of
[`flashinfer/__init__.py`](../../../flashinfer/__init__.py) are callable:

**Attention:**

- `single_prefill_with_kv_cache` / `single_prefill_with_kv_cache_return_lse`
- `BatchPrefillWithPagedKVCacheWrapper`, `BatchPrefillWithRaggedKVCacheWrapper`
- `single_decode_with_kv_cache`
- `BatchDecodeWithPagedKVCacheWrapper`, `CUDAGraphBatchDecodeWithPagedKVCacheWrapper`

**Other:**

- Normalization (`rmsnorm`, `fused_add_rmsnorm`, `gemma_rmsnorm`, …)
- RoPE (`apply_rope_*`, `apply_llama31_rope_*`)
- Sampling (`sampling_from_probs`, `top_k_*`, `top_p_*`, `min_p_sampling_from_probs`, …)
- Paged KV management (`append_paged_kv_cache`, `get_batch_indices_positions`, …)
- Quantization (`packbits`, `segment_packbits`)
- Activation (`silu_and_mul`, `gelu_and_mul`, `gelu_tanh_and_mul`)

**Not available on ROCm:** MLA, cascade, POD, FP4 quantization, TRT-LLM/CUTLASS MoE, cuDNN
backends. Do not attempt to benchmark these — the symbol simply is not re-exported in the
`IS_HIP` branch.

**Backends that exist per op:**

| Op family | Default (HIP) backend | AITER backend available? | How to select AITER |
| --- | --- | --- | --- |
| Single prefill | yes | yes (CK FMHA) | `backend="aiter"` kwarg |
| Batch prefill (paged / ragged) | yes | yes (CK FMHA) | `backend="aiter"` kwarg |
| Decode (single / batch / CUDA-graph) | yes | no | n/a |
| All others (norm, rope, sampling, …) | yes | no | n/a |

**AITER caveats** (see [`README.md`](../../../README.md) and
[`flashinfer/prefill_rocm.py`](../../../flashinfer/prefill_rocm.py)):

- `kv_layout="NHD"` only.
- Batch prefill with AITER's CK FMHA requires `page_size ∈ {1, 16, 1024}`.
- `amd-aiter` must be importable (usually `pip install amd-aiter --index-url https://pypi.amd.com/simple/`).

Trying to benchmark an unsupported config under `backend="aiter"` will raise a Python error
*before* the kernel launches, not silently fall back.

## Method 1: In-script timing with `bench_gpu_time`

For a quick perf check of one op, call
[`flashinfer.testing.bench_gpu_time`](../../../flashinfer/testing/utils.py) directly. On ROCm it
falls through to the `bench_gpu_time_with_cuda_event` path automatically.

```python
import torch
import flashinfer
from flashinfer.testing import bench_gpu_time

seq_len       = 1024
num_qo_heads  = 32
num_kv_heads  = 8      # GQA 4:1
head_dim      = 128
dtype         = torch.bfloat16

q = torch.randn(seq_len, num_qo_heads, head_dim, dtype=dtype, device="cuda")
k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")
v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device="cuda")


def run_default():
    return flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)


def run_aiter():
    return flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=True, backend="aiter",
    )


def report(label, fn):
    # enable_cupti=True is harmless on ROCm — it is silently ignored and the
    # CUDA-events path is used. Passing it makes the script portable to CUDA hosts.
    median_ms, std_ms = bench_gpu_time(
        fn, args=(), enable_cupti=True, num_iters=30, dry_run_iters=5,
    )
    print(f"{label:12s}  median={median_ms:.3f} ms  std={std_ms:.3f} ms")


report("hip-default", run_default)
report("aiter",        run_aiter)
```

Typical output on an MI300X (numbers are illustrative — your exact values will depend on ROCm
version, driver, and HIP-SDMA settings):

```text
hip-default  median=0.182 ms  std=0.004 ms
aiter        median=0.146 ms  std=0.003 ms
```

**Important arguments:**

| Arg | Purpose | Default |
| --- | --- | --- |
| `num_iters` | Measured iterations | 30 |
| `dry_run_iters` | Warmup iterations | 5 |
| `enable_cupti` | CUDA only; ignored on ROCm | False |
| `l2_flush` / `rotate_buffers` | Flush L2 between iterations for memory-bound kernels | varies |

## Method 2: `rocm_profiler` (recommended for optimization work)

For anything you intend to optimize, use the in-repo
[`rocm_profiler/rocm_profiler.py`](../../../rocm_profiler/rocm_profiler.py). It:

1. Runs repeated GPU launches in the current process to get a median kernel time.
2. Re-exec's the same driver script under `rocprofv3` as a subprocess (recognized by the
   `_ROCM_PROFILER_INTERNAL` env sentinel) to collect hardware counters with one
   warmup + one profiled launch.
3. Produces a two-panel log-log **roofline plot** combining the timing and counter data.

All outputs are written under `benchmarks/rocm_benchmarks/` (gitignored).

### Minimal driver script

Start from the working example at
[`benchmarks/rocm_benchmarks/bench_fa2_prefill.py`](../../../benchmarks/rocm_benchmarks/bench_fa2_prefill.py)
and adapt:

```python
# my_bench.py
import torch
import flashinfer
from rocm_profiler import RocmProfiler, KernelConfig

B, S, H_Q, H_KV, D = 1, 1024, 32, 8, 128
dtype = torch.bfloat16
q = torch.randn(S, H_Q, D, dtype=dtype, device="cuda")
k = torch.randn(S, H_KV, D, dtype=dtype, device="cuda")
v = torch.randn(S, H_KV, D, dtype=dtype, device="cuda")

configs = [
    KernelConfig(
        name="s1024_causal",
        run_fn=lambda: flashinfer.single_prefill_with_kv_cache_return_lse(
            q, k, v, causal=True
        ),
        # FLOPs = 2 * S * S * H_Q * D (attention mat-muls), matches the formula
        # used in benchmarks/rocm_benchmarks/bench_fa2_prefill.py.
        theoretical_flops=2 * S * S * H_Q * D,
        theoretical_bytes=(S * H_Q + 2 * S * H_KV) * D * dtype.itemsize,
        label="seq=1024 causal",
    ),
]

profiler = RocmProfiler(
    configs=configs,
    counters="roofline",            # or "compute", "memory", "occupancy", "stall", "basic"
    kernel_name_regex="SinglePrefill",
    output_dir="benchmarks/rocm_benchmarks",
    label="my_bench",
)

if __name__ == "__main__":
    profiler.run()
```

### Run it

```bash
# Full pipeline: timing + counter collection + roofline PNG
python my_bench.py

# Change the counter preset (see header of rocm_profiler.py for the full list)
python my_bench.py --counters occupancy
python my_bench.py --counters stall
python my_bench.py --counters memory

# Timing only (no rocprofv3 at all — fast sanity check)
python my_bench.py --timing-only

# Run profiling but skip the roofline plot
python my_bench.py --skip-roofline

# Regenerate the roofline plot from existing CSVs (no GPU required)
python my_bench.py --replot

# List all built-in counter presets
python my_bench.py --list-presets
```

### Outputs

```text
benchmarks/rocm_benchmarks/<label>_timing.csv             # median + std per config
benchmarks/rocm_benchmarks/<label>_counters.yml           # rocprofv3 input spec
benchmarks/rocm_benchmarks/<label>_counter_collection.csv # raw counters
benchmarks/rocm_benchmarks/<label>_roofline.png           # only for counters=roofline
```

### Counter presets worth knowing

| Preset | What it shows | Typical use |
| --- | --- | --- |
| `roofline` (default) | `FetchSize`, `WriteSize`, MFMA ops, TCC DRAM requests | Is the kernel compute- or memory-bound? |
| `compute` | MFMA ops + cycle counters | Matrix-core throughput on CDNA3/4 |
| `memory` | L2 and DRAM bandwidth breakdown | L2 hit-rate, HBM traffic |
| `occupancy` | `SQ_WAVES`, `SQ_BUSY_CYCLES`, `SQ_VALU_MFMA_BUSY_CYCLES`, `SQ_WAIT_INST_ANY`, `SQ_INSTS_LDS` | Wavefront density, scheduler efficiency |
| `stall` | `SQ_WAIT_INST_VMEM`, `SQ_WAIT_INST_LDS`, `SQ_BUSY_CYCLES` | Diagnose memory stalls |
| `basic` | `FetchSize` / `WriteSize` | Minimal baseline |

You can also pass a path to a `rocprofv3`-native YAML if you need a counter combination that is
not in the preset list.

## Method 3: Raw `rocprofv3` invocation

If you need full control over the counter set, bypass the Python wrapper and use `rocprofv3`
directly. This also works against any standalone Python script.

```bash
# Timeline + per-kernel stats
rocprofv3 --stats --kernel-trace \
    --output-format csv \
    --output-directory rpf-out \
    -- python my_bench.py

# Hardware counters (supply your own pmc / counter-input file)
cat > my_counters.txt <<'EOF'
pmc: SQ_WAVES SQ_BUSY_CYCLES SQ_WAIT_INST_VMEM
EOF
rocprofv3 -i my_counters.txt \
    --output-format csv \
    --output-directory rpf-counters \
    -- python my_bench.py
```

Kernel-name filtering is available via `--kernel-rename` and regex selection via
`--kernel-include-regex` in recent `rocprofv3` versions.

## Reference checking

When comparing the HIP-default and `backend="aiter"` paths (or any two backends), always verify
numerical parity before trusting perf numbers:

```python
ref = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)                   # HIP
got = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, backend="aiter")  # AITER

torch.testing.assert_close(got.float(), ref.float(), rtol=1e-2, atol=1e-2)
```

Loose BF16 tolerances are expected; tighten for FP32-only ops.

## Troubleshooting

### Inconsistent results (large std)

1. Raise `dry_run_iters` to 10–20 so the kernel cache and clocks settle.
2. Raise `num_iters` to 50+ for sub-100-µs kernels.
3. Pin the GPU clock:

   ```bash
   # Query supported clocks
   rocm-smi --showclocks
   # Lock SCLK / MCLK (requires sudo, restores on reboot)
   sudo rocm-smi --setsclk 7
   sudo rocm-smi --setmclk 3
   ```

4. Disable ECC scrubbing interference: `sudo rocm-smi --resetprofile` between runs.

### Kernel name does not match in `rocm_profiler`

The `kernel_name_regex` you pass to `RocmProfiler` must match the mangled kernel name emitted by
`rocprofv3`. If no rows appear in `<label>_counter_collection.csv`:

```bash
# 1. Dry-run to see what kernels are launched
rocprofv3 --stats --kernel-trace --output-format csv \
    --output-directory rpf-dbg -- python my_bench.py

# 2. Inspect rpf-dbg/*_kernel_stats.csv and copy the name prefix into your driver.
```

### AITER backend errors

If `backend="aiter"` raises before any timing runs, it is usually one of:

- `page_size` not in `{1, 16, 1024}` (batch prefill + CK FMHA path).
- `kv_layout != "NHD"`.
- `amd-aiter` not installed.

Fix the call or drop back to the default HIP backend for that config.

### `rocm_profiler` hangs or produces empty CSV

- Check that `rocprofv3` is on `PATH` and executable: `which rocprofv3`.
- Make sure the driver script prints something from the `if __name__ == "__main__":` block —
  the wrapper uses script output as a heartbeat.
- Run with `--timing-only` first to confirm the kernel path itself works before involving
  `rocprofv3`.

## Best practices

1. **Record the arch and ROCm version** alongside every perf number:

   ```python
   import torch
   props = torch.cuda.get_device_properties(0)
   print(props.name, props.gcnArchName, torch.version.hip)
   ```

   A `seq=1024` FA2 number on MI300X (`gfx942`, ROCm 7.2) is not comparable to one on MI350X
   (`gfx950`, ROCm 7.0.2).

2. **Always warm up.** First-call JIT compile will dominate the first measurement otherwise.
   Use `dry_run_iters >= 5` and explicitly call the kernel once before timing in scripts that
   measure the first iteration separately.

3. **Verify correctness before performance.** A kernel that silently writes junk is always
   faster than one that works.

4. **Compare against the AITER backend where it exists.** For single / batch prefill on ROCm,
   AITER's CK FMHA is often the competitive lower bound.

5. **Prefer the `roofline` counter preset to start.** It instantly tells you whether further
   optimization should target arithmetic intensity (MFMA ops) or HBM bandwidth (TCC DRAM
   requests).

## Related documentation

- [`CLAUDE.md`](../../../CLAUDE.md) — project overview and JIT architecture.
- [`.claude/skills/add-rocm-kernel/SKILL.md`](../add-rocm-kernel/SKILL.md) — author a new kernel
  to benchmark.
- [`.claude/skills/debug-rocm-crash/SKILL.md`](../debug-rocm-crash/SKILL.md) — when a kernel
  crashes during timing.
- [`benchmarks/rocm_benchmarks/bench_fa2_prefill.py`](../../../benchmarks/rocm_benchmarks/bench_fa2_prefill.py)
  — a real, working driver script to copy from.
- [`rocm_profiler/rocm_profiler.py`](../../../rocm_profiler/rocm_profiler.py) — full API docs in
  the module header.
- `rocprofv3` docs: <https://rocm.docs.amd.com/projects/rocprofiler-sdk/>.
