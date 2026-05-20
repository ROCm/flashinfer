---
name: benchmark-kernel
description: Guide for benchmarking FlashInfer+ROCm kernels on AMD Instinct (CDNA3/CDNA4)
---

# Benchmarking FlashInfer+ROCm Kernels

For a real driver script to copy, see
[`benchmarks/rocm_benchmarks/bench_fa2_prefill.py`](../../../benchmarks/rocm_benchmarks/bench_fa2_prefill.py) and [`benchmarks/rocm_benchmarks/bench_aiter_prefill.py`](../../../benchmarks/rocm_benchmarks/bench_aiter_prefill.py)
For the in-repo profiler wrapper, see [`rocm_profiler/rocm_profiler.py`](../../../rocm_profiler/rocm_profiler.py).

## Timing method matrix

| Method | When | How |
| --- | --- | --- |
| `flashinfer.testing.bench_gpu_time` | Quick in-loop check (kernels ≳ 50 µs) | Falls through to PyTorch `torch.cuda.Event` (HIP events under ROCm) automatically. |
| `rocm_profiler` (`RocmProfiler`) | Anything you intend to optimize | Two-phase: in-process median timing, then re-execs the same script under `rocprofv3` (sentinel: `_ROCM_PROFILER_INTERNAL`) for hardware counters. Produces roofline PNG. |
| `rocprofv3` directly | Full control over counter set | `rocprofv3 --stats --kernel-trace -- python script.py`; or `-i pmc.txt` for custom counters. |
| `omnitrace` | Host + device timeline when Python overhead is suspect | Installed separately. |

## Non-obvious gotchas

- **CUPTI is NVIDIA-only and `enable_cupti=True` WILL fail on ROCm.** [`flashinfer/testing/utils.py:1010`](../../../flashinfer/testing/utils.py) routes `enable_cupti=True` straight to `bench_gpu_time_with_cupti` with no HIP guard; `cupti-python` is not installable on ROCm. Leave `enable_cupti=False` (the default) — `bench_gpu_time` then uses `torch.cuda.Event` (HIP events under the hood).
- **AITER backend constraints, accurately:**
  - Explicit `backend="aiter"` + `kv_layout != "NHD"` → `ValueError` at `plan()` time. Raised in the prefill wrapper, e.g. [`prefill_rocm.py:1978`](../../../flashinfer/prefill_rocm.py) (single/paged) and the batch-paged wrapper around line 2920. Not raised by auto-selection — that path silently falls back to `fa2`.
  - Explicit `backend="aiter"` on non-gfx942/gfx950 → `RuntimeError`.
  - `amd-aiter` not importable → `ImportError`.
  - **"Native" page sizes** (no flat-gather): `{128, 256, 1024}` for `amd-aiter >= 0.1.10`, else `{16, 1024}` — see `_aiter_native_page_sizes()` in [`prefill_rocm.py:59`](../../../flashinfer/prefill_rocm.py). **Non-native page sizes are NOT rejected** — they go through a flat-gather code path. So the "{1, 16, 1024}" guidance from older docs is wrong.
  - Auto-selection (no explicit `backend=`) silently falls back to `fa2` for any of: `kv_layout != "NHD"`, custom mask, dtype not in `{fp16, bf16}`, `dtype_q != dtype_kv`, `head_dim_qk != head_dim_vo`, `pos_encoding_mode != "NONE"`, or `amd-aiter` not importable. See `_auto_select_prefill_backend()` in [`prefill_rocm.py:311`](../../../flashinfer/prefill_rocm.py) for the authoritative list.
- **Always verify numerical parity before trusting perf numbers.** Compare default-HIP vs AITER outputs with `torch.testing.assert_close(rtol=1e-2, atol=1e-2)` for BF16/FP16 first.
- **`gcnArchName` is the unambiguous arch marker.** Device strings show `cuda:0` on AMD too. Record `torch.cuda.get_device_properties(0).gcnArchName` and `torch.version.hip` alongside every number — a `gfx942` / ROCm 7.2 result is not comparable to a `gfx950` / ROCm 7.0.2 result.

## What can actually be benchmarked on ROCm

Only the APIs in the `IS_HIP` branch of [`flashinfer/__init__.py`](../../../flashinfer/__init__.py) are callable. **Not** available: MLA, cascade, POD, FP4, MoE, cuDNN backends. Don't try to import them.

AITER backend available for: single prefill, batch prefill (paged + ragged) — opt in via `backend="aiter"`. Not available for decode, norm, rope, sampling, etc.

## `rocm_profiler` counter presets

Pass via `RocmProfiler(counters=...)` or `--counters` on the driver script.

| Preset | What it shows | Use for |
| --- | --- | --- |
| `roofline` (default) | `FetchSize`, `WriteSize`, MFMA ops, TCC DRAM requests | "Am I compute- or memory-bound?" |
| `compute` | MFMA ops + cycle counters | Matrix-core throughput |
| `memory` | L2 + DRAM breakdown | L2 hit-rate, HBM traffic |
| `occupancy` | `SQ_WAVES`, `SQ_BUSY_CYCLES`, `SQ_VALU_MFMA_BUSY_CYCLES`, `SQ_INSTS_LDS` | Wavefront density |
| `stall` | `SQ_WAIT_INST_VMEM`, `SQ_WAIT_INST_LDS` | Diagnose memory stalls |
| `basic` | `FetchSize` / `WriteSize` | Minimal baseline |

Or pass a path to a `rocprofv3`-native YAML for a custom counter set.

Driver script flags: `--timing-only` (skip rocprofv3), `--skip-roofline`, `--replot` (regen PNG from existing CSVs, no GPU), `--list-presets`.

Output (under `benchmarks/rocm_benchmarks/`, gitignored):

```text
<label>_timing.csv             # median + std per config
<label>_counter_collection.csv # raw counters
<label>_roofline.png           # only for counters=roofline
```

## Reproducibility checklist

1. **Warm up.** `dry_run_iters >= 5`; raise to 10–20 if std is high. First call includes JIT compile.
2. **Pin clocks** for sub-100-µs kernels:

   ```bash
   rocm-smi --showclocks
   sudo rocm-smi --setsclk 7
   sudo rocm-smi --setmclk 3
   ```

3. **Record arch + ROCm version** in the log: `print(props.name, props.gcnArchName, torch.version.hip)`.
4. **Isolate the GPU:** `HIP_VISIBLE_DEVICES=N` (or `ROCR_VISIBLE_DEVICES=N`, one layer deeper).

## Troubleshooting `rocm_profiler`

- **Empty `_counter_collection.csv`:** `kernel_name_regex` doesn't match the mangled name. Run `rocprofv3 --stats --kernel-trace -- python my_bench.py` first and copy the prefix from `*_kernel_stats.csv`.
- **Hang or no output:** confirm `which rocprofv3` is on `PATH`; the wrapper uses script `print()` output as a heartbeat — make sure the `if __name__ == "__main__":` block prints something.
- **Use `--timing-only` first** to verify the kernel path works before involving `rocprofv3`.
