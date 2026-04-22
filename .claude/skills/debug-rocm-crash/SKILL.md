---
name: debug-rocm-crash
description: Tutorial for debugging HIP/ROCm kernel crashes in FlashInfer+ROCm using API logging plus HIP/ROCm runtime tooling
---

# Tutorial: Debugging ROCm Crashes in FlashInfer+ROCm

This guide shows how to debug HIP/ROCm kernel crashes and errors in the `amd-flashinfer` fork
(CDNA3 `gfx942`, CDNA4 `gfx950`) using the `@flashinfer_api` logging decorator combined with
ROCm's own debugging tools.

If you are used to upstream's `debug-cuda-crash` skill, the Python logging half is identical —
`@flashinfer_api`, `FLASHINFER_LOGLEVEL`, `FLASHINFER_LOGDEST` all work unchanged on HIP. The
CUDA-tooling half (`compute-sanitizer`, `cuda-gdb`, `CUDA_LAUNCH_BLOCKING`) is rewritten below
using the ROCm equivalents.

## Goal

When your code crashes on an AMD Instinct GPU with errors like:

- `HIP error: the operation cannot be performed in the present state`
- `hipErrorIllegalAddress`
- `Memory access fault by GPU node-N (Agent handle: ...) on address 0x...`
- `hipErrorOutOfMemory`
- `RuntimeError: CUDA error: an illegal memory access was encountered` (PyTorch masquerades HIP
  errors as CUDA errors)

… you want to:

- Capture input tensors BEFORE the crash (so the crash itself doesn't take the evidence with it).
- Pinpoint exactly which kernel launch faulted.
- Understand whether the bug is a shape mismatch, a bad page table / KV config, an AITER
  limitation, or a genuine kernel bug.

## Why use API logging?

**Problem:** HIP faults frequently terminate the process with little more than a hex address,
leaving no Python-level context.

**Solution:** `@flashinfer_api` logs inputs (shape, dtype, device, strides, optionally min/max/mean
and NaN/Inf counts) BEFORE the kernel runs. If the kernel crashes, the last log entry shows you
exactly what data it received.

## Step 1: Enable API logging

### Basic (function names only)

```bash
export FLASHINFER_LOGLEVEL=1
export FLASHINFER_LOGDEST=stdout

python my_script.py
```

Output:

```text
[2026-04-21 10:30:45] FlashInfer API Call: single_prefill_with_kv_cache
```

### Detailed (inputs / outputs + metadata)

```bash
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=debug.log

python my_script.py
```

Example output in `debug.log`:

```text
================================================================================
[2026-04-21 10:30:45] FlashInfer API Logging — System Information
================================================================================
FlashInfer version: 0.5.3+amd.1
HIP / ROCm version: 7.1.1
GPU 0: AMD Instinct MI300X
  gcnArchName: gfx942:sramecc+:xnack-
PyTorch version: 2.9.1+rocm7.1
================================================================================

[2026-04-21 10:30:46] FlashInfer API Call: batch_decode_with_paged_kv_cache
--------------------------------------------------------------------------------
Positional input arguments:
  arg[0]:
    Tensor(
      shape=(32, 8, 128)
      dtype=torch.bfloat16
      device=cuda:0
      requires_grad=False
      is_contiguous=True
    )
Keyword input arguments:
  paged_kv_cache=
    Tensor(
      shape=(1024, 2, 8, 128)
      dtype=torch.bfloat16
      device=cuda:0
      ...
    )
```

Even though the device string shows `cuda:0`, the underlying device is an AMD GPU — this is
expected because PyTorch's ROCm build reuses the `cuda` namespace. The `gcnArchName` line above
is the unambiguous ROCm marker.

### Full (with tensor statistics)

```bash
export FLASHINFER_LOGLEVEL=5
export FLASHINFER_LOGDEST=debug.log

python my_script.py
```

Adds:

```text
  Tensor(
    shape=(32, 8, 128)
    dtype=torch.bfloat16
    device=cuda:0
    requires_grad=False
    is_contiguous=True
    min=-3.125000
    max=4.250000
    mean=0.015625
    nan_count=0
    inf_count=0
  )
```

Use level 5 when diagnosing numerical issues (NaN/Inf propagation). Note that HIP-graph capture
paths auto-skip statistics; that is intentional and shows up as
`[statistics skipped: HIP graph capture in progress]`.

## Step 2: Force deterministic kernel launches before debugging

HIP async launches make Python tracebacks point at the wrong line. Set these env vars **before**
running your script:

```bash
export HIP_LAUNCH_BLOCKING=1
export AMD_SERIALIZE_KERNEL=3
```

- `HIP_LAUNCH_BLOCKING=1` — force every HIP API call to be synchronous.
- `AMD_SERIALIZE_KERNEL=3` — also serialize kernel launches through the queue. This is the
  single most useful knob for `Memory access fault by GPU node-N` errors, because it pins the
  fault to the *actual* faulting kernel rather than whichever subsequent launch happened to
  finish first.

Both are zero-overhead when there's no bug to chase, so enabling them in `pytest` runs while
iterating on a new kernel is reasonable.

## Step 3: Common ROCm errors and how to debug them

### Error 1: Illegal memory access / GPU memory fault

**Error messages** (any of these indicate the same class of bug):

```text
RuntimeError: CUDA error: an illegal memory access was encountered
HIP error: hipErrorIllegalAddress
Memory access fault by GPU node-1 (Agent handle: 0x...) on address 0x7f...
VM_CONTEXT1_PROTECTION_FAULT_STATUS ... NO_RETRY: 0x0
```

**Recipe:**

```bash
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=crash.log
export AMD_SERIALIZE_KERNEL=3
export HIP_LAUNCH_BLOCKING=1
python my_script.py
```

In `crash.log`, find the **last** `FlashInfer API Call:` entry — that is the kernel that took
the process down. Check:

- Tensor **shapes** match what the kernel expects (head_dim, num_heads).
- All tensors are on the same device (both `cuda:0`, not mixed `cuda:0` + `cpu`).
- `is_contiguous=True` where required; non-contiguous strides are a classic cause of
  out-of-bounds reads.
- For paged-KV wrappers: `kv_indices` / `kv_indptr` values are within `[0, num_pages)`.

**Common root causes in this fork:**

- Wrong `head_dim_qk` / `head_dim_vo` mismatch between `q` and the KV cache.
- CPU tensor accidentally passed to a GPU API.
- Non-contiguous `q`/`k`/`v` from a `.transpose()` or `.view()` chain — add a `.contiguous()`.
- Out-of-range `kv_indices` — often off-by-one when building page tables by hand.
- **AITER-specific:** see dedicated section below.

### Error 2: AITER backend crash (`backend="aiter"`)

When using `backend="aiter"` on single or batch prefill, watch for two very specific gotchas
(both documented in [`README.md`](../../../README.md) and enforced by the code in
[`flashinfer/prefill_rocm.py`](../../../flashinfer/prefill_rocm.py)):

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ValueError` raised *before* any kernel launch | `page_size` is not in `{1, 16, 1024}` (batch prefill + CK FMHA) | Re-plan with one of the supported page sizes, or drop `backend="aiter"` for that call. |
| `ValueError` about KV layout | `kv_layout != "NHD"` | Switch to `NHD` or use the default HIP backend. |
| Hard GPU fault mid-kernel, no Python exception | `amd-aiter` version mismatch vs. the ROCm build | Reinstall `amd-aiter` matching your ROCm version (`--extra-index-url https://pypi.amd.com/rocm-<version>/simple`). |
| `ModuleNotFoundError: aiter` | `amd-aiter` not installed | `pip install amd-aiter --index-url https://pypi.amd.com/simple/`. |

If API logging shows a correct-looking call to a prefill API but the process dies with a GPU
fault and no Python traceback, **disable the AITER backend** as a first step to see whether the
bug is in AITER or in our side of the port.

### Error 3: NaN / Inf values

```text
RuntimeError: ... returned nan or inf
```

```bash
export FLASHINFER_LOGLEVEL=5
export FLASHINFER_LOGDEST=nan.log
python my_script.py
```

Check `nan_count` / `inf_count` in the log. On CDNA3/4 the most common sources are:

- FP8 path overflow — the `_fnuz` variants used on AMD
  (`__hip_fp8_e4m3_fnuz`, `__hip_fp8_e5m2_fnuz`) have a different representable range than
  NVIDIA's `__nv_fp8_e4m3`. A scale factor calibrated against an NVIDIA reference will
  routinely overflow on ROCm.
- A previous op producing `-inf` / `inf` that is then fed into `exp` (online softmax).
- Uninitialized memory — `torch.empty(...)` vs `torch.zeros(...)`.

### Error 4: Out of memory

```text
RuntimeError: HIP out of memory.
```

```bash
rocm-smi --showmeminfo vram --showpids
export FLASHINFER_LOGLEVEL=3
python my_script.py
```

Look for unexpectedly large tensor shapes in the last log entry. If the process keeps getting
OOM-killed on healthy-looking shapes, check:

- Zombie processes holding VRAM: `rocm-smi --showpids` and `kill -9` them.
- JIT cache compile spike — set `MAX_JOBS=1` to cap concurrent ninja jobs during AOT builds.
- Another tenant on the same GPU — pin to a single GPU with `HIP_VISIBLE_DEVICES=N`.

### Error 5: Wrong dtype

```text
RuntimeError: expected scalar type BFloat16 but found Half
```

```bash
export FLASHINFER_LOGLEVEL=3
python my_script.py
```

In the log, look for the mismatching `dtype=` field. On ROCm, confirm:

- If the op supports FP8 on your arch: `gfx942`/`gfx950` use the `_fnuz` FP8 variants — a
  callsite that expects `torch.float8_e4m3fn` (NVIDIA's OCP FP8) will mis-dispatch. The
  PyTorch dtype used on ROCm for `__hip_fp8_e4m3_fnuz` is `torch.float8_e4m3fnuz`.

## Step 4: Multi-GPU / multi-process debugging

For multi-rank runs use the `%i` pattern in the log destination:

```bash
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=debug_rank_%i.log
export HIP_VISIBLE_DEVICES=0,1,2,3      # restrict to specific GPUs
# (or ROCR_VISIBLE_DEVICES — same effect, but applied earlier in the stack)

torchrun --nproc_per_node=4 my_script.py
```

This produces `debug_rank_<pid>.log` per process. Use `HIP_VISIBLE_DEVICES` instead of
`CUDA_VISIBLE_DEVICES` when you need to isolate a specific AMD device.

If a specific GPU is misbehaving (ECC errors, firmware stuck), check it with
`rocm-smi --showreset --showuniqueid --showproductname` and open `dmesg -wH` in another
terminal.

## Step 5: Advanced debugging with ROCm tools

### `rocgdb` (CUDA-GDB equivalent)

```bash
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=debug.log
export AMD_SERIALIZE_KERNEL=3
export HIP_LAUNCH_BLOCKING=1

rocgdb --args python my_script.py
```

Inside `rocgdb`:

```text
(rocgdb) catch throw
(rocgdb) run
(rocgdb) bt            # stack trace at the crash point
(rocgdb) info agents   # list GPUs
(rocgdb) info wavefronts
```

For attaching to a running process (e.g. a hang), set before you launch your script:

```bash
export ROCM_DEBUG_WAIT_FOR_DEBUGGER=1
```

Then `rocgdb -p <pid>` attaches; no debugger attached → the process waits at the first GPU API
call.

### HIP / HSA runtime tracing

```bash
export AMD_LOG_LEVEL=3         # HIP API + stream trace
# export AMD_LOG_LEVEL=4       # very verbose, includes arg decoding
export HSA_ENABLE_DEBUG=1      # one layer below HIP (runtime queues, agents)
python my_script.py 2> hip.trace
```

Grep for `hipLaunchKernel`, `hipMemcpy`, and `error` in `hip.trace`. The trace is linear with
`HIP_LAUNCH_BLOCKING=1`, which makes it possible to correlate each FlashInfer API call with the
exact underlying HIP launches.

### Device state snapshots with `rocm-smi`

Leave this running in another terminal while reproducing a hang:

```bash
watch -n 1 'rocm-smi --showuse --showmeminfo vram --showpids --showprofile'
```

Watch for:

- GPU stuck at 100% but no `SQ` activity — kernel is looping.
- VRAM pinned high after your process exits — another process is still holding it.
- Throttling indicators (`POWERCAP`, `THERMAL`) — reproduce on a cooler box before filing a
  kernel bug.

### `dmesg` for firmware-level faults

```bash
sudo dmesg -T | grep -i -E "amdgpu|kfd|vm_fault" | tail -100
```

`VM_CONTEXT1_PROTECTION_FAULT_STATUS` entries here tell you page-fault class, access type, and
the offending address — useful when the Python log only says `hipErrorIllegalAddress`.

## Step 6: Kernel-level debugging with `printf`

`printf()` works inside HIP device code exactly the way it does on CUDA:

```cpp
__global__ void MyKernel(const float* __restrict__ input,
                         float* __restrict__ output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Print from one thread per block to avoid flood
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("n=%d, input[0]=%f\n", n, input[0]);
  }

  if (idx < n) {
    output[idx] = input[idx] * 2.0f;
  }
}
```

Flush after the launch from Python:

```python
my_kernel(input, output)
torch.cuda.synchronize()  # Flushes device printf buffer on ROCm too
```

### Warp / wavefront considerations

The wavefront size on CDNA3 and CDNA4 is **64** (not 32 as on NVIDIA). Adjust any
representative-thread logic accordingly:

```cpp
// CDNA3/4: wavefront size = 64
if (threadIdx.x % 64 == 0) {
  printf("Wavefront %d processing\n", threadIdx.x / 64);
}
```

Common mistake ported blindly from a CUDA example:

```cpp
// ❌ Assumes warp size 32; prints from thread 32 of a CDNA wavefront
if (threadIdx.x % 32 == 0) {
  printf("...");
}
```

Use `warpSize` (a built-in `unsigned int`) when writing portable code.

### Device asserts

```cpp
assert(value >= 0.0f && "Value must be non-negative");
```

Build with JIT debug flags to make these trip reliably:

```bash
export FLASHINFER_JIT_VERBOSE=1
```

(Unlike upstream there is no `FLASHINFER_JIT_DEBUG=1` `-O0 -g -G` mode on the HIP side yet;
`-O0` is not wired into `hipcc` invocations. Add `-g` via `extra_cuda_cflags` temporarily in
the JIT generator while debugging.)

## Environment Variables Reference

### FlashInfer logging

| Variable | Values | Description |
| --- | --- | --- |
| `FLASHINFER_LOGLEVEL` | `0` | No logging (default). Zero overhead. |
| | `1` | Function names only. |
| | `3` | Inputs/outputs with shape/dtype/device/strides. |
| | `5` | + min/max/mean/nan/inf statistics. |
| `FLASHINFER_LOGDEST` | `stdout` | Console (default). |
| | `stderr` | Stderr. |
| | `<path>` | File. |
| | `log_%i.txt` | Multi-process; `%i` expands to PID. |
| `FLASHINFER_JIT_VERBOSE` | `1` | Print every `hipcc` invocation and build command. |

### HIP / ROCm runtime

| Variable | Effect |
| --- | --- |
| `HIP_LAUNCH_BLOCKING=1` | Force synchronous launches (stack traces pin the faulting kernel). |
| `AMD_SERIALIZE_KERNEL=3` | Serialize kernel launches through the queue. |
| `AMD_LOG_LEVEL=3` (or `4`) | HIP API trace. |
| `HSA_ENABLE_DEBUG=1` | HSA runtime trace. |
| `HIP_VISIBLE_DEVICES=0,1` | Restrict visible GPUs (preferred on ROCm). |
| `ROCR_VISIBLE_DEVICES=0,1` | Same as above, applied one layer deeper. |
| `ROCM_DEBUG_WAIT_FOR_DEBUGGER=1` | Block until `rocgdb` attaches. |

## Best practices

### 1. Always start with `FLASHINFER_LOGLEVEL=3`

```bash
export FLASHINFER_LOGLEVEL=3
```

Gives you tensor metadata without overwhelming output.

### 2. Combine with `AMD_SERIALIZE_KERNEL=3` on first reproduction

```bash
export FLASHINFER_LOGLEVEL=3
export AMD_SERIALIZE_KERNEL=3
export HIP_LAUNCH_BLOCKING=1
```

This is the single most useful env combination for debugging an unknown HIP fault.

### 3. Log to a file for crashes

```bash
export FLASHINFER_LOGDEST=crash.log
```

Console output can be lost when the process SIGKILLs itself on a GPU fault.

### 4. Compare before / after on the last API call

- Last successful `FlashInfer API Call:` with **both** inputs and outputs logged — OK.
- Last `FlashInfer API Call:` with inputs logged but **no outputs** — that's your crash site.

### 5. Disable logging in production

```bash
unset FLASHINFER_LOGLEVEL     # or export FLASHINFER_LOGLEVEL=0
```

The `@flashinfer_api` decorator short-circuits to a zero-overhead path when disabled.

## Troubleshooting the debugger itself

### No logs appear

- Verify the API you're calling actually has `@flashinfer_api` on it — decoration coverage is
  a work in progress; a handful of low-level APIs may not be wrapped yet.
- Check the env vars are exported in the right shell:

  ```bash
  echo $FLASHINFER_LOGLEVEL  # expect "3"
  echo $FLASHINFER_LOGDEST   # expect path or "stdout"
  ```

### Statistics skipped at level 5

```text
[statistics skipped: HIP graph capture in progress]
```

Expected: min/max/mean/nan/inf would require synchronization that is illegal during graph
capture. Temporarily drop to `FLASHINFER_LOGLEVEL=3` if you need inputs from inside a captured
graph.

### `rocgdb` exits immediately with `no symbol table loaded`

`pip install`-installed HIP binaries are often stripped. Reinstall with
`-DCMAKE_BUILD_TYPE=RelWithDebInfo` or add `"-g"` to `extra_cuda_cflags` in the JIT generator
for the op you are debugging, clear `~/.cache/flashinfer/`, and retry.

## Quick examples

### Debug shape mismatch

```bash
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=stdout
python my_script.py
# Read tensor shapes in stdout
```

### Debug NaN / Inf

```bash
export FLASHINFER_LOGLEVEL=5
export FLASHINFER_LOGDEST=nan.log
python my_script.py
# Grep "nan_count=" / "inf_count=" in nan.log
```

### Debug a hard GPU fault

```bash
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=gpu_fault.log
export AMD_SERIALIZE_KERNEL=3
export HIP_LAUNCH_BLOCKING=1
python my_script.py
# Last entry in gpu_fault.log is the faulting call.
# Also check `sudo dmesg -T | tail -50` for VM_CONTEXT1_PROTECTION_FAULT_STATUS.
```

### Debug multi-GPU

```bash
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=rank_%i.log
export HIP_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 train.py
# Inspect rank_*.log files per process.
```

### Full `rocgdb` session

```bash
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=debug.log
export AMD_SERIALIZE_KERNEL=3
rocgdb --args python my_script.py
# (rocgdb) catch throw
# (rocgdb) run
# (rocgdb) bt
```

## Example: full debug session

### Your code crashes

```python
import torch
import flashinfer

q  = torch.randn(32, 8, 128, dtype=torch.bfloat16, device="cuda")
kv = torch.randn(1024, 2, 8, 64, dtype=torch.bfloat16, device="cuda")   # wrong head_dim!

out = flashinfer.single_decode_with_kv_cache(q, kv[:, 0], kv[:, 1])     # crashes
```

Output:

```text
Memory access fault by GPU node-1 (Agent handle: 0x...) on address 0x7f9d...
```

### Enable logging + deterministic launches

```bash
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=debug.log
export AMD_SERIALIZE_KERNEL=3
export HIP_LAUNCH_BLOCKING=1
python test.py
```

### Read `debug.log`

```text
[...] FlashInfer API Call: single_decode_with_kv_cache
Positional input arguments:
  arg[0]:
    Tensor(shape=(32, 8, 128), dtype=torch.bfloat16, device=cuda:0, ...)
  arg[1]:
    Tensor(shape=(1024, 8, 64), dtype=torch.bfloat16, device=cuda:0, ...)   # ← head_dim=64, not 128
  arg[2]:
    Tensor(shape=(1024, 8, 64), dtype=torch.bfloat16, device=cuda:0, ...)   # ← also wrong
```

### Fix

```python
kv = torch.randn(1024, 2, 8, 128, dtype=torch.bfloat16, device="cuda")  # fixed
```

### Success

```bash
python test.py
# No crash; debug.log shows both the call and the output tensor.
```

## Summary

1. Before anything else:

   ```bash
   export FLASHINFER_LOGLEVEL=3
   export FLASHINFER_LOGDEST=debug.log
   export AMD_SERIALIZE_KERNEL=3
   export HIP_LAUNCH_BLOCKING=1
   ```

2. Reproduce the crash. Inputs are logged BEFORE each kernel runs, so the last entry tells you
   which call faulted.

3. If the shape/dtype/device picture in the log looks correct, escalate to
   `AMD_LOG_LEVEL=3`, then to `rocgdb`, then to `dmesg` for VM-level faults.

4. For AITER crashes, check the layout/page-size invariants first — they cover a large fraction
   of "illegal address" reports in practice.

5. Disable logging when done:

   ```bash
   export FLASHINFER_LOGLEVEL=0
   ```

## Related documentation

- [`CLAUDE.md`](../../../CLAUDE.md) — project overview; see the "Debugging" and "API Logging"
  sections for background.
- [`.claude/skills/add-rocm-kernel/SKILL.md`](../add-rocm-kernel/SKILL.md) — when you are
  debugging a kernel you just wrote.
- [`.claude/skills/benchmark-kernel/SKILL.md`](../benchmark-kernel/SKILL.md) — when the crash
  only happens under profiling.
- ROCm debugging documentation: <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/debugging.html>
- `rocgdb` user guide: <https://rocm.docs.amd.com/projects/llvm-project/en/latest/reference/rocgdb.html>
- Upstream's [`debug-cuda-crash` skill](https://github.com/flashinfer-ai/flashinfer/blob/main/.claude/skills/debug-cuda-crash/SKILL.md) —
  the source this tutorial was adapted from; useful when cross-referencing a bug that reproduces
  on both backends.
