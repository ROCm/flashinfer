# Contributing to FlashInfer+ROCm

This is the **AMD ROCm port** of FlashInfer (`amd-flashinfer`), targeting
AMD Instinct GPUs — gfx942 (MI300X / MI325X, CDNA3) and gfx950
(MI355X, CDNA4). The upstream CUDA repo at
<https://github.com/flashinfer-ai/flashinfer> uses different env vars,
paths, and toolchains; its contribution guide does not transfer here.

For project overview, install steps, running tests, AITER setup, and
environment variables, see [`README.md`](README.md). This document
covers only what's specific to contributing code.

# Code Structure

```text
flashinfer/
├── include/                  # framework-agnostic kernel headers (raw pointers only)
│   ├── flashinfer/           # FlashInfer kernel implementations
│   └── gpu_iface/backend/    # GPU abstraction layer — cuda/ and hip/ shims
├── csrc/                     # upstream CUDA op registration (PyTorch bindings)
├── flashinfer/
│   ├── csrc_rocm/            # HIP op registration (PyTorch bindings) — the ROCm analog of csrc/
│   ├── jit/                  # Python JIT compilation infra (cpp_ext_hip.py is the HIP entry)
│   └── *.py                  # Python user-facing API (e.g. attention.py, mla_rocm.py)
├── tests/rocm_tests/         # HIP test suite (test_*_hip.py)
├── benchmarks/rocm_benchmarks/  # ROCm-specific benchmarks
└── 3rdparty/                 # vendored dependencies (cutlass, composable_kernel, …)
```

**Framework separation.** `include/` files must remain framework-agnostic
— no PyTorch headers, raw pointers only. PyTorch tensor handling for HIP
ops lives in `flashinfer/csrc_rocm/`. Violating this causes subtle build
failures because the same headers are pulled into the JIT compilation
pipeline that has no PyTorch on its include path.

**`csrc/` vs `flashinfer/csrc_rocm/`.** `csrc/` is the upstream CUDA op
registration tree — keep it in sync with upstream where possible to
reduce merge conflicts. New HIP-specific op bindings go in
`flashinfer/csrc_rocm/`, with a `_hip` or `_aiter` suffix when the file
routes to a HIP-specific code path or to AITER.

**`include/gpu_iface/`.** Hides CUDA/HIP divergence behind a common
header surface (`math_ops.hpp`, `mma_ops.hpp`, `memory_ops.hpp`, …).
When you need a new intrinsic, add the abstraction in `gpu_iface/` and
provide a HIP implementation under `gpu_iface/backend/hip/`. Don't
reach for `hipcub`, `__hip_*`, or inline asm from inside
`include/flashinfer/` — go through `gpu_iface`.

# Adding a Kernel

1. **Kernel implementation** — framework-agnostic header(s) in
   `include/flashinfer/`, using `gpu_iface/` for any CUDA/HIP-divergent
   intrinsic.
2. **PyTorch binding** — register the op in `flashinfer/csrc_rocm/`.
   The only layer that may include Torch headers.
3. **JIT generator** — add the op's JIT spec in `flashinfer/jit/*.py`.
4. **Python interface** — expose the user-facing API in `flashinfer/*.py`.
5. **Tests** — `test_*_hip.py` under `tests/rocm_tests/`. Reuse the
   fixtures in `tests/rocm_tests/conftest.py`.
6. **(Optional) Benchmark** — script under `benchmarks/rocm_benchmarks/`.
7. **Pre-commit** — `pre-commit run -a` before submitting.

A step-by-step Claude Code skill (`add-rocm-kernel`) walks through this
with concrete examples.

# Build / JIT Gotchas

**JIT cache silently sticky.** `JitSpec.build()` only writes
`build.ninja` when the file is missing, so changing env vars
(`FLASHINFER_ROCM_ARCH_LIST`, extra cflags) is a **silent no-op**
unless you either `rm -rf ~/.cache/flashinfer/` or call
`spec.write_ninja()` explicitly. When debugging build flags, always
clear the cache first.

**Debug builds.** `FLASHINFER_JIT_DEBUG=1` is a no-op on ROCm/HIP — it
only injects debug flags on the CUDA branch. To get a debug build on
ROCm, append `"-O0", "-g"` via `extra_cuda_cflags` in the op's JIT
generator (the HIP path injects `-O3` before `extra_cuda_cflags`, so
the trailing `-O0` is what actually overrides it on the hipcc command
line) and clear `~/.cache/flashinfer/`.

# Pre-Commit

```bash
pre-commit install   # one-time, installs the git hook
pre-commit run -a    # run on all files
```

CI rejects PRs that don't pass `pre-commit run -a`.

# Submitting Changes

Open PRs against the `amd-integration` branch of
[`ROCm/flashinfer`](https://github.com/ROCm/flashinfer). For PR
description conventions (sections, benchmarks, test plan), see the
"PR Description" section of [`CLAUDE.md`](CLAUDE.md).
