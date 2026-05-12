# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **This is the AMD ROCm port of FlashInfer** (`amd-flashinfer`), targeting AMD Instinct GPUs (CDNA3/CDNA4). CUDA/NVIDIA references from the upstream project do not apply here. The upstream repository lives at <https://github.com/flashinfer-ai/flashinfer>.

## Project Overview

FlashInfer+ROCm is a GPU kernel library for LLM serving on AMD hardware that uses **JIT (Just-In-Time) compilation by default**. This means kernel code changes are automatically picked up without reinstalling the package — extremely convenient for development.

## Quick Reference

| Task | Command |
|------|---------|
| Install for development | `python -m pip install --no-build-isolation -ve.` |
| Initialize submodules | `git submodule update --init --recursive` |
| Run all ROCm tests | `pytest tests/rocm_tests/` |
| Run specific test | `pytest tests/rocm_tests/test_file.py::test_function` |
| Run tests (skip slow) | `pytest tests/rocm_tests/ -m "not slow"` |
| Run benchmark | `python benchmarks/rocm_benchmarks/bench_fa2_prefill.py` |
| Run linting | `pre-commit run -a` |
| Install pre-commit hooks | `pre-commit install` |
| Clear JIT cache | `rm -rf ~/.cache/flashinfer/` |
| Enable verbose JIT logging | `export FLASHINFER_JIT_VERBOSE=1` |
| Enable debug build | `export FLASHINFER_JIT_DEBUG=1` |
| Set target architectures | `export FLASHINFER_ROCM_ARCH_LIST="gfx942,gfx950"` |
| Limit parallel ninja jobs | `export MAX_JOBS=4` |

## Quick Start for Development

### Installation

```bash
git clone --recursive https://github.com/ROCm/flashinfer.git
cd flashinfer
python -m pip install --no-build-isolation -ve.
```

**Important**: Torch must be installed from AMD's ROCm repository, **not** from PyPI:

```bash
pip install torch==<torch-version> -f https://repo.radeon.com/rocm/manylinux/rocm-rel-<rocm-version>
```

Replace `<torch-version>` and `<rocm-version>` with the versions listed in the
[GPU and ROCm Support](README.md#gpu-and-rocm-support) table in `README.md`.

If you forgot `--recursive` when cloning:

```bash
git submodule update --init --recursive
```

That's it! You can now:

- Run all tests and benchmarks
- Modify kernel source code in `include/` without reinstalling
- Changes are JIT-compiled on next use

### How JIT Compilation Works

When you call a FlashInfer API:

1. **First call**: Generates specialized HIP/C++ code based on parameters (dtype, head_dim, etc.), compiles it with ninja + hipcc, caches the `.so` file
2. **Subsequent calls**: Uses cached compiled module
3. **After kernel changes**: Automatically detects changes and recompiles

**No manual rebuild step needed** — just edit `.cuh` files and run your code again.

**Important pitfall**: `JitSpec.build()` only writes `build.ninja` when the file is missing. If you change compiler flags via env vars (`FLASHINFER_ROCM_ARCH_LIST`, extra cflags), those changes are silent no-ops unless you force `spec.write_ninja()` before `build()`.

### Docker Images

AMD publishes validated Docker images on Docker Hub:

```bash
docker run -it --privileged --network=host --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --shm-size 128G rocm/flashinfer:<tag>
```

See [Docker Hub](https://hub.docker.com/r/rocm/flashinfer/tags) for available tags.

## Testing

Tests live in `tests/rocm_tests/`. The test worker count is automatically halved relative to physical GPU count to avoid HSA/HIPBLAS flakiness under concurrent load.

```bash
# All tests
pytest tests/rocm_tests/

# Skip heavy tests (fast iteration)
pytest tests/rocm_tests/ -m "not slow"

# Specific file
pytest tests/rocm_tests/test_single_prefill_kernels_hip.py

# Specific test
pytest tests/rocm_tests/test_single_prefill_kernels_hip.py::test_single_prefill_with_kv_cache

# Parallel across GPUs
pytest tests/rocm_tests/ -n auto
```

### Markers

| Marker | Meaning |
|--------|---------|
| `slow` | Heavy tests (1M-trial sampling, 4GB tensor cases). Exclude with `-m "not slow"` for fast iteration; nightly runs include them. |

### Skipping Tests Based on GPU Architecture

Use utilities from `flashinfer.hip_utils` and the `HAS_AITER` flag for conditional skipping:

```python
import pytest
from flashinfer.aiter_utils import HAS_AITER
from flashinfer.hip_utils import FLASHINFER_SUPPORTED_ROCM_ARCHS

# Skip if AITER not installed
@pytest.mark.skipif(not HAS_AITER, reason="requires AITER")
def test_aiter_prefill(): ...

# Skip on unsupported arch
@pytest.mark.skipif("gfx942" not in FLASHINFER_SUPPORTED_ROCM_ARCHS, reason="requires gfx942")
def test_cdna3_kernel(): ...
```

## Benchmarking

ROCm benchmarks live in `benchmarks/rocm_benchmarks/`. For GPU timing use `rocprof` or the `rocm_profiler` module in this repo.

```bash
python benchmarks/rocm_benchmarks/bench_fa2_prefill.py
```

For counter-based profiling (HBM bandwidth, LDS stalls, MFMA utilization):

```bash
# Validate profiler attribution first
python benchmarks/rocm_benchmarks/validate_profiler_attribution.py

# Cycle decomposition
python benchmarks/rocm_benchmarks/cycle_decomp.py
```

## Code Linting

```bash
pre-commit run -a        # Run all hooks
pre-commit install       # Run on every commit
```

## Architecture: JIT Compilation System

FlashInfer+ROCm's JIT system follows the same three-layer pattern as upstream but uses hipcc instead of nvcc.

### Layer 1: JitSpec (flashinfer/jit/core.py)

`JitSpec` defines compilation metadata:

- `name`: Unique identifier (URI hash from parameters)
- `sources`: List of `.cu`/`.cpp` files to compile (HIP uses `.cu` extension)
- `extra_cuda_cflags`, `extra_cflags`, `extra_ldflags`: Compiler flags (passed to hipcc)

### JIT Directory Rules

**NEVER write to package directories** — they may be read-only after installation.

| Directory | Writable | Use for |
|-----------|----------|---------|
| `FLASHINFER_GEN_SRC_DIR` | ✓ Yes | Generated source files (Jinja output, copied `.cu` files) |
| `FLASHINFER_JIT_DIR` | ✓ Yes | Compiled `.so` outputs |
| `FLASHINFER_CSRC_DIR` | ✗ No | Read-only source templates |
| `FLASHINFER_AOT_DIR` | ✗ No | Read-only pre-compiled binaries |

### Compilation Context: Architecture-Specific Compilation

FlashInfer uses `CompilationContext` to manage ROCm architecture targets.

**How it works:**

- Auto-detects GPUs or reads `FLASHINFER_ROCM_ARCH_LIST` environment variable (default: `gfx942`)
- JIT modules compile for architectures in `FLASHINFER_SUPPORTED_ROCM_ARCHS = ["gfx942", "gfx950"]`
- If GPU not supported → `RuntimeError: No supported ROCm architectures found`
- Cache path: `~/.cache/flashinfer/<version>/<gfx_arch>/`

### Layer 2: Code Generation

Every `gen_*_module()` function in `flashinfer/jit/` follows this pattern:

```python
def gen_some_module(dtype_in, dtype_out, ...):
    uri = get_some_uri(dtype_in, dtype_out, ...)
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri

    # Optional: render Jinja template for type specialization
    with open(jit_env.FLASHINFER_CSRC_DIR / "some_customize_config.jinja") as f:
        template = jinja2.Template(f.read())
    config_content = template.render(dtype_in=dtype_map[dtype_in], ...)
    write_if_different(gen_directory / "some_config.inc", config_content)

    # Copy source files to gen directory
    sources = []
    for fname in ["some_kernel.cu", "some_jit_binding.cu"]:
        shutil.copy(jit_env.FLASHINFER_CSRC_DIR / fname, gen_directory / fname)
        sources.append(gen_directory / fname)

    return gen_jit_spec(uri, sources, extra_cuda_cflags=[...])
```

If no type specialization is needed, skip the Jinja step and copy source files directly.

### Layer 3: Compilation and Loading

`JitSpec` methods:

- `write_ninja()` — Generates `build.ninja` file (using hipcc)
- `build()` — Executes `ninja` to compile sources; **only writes ninja file if missing**
- `build_and_load()` — Compiles and loads the `.so` via PyTorch's `cpp_extension`

## Directory Structure

```text
flashinfer/
├── include/flashinfer/           # Header-only HIP/C++ kernel templates
│   ├── attention/generic/        # Attention kernels (prefill, decode, cascade)
│   ├── gpu_iface/                # GPU abstraction layer (HIP/CUDA iface)
│   └── [...]
│
├── flashinfer/csrc_rocm/         # ROCm kernel launchers + PyTorch bindings
│   ├── *.cu                      # Kernel launcher implementations
│   ├── *_jit_pybind.cu           # PyTorch pybind exports
│   ├── *_customize_config.jinja  # Type config templates (optional)
│   └── [...]
│
├── flashinfer/                   # Python package
│   ├── jit/
│   │   ├── core.py               # JitSpec, compilation infrastructure
│   │   ├── env.py                # Workspace paths (HIP branch)
│   │   └── [...]
│   ├── hip_utils.py              # ROCm utilities, arch detection, version checks
│   ├── aiter_utils.py            # AITER backend helpers + HAS_AITER flag
│   ├── prefill_rocm.py           # ROCm prefill Python API
│   └── *.py                      # Other high-level Python APIs
│
├── tests/rocm_tests/             # ROCm test suite
│   ├── conftest.py               # xdist worker count, GPU detection
│   ├── test_*_hip.py             # Per-kernel test files
│   └── [...]
│
├── benchmarks/rocm_benchmarks/   # ROCm benchmarking scripts
│   ├── bench_fa2_prefill.py
│   ├── validate_profiler_attribution.py
│   ├── cycle_decomp.py
│   └── [...]
│
└── CMakeLists.txt                # scikit-build-core build system
```

### Critical Rule: Framework Separation

**Torch headers MUST NOT be included in `include/` directory files.**

- `include/`: Framework-agnostic HIP kernels (accept raw pointers, no Torch types)
- `flashinfer/csrc_rocm/`: PyTorch bindings (tensor handling, pybind exports)

## Adding a New Operation (ROCm)

1. Write kernel in `include/flashinfer/attention/generic/new_op.cuh` (framework-agnostic)
2. Write launcher in `flashinfer/csrc_rocm/new_op.cu` (PyTorch tensor handling)
3. Create PyTorch pybind binding in `flashinfer/csrc_rocm/new_op_jit_pybind.cu`
4. (Optional) Create Jinja template for type specialization
5. Write JIT module generator in `flashinfer/jit/`
6. Write Python API in `flashinfer/new_op_rocm.py` with `@functools.cache`
7. Write tests in `tests/rocm_tests/test_new_op_hip.py`
8. Register in `flashinfer/aot.py` for AOT compilation
9. Export in `flashinfer/__init__.py`

**Example implementations:**

- **Simple**: `flashinfer/csrc_rocm/norm.cu` — no Jinja, good starting point
- **Moderate**: `flashinfer/csrc_rocm/rope.cu` — with type dispatch
- **Complex**: `flashinfer/csrc_rocm/batch_prefill.cu` — plan-run pattern, JIT config

## Key Architectural Patterns

### Module Caching

Two-level caching to avoid recompilation:

1. **Python-level** (`@functools.cache`): In-memory cache of loaded modules
2. **File-level** (`~/.cache/flashinfer/`): Compiled `.so` files on disk

URI computed as: `hash(operation_type + parameters + source_hashes + flags + rocm_arch)`

**Cache management:**

- Clear cache: `rm -rf ~/.cache/flashinfer/`
- Override location: `export FLASHINFER_WORKSPACE_BASE="/scratch"`

### AITER Backend

Some kernels (notably prefill attention) have an AITER backend in addition to the FA2 backend. AITER is an external AMD library for highly optimized attention on CDNA hardware.

```bash
# Install AITER from source
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter && python3 setup.py develop
```

Check availability in code:

```python
from flashinfer.aiter_utils import HAS_AITER
```

### Dispatch Macros

Handle combinatorial parameter spaces:

```cpp
DISPATCH_DTYPE(input_dtype, DTypeIn, {
  DISPATCH_DTYPE(output_dtype, DTypeOut, {
    DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
      LaunchKernel<DTypeIn, DTypeOut, BLOCK_SIZE>(...);
    });
  });
});
```

## Debugging

### Enable JIT Logging

```bash
export FLASHINFER_JIT_VERBOSE=1      # Verbose JIT output
export FLASHINFER_JIT_DEBUG=1        # Debug symbols, -O0
```

### Inspect Generated Code

```bash
# Generated sources
ls -la ~/.cache/flashinfer/*/generated/

# Compiled modules
ls -la ~/.cache/flashinfer/*/cached_ops/

# Build files
cat ~/.cache/flashinfer/*/cached_ops/*/build.ninja
```

### Environment Variables

```bash
# Architecture
export FLASHINFER_ROCM_ARCH_LIST="gfx942,gfx950"  # Target architectures (comma-separated)

# Compilation
export MAX_JOBS=4                                   # Parallel ninja jobs
export FLASHINFER_JIT_VERBOSE=1                     # Verbose hipcc output
export FLASHINFER_JIT_DEBUG=1                       # Debug build (-O0, debug symbols)

# Cache
export FLASHINFER_WORKSPACE_BASE="/scratch"         # Custom cache directory
```

### GPU Profiling

ROCm profiling uses `rocprof` or the `rocm_profiler` module in this repo (not CUPTI):

```bash
# Basic profiling
rocprof --stats python my_script.py

# Counter collection
rocprof --hsa-trace python my_script.py
```

For per-dispatch attribution, see `benchmarks/rocm_benchmarks/validate_profiler_attribution.py` — the `FetchSize` counter reports L2↔HBM bytes per dispatch accurately.

## Development Workflow

### Typical Development Loop

1. Edit kernel code in `include/flashinfer/attention/generic/some_kernel.cuh`
2. Run test: `pytest tests/rocm_tests/test_some_kernel_hip.py::test_specific_case`
3. FlashInfer detects changes and recompiles automatically
4. No `pip install` needed!

### Modifying Existing Kernels

- **Kernel templates**: `include/flashinfer/**/*.cuh` — changes picked up on next JIT compile
- **Launcher code**: `flashinfer/csrc_rocm/*.cu` — may need changes if adding new template parameters
- **Jinja templates**: `flashinfer/csrc_rocm/*.jinja` — update if adding new config parameters
- **Python API**: `flashinfer/*.py` — update if changing function signatures

## Build System Details

- **Build backend**: `scikit-build-core` (PEP 517) with CMake
- **Compiler**: `hipcc` (invoked by ninja through PyTorch's `cpp_extension`)
- **Version**: `<upstream_version>+amd.<n>` (e.g., `0.2.5+amd.2`)

## Supported GPU Architectures

FlashInfer+ROCm currently supports and actively tests on:

| Architecture | GPU family | Notes |
|---|---|---|
| `gfx942` | CDNA3 (MI300X, MI325X) | Primary target, AITER backend available |
| `gfx950` | CDNA4 (MI355X) | Supported |

## External Documentation Resources

### Core Dependencies

- **ROCm / HIP**: [ROCm documentation](https://rocm.docs.amd.com/)
- **HIP ISA**: [CDNA ISA reference guides](https://rocm.docs.amd.com/projects/llvm-project/en/latest/)
- **Composable Kernel (CK)**: [CK repository](https://github.com/ROCmSoftwarePlatform/composable_kernel) — reference for LDS layout, sched_group_barrier patterns, and tiling strategies on CDNA
- **AITER**: [AITER repository](https://github.com/ROCm/aiter) — AMD's optimized attention library; useful as a performance reference for gfx942
- **HipKittens**: [HipKittens paper (arxiv 2511.08083)](https://arxiv.org/abs/2511.08083) — flash-attention on AMD; notes that producer/consumer patterns underperform on CDNA, recommends 4-wave interleave

### When to Consult These Docs

- **Understanding CDNA3 LDS/MFMA behavior** → CK source (`qr_ks_vs.hpp`) and CDNA3 ISA guide
- **Checking optimal sched_group_barrier ratios** → CK `qr_ks_vs` for the specific hdim/dtype combination
- **Benchmarking against a reference** → AITER `mha_fwd` for fp16/hdim=256 attention
- **Writing inline assembly** → CDNA ISA for instruction syntax and operand constraints

## Release Versioning

FlashInfer+ROCm follows `<upstream_version>+amd.<n>` (e.g., `0.2.5+amd.2`), tying each release to the corresponding upstream FlashInfer tag.

## Plan Files

When a plan is created and approved, save it to the Claude Code project memory directory for this repo (visible via `/memory` in Claude Code):

**Naming:** `plan_<short_descriptive_slug>.md` — use the subject of the plan, not a random name.
Examples: `plan_mla_hip_port.md`, `plan_rope_cdna3_tuning.md`, `plan_pod_hip_port.md`

**Index:** After saving the file, add a one-line entry to `MEMORY.md` in that same directory:
`- [Plan: <title>](plan_<slug>.md) — <one-line summary>`

This ensures every plan is findable by name across sessions without being committed to git.

---

> Because practical engineering involves the accumulated experience of trial and error, match the coding style, efficiency, complexity, and defensiveness by learning from existing code as much as possible. Document intentional departures with rationale. For performance-critical hot paths, leave justification for special algorithmic choices and alternatives considered in a comment for review.
>
> **Keep documentation in sync with code changes:** When modifying code referenced in this document, update the corresponding documentation immediately. New patterns or conventions should be documented; deprecated approaches should be removed or marked deprecated.
