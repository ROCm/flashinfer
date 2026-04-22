---
name: add-rocm-kernel
description: Step-by-step tutorial for adding new HIP/ROCm kernels to FlashInfer+ROCm (amd-flashinfer)
---

# Tutorial: Adding a New Kernel to FlashInfer+ROCm

This tutorial walks through adding a simple element-wise scale operation to the ROCm port of
FlashInfer. We implement `scale(x, factor) = x * factor` to illustrate the complete workflow on
CDNA3 (`gfx942`) and CDNA4 (`gfx950`).

If you are used to upstream's `add-cuda-kernel` tutorial, note the following ROCm-specific
differences up front:

| Concern | Upstream CUDA | This ROCm port |
| --- | --- | --- |
| Launcher directory | `csrc/` | [`flashinfer/csrc_rocm/`](../../../flashinfer/csrc_rocm/) |
| Bindings | TVM-FFI (`TVM_FFI_DLL_EXPORT_TYPED_FUNC`) | Plain Torch extension (`TORCH_LIBRARY_FRAGMENT`) |
| Tensor type | `tvm::ffi::TensorView` | `at::Tensor` |
| Stream | `get_stream(device)` | `at::hip::getCurrentHIPStream()` |
| Compiler | `nvcc` | `hipcc` (amdclang++) |
| Arch env var | `FLASHINFER_CUDA_ARCH_LIST` | `FLASHINFER_ROCM_ARCH_LIST` |
| AOT registration | `flashinfer/aot.py` | [`flashinfer/aot_hip.py`](../../../flashinfer/aot_hip.py) |
| Tests directory | `tests/` | [`tests/rocm_tests/`](../../../tests/rocm_tests/) |

## Goal

Add a new operation that scales each element of a tensor by a scalar factor:

- Input: tensor `x` and scalar `factor`
- Output: `x * factor` (element-wise)
- Support FP16 and BF16
- Compile for both `gfx942` and `gfx950`

## Step 1: Define the HIP kernel in `include/`

Create `include/flashinfer/scale.cuh`. **Do not include `<torch/...>` headers here.** The file
must stay framework-agnostic so the same header can compile under CUDA (upstream) and HIP (this
port). For anything that differs between the two platforms, reach for
[`include/gpu_iface/`](../../../include/gpu_iface/).

```cpp
#pragma once

#include "gpu_iface/platform.hpp"
#include "gpu_iface/gpu_runtime_compat.hpp"
#include "gpu_iface/vec_dtypes.hpp"

namespace flashinfer {

/*!
 * \brief Element-wise scale kernel.
 * \tparam T Data type (half / __hip_bfloat16 / float)
 */
template <typename T>
__global__ void ScaleKernel(const T* __restrict__ input, T* __restrict__ output,
                            T factor, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = input[idx] * factor;
  }
}

/*!
 * \brief Launch scale kernel (platform-agnostic).
 */
template <typename T>
gpuError_t ScaleLauncher(const T* input, T* output, T factor, int n,
                         gpuStream_t stream = nullptr) {
  const int threads = 256;
  const int blocks  = (n + threads - 1) / threads;

  ScaleKernel<T><<<blocks, threads, 0, stream>>>(input, output, factor, n);

  return gpuGetLastError();
}

}  // namespace flashinfer
```

**Key points:**

- No `<cuda_runtime.h>` / `<hip/hip_runtime.h>` includes — these are pulled in transitively by
  `gpu_iface/platform.hpp` based on whether the TU is being compiled for CUDA or HIP.
- `gpuError_t`, `gpuStream_t`, and `gpuGetLastError()` come from
  [`include/gpu_iface/gpu_runtime_compat.hpp`](../../../include/gpu_iface/gpu_runtime_compat.hpp)
  and alias to either the CUDA or HIP symbols depending on the backend macro.
- `__global__` and the `<<<...>>>` launch syntax are supported on both HIP and CUDA without
  translation.
- Template on dtype; the concrete dtype is selected in the launcher via a dispatch macro.

### When to add something to `gpu_iface`

If your kernel needs a primitive that differs meaningfully between CUDA and HIP (an MMA
intrinsic, a cross-lane shuffle, a memory fence, a warp-wide reduction, a dtype container), add
it to the appropriate `include/gpu_iface/backend/{cuda,hip}/` file and expose a shared name from
the top-level `gpu_iface/` header — do **not** duplicate the whole kernel under `csrc_rocm/`.

Representative HIP-side files already in use:

- [`include/gpu_iface/backend/hip/vec_dtypes_hip.h`](../../../include/gpu_iface/backend/hip/vec_dtypes_hip.h)
- [`include/gpu_iface/backend/hip/mma_hip.h`](../../../include/gpu_iface/backend/hip/mma_hip.h)
- [`include/gpu_iface/backend/hip/memory_ops_hip.h`](../../../include/gpu_iface/backend/hip/memory_ops_hip.h)
- [`include/gpu_iface/backend/hip/math_hip.h`](../../../include/gpu_iface/backend/hip/math_hip.h)

## Step 2: Create the launcher in `flashinfer/csrc_rocm/`

Create `flashinfer/csrc_rocm/scale.cu`. This is the file that bridges PyTorch tensors to the
framework-agnostic kernel above.

```cpp
#include <cstdint>
#include <flashinfer/scale.cuh>

#include "pytorch_extension_utils.h"

using namespace flashinfer;

void scale(at::Tensor& output, at::Tensor& input, double factor) {
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  TORCH_CHECK(input.sizes() == output.sizes(),
              "scale: output shape must match input shape");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(),
              "scale: output dtype must match input dtype");

  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(input.device());
  const hipStream_t stream = at::hip::getCurrentHIPStream();
  const int n = static_cast<int>(input.numel());

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, [&] {
    hipError_t status = ScaleLauncher<c_type>(
        static_cast<c_type*>(input.data_ptr()),
        static_cast<c_type*>(output.data_ptr()),
        static_cast<c_type>(factor),
        n,
        stream);
    TORCH_CHECK(status == hipSuccess,
                "scale failed: " + std::string(hipGetErrorString(status)));
    return true;
  });
}
```

**Key points:**

- Include [`pytorch_extension_utils.h`](../../../flashinfer/csrc_rocm/pytorch_extension_utils.h)
  for `at::Tensor`, the `CHECK_*` macros, and the `DISPATCH_PYTORCH_DTYPE_TO_CTYPE_*` family.
- Use `c10::hip::OptionalHIPGuardMasqueradingAsCUDA` — PyTorch's ROCm build "masquerades" as
  CUDA, so device guards and streams are exposed through the HIP-prefixed namespaces.
- Acquire the current HIP stream with `at::hip::getCurrentHIPStream()`, not
  `c10::cuda::getCurrentCUDAStream()`.
- Dispatch macro: `DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16` covers FP16+BF16. For a dispatch that
  also covers FP32 or FP8, use the other `DISPATCH_PYTORCH_DTYPE_TO_CTYPE_*` variants defined
  in `pytorch_extension_utils.h`.
- Error handling uses `TORCH_CHECK(cond, msg)` — the PyTorch extension idiom. There is no
  `TVM_FFI_THROW` on this path.

### Validation helpers available

From [`flashinfer/csrc_rocm/pytorch_extension_utils.h`](../../../flashinfer/csrc_rocm/pytorch_extension_utils.h):

- `CHECK_INPUT(tensor)` — validates CUDA/HIP + contiguous.
- `CHECK_LAST_DIM_CONTIGUOUS_INPUT(tensor)` — validates CUDA/HIP + last-dim-contiguous.
- `CHECK_EQ(a, b)`, `CHECK_DIM(n, tensor)` — shape / rank sanity checks.
- `DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16` — FP16 + BF16
- `DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16_FP32` — FP16 + BF16 + FP32
- `DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8` — E4M3 + E5M2 (the `_fnuz` variants on CDNA3/4)

For a worked-out reference, read [`flashinfer/csrc_rocm/norm.cu`](../../../flashinfer/csrc_rocm/norm.cu)
(kept intentionally simple) and compare against the more involved
[`flashinfer/csrc_rocm/batch_prefill.cu`](../../../flashinfer/csrc_rocm/batch_prefill.cu) (plan-run
pattern, multiple backends, FP8 path).

## Step 3: Create the Torch-extension binding

Create `flashinfer/csrc_rocm/flashinfer_scale_binding.cu`. This is the file that exports the
launcher to Python.

```cpp
#include "pytorch_extension_utils.h"

void scale(at::Tensor& output, at::Tensor& input, double factor);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  // Element-wise scale: output = input * factor
  m.def("scale", scale);
}
```

**Key points:**

- The `TORCH_EXTENSION_NAME` macro is defined by PyTorch's build system and resolves to the
  unique module name for this JIT build — `TORCH_LIBRARY_FRAGMENT` registers `scale` under that
  namespace.
- `pytorch_extension_utils.h` also emits a `PyInit_<name>` stub so the resulting `.so` is
  importable as a Python module (see the bottom of
  [`pytorch_extension_utils.h`](../../../flashinfer/csrc_rocm/pytorch_extension_utils.h)).
- Compare with [`flashinfer/csrc_rocm/flashinfer_norm_binding.cu`](../../../flashinfer/csrc_rocm/flashinfer_norm_binding.cu)
  for the exact pattern.

**Do not write:**

- `TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, scale)` — that's the upstream TVM-FFI pattern; it does
  not work on this port.
- `PYBIND11_MODULE(...)` — we use the `TORCH_LIBRARY_FRAGMENT` flavor which integrates with
  `torch.library` and thus with `torch.compile`.

## Step 4: (Optional) Jinja type specialization

For operations that benefit from compile-time type specialization (you want one `.so` per dtype
combination rather than runtime dispatch), add a Jinja template next to the launcher:

`flashinfer/csrc_rocm/scale_customize_config.jinja`:

```jinja
#pragma once

using DTypeIn  = {{ dtype_in }};
using DTypeOut = {{ dtype_out }};
constexpr int SCALE_BLOCK_SIZE = {{ block_size }};
```

The JIT module generator (Step 5) renders this to a concrete `.inc` file before invoking
`hipcc`. See [`flashinfer/csrc_rocm/batch_prefill_customize_config.jinja`](../../../flashinfer/csrc_rocm/batch_prefill_customize_config.jinja)
for a non-trivial example.

**When to skip Jinja:** for a kernel like our `scale` example, where the dtype is picked via
`DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16` at runtime, there is no benefit. Skip this step entirely.

## Step 5: Write the JIT module generator

Create `flashinfer/jit/scale.py`:

```python
"""
Copyright (c) 2026 by FlashInfer+ROCm team.
SPDX-License-Identifier: Apache-2.0
"""

from . import env as jit_env
from .core import JitSpec, gen_jit_spec


def gen_scale_module() -> JitSpec:
    """JitSpec for the element-wise scale op.

    No Jinja / type specialization is needed here because the dtype dispatch
    happens inside DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16 at runtime.
    """
    extra_flags = [
        "-DENABLE_BF16",
    ]
    return gen_jit_spec(
        "scale",
        [
            jit_env.FLASHINFER_CSRC_DIR / "scale.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_scale_binding.cu",
        ],
        extra_cuda_cflags=extra_flags,
    )
```

**Key points:**

- `jit_env.FLASHINFER_CSRC_DIR` resolves to `flashinfer/csrc_rocm/` on HIP, via
  [`flashinfer/get_include_paths.py::get_csrc_dir()`](../../../flashinfer/get_include_paths.py).
  This is a conscious divergence from upstream — do **not** reach for a hard-coded `csrc/`.
- `extra_cuda_cflags` is still the kwarg name even on HIP (for source-compat with upstream);
  internally [`flashinfer/jit/core.py`](../../../flashinfer/jit/core.py) maps it to flags passed
  to `hipcc`.
- `gen_jit_spec` on HIP automatically prepends the output of
  `current_compilation_context.get_hipcc_flags_list()` — that is, `--offload-arch=gfxNNN` for
  every target arch plus the common HIP defines (`-DFLASHINFER_ENABLE_HIP`, etc.). You do not
  need to add `--offload-arch` yourself unless you are overriding a built-in default.
- If your kernel must **only** run on one arch, add a runtime check (e.g. via
  `FLASHINFER_SUPPORTED_ROCM_ARCHS` in [`flashinfer/hip_utils.py`](../../../flashinfer/hip_utils.py))
  at the Python API layer. There is no HIP-side equivalent of upstream's
  `supported_major_versions=[...]` mechanism yet.

### Register the generator for re-export

Add the import to the `IS_HIP` branch of
[`flashinfer/jit/__init__.py`](../../../flashinfer/jit/__init__.py):

```python
elif IS_HIP:
    # ...
    from .scale import gen_scale_module as gen_scale_module
```

Place it alphabetically among the existing `from .norm import ...`, `from .rope import ...`
lines.

## Step 6: Write the Python API

Create `flashinfer/scale.py`:

```python
"""
Copyright (c) 2026 by FlashInfer+ROCm team.
SPDX-License-Identifier: Apache-2.0
"""

import functools
from typing import Optional

import torch

from .jit.scale import gen_scale_module


@functools.cache
def _get_scale_module():
    """Compile + load the scale module exactly once per process."""
    return gen_scale_module().build_and_load()


def scale(
    input: torch.Tensor,
    factor: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Element-wise ``output = input * factor``.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor on an AMD GPU. Must be FP16 or BF16 and contiguous.
    factor : float
        Scalar multiplier.
    out : Optional[torch.Tensor]
        Pre-allocated output tensor. If ``None``, a new tensor is allocated.

    Returns
    -------
    torch.Tensor
        ``input * factor`` with the same shape/dtype/device as ``input``.

    Examples
    --------
    >>> import torch, flashinfer
    >>> x = torch.randn(1024, dtype=torch.float16, device="cuda")
    >>> y = flashinfer.scale(x, 2.0)
    >>> torch.allclose(y, x * 2.0)
    True
    """
    if out is None:
        out = torch.empty_like(input)

    module = _get_scale_module()
    module.scale(out, input, float(factor))
    return out
```

**Key points:**

- `@functools.cache` caches the compiled module in memory so subsequent calls skip the JIT
  cache lookup entirely.
- **Destination-passing style**: accept an optional `out=` so perf-sensitive callers can avoid
  an extra allocation.
- On ROCm, `input.device.type == "cuda"` — PyTorch's ROCm build reuses the CUDA namespace. Do
  not test for `"hip"`; it will never be true in practice.
- If you want API logging, add `@flashinfer_api` above `def scale(...)`. See the
  [`debug-rocm-crash`](../debug-rocm-crash/SKILL.md) skill.

### Expose from the package

Add the export to the `IS_HIP` branch of
[`flashinfer/__init__.py`](../../../flashinfer/__init__.py):

```python
elif IS_HIP:
    # ...
    from .scale import scale as scale
```

## Step 7: Write tests

Create `tests/rocm_tests/test_scale_hip.py`:

```python
"""
Copyright (c) 2026 by FlashInfer+ROCm team.
SPDX-License-Identifier: Apache-2.0
"""

import pytest
import torch

import flashinfer
from flashinfer.hip_utils import FLASHINFER_SUPPORTED_ROCM_ARCHS


def _current_arch() -> str:
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1024,), (32, 128), (8, 32, 128)])
@pytest.mark.parametrize("factor", [0.5, 1.0, 2.0, -3.25])
def test_scale_correctness(shape, dtype, factor):
    assert _current_arch() in FLASHINFER_SUPPORTED_ROCM_ARCHS, (
        "Test requires a FlashInfer-supported AMD GPU"
    )

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    y = flashinfer.scale(x, factor)

    ref = x.float() * factor
    torch.testing.assert_close(y.float(), ref, rtol=1e-2, atol=1e-2)


def test_scale_inplace_out():
    x = torch.randn(64, 64, dtype=torch.float16, device="cuda")
    out = torch.empty_like(x)
    y = flashinfer.scale(x, 3.0, out=out)

    assert y.data_ptr() == out.data_ptr()
    torch.testing.assert_close(y.float(), x.float() * 3.0, rtol=1e-2, atol=1e-2)
```

**Key points:**

- Test files under [`tests/rocm_tests/`](../../../tests/rocm_tests/) are named `test_*_hip.py`
  by convention.
- The repo's [`tests/rocm_tests/conftest.py`](../../../tests/rocm_tests/conftest.py) hooks into
  `pytest-xdist` so `pytest -n auto` only spawns workers for
  FlashInfer-supported GPUs. You do not need to parametrize over devices yourself.
- Use FP32 for reference math to avoid dtype-mismatch asserts with `assert_close`.
- Keep tolerances loose enough for BF16 (`rtol=1e-2`, `atol=1e-2`); tighten for FP32-only ops.

Run it:

```bash
pytest tests/rocm_tests/test_scale_hip.py -v
# Or only on GPU 0
HIP_VISIBLE_DEVICES=0 pytest tests/rocm_tests/test_scale_hip.py -v
```

## Step 8: Register for AOT (optional)

If your op should also be available in pre-compiled wheels (the
[`amd-flashinfer-jit-cache/`](../../../amd-flashinfer-jit-cache/) package), register the JIT
generator in [`flashinfer/aot_hip.py`](../../../flashinfer/aot_hip.py). Add a generator that
yields your `JitSpec`, and reference it from the main AOT-compile loop.

Pattern (see existing `gen_fa2` in that file):

```python
def gen_scale() -> Iterator:
    from .jit.scale import gen_scale_module
    yield gen_scale_module()
```

Then AOT compile with:

```bash
cd amd-flashinfer-jit-cache
export FLASHINFER_ROCM_ARCH_LIST="gfx942,gfx950"
python -m build --no-isolation --wheel
```

The resulting wheel ships a pre-compiled `.so` per arch, indexed by the URI hash.

## CDNA3 vs CDNA4 — what to watch for

Both `gfx942` (CDNA3, MI300X/MI325X) and `gfx950` (CDNA4, MI350X/MI355X) are Matrix Core
architectures, but they are not fully compatible:

| Concern | CDNA3 (`gfx942`) | CDNA4 (`gfx950`) |
| --- | --- | --- |
| MFMA intrinsics | `__builtin_amdgcn_mfma_*` family (F16, BF16, I8, FP8) | Same family **plus** new CDNA4-only instructions (wider FP8 MFMAs, additional block sizes) |
| FP8 format | `__hip_fp8_e4m3_fnuz`, `__hip_fp8_e5m2_fnuz` (FNUZ biasing) | Same FNUZ variants (OCP FP8 support depends on ROCm version) |
| LDS capacity | 64 KB / CU | 160 KB / XCD on some configs — **do not** assume identical block/tile sizes |
| Wavefront size | 64 | 64 |

Practical implications when authoring a new kernel:

- If you use MFMA intrinsics, guard them on the arch macro (`__gfx942__`, `__gfx950__`) or
  behind the `FLASHINFER_SUPPORTED_ROCM_ARCHS` check at the Python level.
- Do not hard-code LDS tile sizes. Either parameterize the kernel (Jinja) or query the device
  properties at plan time (e.g. `torch.cuda.get_device_properties(dev).shared_memory_per_block`).
- FP8: on both arches, the `_fnuz` variants are the safe default. Bit-exact parity with NVIDIA
  `__nv_fp8_e4m3` is **not** guaranteed — reference tests must account for the FNUZ
  representation.

When in doubt, look at how
[`flashinfer/prefill_rocm.py`](../../../flashinfer/prefill_rocm.py) and
[`flashinfer/csrc_rocm/batch_prefill.cu`](../../../flashinfer/csrc_rocm/batch_prefill.cu) handle
per-arch specialization.

## Reference implementations in this repo

| Complexity | Files |
| --- | --- |
| Simple, no Jinja | [`flashinfer/norm.py`](../../../flashinfer/norm.py) + [`flashinfer/csrc_rocm/norm.cu`](../../../flashinfer/csrc_rocm/norm.cu) + [`flashinfer/csrc_rocm/flashinfer_norm_binding.cu`](../../../flashinfer/csrc_rocm/flashinfer_norm_binding.cu) + [`flashinfer/jit/norm.py`](../../../flashinfer/jit/norm.py) |
| Moderate, with Jinja | [`flashinfer/csrc_rocm/single_prefill.cu`](../../../flashinfer/csrc_rocm/single_prefill.cu) + [`flashinfer/csrc_rocm/single_prefill_customize_config.jinja`](../../../flashinfer/csrc_rocm/single_prefill_customize_config.jinja) + [`flashinfer/csrc_rocm/single_prefill_kernel_inst.jinja`](../../../flashinfer/csrc_rocm/single_prefill_kernel_inst.jinja) |
| Complex (plan-run, AITER, FP8) | [`flashinfer/prefill_rocm.py`](../../../flashinfer/prefill_rocm.py) + [`flashinfer/csrc_rocm/batch_prefill.cu`](../../../flashinfer/csrc_rocm/batch_prefill.cu) |

## Summary checklist

When adding a new op, verify each box:

- [ ] Header in `include/flashinfer/` — no Torch/HIP-runtime includes; uses `gpu_iface/` for
      platform-differing primitives.
- [ ] Launcher in `flashinfer/csrc_rocm/<name>.cu` with `#include "pytorch_extension_utils.h"`,
      `at::Tensor` inputs, `at::hip::getCurrentHIPStream()`, and a `DISPATCH_PYTORCH_DTYPE_*`
      block.
- [ ] Binding in `flashinfer/csrc_rocm/flashinfer_<name>_binding.cu` using
      `TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m)`.
- [ ] (Optional) Jinja template for type specialization.
- [ ] JIT generator in `flashinfer/jit/<name>.py` returning a `JitSpec` via `gen_jit_spec`.
- [ ] Import exposed from the `IS_HIP` branches of `flashinfer/jit/__init__.py` **and**
      `flashinfer/__init__.py`.
- [ ] Python API with `@functools.cache`, destination-passing style, FP16/BF16 support,
      and optional `@flashinfer_api`.
- [ ] Tests in `tests/rocm_tests/test_<name>_hip.py`.
- [ ] (Optional) AOT registration in `flashinfer/aot_hip.py`.
- [ ] Run `pre-commit run -a` before committing.

## Related documentation

- [`CLAUDE.md`](../../../CLAUDE.md) — project overview, JIT architecture, feature matrix.
- [`.claude/skills/benchmark-kernel/SKILL.md`](../benchmark-kernel/SKILL.md) — how to benchmark
  the kernel you just added.
- [`.claude/skills/debug-rocm-crash/SKILL.md`](../debug-rocm-crash/SKILL.md) — debugging recipes
  when `TORCH_CHECK` fires or the GPU faults.
- Upstream's [`add-cuda-kernel` skill](https://github.com/flashinfer-ai/flashinfer/blob/main/.claude/skills/add-cuda-kernel/SKILL.md)
  — the source this tutorial was adapted from. Useful when you are porting a kernel from
  upstream CUDA and want to see the "before" picture.
