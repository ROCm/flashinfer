<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/flashinfer-ai/web-data/blob/main/logo/FlashInfer-black-background.png?raw=true">
    <img alt="FlashInfer" src="https://github.com/flashinfer-ai/web-data/blob/main/logo/FlashInfer-white-background.png?raw=true" width=55%>
  </picture>
</p>
<h1 align="center">
Kernel Library for LLM Serving
</h1>

<p align="center">
| <a href="https://flashinfer.ai"><b>Blog</b></a> | <a href="https://docs.flashinfer.ai"><b>Documentation</b></a> | <a href="https://join.slack.com/t/flashinfer/shared_invite/zt-379wct3hc-D5jR~1ZKQcU00WHsXhgvtA"><b>Slack</b></a> |  <a href="https://github.com/orgs/flashinfer-ai/discussions"><b>Discussion Forum</b></a> |
</p>

[![Build Status](https://ci.tlcpack.ai/job/flashinfer-ci/job/main/badge/icon)](https://ci.tlcpack.ai/job/flashinfer-ci/job/main/)
[![Documentation](https://github.com/flashinfer-ai/flashinfer/actions/workflows/build-doc.yml/badge.svg)](https://github.com/flashinfer-ai/flashinfer/actions/workflows/build-doc.yml)

FlashInfer is a library and kernel generator for Large Language Models that provides high-performance implementation of LLM GPU kernels such as FlashAttention, SparseAttention, PageAttention, Sampling, and more. FlashInfer focuses on LLM serving and inference, and delivers state-of-the-art performance across diverse scenarios.

Check our [v0.2 release blog](https://flashinfer.ai/2024/12/16/flashinfer-v02-release.html) for new features!

The core features of FlashInfer include:

| Kernel Type | FP16 / BF16 | FP8 (E4M3, E5M2) | Has AITER backend | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Decode Attention** | ✅ | ✅ | No | Supports MHA, GQA, and MQA |
| **Prefill Attention** | ✅ | WIP | ✅ | Supports MHA, GQA, and MQA |
| **Cascade Attention** | TBD | TBD | No |  Not Yet Ported |
| **MLA** | TBD | TBD | No | Not Yet Ported |
| **POD** | TBD | TBD | No | Not Yet Ported |
| **Positional Encoding** | TBD | TBD | No | Not Yet Ported |
| **Sampling** | ✅ | TBD | No | Supports Top-K/Top-P Sampling/OnlineSoftmax/SamplingFromLogits |
| **Logits Processor** | ✅ | TBD | No |  |
| **Normalization** | ✅ | TBD | No | Supports RMS-Norm/Layer-Norm |

FlashInfer supports PyTorch, TVM and C++ (header-only) APIs, and can be easily integrated into existing projects.

## News

- [Mar 10, 2025] [Blog Post](https://flashinfer.ai/2025/03/10/sampling.html) Sorting-Free GPU Kernels for LLM Sampling, which explains the design of sampling kernels in FlashInfer.
- [Mar 1, 2025] Checkout flashinfer's [intra-kernel profiler](https://github.com/flashinfer-ai/flashinfer/tree/main/profiler) for visualizing the timeline of each threadblock in GPU kernels.
- [Dec 16, 2024] [Blog Post](https://flashinfer.ai/2024/12/16/flashinfer-v02-release.html) FlashInfer 0.2 - Efficient and Customizable Kernels for LLM Inference Serving
- [Sept 2024] We've launched a [Slack](https://join.slack.com/t/flashinfer/shared_invite/zt-2r93kj2aq-wZnC2n_Z2~mf73N5qnVGGA) workspace for Flashinfer users and developers. Join us for timely support, discussions, updates and knowledge sharing!
- [Jan 31, 2024] [Blog Post](https://flashinfer.ai/2024/01/08/cascade-inference.html) Cascade Inference: Memory-Efficient Shared Prefix Batch Decoding
- [Jan 31, 2024] [Blog Post](https://flashinfer.ai/2024/01/03/introduce-flashinfer.html) Accelerating Self-Attentions for LLM Serving with FlashInfer

## Getting Started

Using our PyTorch API is the easiest way to get started:

### Install from PyPI

FlashInfer is available as a Python package for Linux. Install the core package with:

```bash
pip install flashinfer-python
```

**Package Options:**

- **flashinfer-python**: Core package that compiles/downloads kernels on first use
- **flashinfer-cubin**: Pre-compiled kernel binaries for all supported GPU architectures
- **flashinfer-jit-cache**: Pre-built kernel cache for specific CUDA versions

**For faster initialization and offline usage**, install the optional packages to have most kernels pre-compiled:

```bash
pip install flashinfer-python flashinfer-cubin
# JIT cache package (replace cu129 with your CUDA version: cu128, cu129, or cu130)
pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu129
```

This eliminates compilation and downloading overhead at runtime.

### Install from Source

Build the core package from source:

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
python -m pip install -v .
```

**For development**, install in editable mode:

```bash
python -m pip install --no-build-isolation -e . -v
```

**Build optional packages:**

`flashinfer-cubin`:

```bash
cd flashinfer-cubin
python -m build --no-isolation --wheel
python -m pip install dist/*.whl
```

`flashinfer-jit-cache` (customize `FLASHINFER_CUDA_ARCH_LIST` for your target GPUs):

```bash
export FLASHINFER_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a 10.0a 10.3a 11.0a 12.0f"
cd flashinfer-jit-cache
python -m build --no-isolation --wheel
python -m pip install dist/*.whl
```

For more details, see the [Install from Source documentation](https://docs.flashinfer.ai/installation.html#install-from-source).

### Install Nightly Build

Nightly builds are available for testing the latest features:

```bash
# Core and cubin packages
pip install -U --pre flashinfer-python --index-url https://flashinfer.ai/whl/nightly/ --no-deps # Install the nightly package from custom index, without installing dependencies
pip install flashinfer-python  # Install flashinfer-python's dependencies from PyPI
pip install -U --pre flashinfer-cubin --index-url https://flashinfer.ai/whl/nightly/
# JIT cache package (replace cu129 with your CUDA version: cu128, cu129, or cu130)
pip install -U --pre flashinfer-jit-cache --index-url https://flashinfer.ai/whl/nightly/cu129
```

### Verify Installation

After installation, verify that FlashInfer is correctly installed and configured:

```bash
flashinfer show-config
```

This command displays:

- FlashInfer version and installed packages (flashinfer-python, flashinfer-cubin, flashinfer-jit-cache)
- PyTorch and CUDA version information
- Environment variables and artifact paths
- Downloaded cubin status and module compilation status

### Trying it out

```bash
FLASHINFER_HIP_ARCHITECTURES=gfx942 python -m pip install --no-build-isolation -ve.
```

**Note:** The `--no-deps` flag assumes dependencies are pre-installed. Omit it
to download dependencies during build. AOT builds take longer and use more disk
space but avoid JIT compilation at runtime.

### Running Tests

The Python tests suite can be run with pytest:

```bash
# Run default tests (configured in pyproject.toml)
pytest

# Run specific test file
pytest tests/test_decode_kernels_hip.py

# Run with pattern matching
pytest -k "test_decode_kernels_hip"

# Verbose output
pytest -v

# if the system has multiple GPUs, run tests parallely on the multiple GPUS
pytest -n auto # automatically select all available GPUs
pytest -n 4 # Run on the specified number of GPUs
```

The default test configuration is specified in [pyproject.toml](pyproject.toml) under the `testpaths` setting.

## AITER Support

FlashInfer+ROCm has experimental support to use [AITER](https://github.com/ROCm/aiter) as a
backend. The `aiter` backend currently is enabled for the `single_prefill` and `batch_prefill`
kernels. To use AITER as the backend for these kernels, please set `backend=aiter` keyword
argument when invoking the kernels. Additionally, AITER should also be installed on your system and
you may follow one of the below ways to do so.

**Install AITER by building from source**

```bash
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
python3 setup.py develop
```
**Install AITER wheel package from https://pypi.amd.com/simple/**

```bash
pip install amd-aiter --index-url https://pypi.amd.com/simple/
```

### Known Limitations:

The AITER backed only supports `NHD` kv_layout.

### Single Prefill Example

This section provides an example on how to use Single Prefill with AITER

```python
import torch
import flashinfer

kv_len = 2048
num_kv_heads = 32
head_dim = 128

k = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)
v = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)

  # NHD Layout
  k = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0",dtype=torch.float16)
  v = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16)

num_qo_heads = 32
q = torch.randn(num_qo_heads, head_dim).half().to(0)

  # Call flashinfer API
  logits_soft_cap = logits_soft_cap if logits_soft_cap > 0 else None
  if return_lse:
    o, lse = flashinfer.single_prefill_with_kv_cache_return_lse(
        q,
        k,
        v,
        causal=causal,
        kv_layout=kv_layout,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
        backend="aiter" # Pass the backend = aiter flag to enable # AITER computation
    )
    print(f"  FlashInfer output shape: {o.shape}, LSE shape: {lse.shape}")

  else:
    o = flashinfer.single_prefill_with_kv_cache(
        q,
        k,
        v,
        causal=causal,
        kv_layout=kv_layout,
        pos_encoding_mode=pos_encoding_mode,
        logits_soft_cap=logits_soft_cap,
        backend="aiter" # Pass the backend = aiter flag to enable # AITER computation
    )
    print(f"  FlashInfer output shape: {o.shape}")

# append attention
append_qo_len = 128
q = torch.randn(append_qo_len, num_qo_heads, head_dim).half().to(0) # append attention, the last 128 tokens in the KV-Cache are the new tokens
o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True) # append attention without RoPE on-the-fly, apply causal mask
o_rope_on_the_fly = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, pos_encoding_mode="ROPE_LLAMA") # append attention with LLaMA style RoPE on-the-fly, apply causal mask

# prefill attention
qo_len = 2048
q = torch.randn(qo_len, num_qo_heads, head_dim).half().to(0) # prefill attention
o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=False) # prefill attention without RoPE on-the-fly, do not apply causal mask
```
