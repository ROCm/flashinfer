# FlashInfer+ROCm: An AMD ROCm port of FlashInfer

FlashInfer+ROCm is an AMD ROCm port of the [FlashInfer](https://github.com/flashinfer-ai/flashinfer) library of fast attention, RoPE, RMSNorm, sampling, and logits-processor kernels for LLM inference on AMD Instinct GPUs. This README is aimed at library consumers: developers embedding FlashInfer kernels into their own training or serving stack.

**Status:** Active development, attention (single/batch prefill and decode) is the primary focus. See [CHANGELOG.md](CHANGELOG.md) for the full history.

**Versioning:** Release tags use the form `<upstream_version>+amd.<n>` (for example, `0.5.3+amd.1` is the first AMD release based on upstream `v0.5.3`).

## Minimal usage

```python
import torch
import flashinfer

# Device is still "cuda" on PyTorch+ROCm.
q = torch.randn(1024, 32, 128, dtype=torch.float16, device="cuda")
k = torch.randn(1024,  8, 128, dtype=torch.float16, device="cuda")  # GQA 4:1
v = torch.randn(1024,  8, 128, dtype=torch.float16, device="cuda")

# Default backend = "fa2". Use backend="aiter" or "fa3_cdna3" to switch.
o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
```

## Table of Contents

* [Feature Support Matrix](#feature-support-matrix)
* [GPU, ROCm, and PyTorch Support](#gpu-rocm-and-pytorch-support)
* [Getting Started](#getting-started)
  * [Option 1: Pre-built Docker Image](#option-1-pre-built-docker-image)
  * [Option 2: Install from a Wheel Package](#option-2-install-from-a-wheel-package)
  * [Running the Examples](#running-the-examples)
* [Build from Source](#build-from-source)
  * [Development Environment](#development-environment)
  * [Building and Installing a Wheel](#building-and-installing-a-wheel)
  * [Running Tests](#running-tests)
* [Prefill Backends](#prefill-backends)
* [Contributing and License](#contributing-and-license)

## Feature Support Matrix

| Kernel | FP16 / BF16 | FP8 (E4M3, E5M2) | Backends | Notes |
| :--- | :---: | :---: | :--- | :--- |
| **Decode attention** | Yes | Yes | `fa2` | MHA, GQA, MQA |
| **Prefill attention** | Yes | WIP | `fa2`, `aiter`, `fa3_cdna3` | MHA, GQA, MQA |
| **RoPE** (incl. Llama 3.1, fused RoPE+FP8+paged-KV append) | Yes | - | `fa2` | |
| **RMSNorm / LayerNorm / Gemma variants** | Yes | - | `fa2` | |
| **Sampling** | Yes | - | `fa2` | Top-K, Top-P, OnlineSoftmax, SamplingFromLogits |
| **Logits processor** | Yes | - | `fa2` | |
| **Quantization** (`packbits`, `segment_packbits`) | Yes | - | `fa2` | |
| Cascade, MLA, POD, PosEncoding-mode variants | - | - | - | Not yet ported |

## GPU, ROCm, and PyTorch Support

**GPU architectures:** gfx942 (CDNA3 — MI300X, MI325X), gfx950 (CDNA4 — MI355X).

**ROCm:** 7.0.2, 7.1.1, 7.2.

**PyTorch+ROCm:** 2.8.0, 2.9.1. Install the matching wheel from `repo.radeon.com`:

```bash
pip install torch==2.9.1 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2
```

Other versions may work but are not tested. Replace `7.2` with the ROCm version you need; see <https://repo.radeon.com/rocm/manylinux/> for the full list.

## Getting Started

### Option 1: Pre-built Docker Image

AMD validates and publishes [FlashInfer images](https://hub.docker.com/r/rocm/flashinfer/tags) on Docker Hub:

| Docker image | ROCm | FlashInfer | PyTorch | Ubuntu | Python | GPU |
| --- | --- | --- | --- | --- | --- | --- |
| `rocm/flashinfer:flashinfer-0.5.3.amd1_rocm7.2_ubuntu24.04_py3.12_pytorch2.9.1` | 7.2.0 | v0.5.3 | 2.9.1 | 24.04 | 3.12 | MI355X, MI325X, MI300X |
| `rocm/flashinfer:flashinfer-0.5.3.amd1_rocm7.0.2_ubuntu24.04_py3.12_pytorch2.9.1` | 7.0.2 | v0.5.3 | 2.9.1 | 24.04 | 3.12 | MI355X, MI325X, MI300X |
| `rocm/flashinfer:flashinfer-0.2.5.amd2_rocm7.1.1_ubuntu24.04_py3.12_pytorch2.8` | 7.1.1 | v0.2.5 | 2.8.0 | 24.04 | 3.12 | MI325X, MI300X |

Start a container:

```bash
docker run -it --privileged --network=host --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --shm-size 128G --name=<container-name> <docker-image-tag>
```

Verify:

```bash
micromamba activate base  # env name may vary per image
python -c "import flashinfer; print(flashinfer.__version__)"
# expected: 0.5.3+amd.1
```

### Option 2: Install from a Wheel Package

```bash
pip install amd-flashinfer --index-url https://pypi.amd.com/simple/
pip install torch==2.9.1 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2
```

> `torch` is deliberately not a declared dependency because the ROCm wheel must come from `repo.radeon.com`, not PyPI. Installing without `-f` will pull a non-ROCm build.

### Running the Examples

```bash
wget https://raw.githubusercontent.com/ROCm/flashinfer/amd-integration/examples/single_prefill_example.py
wget https://raw.githubusercontent.com/ROCm/flashinfer/amd-integration/examples/batch_prefill_example.py
wget https://raw.githubusercontent.com/ROCm/flashinfer/amd-integration/examples/batch_decode_example.py
python single_prefill_example.py
```

An end-to-end recommendation-system notebook that exercises the full public API is also available at [`examples/recommendation_system_flashinfer_rocm.ipynb`](examples/recommendation_system_flashinfer_rocm.ipynb).

## Build from Source

### Development Environment

```bash
docker build \
  --build-arg ROCM_VERSION=7.2 \
  --build-arg PY_VERSION=3.12 \
  --build-arg TORCH_VERSION=2.9.1 \
  --build-arg USERNAME=$USER \
  --build-arg USER_UID=$(id -u) \
  --build-arg USER_GID=$(id -g) \
  -t flashinfer-0.5.3.amd1_rocm7.2_ubuntu24.04_py3.12_pytorch2.9.1 \
  -f .devcontainer/rocm/Dockerfile .

docker run -it \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --privileged --shm-size=128G --network=host \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  -v $PWD:/workspace --name flashinfer-dev-container \
  flashinfer-0.5.3.amd1_rocm7.2_ubuntu24.04_py3.12_pytorch2.9.1
```

### Building and Installing a Wheel

```bash
# Editable install (JIT kernels compile on first use)
python -m pip install --no-build-isolation -ve .

# Wheel build
python -m pip wheel . --wheel-dir=./dist/ --no-deps --no-build-isolation -v
pip install dist/amd_flashinfer-*.whl
```

### Running Tests

```bash
pytest                               # curated set from pyproject.toml
pytest tests/rocm_tests              # AMD-authored HIP tests
pytest -n auto                       # across all visible GPUs
pytest -k "test_batch_decode_kernels_hip"
```

The default test set is pinned in `[tool.pytest.ini_options]` in [pyproject.toml](pyproject.toml).

## Prefill Backends

All prefill entry points accept a `backend=` keyword (default `"auto"`, which resolves to `"fa2"`).

- **`fa2` (default)** — In-tree HIP port of FlashAttention-2. Broadest coverage: paged + ragged, single + batch, fp16/bf16.
- **`aiter`** — Wraps [AITER](https://github.com/ROCm/aiter) CK FMHA (`flash_attn_varlen_func`, `mha_batch_prefill_func`). NHD layout only; paged batch-prefill `page_size ∈ {16, 1024}` (or `{128, 256, 1024}` on `amd-aiter==0.1.10`).
- **`fa3_cdna3`** — MI300X-optimized single-prefill kernel for chunked prefill, `head_dim=256`, `q_len != kv_len`. Experimental; see [`benchmarks/rocm_benchmarks/bench_fa3_cdna3.py`](benchmarks/rocm_benchmarks/bench_fa3_cdna3.py) and [`examples/single_prefill_example.py`](examples/single_prefill_example.py).

Install AITER if you plan to use `backend="aiter"` outside the prebuilt Docker image:

```bash
pip install amd-aiter --index-url https://pypi.amd.com/simple/
# or: git clone --recursive https://github.com/ROCm/aiter.git && cd aiter && python3 setup.py develop
```

Example:

```python
o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, backend="aiter")
```

## Contributing and License

See [CONTRIBUTING.md](CONTRIBUTING.md). Run `pre-commit run -a` and `pytest` before opening a PR.

Upstream project: [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer). Released under the Apache-2.0 License — [LICENSE](LICENSE), [NOTICE](NOTICE).
