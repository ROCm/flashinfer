# FlashInfer+ROCm: An AMD ROCm port of FlashInfer

FlashInfer+ROCm is a port of the [FlashInfer](https://github.com/flashinfer-ai/flashinfer) library
that adds support for AMD Instinct GPUs. The project is in active development with current focus on
porting attention kernels to ROCm.

**Versioning:** The release tag format `<upstream_version>+amd` ties each FlashInfer+ROCm release
to its corresponding upstream tag (e.g., `0.2.5+amd.2` is second release of amd-flashinfer based on upstream version `v0.2.5`).

## Table of Contents

* [Feature Support Matrix](#feature-support-matrix)
* [GPU and ROCm Support](#gpu-and-rocm-support)
* [Getting Started](#getting-started)
  * [Option 1: Get a Pre-built Docker Image](#option-1-get-a-pre-built-docker-image)
  * [Option 2: Install from a Wheel Package](#option-2-install-from-a-wheel-package)
  * [Trying the Examples](#trying-the-examples)
* [Build from Source](#build-from-source)
  * [Setting up a Development Environment](#setting-up-a-development-environment)
  * [Building and Installing a Wheel Package](#building-and-installing-a-wheel-package)
  * [Running Tests](#running-tests)
* [AITER Support](#aiter-support)
  * [Single Prefill AITER example](#single-prefill-example)

## Feature Support Matrix

| Kernel Type | FP16 / BF16 | FP8 (E4M3, E5M2) | Has AITER backend | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Decode Attention** | ✅ | ✅ | No | Supports MHA, GQA, and MQA |
| **Prefill Attention** | ✅ | WIP | ✅ | Supports MHA, GQA, and MQA |
| **Cascade Attention** | TBD | TBD | No | Not Yet Ported |
| **MLA** | TBD | TBD | No | Not Yet Ported |
| **POD** | TBD | TBD | No | Not Yet Ported |
| **Positional Encoding** | TBD | TBD | No | Not Yet Ported |
| **Sampling** | ✅ | TBD | No | Supports Top-K/Top-P Sampling/OnlineSoftmax/SamplingFromLogits |
| **Logits Processor** | ✅ | TBD | No | |
| **Normalization** | ✅ | TBD | No | Supports RMS-Norm/Layer-Norm |

## GPU and ROCm Support

**Supported GPU:** gfx942 (CDNA3 architecture), gfx950 (CDNA4 architecture)

**Supported ROCm versions:** 7.0.2, 7.1.1, 7.2

## Torch Version Support

**Torch+ROCm:** 2.8.0, 2.9.1

**Note**: Other versions may work but have not been tested. Refer to <https://repo.radeon.com/rocm/manylinux/rocm-rel-{rocm-version}/> (replacing `{rocm-version}` with the desired ROCm version, e.g., `7.0.2`) for available versions.

## Getting Started

### Option 1: Get a Pre-built Docker Image

AMD validates and publishes [FlashInfer images](https://hub.docker.com/r/rocm/flashinfer/tags)
with ROCm backends on Docker Hub. The following Docker image tag and associated
inventories represent the latest available FlashInfer version from the official Docker Hub.

| Docker image | ROCm | FlashInfer | PyTorch | Ubuntu | Python | GPU |
| ------------ | ---- | ---------- | ------- | ------ | ------ | --- |
| rocm/flashinfer:flashinfer-0.5.3.amd1_rocm7.2_ubuntu24.04_py3.12_pytorch2.9.1 |7.2.0 | v0.5.3 | 2.9.1 | 24.04 | 3.12 | MI355x, MI325X, MI300X |
| rocm/flashinfer:flashinfer-0.5.3.amd1_rocm7.0.2_ubuntu24.04_py3.12_pytorch2.9.1 | 7.0.2 | v0.5.3 | 2.9.1 | 24.04 | 3.12 | MI355x, MI325X, MI300X |
| rocm/flashinfer:flashinfer-0.2.5.amd2_rocm7.1.1_ubuntu24.04_py3.12_pytorch2.8 | 7.1.1 | v0.2.5 | 2.8.0 | 24.04 | 3.12 | MI325X, MI300X |

**Start a container:**

```bash
docker run -it --privileged --network=host --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --shm-size 128G --name=<container-name> <docker-image-tag>
```

**Activate the environment and verify:**

```bash
# Activate micromamba environment (Note: env name may vary based on the image)
micromamba activate base

# Verify installation
python -c "import flashinfer; print(flashinfer.__version__)"
```

Expected output: `0.5.3+amd.1` (with a possible JIT backend message)

### Option 2: Install from a Wheel Package

Install from AMD's package repository:

```bash
pip install amd-flashinfer --index-url https://pypi.amd.com/simple/
```

Install the needed ROCm-enabled torch package from <https://repo.radeon.com>:

```bash
pip install torch==2.9.1 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2
```

**NOTE**: The torch version should be exactly as available on repo.radeon.com otherwise a non-ROCm
torch version will get installed from pypi.

### Trying the Examples

Download and run example scripts from the repository:

```bash
# Download a single example
wget https://raw.githubusercontent.com/ROCm/flashinfer/amd-integration/examples/single_prefill_example.py
python single_prefill_example.py

# Download all examples
for example in single_prefill_example.py batch_prefill_example.py batch_decode_example.py; do
  wget https://raw.githubusercontent.com/ROCm/flashinfer/amd-integration/examples/$example
done
```

**Available examples:**

* `single_prefill_example.py` - Single-sequence prefill attention
* `batch_prefill_example.py` - Batched prefill attention
* `batch_decode_example.py` - Batched decode attention
* `examples/amd_flashinfer_rocm_tutorial.ipynb` - Jupyter tutorial: environment verification (`hip_utils`), AITER-backed prefill examples, and `logits_processor` on ROCm
* `examples/run_jupyter_server.sh` - Start JupyterLab from the repo root (run inside your ROCm/FlashInfer environment or Docker container)

## Build from Source

### Setting up a Development Environment

Build the development Docker image with the repository's Dockerfile:

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
```

<!-- markdownlint-disable MD033 -->
<details>
<summary>Build argument descriptions</summary>

* `ROCM_VERSION`: ROCm version (default: 7.2)
* `PY_VERSION`: Python version (default: 3.12)
* `TORCH_VERSION`: PyTorch version (default: 2.9.1)
* `USERNAME`: Username inside container (default: devuser)
* `USER_UID`: User ID for matching host permissions
* `USER_GID`: Group ID for matching host permissions

</details>
<!-- markdownlint-enable MD033 -->

**Run the development container:**

```bash
docker run -it \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --privileged --shm-size=128G --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -v $PWD:/workspace \
  --name flashinfer-dev-container \
  flashinfer-0.5.3.amd1_rocm7.2_ubuntu24.04_py3.12_pytorch2.9.1
```

<!-- markdownlint-disable MD033 -->
<details>
<summary>Docker run argument descriptions</summary>

* `--cap-add=SYS_PTRACE`: Enables debugging
* `--security-opt seccomp=unconfined`: Relaxes security for development
* `--ipc=host`: Shares host IPC for better performance
* `--privileged`: Required for GPU access
* `--shm-size=128G`: Shared memory size (adjust as needed)
* `--network=host`: Uses host networking
* `--device=/dev/kfd --device=/dev/dri`: Exposes AMD GPU devices
* `--group-add video --group-add render`: GPU access groups
* `-v <host-path>:<container-path>`: Mounts source code

</details>
<!-- markdownlint-enable MD033 -->

**Note:** Environment name varies based on Python, PyTorch, and ROCm versions.

### Building and Installing a Wheel Package

**Build with JIT (Just-in-Time) compilation only:**

```bash
python -m pip wheel . --wheel-dir=./dist/ --no-deps --no-build-isolation -v
cd dist && pip install amd_flashinfer-*.whl
```

**Editable install for development:**

```bash
python -m pip install --no-build-isolation -ve.
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

# To run tests parallely on multiple GPUs
pytest -n auto # Uses all available GPUs
pytest -n 2 # Use only two GPUs
```

The default test configuration is specified in [pyproject.toml](pyproject.toml) under the `testpaths` setting.

#### Recommended invocation on AMD CPX systems

`pytest-rerunfailures` (declared in the `dev` extra — `pip install -e ".[dev]"`)
absorbs the residual transient HIP runtime crashes. Then for the full suite:

```bash
# Fast path — skips heavy 1M-trial sampling-frequency tests and 4 GB
# speculative-sampling cases (~7 min on a CPX 8-card host):
pytest -n auto --reruns 2 -m "not slow"

# Full coverage — including the slow tests (~20 min):
pytest -n auto --reruns 2

# Slow path only (~13 min):
pytest -n auto --reruns 2 -m "slow"
```

**Notes**

* `pytest -n auto` for the `tests/rocm_tests/` suite spawns **half as many xdist workers as physical AMD cards** (e.g. 4 workers on a CPX-mode 8-card MI308X / MI325X host). One worker per physical card was tried first but produced sporadic failures across rope, single_prefill, and logits_cap under residual concurrent load; halving the count produces reliable green runs. Each worker is pinned to its card via `HIP_VISIBLE_DEVICES`. On non-CPX systems the helper applies the same halving; users who want every device used can pass an explicit `-n N`.
* `--reruns 2` (from `pytest-rerunfailures`) absorbs the residual ~0.01 % of transient HIP runtime crashes (HSA exceptions, HIPBLAS handle-pool exhaustion, intermittent generator non-determinism) that worker pinning cannot fully eliminate. Successful tests are not duplicated; only failed tests are retried.
* The `slow` marker is registered in [pyproject.toml](pyproject.toml). It tags the 1M-trial sampling-frequency tests, the 4 GB-tensor speculative-sampling cases, and the entire `TestLogitsPipeCompilationHIP` class (every test there runs the sampling kernel twice per case for compile=True/False).
* The reference attention helper in `tests/attention_reference.py` wraps `torch.matmul` in a `_hipblas_safe_matmul` retry helper that catches `HIPBLAS_STATUS_ALLOC_FAILED` and retries with a short back-off — needed under heavy concurrent xdist load.

## AITER Support

FlashInfer+ROCm has experimental support to use [AITER](https://github.com/ROCm/aiter) as a
backend. The `aiter` backend currently is enabled for the `single_prefill` and `batch_prefill`
kernels only. To use AITER as the backend for these kernels, please set `backend="aiter"` keyword
argument when invoking the kernels. Unless you are using the prebuilt docker image, AITER should also be installed on your system. You may follow one of the following ways to do so.

### Install AITER from source

```bash
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
python3 setup.py develop
```

### Install AITER wheel package

Wheel packages are available from AMD's PyPI index: [pypi.amd.com/simple](https://pypi.amd.com/simple/).

```bash
pip install amd-aiter --index-url https://pypi.amd.com/simple/
```

### Known Limitations

The AITER backend has the following constraints. With `backend="auto"` (the
default), the wrapper inspects the call and falls back to `fa2` with a
one-time `logger.warning` when any of the first group is violated; with
`backend="aiter"` the call will error or, for the second group, run but
ignore the unsupported argument.

**Conditions that fall back to `fa2` under `backend="auto"`:**

* GPU is not gfx942 or gfx950
* `kv_layout` is not `NHD`
* a custom attention mask tensor is supplied
* `q_dtype` is not `float16` / `bfloat16` (no fp32, fp8, or int8)
* `q_dtype != kv_dtype` (mixed-precision Q/KV is unsupported)
* `head_dim_qk != head_dim_vo` (e.g. DeepSeek-style MLA with 192/128 head dims)
* the `aiter` Python package is not importable

**Features silently ignored on the AITER path** (the kwargs are accepted by
the FlashInfer wrapper but not forwarded to AITER, which can produce wrong
results — pass `backend="fa2"` explicitly if you need any of these):

* ALiBi slopes (`maybe_alibi_slopes`)
* in-kernel positional encoding modes (`pos_encoding_mode`, `rope_scale`,
  `rope_theta`)
* attention sinks (`sinks`)
* multi-modal / prefix-cache helpers (`maybe_prefix_len_ptr`,
  `maybe_token_pos_in_items_ptr`, `maybe_max_item_len_ptr`)
* FP8 dequant scales (`scale_q` / `scale_k` / `scale_v`)
* `use_fp16_qk_reduction`, `enable_pdl`

**Other notes:**

* Batch prefill: AITER's CK FMHA kernels natively support page sizes
  `{16, 1024}` (or `{128, 256, 1024}` on `amd-aiter==0.1.10`). Other page
  sizes still work but go through an extra GPU gather to flatten paged KV
  before the AITER call.
* Ragged (non-paged) KV is not yet implemented on the AITER batch-prefill
  path. `BatchPrefillWithRaggedKVCacheWrapper` therefore forces the backend
  to `fa2` regardless of whether you pass `backend="auto"` or
  `backend="aiter"` (a warning is logged in the latter case).

### Single Prefill Example

This section provides an example on how to use Single Prefill with AITER.

```python
import torch
import flashinfer

# Configuration
seq_len = 1024        # Prompt length
num_qo_heads = 32     # Number of query/output heads
num_kv_heads = 8      # Number of KV heads (GQA with 4:1 ratio)
head_dim = 128

# Create Q, K, V tensors (NHD layout: sequence, heads, dimension)
q = torch.randn(seq_len, num_qo_heads, head_dim, dtype=torch.float16, device="cuda")
k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")

# Run single prefill attention with causal masking
output = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True,  backend="aiter")
```
