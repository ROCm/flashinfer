# FlashInfer+ROCm: An AMD ROCm port of FlashInfer

FlashInfer+ROCm is a port of the [FlashInfer](https://github.com/flashinfer-ai/flashinfer) library
that adds support for AMD Instinct GPUs. The project is in active development with current focus on
porting attention kernels to ROCm.

**Versioning:** The release tag format `<upstream_version>+amd` ties each FlashInfer+ROCm release
to its corresponding upstream tag (e.g., `0.2.5+amd.2` is based on upstream `v0.2.5`).

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
| **Cascade Attention** | TBD | TBD | No |  Not Yet Ported |
| **MLA** | TBD | TBD | No | Not Yet Ported |
| **POD** | TBD | TBD | No | Not Yet Ported |
| **Positional Encoding** | TBD | TBD | No | Not Yet Ported |
| **Sampling** | ✅ | TBD | No | Supports Top-K/Top-P Sampling/OnlineSoftmax/SamplingFromLogits |
| **Logits Processor** | ✅ | TBD | No |  |
| **Normalization** | ✅ | TBD | No | Supports RMS-Norm/Layer-Norm |

## GPU and ROCm Support

**Supported GPU:** gfx942 (CDNA3 architecture)

**Supported ROCm versions:** 6.4.1, 7.1.1

## Torch Version Support

**Torch+ROCm:** 2.7.1, 2.8.0

**Note**: Other versions may work but have not been tested. Refer to <https://repo.radeon.com/rocm/manylinux/rocm-rel-{rocm-version}/> (replacing `{rocm-version}` with the desired ROCm version, e.g., `6.4.1`) for available versions.

## Getting Started

### Option 1: Get a Pre-built Docker Image

AMD validates and publishes [FlashInfer images](https://hub.docker.com/r/rocm/flashinfer/tags)
with ROCm backends on Docker Hub. The following Docker image tag and associated
inventories represent the latest available FlashInfer version from the official Docker Hub.

| Docker image | ROCm | FlashInfer | PyTorch | Ubuntu | Python | GPU |
| ------------ | ---- | ---------- | ------- | ------ | ------ | --- |
| rocm/flashinfer:flashinfer-0.2.5.amd2_rocm7.1.1_ubuntu24.04_py3.12_pytorch2.8 | [7.1.1](https://repo.radeon.com/rocm/apt/7.1.1/) | [v0.2.5](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.2.5) | [2.8.0](https://github.com/ROCm/pytorch/releases/tag/v2.8.0) | 24.04 | [3.12](https://www.python.org/downloads/release/python-3129/) | MI325X, MI300X |
| rocm/flashinfer:flashinfer-0.2.5_rocm6.4_ubuntu24.04_py3.12_pytorch2.7 | [6.4.1](https://repo.radeon.com/rocm/apt/6.4.1/) | [v0.2.5](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.2.5) | [2.7.1](https://github.com/ROCm/pytorch/releases/tag/v2.7.1) | 24.04 | [3.12](https://www.python.org/downloads/release/python-3129/) | MI300X |

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

Expected output: `0.2.5+amd.2` (with a possible JIT backend message)

### Option 2: Install from a Wheel Package

Install from AMD's package repository:

```bash
pip install amd-flashinfer --index-url https://pypi.amd.com/simple/
```

Install the needed ROCm-enabled torch package from <https://repo.radeon.com>:

```bash
pip install torch==2.8.0 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1
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

## Build from Source

### Setting up a Development Environment

Build the development Docker image with the repository's Dockerfile:

```bash
docker build \
  --build-arg ROCM_VERSION=7.1.1 \
  --build-arg PY_VERSION=3.12 \
  --build-arg TORCH_VERSION=2.8.0 \
  --build-arg USERNAME=$USER \
  --build-arg USER_UID=$(id -u) \
  --build-arg USER_GID=$(id -g) \
  -t flashinfer-0.2.5_rocm7.1.1_ubuntu24.04_py3.12_pytorch2.8.0 \
  -f .devcontainer/rocm/Dockerfile .
```

<!-- markdownlint-disable MD033 -->
<details>
<summary>Build argument descriptions</summary>

* `ROCM_VERSION`: ROCm version (default: 7.1.1)
* `PY_VERSION`: Python version (default: 3.12)
* `TORCH_VERSION`: PyTorch version (default: 2.8.0)
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
  flashinfer-0.2.5_rocm7.1.1_ubuntu24.04_py3.12_pytorch2.8.0
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

**Build with AOT (Ahead-of-Time) compiled kernels:**

```bash
FLASHINFER_HIP_ARCHITECTURES=gfx942 FLASHINFER_AOT_TORCH_EXTS=ON \
  python -m pip wheel . --wheel-dir=./dist/ --no-deps --no-build-isolation -v
cd dist && pip install amd_flashinfer-*.whl
```

**Build with JIT (Just-in-Time) compilation only:**

```bash
FLASHINFER_HIP_ARCHITECTURES=gfx942 \
  python -m pip wheel . --wheel-dir=./dist/ --no-deps --no-build-isolation -v
cd dist && pip install amd_flashinfer-*.whl
```

**Editable install for development:**

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
```

The default test configuration is specified in [pyproject.toml](pyproject.toml) under the `testpaths` setting.

## AITER Support

FlashInfer+ROCm has experimental support to use [AITER](https://github.com/ROCm/aiter) as a
backeend. The `aiter` backend currently is enabled for the `single_prefill` and `batch_prefill`
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

def single_prefill_with_kv_cache_aiter_example():
  qo_len = 128
  kv_len = 128
  num_qo_heads = 1
  num_kv_heads = 1
  head_dim = 64
  causal = False
  kv_layout = "NHD"
  pos_encoding_mode = "NONE"
  logits_soft_cap = 8.0
  return_lse = False

  q = torch.randn(qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16)

  # NHD Layout
  k = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0",dtype=torch.float16)
  v = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float16)


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

if __name__ == "__main__":
  single_prefill_with_kv_cache_aiter_example()

```
