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
* [For Developers](#for-developers)
  * [Setting up a Development Environment](#setting-up-a-development-environment)
  * [Building and Installing a Wheel Package](#building-and-installing-a-wheel-package)
  * [Running Tests](#running-tests)


## Feature Support Matrix

| Kernel Type | FP16 / BF16 | FP8 (E4M3, E5M2) | Notes |
| :--- | :---: | :---: | :--- |
| **Decode Attention** | ✅ | ✅ | Supports MHA, GQA, and MQA |
| **Prefill Attention** | ✅ | WIP | Supports MHA, GQA, and MQA |
| **Cascade** | WIP | WIP | Not Yet Ported |
| **MLA** | TBD | TBD | Not Yet Ported |
| **POD** | TBD | TBD | Not Yet Ported |
| **Positional Encoding** | TBD | TBD | Not Yet Ported |
| **Sampling** | TBD | TBD | Top-K/Top-P Sampling Not Yet Ported |
| **Normalization** | TBD | TBD | RMS-Norm/Layer-Norm Not Yet Ported |

## GPU and ROCm Support

**Supported GPU:** gfx942 (CDNA3 architecture)

**Supported ROCm versions:** 6.3.2, 6.4.1, 7.0.2, 7.1.1

## Torch Version Support

**Torch+ROCm:** 2.7.1, 2.8.0

**Note**: Other versions may work but have not been tested. Refer to https://repo.radeon.com/rocm/manylinux/rocm-rel-{rocm-version}/ (replacing `{rocm-version}` with the desired ROCm version, e.g., `6.4.1`) for available versions.

## Getting Started
### Option 1: Get a Pre-built Docker Image


AMD validates and publishes [FlashInfer images](https://hub.docker.com/r/rocm/flashinfer/tags)
with ROCm backends on Docker Hub. The following Docker image tag and associated
inventories represent the latest available FlashInfer version from the official Docker Hub.

| Docker image | ROCm | FlashInfer | PyTorch | Ubuntu | Python | GPU |
|--------------|------|------------|---------|--------|--------|-----|
| rocm/flashinfer:flashinfer-0.2.5.amd2_rocm7.1.1_ubuntu24.04_py3.12_pytorch2.8  | [7.1.1](https://repo.radeon.com/rocm/apt/7.1.1/) | [v0.2.5](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.2.5) | [2.8.0](https://github.com/ROCm/pytorch/releases/tag/v2.8.0) | 24.04 | [3.12](https://www.python.org/downloads/release/python-3129/) | MI325X, MI300X |
| rocm/flashinfer:flashinfer-0.2.5.amd2_rocm7.0.2_ubuntu24.04_py3.12_pytorch2.8 | [7.0.2](https://repo.radeon.com/rocm/apt/7.0.2/) | [v0.2.5](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.2.5) | [2.8.0](https://github.com/ROCm/pytorch/releases/tag/v2.8.0) | 24.04 | [3.12](https://www.python.org/downloads/release/python-3129/) | MI325X, MI300X |
| rocm/flashinfer:flashinfer-0.2.5.amd2_rocm6.4.4_ubuntu24.04_py3.12_pytorch2.7.1 | [6.4.4](https://repo.radeon.com/rocm/apt/6.4.4/) | [v0.2.5](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.2.5) | [2.7.1](https://github.com/ROCm/pytorch/releases/tag/v2.7.1) | 24.04 | [3.12](https://www.python.org/downloads/release/python-3129/) | MI325X, MI300X |
| rocm/flashinfer:flashinfer-0.2.5_rocm6.4_ubuntu24.04_py3.12_pytorch2.7 | [6.4.1](https://repo.radeon.com/rocm/apt/6.4.1/) | [v0.2.5](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.2.5) | [2.7.1](https://github.com/ROCm/pytorch/releases/tag/v2.7.1) | 24.04 | [3.12](https://www.python.org/downloads/release/python-3129/) | MI300X |

**Start a container:**

```bash
docker run -it --privileged --network=host --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --shm-size 128G --name=<container-name> <docker-image-tag>
```

**Activate the environment and verify:**

```bash
# Activate micromamba environment (name varies by image)
micromamba activate flashinfer-py3.12-torch2.7.1-rocm6.4.1

# Verify installation
python -c "import flashinfer; print(flashinfer.__version__)"
```

Expected output: `0.2.5+rocm.1` (with a possible JIT backend message)

### Option 2: Install from a Wheel Package

Install from AMD's package repository:

```bash
pip install amd-flashinfer --index-url https://pypi.amd.com/simple/
```

Install a ROCm-enabled torch package from https://repo.radeon.com:

```bash
pip install torch==2.7.1 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1
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
- `single_prefill_example.py` - Single-sequence prefill attention
- `batch_prefill_example.py` - Batched prefill attention
- `batch_decode_example.py` - Batched decode attention


## For Developers

### Setting up a Development Environment

Build the development Docker image with the repository's Dockerfile:

```bash
docker build \
  --build-arg ROCM_VERSION=6.4.1 \
  --build-arg PY_VERSION=3.12 \
  --build-arg TORCH_VERSION=2.7.1 \
  --build-arg USERNAME=$USER \
  --build-arg USER_UID=$(id -u) \
  --build-arg USER_GID=$(id -g) \
  -t flashinfer-0.2.5_rocm6.4_ubuntu24.04_py3.12_pytorch2.7 \
  -f .devcontainer/rocm/Dockerfile .
```

<details>
<summary>Build argument descriptions</summary>

- `ROCM_VERSION`: ROCm version (default: 7.0.2)
- `PY_VERSION`: Python version (default: 3.12)
- `TORCH_VERSION`: PyTorch version (default: 2.7.1)
- `USERNAME`: Username inside container (default: devuser)
- `USER_UID`: User ID for matching host permissions
- `USER_GID`: Group ID for matching host permissions
</details>

**Run the development container:**

```bash
docker run -it \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --privileged --shm-size=128G --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -v $PWD:/workspace \
  --name flashinfer-dev-container \
  flashinfer-0.2.5_rocm6.4_ubuntu24.04_py3.12_pytorch2.7
```

<details>
<summary>Docker run argument descriptions</summary>

- `--cap-add=SYS_PTRACE`: Enables debugging
- `--security-opt seccomp=unconfined`: Relaxes security for development
- `--ipc=host`: Shares host IPC for better performance
- `--privileged`: Required for GPU access
- `--shm-size=128G`: Shared memory size (adjust as needed)
- `--network=host`: Uses host networking
- `--device=/dev/kfd --device=/dev/dri`: Exposes AMD GPU devices
- `--group-add video --group-add render`: GPU access groups
- `-v <host-path>:<container-path>`: Mounts source code
</details>

**Activate the micromamba environment:**

```bash
micromamba activate flashinfer-py3.12-torch2.7.1-rocm6.4.1
```

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
