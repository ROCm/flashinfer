# FlashInfer+ROCm: An AMD ROCm port of FlashInfer

<p align="center">
| <a href="https://flashinfer.ai"><b>Blog</b></a> | <a href="https://docs.flashinfer.ai"><b>Documentation</b></a> | <a href="https://join.slack.com/t/flashinfer/shared_invite/zt-379wct3hc-D5jR~1ZKQcU00WHsXhgvtA"><b>Slack</b></a> |  <a href="https://github.com/orgs/flashinfer-ai/discussions"><b>Discussion Forum</b></a> |
</p>

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

## Feature Support Matrix

| Kernel Type | FP16 / BF16 | FP8 (E4M3, E5M2) | Notes |
| :--- | :---: | :---: | :--- |
| **Decode Attention** | ✅ | ✅ | Supports MHA, GQA, and MQA |
| **Prefill Attention** | ✅ | WIP | Supports MHA, GQA, and MQA |
| **Cascade Attention** | WIP | WIP | Not Yet Ported |
| **MLA** | TBD | TBD | Not Yet Ported |
| **POD** | TBD | TBD | Not Yet Ported |
| **Positional Encoding** | TBD | TBD | Not Yet Ported |
| **Sampling** | TBD | TBD | Top-K/Top-P Sampling Not Yet Ported |
| **Normalization** | TBD | TBD | RMS-Norm/Layer-Norm Not Yet Ported |

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

FlashInfer is available as a Python package for Linux on PyPI. You can install it with the following command:

```bash
pip install amd-flashinfer --index-url https://pypi.amd.com/simple/
```

Install the needed ROCm-enabled torch package from <https://repo.radeon.com>:

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
python -m pip install -v .

# for development & contribution, install in editable mode
python -m pip install --no-build-isolation -e . -v
```

To pre-compile essential kernels ahead-of-time (AOT), run the following command:

```bash
# Set target CUDA architectures
export FLASHINFER_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a 10.0a"
# Build AOT kernels. Will produce AOT kernels in aot-ops/
python -m flashinfer.aot
# Build AOT wheel
python -m build --no-isolation --wheel
# Install AOT wheel
python -m pip install dist/flashinfer_*.whl
```

**Available examples:**

* `single_prefill_example.py` - Single-sequence prefill attention
* `batch_prefill_example.py` - Batched prefill attention
* `batch_decode_example.py` - Batched decode attention

## Build from Source

### Setting up a Development Environment

kv_len = 2048
num_kv_heads = 32
head_dim = 128

k = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)
v = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)

# decode attention

num_qo_heads = 32
q = torch.randn(num_qo_heads, head_dim).half().to(0)

o = flashinfer.single_decode_with_kv_cache(q, k, v) # decode attention without RoPE on-the-fly
o_rope_on_the_fly = flashinfer.single_decode_with_kv_cache(q, k, v, pos_encoding_mode="ROPE_LLAMA") # decode with LLaMA style RoPE on-the-fly

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

Check out [documentation](https://docs.flashinfer.ai/) for usage of batch decode/append/prefill kernels and shared-prefix cascading kernels.

## Custom Attention Variants

Starting from FlashInfer v0.2, users can customize their own attention variants with additional parameters. For more details, refer to our [JIT examples](https://github.com/flashinfer-ai/flashinfer/blob/main/tests/test_jit_example.py).

## C++ API and TVM Bindings

</details>
<!-- markdownlint-enable MD033 -->

## GPU Support

FlashInfer currently provides support for NVIDIA SM architectures 80 and higher and beta support for 103, 110, 120, and 121.

## Adoption

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
