# AMD FlashInfer JIT Cache

This package contains pre-compiled HIP kernels for FlashInfer on AMD ROCm platforms.

## Purpose

The `amd-flashinfer-jit-cache` package provides ahead-of-time (AOT) compiled kernels to significantly reduce initialization time when using FlashInfer with AMD GPUs. Without this package, FlashInfer will compile kernels just-in-time (JIT) during runtime, which can take several minutes on first use.

## Installation

This package is intended to be installed alongside the main `amd-flashinfer` package:

```bash
pip install amd-flashinfer amd-flashinfer-jit-cache
```

## Architecture Support

This package is built specifically for the **AMD MI300 series (gfx942)** architecture.

Check the package version and tags to ensure compatibility with your GPU architecture.

## Development

To build this package from source:

```bash
cd amd-flashinfer-jit-cache
python -m build --wheel
```

The build process will:
1. Generate kernel specifications using `flashinfer.aot_hip`
2. Compile kernels for the gfx942 architecture
3. Package compiled `.so` files into the wheel

## Environment Variables

- `FLASHINFER_ROCM_ARCH_LIST`: Target architecture (default: "gfx942")
- `HIP_PATH`: Path to ROCm/HIP installation (auto-detected if not set)

## License

Apache License 2.0
