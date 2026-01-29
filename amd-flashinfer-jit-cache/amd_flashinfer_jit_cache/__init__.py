# SPDX-FileCopyrightText : 2025 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier : Apache-2.0

"""AMD FlashInfer JIT Cache Package

This package provides pre-compiled HIP kernels for FlashInfer on AMD ROCm platforms.
"""

from pathlib import Path

# Get the path to the AOT modules directory within this package
jit_cache_dir = Path(__file__).parent / "jit_cache"


def get_jit_cache_dir() -> str:
    """Get the path to the jit_cache directory containing pre-compiled kernels.

    Returns:
        Path: Absolute path to the jit_cache directory
    """
    return str(jit_cache_dir)


try:
    from ._version import __version__ as __version__
except (ModuleNotFoundError, ImportError):
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "get_jit_cache_dir",
]
