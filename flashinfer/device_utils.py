# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Device detection and capability utilities for CUDA/ROCm backends.

This module provides a central location for device detection and backend
selection, avoiding scattered checks throughout the codebase.
"""

from typing import Optional

from torch import version


def is_hip_available() -> bool:
    """
    Check if ROCm/HIP backend is available.

    Returns:
        bool: True if PyTorch was built with ROCm/HIP support
    """
    return hasattr(version, "hip") and version.hip is not None


def is_cuda_available() -> bool:
    """
    Check if CUDA backend is available (and not HIP).

    Returns:
        bool: True if PyTorch was built with CUDA (not ROCm) support
    """
    return hasattr(version, "cuda") and version.cuda is not None


# Global constants - evaluated once at module import
# Use these throughout the codebase for device-specific logic
IS_HIP = is_hip_available()
IS_CUDA = is_cuda_available()


def get_device_backend() -> str:
    """
    Get the current device backend.

    Returns:
        str: One of 'hip', 'cuda', or 'cpu'
    """
    if IS_HIP:
        return "hip"
    elif IS_CUDA:
        return "cuda"
    return "cpu"


def get_backend_version() -> Optional[str]:
    """
    Get the version string of the current backend (CUDA or HIP).

    Returns:
        Optional[str]: Version string (e.g., "12.4" for CUDA or "6.4.0" for ROCm)
                      Returns None if neither backend is available
    """
    if IS_HIP:
        return version.hip
    elif IS_CUDA:
        return version.cuda
    return None


def get_backend_name() -> str:
    """
    Get a human-readable name for the current backend.

    Returns:
        str: "ROCm/HIP", "CUDA", or "CPU"
    """
    if IS_HIP:
        return "ROCm/HIP"
    elif IS_CUDA:
        return "CUDA"
    return "CPU"
    