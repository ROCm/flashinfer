# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
AITER utilities for ROCm.

This module provides utilities for AITER, a library for efficient attention operations.
"""

import torch

from .hip_utils import FLASHINFER_SUPPORTED_ROCM_ARCHS


def is_aiter_supported(device: torch.device) -> bool:
    """Return True when the given device is an AMD GPU that AITER targets.

    Checks arch name (gfx942/gfx950) rather than whether the aiter package
    is currently installed.  Use _require_aiter_runtime() (in prefill_rocm.py)
    to also verify the package is importable at runtime.
    """
    if torch.version.hip is None:
        return False
    try:
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    except Exception:
        return False
    return arch in FLASHINFER_SUPPORTED_ROCM_ARCHS


def get_aiter_mha_module():
    from aiter.ops import mha as aiter_mha_module

    return aiter_mha_module
