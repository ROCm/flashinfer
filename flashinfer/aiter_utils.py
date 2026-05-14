# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import functools

import torch

from .hip_utils import FLASHINFER_SUPPORTED_ROCM_ARCHS


def is_aiter_supported(device: torch.device) -> bool:
    """Return True when the given device is an AMD GPU that AITER targets (gfx942/gfx950)."""
    if torch.version.hip is None:
        return False
    try:
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    except Exception:
        return False
    return arch in FLASHINFER_SUPPORTED_ROCM_ARCHS


@functools.cache
def get_aiter_mha_module():
    from aiter.ops import mha as aiter_mha_module

    return aiter_mha_module
