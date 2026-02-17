# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
AITER utilities for ROCm.

This module provides utilities for AITER, a library for efficient attention operations.
"""

HAS_AITER = False

try:
    import aiter
    HAS_AITER = True
except ImportError:
    pass

if HAS_AITER:
    def get_aiter_mha_module():
        from aiter.ops import mha as aiter_mha_module
        return aiter_mha_module
