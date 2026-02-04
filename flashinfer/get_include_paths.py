# SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
#
# SPDX - License - Identifier : Apache - 2.0

import os
import pathlib
from sysconfig import get_path


def _get_package_root_dir():
    """Return the root directory of the flashinfer package.

    Returns
    -------
    package_root_dir : str
        Path to the root directory of the flashinfer package.
    """
    platlib = get_path("platlib")
    return os.path.join(platlib, "flashinfer")


def get_include():
    """Return the directory containing the header files needed by the JIT.

    The `include` dir in the splatlib/flashinfer directory contains the header
    files needed by the JIT to compile the C++ code. The include path contains
    all flashinfer, cutlass, and Cute headers and any future dependencies.

    Returns
    -------
    include_dir : str
        Path to include and Cutlass header files.
    """
    include_dir = os.path.join(_get_package_root_dir(), "include")
    return str(include_dir)


def get_csrc_dir():
    """Return the directory containing the C++/CUDA source files used by jit.

    Returns
    -------
    csrc_dir : str
        Path to flashinfer's C++/ROCm source files.
    """
    csrc_dir = pathlib.Path(__file__).parent / "csrc_rocm"
    return str(csrc_dir)
