# SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
#
# SPDX - License - Identifier : Apache - 2.0

import pathlib

_INCLUDE_MARKER = pathlib.Path("flashinfer") / "attention" / "generic" / "prefill.cuh"


def _include_dir_has_headers(include_root: pathlib.Path) -> bool:
    return (include_root / _INCLUDE_MARKER).is_file()


def _get_package_root_dir():
    """Return the root directory of the flashinfer package.

    Uses the location of this file so the path is correct for both regular
    installs (site-packages/flashinfer/) and editable installs. With
    scikit-build-core's default ``redirect`` editable mode, Python sources
    are loaded from the source tree while compiled artifacts may live under
    ``_skbuild/editable``.

    Returns
    -------
    package_root_dir : str
        Path to the root directory of the flashinfer package.
    """
    return str(pathlib.Path(__file__).parent)


def get_include():
    """Return the directory containing the header files needed by the JIT.

    Prefer ``<package>/flashinfer/include`` (wheel install or editable symlink from
    CMake). If that directory does not contain the JIT headers yet, fall back to
    ``<repo>/include`` so running from a source tree with ``PYTHONPATH`` works
    without running the editable install symlink step.

    Returns
    -------
    include_dir : str
        Path to include and Cutlass header files.
    """
    package_dir = pathlib.Path(_get_package_root_dir()).resolve()
    pkg_include = package_dir / "include"
    repo_include = package_dir.parent / "include"
    if _include_dir_has_headers(pkg_include):
        return str(pkg_include)
    if _include_dir_has_headers(repo_include):
        return str(repo_include)
    return str(pkg_include)


def get_csrc_dir():
    """Return the directory containing the C++/CUDA source files used by jit.

    Returns
    -------
    csrc_dir : str
        Path to flashinfer's C++/ROCm source files.
    """
    csrc_dir = pathlib.Path(__file__).parent / "csrc_rocm"
    return str(csrc_dir)
