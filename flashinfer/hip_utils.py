# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0


def get_system_rocm_version():
    """
    Attempt to detect the system ROCm version.

    Returns:
        str: ROCm version like "6.4" or "7.0", or None if not detectable
    """
    import os
    import re
    import subprocess

    # Method 1: Try /opt/rocm/.info/version (most reliable)
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    version_file = os.path.join(rocm_path, ".info", "version")
    try:
        with open(version_file, "r") as f:
            version = f.read().strip()
            # Convert "6.4.0" to "6.4"
            return ".".join(version.split(".")[:3])
    except (FileNotFoundError, IOError):
        pass

    # Method 2: Try amd-smi command
    try:
        result = subprocess.run(
            ["amd-smi"], capture_output=True, text=True, timeout=5, check=False
        )
        if result.returncode == 0:
            match = re.search(r"ROCm version:\s*(\d+\.\d+\.\d)", result.stdout)
            if match:
                return match.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Method 3: Try dpkg (Ubuntu/Debian)
    try:
        result = subprocess.run(
            ["dpkg", "-l", "rocm-core"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            match = re.search(r"rocm-core\s+(\d+\.\d+\.\d)", result.stdout)
            if match:
                return match.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def validate_rocm_arch(arch_list: str = None, verbose: bool = False) -> str:
    """
    Validate ROCm architecture against system ROCm version.

    Args:
        arch_list: Comma-separated list of architectures (e.g., "gfx942,gfx90a").
                   If None, reads from FLASHINFER_ROCM_ARCH_LIST env var or defaults to "gfx942"
        verbose: Whether to print validation messages

    Returns:
        Validated architecture list string

    Raises:
        RuntimeError: If ROCm not found or architectures not supported
    """
    import os

    # ROCm compatibility matrix: version -> supported gfx architectures
    # Refer: https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html
    ROCM_COMPAT_MATRIX = {
        "7.2": [
            "gfx950",
            "gfx1201",
            "gfx1200",
            "gfx1101",
            "gfx1100",
            "gfx1030",
            "gfx942",
            "gfx90a",
            "gfx908",
        ],
        "7.1": [
            "gfx950",
            "gfx1201",
            "gfx1200",
            "gfx1101",
            "gfx1100",
            "gfx1030",
            "gfx942",
            "gfx90a",
            "gfx908",
        ],
        "7.0": [
            "gfx950",
            "gfx1201",
            "gfx1200",
            "gfx1101",
            "gfx1100",
            "gfx1030",
            "gfx942",
            "gfx90a",
            "gfx908",
        ],
        "6.4": ["gfx1100", "gfx1030", "gfx942", "gfx90a", "gfx908"],
        "6.3": ["gfx1100", "gfx1030", "gfx942", "gfx90a", "gfx908"],
    }

    # Get architecture list from parameter, env var, or default
    if arch_list is None:
        arch_list = os.environ.get("FLASHINFER_ROCM_ARCH_LIST", "gfx942")

    # Validate system has ROCm installed
    system_rocm_version = get_system_rocm_version()
    if system_rocm_version is None:
        raise RuntimeError(
            "Could not detect ROCm installation. Please ensure ROCm is installed and "
            "accessible (check ROCM_PATH or /opt/rocm)."
        )

    # Parse version to major.minor for compatibility check
    version_parts = system_rocm_version.split(".")
    rocm_version_key = f"{version_parts[0]}.{version_parts[1]}"

    # Validate architectures against compatibility matrix
    requested_archs = [arch.strip() for arch in arch_list.split(",")]
    supported_archs = ROCM_COMPAT_MATRIX.get(rocm_version_key, [])

    if not supported_archs:
        raise RuntimeError(
            f"ROCm version {system_rocm_version} does not support the provided architectures: {', '.join(requested_archs)}.\n"
            f"See compatibility matrix: https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html"
        )

    # Check each requested arch is supported
    unsupported = [arch for arch in requested_archs if arch not in supported_archs]
    if unsupported:
        raise RuntimeError(
            f"ROCm version {system_rocm_version} does not support the provided arch: {', '.join(unsupported)}.\n"
            f"See compatibility matrix: https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html"
        )

    if verbose:
        print(f"Validated ROCm {system_rocm_version} with architecture(s): {arch_list}")

    return arch_list
