# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# AMDGPU archs supported by amd-flashinfer
FLASHINFER_SUPPORTED_ROCM_ARCHS = ["gfx942"]


def get_rocm_home():
    """
    Get the ROCM_HOME directory from environment variables or default path.

    Returns:
        str: Path to ROCm installation (e.g., "/opt/rocm")
    """
    import os

    return os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME") or "/opt/rocm"


def is_therock_build() -> bool:
    """
    Check if ROCm was built by using TheRock build system.

    Returns:
        bool: True if TheRock manifest exists, False otherwise
    """
    import os

    rocm_home = get_rocm_home()
    manifest_path = os.path.join(rocm_home, "share", "therock", "therock_manifest.json")
    return os.path.isfile(manifest_path)


def get_system_rocm_version_from_info_file():
    """
    Try to get ROCm version from /opt/rocm/.info/version file.

    Returns:
        str: ROCm version like "7.1.0" or None if not found
    """
    import os

    rocm_home = get_rocm_home()
    version_file = os.path.join(rocm_home, ".info", "version")
    try:
        with open(version_file, "r") as f:
            version = f.read().strip()
            return ".".join(version.split(".")[:3])
    except (FileNotFoundError, IOError):
        return None


def get_system_rocm_version_from_hipconfig():
    """
    Try to get ROCm version from hipconfig --version command.

    Returns:
        str: ROCm version like "7.1.0" or None if not found
    """
    import re
    import subprocess

    try:
        result = subprocess.run(
            ["hipconfig", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False
        )
        if result.returncode == 0:
            match = re.search(r"(\d+\.\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def get_system_rocm_version_from_amd_smi():
    """
    Try to get ROCm version from amd-smi command.

    Returns:
        str: ROCm version like "7.1.0" or None if not found
    """
    import re
    import subprocess

    try:
        result = subprocess.run(
            ["amd-smi"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False
        )
        if result.returncode == 0:
            match = re.search(r"ROCm version:\s*(\d+\.\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def get_system_rocm_version_from_dpkg():
    """
    Try to get ROCm version from dpkg (Ubuntu/Debian package manager).

    Returns:
        str: ROCm version like "7.1.0" or None if not found
    """
    import re
    import subprocess

    try:
        result = subprocess.run(
            ["dpkg", "-l", "rocm-core"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            match = re.search(r"rocm-core\s+(\d+\.\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def get_system_rocm_version():
    """
    Attempt to detect the system ROCm version.

    For standard builds, tries methods in order of reliability.
    For TheRock builds, prioritizes hipconfig as it's more reliable.

    Returns:
        str: ROCm version like "7.1.0" or None if not detectable
    """
    # For TheRock builds, prioritize hipconfig
    if is_therock_build():
        return get_system_rocm_version_from_hipconfig()

    # Try standard detection methods in order of reliability
    detection_methods = [
        get_system_rocm_version_from_info_file,
        get_system_rocm_version_from_amd_smi,
        get_system_rocm_version_from_dpkg,
        get_system_rocm_version_from_hipconfig,
    ]

    for method in detection_methods:
        version = method()
        if version:
            return version
        print(f"ROCm version not found using {method.__name__}. Trying next method...")

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
    #        https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md#rocm-on-linux
    # Update lists for adding or removing a version or arch
    # Add new tuple for adding a new version group
    _ROCM_ARCH_GROUPS = [
        (
            ["7.12", "7.11", "7.10"],
            ["gfx950", "gfx942", "gfx90a", "gfx908", "gfx906",
             "gfx1201", "gfx1200", "gfx1151", "gfx1150",
             "gfx1102", "gfx1101", "gfx1100", "gfx1030"],
        ),
        (
            ["7.2", "7.1", "7.0"],
            ["gfx950", "gfx1201", "gfx1200", "gfx1101", "gfx1100",
             "gfx1030", "gfx942", "gfx90a", "gfx908"],
        ),
        (
            ["6.4", "6.3"],
            ["gfx1100", "gfx1030", "gfx942", "gfx90a", "gfx908"],
        ),
    ]

    # Build the compatibility matrix
    ROCM_COMPAT_MATRIX = {
        version: archs
        for versions, archs in _ROCM_ARCH_GROUPS
        for version in versions
    }

    # Get architecture list from parameter, env var, or default
    if arch_list is None:
        arch_list = os.environ.get("FLASHINFER_ROCM_ARCH_LIST", "gfx942")

    # Validate system has ROCm installed
    system_rocm_version = get_system_rocm_version()
    if system_rocm_version is None:
        raise RuntimeError(
            "Could not detect ROCm installation. Please ensure ROCm is installed and "
            "accessible (check ROCM_PATH, ROCM_HOME or /opt/rocm)."
        )

    # Parse version to major.minor for compatibility check
    version_parts = system_rocm_version.split(".")
    rocm_version_key = f"{version_parts[0]}.{version_parts[1]}"

    # Validate architectures against compatibility matrix
    requested_archs = [arch.strip() for arch in arch_list.split(",")]
    supported_archs = ROCM_COMPAT_MATRIX.get(rocm_version_key, [])

    if not supported_archs:
        raise RuntimeError(
            f"ROCm version {system_rocm_version} is not recognized in the ROCm "
            f"compatibility matrix. Requested architectures: {', '.join(requested_archs)}.\n"
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


def validate_flashinfer_rocm_arch(
    arch_list: str = None, torch_cpp_ext_module=None, verbose: bool = False
) -> tuple:
    """
    Comprehensive ROCm architecture validation for FlashInfer compilation.

    Validates in order:
    1. System ROCm version supports the architectures (ROCM_COMPAT_MATRIX)
    2. FlashInfer has AMD ports for the architectures (FLASHINFER_SUPPORTED_ROCM_ARCHS)
    3. PyTorch was compiled with the architectures (torch.utils.cpp_extension)

    Args:
        arch_list: Comma-separated list (e.g., "gfx942,gfx90a") or None for default
        torch_cpp_ext_module: torch.utils.cpp_extension module for PyTorch validation
        verbose: Print validation messages

    Returns:
        tuple: (arch_flags, arch_set)
            - arch_flags: List like ["--offload-arch=gfx942"]
            - arch_set: Set like {"gfx942"}

    Raises:
        RuntimeError: If any validation step fails with clear error message
    """
    import os

    # Get architecture list from parameter, env var, or default
    if arch_list is None:
        arch_list = os.environ.get("FLASHINFER_ROCM_ARCH_LIST", "gfx942")

    # Step 1: Validate against system ROCm version (reuse existing logic)
    validated_arch_list = validate_rocm_arch(arch_list=arch_list, verbose=verbose)
    requested_archs = [arch.strip() for arch in validated_arch_list.split(",")]

    # Step 2: Validate against AMD-ported FlashInfer architectures
    unsupported_by_flashinfer = [
        arch for arch in requested_archs if arch not in FLASHINFER_SUPPORTED_ROCM_ARCHS
    ]
    if unsupported_by_flashinfer:
        raise RuntimeError(
            f"FlashInfer does not support the following ROCm architectures: {', '.join(unsupported_by_flashinfer)}.\n"
            f"Currently supported by FlashInfer: {', '.join(FLASHINFER_SUPPORTED_ROCM_ARCHS)}"
        )

    # Step 3: Validate against PyTorch's available architectures (if module provided)
    arch_flags = [f"--offload-arch={arch}" for arch in requested_archs]
    if torch_cpp_ext_module is not None:
        pytorch_arch_flags = torch_cpp_ext_module._get_rocm_arch_flags()
        missing_in_pytorch = [
            flag for flag in arch_flags if flag not in pytorch_arch_flags
        ]
        if missing_in_pytorch:
            raise RuntimeError(
                f"PyTorch does not support the following architectures: {', '.join(missing_in_pytorch)}.\n"
                f"PyTorch was compiled with: {', '.join(pytorch_arch_flags)}"
            )

    if verbose:
        print(f"FlashInfer validated architectures: {', '.join(requested_archs)}")

    # Return both the flags list and the set
    arch_set = set(requested_archs)
    return arch_flags, arch_set


def check_torch_rocm_compatibility() -> None:
    """
    Verify that PyTorch is installed with compatible ROCm support.

    This function checks:
    1. PyTorch is installed
    2. PyTorch has ROCm/HIP support (not CPU-only)
    3. PyTorch ROCm version matches system ROCm version (if detectable)

    Provides helpful error messages to guide users to correct installation.

    Raises:
        ImportError: If PyTorch is not installed
        RuntimeError: If PyTorch doesn't have ROCm support
    """
    import warnings

    from torch import version

    # Check for torch package with rocm support
    if not hasattr(version, "hip") or version.hip is None:
        raise RuntimeError(
            "\n" + "=" * 70 + "\n"
            "ERROR: PyTorch does NOT have ROCm support.\n\n"
            "You installed the CPU-only version from PyPI.\n"
            "amd-flashinfer requires PyTorch compiled with ROCm support.\n\n"
            "Fix this by:\n"
            "  1. Uninstall current PyTorch:\n"
            "     pip uninstall torch\n\n"
            "  2. Install PyTorch for ROCm:\n"
            "     pip install torch==2.7.1 --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/\n\n"
            "See https://github.com/rocm/flashinfer for detailed installation instructions.\n"
            + "=" * 70
        )

    # ROCm version compatibility warning
    torch_rocm = version.hip
    torch_rocm_major_minor = ".".join(torch_rocm.split(".")[:2])

    # Try to detect system ROCm version
    system_rocm = get_system_rocm_version()

    if system_rocm:
        system_rocm_major_minor = ".".join(system_rocm.split(".")[:2])
        if torch_rocm_major_minor != system_rocm_major_minor:
            warnings.warn(
                f"\n{'=' * 70}\n"
                f"WARNING: ROCm version mismatch detected!\n\n"
                f"  System ROCm version: {system_rocm}\n"
                f"  PyTorch ROCm version: {torch_rocm_major_minor}\n\n"
                f"This may cause runtime errors or crashes.\n\n"
                f"To fix, reinstall PyTorch for your ROCm version:\n"
                f"  pip install torch==2.7.1 --index-url "
                f"https://repo.radeon.com/rocm/manylinux/rocm-rel-{system_rocm}/\n\n"
                f"Or if using uv:\n"
                f"  export FLASHINFER_ROCM_VERSION={system_rocm}\n"
                f"  uv pip install torch==2.7.1 --index-url "
                f"https://repo.radeon.com/rocm/manylinux/rocm-rel-{system_rocm}/\n"
                f"{'=' * 70}",
                RuntimeWarning,
                stacklevel=2,
            )
