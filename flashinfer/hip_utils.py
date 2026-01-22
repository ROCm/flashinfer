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
