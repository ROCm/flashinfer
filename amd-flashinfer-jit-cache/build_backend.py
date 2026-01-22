"""
Copyright (c) 2025 by FlashInfer team.
Copyright (c) 2025 by AMD ROCm team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import platform
import sys
from pathlib import Path

from setuptools import build_meta as _orig
from wheel.bdist_wheel import bdist_wheel

# Add parent directory to path to import flashinfer modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip version check when building amd-flashinfer-jit-cache package
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"


def _create_build_metadata():
    """Create build metadata file with version information using setuptools_scm."""
    # Import setuptools_scm to get version
    try:
        from setuptools_scm import get_version

        version = get_version(root="..", relative_to=__file__)
    except Exception as e:
        print(f"Warning: Could not get version from setuptools_scm: {e}")
        version = "0.0.0+unknown"

    build_meta_file = (
        Path(__file__).parent / "amd_flashinfer_jit_cache" / "_build_meta.py"
    )
    build_meta_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if we're in a git repository
    git_dir = Path(__file__).parent.parent / ".git"
    in_git_repo = git_dir.exists()

    # If file exists and not in git repo (installing from sdist), keep existing file
    if build_meta_file.exists() and not in_git_repo:
        print("Build metadata file already exists (not in git repo), keeping it")
        return version

    # In git repo (editable) or file doesn't exist, create/update it
    with open(build_meta_file, "w") as f:
        f.write('"""Build metadata for amd-flashinfer-jit-cache package."""\n')
        f.write(f'__version__ = "{version}"\n')

    print(f"Created build metadata file with version {version}")
    return version


# Create build metadata as soon as this module is imported
_create_build_metadata()


def _compile_jit_cache(output_dir: Path, verbose: bool = True):
    """Compile AOT modules using flashinfer.aot_hip functions directly."""
    from flashinfer import aot_hip

    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Set up build directory
    build_dir = project_root / "build" / "aot_hip"

    # Use the centralized compilation function from aot_hip.py
    aot_hip.compile_and_package_modules(
        out_dir=output_dir,
        build_dir=build_dir,
        project_root=project_root,
        config=None,  # Use default config
        verbose=verbose,
        skip_prebuilt=False,
    )


def _build_aot_modules():
    """Build AOT HIP modules."""
    # First, ensure AOT modules are compiled
    aot_package_dir = Path(__file__).parent / "amd_flashinfer_jit_cache" / "hip_cache"
    aot_package_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Compile AOT modules
        _compile_jit_cache(aot_package_dir)

        # Verify that some modules were actually compiled
        so_files = list(aot_package_dir.rglob("*.so"))
        if not so_files:
            raise RuntimeError("No .so files were generated during AOT compilation")

        print(f"Successfully compiled {len(so_files)} AOT HIP modules")

    except Exception as e:
        print(f"Failed to compile AOT modules: {e}")
        raise


def _prepare_build():
    """Shared preparation logic for both wheel and editable builds."""
    _build_aot_modules()


class PlatformSpecificBdistWheel(bdist_wheel):
    """Custom wheel builder that uses py_limited_api for cp310+."""

    def finalize_options(self):
        super().finalize_options()
        # Force platform-specific wheel (not pure Python)
        self.root_is_pure = False
        # Use py_limited_api for cp310 (Python 3.10+)
        self.py_limited_api = "cp310"

    def get_tag(self):
        # Use py_limited_api tags
        python_tag = "cp310"
        abi_tag = "abi3"  # Stable ABI tag

        # Get platform tag
        machine = platform.machine()
        if platform.system() == "Linux":
            # Use manylinux_2_28 as specified
            if machine == "x86_64":
                plat_tag = "manylinux_2_28_x86_64"
            elif machine == "aarch64":
                plat_tag = "manylinux_2_28_aarch64"
            else:
                plat_tag = f"linux_{machine}"
        else:
            # Fallback for other platforms
            plat_tag = f"{platform.system().lower()}_{machine}"

        return python_tag, abi_tag, plat_tag


class _MonkeyPatchBdistWheel:
    """Context manager to temporarily monkey-patch setuptools bdist_wheel."""

    def __enter__(self):
        import setuptools.command.bdist_wheel as setuptools_bdist_wheel

        self.original_bdist_wheel = setuptools_bdist_wheel.bdist_wheel
        setuptools_bdist_wheel.bdist_wheel = PlatformSpecificBdistWheel
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import setuptools.command.bdist_wheel as setuptools_bdist_wheel

        setuptools_bdist_wheel.bdist_wheel = self.original_bdist_wheel


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build wheel with custom AOT module compilation."""
    print("Building amd-flashinfer-jit-cache wheel...")

    _prepare_build()

    with _MonkeyPatchBdistWheel():
        return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Build editable install with custom AOT module compilation."""
    print("Building amd-flashinfer-jit-cache in editable mode...")

    _prepare_build()

    # Now build the editable install using setuptools
    _orig_build_editable = getattr(_orig, "build_editable", None)
    if _orig_build_editable is None:
        raise RuntimeError("build_editable not supported by setuptools backend")

    result = _orig_build_editable(wheel_directory, config_settings, metadata_directory)

    return result


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    """Prepare metadata with platform-specific wheel tags."""
    with _MonkeyPatchBdistWheel():
        return _orig.prepare_metadata_for_build_wheel(
            metadata_directory, config_settings
        )


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    """Prepare metadata for editable install."""
    with _MonkeyPatchBdistWheel():
        return _orig.prepare_metadata_for_build_editable(
            metadata_directory, config_settings
        )


# Export the required interface
get_requires_for_build_wheel = _orig.get_requires_for_build_wheel
get_requires_for_build_editable = getattr(
    _orig, "get_requires_for_build_editable", None
)
