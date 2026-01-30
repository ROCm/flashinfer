"""
Copyright (c) 2024 by FlashInfer team.

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

# NOTE(lequn): Do not "from .jit.env import xxx".
# Do "from .jit import env as jit_env" and use "jit_env.xxx" instead.
# This helps AOT script to override envs.

import os
import pathlib
import re
import warnings

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ..get_include_paths import get_csrc_dir, get_include


def _get_workspace_dir_name() -> pathlib.Path:
    """Get workspace directory name based on GPU architecture.

    For CUDA: Uses compute capability (e.g., 75_80_89_90)
    For ROCm: Uses gfx architecture (e.g., gfx942)
    Falls back to 'noarch' if detection fails.
    """
    arch = "noarch"

    if HAS_TORCH and torch.cuda.is_available():
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            # ROCm path: extract gfx architecture
            try:
                # Get from environment variable first (set by CompilationContext)
                env_arch_list = os.getenv("FLASHINFER_ROCM_ARCH_LIST")
                if env_arch_list:
                    # Use the first architecture from the list
                    archs = [a.strip() for a in env_arch_list.split(",") if a.strip()]
                    if archs:
                        arch = archs[0].replace("gfx", "gfx")  # Keep gfx prefix
                else:
                    # Auto-detect from current device
                    props = torch.cuda.get_device_properties(
                        torch.cuda.current_device()
                    )
                    gcn_arch = props.gcnArchName
                    # Extract gfx arch (e.g., "gfx942:sramecc+:xnack-" -> "gfx942")
                    match = re.match(r"(gfx\d+)", gcn_arch)
                    if match:
                        arch = match.group(1)
            except Exception:
                pass
        else:
            # CUDA path: extract compute capability
            try:
                from torch.utils.cpp_extension import _get_cuda_arch_flags

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", r".*TORCH_CUDA_ARCH_LIST.*", module="torch"
                    )
                    flags = _get_cuda_arch_flags()
                arch = "_".join(
                    sorted(set(re.findall(r"compute_(\d+)", "".join(flags))))
                )
            except Exception:
                pass

    flashinfer_base = os.getenv(
        "FLASHINFER_WORKSPACE_BASE", pathlib.Path.home().as_posix()
    )
    # e.g.: $HOME/.cache/flashinfer/gfx942/ (ROCm)
    # or:   $HOME/.cache/flashinfer/75_80_89_90/ (CUDA)
    return pathlib.Path(flashinfer_base) / ".cache" / "flashinfer" / arch


# use pathlib
FLASHINFER_WORKSPACE_DIR = _get_workspace_dir_name()
FLASHINFER_JIT_DIR = FLASHINFER_WORKSPACE_DIR / "cached_ops"
FLASHINFER_GEN_SRC_DIR = FLASHINFER_WORKSPACE_DIR / "generated"
FLASHINFER_INCLUDE_DIR = pathlib.Path(get_include())
FLASHINFER_CSRC_DIR = pathlib.Path(get_csrc_dir())
