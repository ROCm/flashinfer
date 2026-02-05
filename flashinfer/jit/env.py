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
from ..device_utils import IS_CUDA, IS_HIP


FLASHINFER_BASE_DIR = pathlib.Path(
    os.getenv("FLASHINFER_WORKSPACE_BASE", pathlib.Path.home().as_posix())
)
FLASHINFER_CACHE_DIR = FLASHINFER_BASE_DIR / ".cache" / "flashinfer"


def _get_workspace_dir_name() -> pathlib.Path:
    if IS_CUDA:
        from torch.utils.cpp_extension import _get_cuda_arch_flags

        try:
            with warnings.catch_warnings():
                # Ignore the warning for TORCH_CUDA_ARCH_LIST not set
                warnings.filterwarnings(
                    "ignore", r".*TORCH_CUDA_ARCH_LIST.*", module="torch"
                )
                flags = _get_cuda_arch_flags()
            arch = "_".join(sorted(set(re.findall(r"compute_(\d+)", "".join(flags)))))
        except Exception:
            arch = "noarch"
        # e.g.: $HOME/.cache/flashinfer/75_80_89_90/
    elif IS_HIP:
        try:
            # Prioritize actual device architecture over PyTorch's build list
            import torch

            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            gcn_arch = props.gcnArchName
            # Extract gfx arch (e.g., "gfx942:sramecc+:xnack-" -> "gfx942")
            match = re.match(r"(gfx\d+)", gcn_arch)
            if match:
                arch = match.group(1)
            else:
                # Fallback to PyTorch's arch list if device detection fails
                from torch.utils.cpp_extension import _get_rocm_arch_flags

                flags = _get_rocm_arch_flags()
                archs = [
                    flag.replace("--offload-arch=", "")
                    for flag in flags
                    if flag.startswith("--offload-arch=")
                ]
                arch = archs[0] if archs else "noarch"
        except Exception:
            arch = "noarch"
        # e.g.: $HOME/.cache/flashinfer/gfx942/
    return FLASHINFER_CACHE_DIR / arch


# use pathlib
FLASHINFER_WORKSPACE_DIR = _get_workspace_dir_name()
FLASHINFER_JIT_DIR = FLASHINFER_WORKSPACE_DIR / "cached_ops"
FLASHINFER_GEN_SRC_DIR = FLASHINFER_WORKSPACE_DIR / "generated"

if IS_CUDA:
    FLASHINFER_CUBIN_DIR = pathlib.Path(
        os.getenv("FLASHINFER_CUBIN_DIR", (FLASHINFER_CACHE_DIR / "cubins").as_posix())
    )
    _package_root = pathlib.Path(__file__).resolve().parents[1]
    FLASHINFER_DATA = _package_root / "data"
    FLASHINFER_INCLUDE_DIR = _package_root / "data" / "include"
    FLASHINFER_CSRC_DIR = _package_root / "data" / "csrc"
    # FLASHINFER_SRC_DIR = _package_root / "data" / "src"
    FLASHINFER_TVM_BINDING_DIR = _package_root / "data" / "tvm_binding"
    FLASHINFER_AOT_DIR = _package_root / "data" / "aot"
    CUTLASS_INCLUDE_DIRS = [
        _package_root / "data" / "cutlass" / "include",
        _package_root / "data" / "cutlass" / "tools" / "util" / "include",
    ]
    SPDLOG_INCLUDE_DIR = _package_root / "data" / "spdlog" / "include"

    def get_nvshmem_include_dirs():
        paths = os.environ.get("NVSHMEM_INCLUDE_PATH")
        if paths is not None:
            return [pathlib.Path(p) for p in paths.split(os.pathsep) if p]

        import nvidia.nvshmem

        path = pathlib.Path(nvidia.nvshmem.__path__[0]) / "include"
        return [path]

    def get_nvshmem_lib_dirs():
        paths = os.environ.get("NVSHMEM_LIBRARY_PATH")
        if paths is not None:
            return [pathlib.Path(p) for p in paths.split(os.pathsep) if p]

        import nvidia.nvshmem

        path = pathlib.Path(nvidia.nvshmem.__path__[0]) / "lib"
        return [path]
elif IS_HIP:
    from ..get_include_paths import _get_package_root_dir, get_csrc_dir, get_include

    # FIXME: Remove once upgrade to v0.4.1 is completed and we move to using
    # amd-flashinfer-jit-cache package.
    _package_root = _get_package_root_dir()
    aot_dir = str(os.path.join(_get_package_root_dir(), "data", "aot"))
    FLASHINFER_AOT_DIR = pathlib.Path(aot_dir)
    FLASHINFER_INCLUDE_DIR = pathlib.Path(get_include())
    FLASHINFER_CSRC_DIR = pathlib.Path(get_csrc_dir())
