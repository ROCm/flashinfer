"""
Copyright (c) 2023 by FlashInfer team.

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
import re
import subprocess
from pathlib import Path
from typing import List, Mapping

import setuptools
from setuptools.dist import Distribution

root = Path(__file__).parent.resolve()
aot_ops_package_dir = root / "build" / "aot-ops-package-dir"
enable_aot = aot_ops_package_dir.is_dir() and any(aot_ops_package_dir.iterdir())


def write_if_different(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def get_version():
    package_version = (root / "version.txt").read_text().strip()
    local_version = os.environ.get("FLASHINFER_LOCAL_VERSION")
    if local_version is None:
        return package_version
    return f"{package_version}+{local_version}"


def generate_build_meta(aot_build_meta: dict) -> None:
    build_meta_str = f"__version__ = {get_version()!r}\n"
    if len(aot_build_meta) != 0:
        build_meta_str += f"build_meta = {aot_build_meta!r}\n"
    write_if_different(root / "flashinfer" / "_build_meta.py", build_meta_str)


ext_modules: List[setuptools.Extension] = []
cmdclass: Mapping[str, type[setuptools.Command]] = {}
install_requires = [
    "numpy",
    "torch",
    "ninja",
    "requests",
    "pynvml",
    "einops",
    "packaging>=24.2",
    "nvidia-cudnn-frontend>=1.13.0",
]
generate_build_meta({})

if enable_aot:
    import torch
    import torch.utils.cpp_extension as torch_cpp_ext
    from packaging.version import Version

    def get_cuda_version() -> Version:
        if torch_cpp_ext.CUDA_HOME is None:
            nvcc = "nvcc"
        else:
            nvcc = os.path.join(torch_cpp_ext.CUDA_HOME, "bin/nvcc")
        txt = subprocess.check_output([nvcc, "--version"], text=True)
        return Version(re.findall(r"release (\d+\.\d+),", txt)[0])

    cuda_version = get_cuda_version()
    torch_full_version = Version(torch.__version__)
    torch_version = f"{torch_full_version.major}.{torch_full_version.minor}"
    install_requires = [req for req in install_requires if not req.startswith("torch ")]
    install_requires.append(f"torch == {torch_version}.*")

    aot_build_meta = {}
    aot_build_meta["cuda_major"] = cuda_version.major
    aot_build_meta["cuda_minor"] = cuda_version.minor
    aot_build_meta["torch"] = torch_version
    aot_build_meta["python"] = platform.python_version()
    aot_build_meta["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST")
    generate_build_meta(aot_build_meta)


class AotDistribution(Distribution):
    def has_ext_modules(self) -> bool:
        return enable_aot

    cutlass = root / "3rdparty" / "cutlass"
    include_dirs = [
        root.resolve() / "libflashinfer/include",
        cutlass.resolve() / "include",  # for group gemm
        cutlass.resolve() / "tools" / "util" / "include",
    ]
    cxx_flags = [
        "-O3",
        "-Wno-switch-bool",
        "-DPy_LIMITED_API=0x03080000",
    ]
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "--threads=1",
        "-Xfatbin",
        "-compress-all",
        "-use_fast_math",
        "-DNDEBUG",
        "-DPy_LIMITED_API=0x03080000",
    ]
    libraries = [
        "cublas",
        "cublasLt",
    ]
    sm90a_flags = "-gencode arch=compute_90a,code=sm_90a".split()
    kernel_sources = [
        "csrc/bmm_fp8.cu",
        "csrc/cascade.cu",
        "csrc/group_gemm.cu",
        "csrc/norm.cu",
        "csrc/page.cu",
        "csrc/quantization.cu",
        "csrc/rope.cu",
        "csrc/sampling.cu",
        "csrc/renorm.cu",
        "csrc/activation.cu",
        "csrc/batch_decode.cu",
        "csrc/batch_prefill.cu",
        "csrc/single_decode.cu",
        "csrc/single_prefill.cu",
        # "csrc/pod.cu",  # Temporarily disabled
        "csrc/flashinfer_ops.cu",
        "csrc/custom_all_reduce.cu",
    ]
    kernel_sm90_sources = [
        "csrc/group_gemm_sm90.cu",
        "csrc/single_prefill_sm90.cu",
        "csrc/batch_prefill_sm90.cu",
        "csrc/flashinfer_ops_sm90.cu",
        "csrc/group_gemm_f16_f16_sm90.cu",
        "csrc/group_gemm_bf16_bf16_sm90.cu",
        "csrc/group_gemm_e4m3_f16_sm90.cu",
        "csrc/group_gemm_e5m2_f16_sm90.cu",
        "csrc/group_gemm_e4m3_bf16_sm90.cu",
        "csrc/group_gemm_e5m2_bf16_sm90.cu",
    ]
    decode_sources = list(gen_dir.glob("*decode_head*.cu"))
    prefill_sources = [
        f for f in gen_dir.glob("*prefill_head*.cu") if "_sm90" not in f.name
    ]
    prefill_sm90_sources = list(gen_dir.glob("*prefill_head*_sm90.cu"))
    ext_modules = [
        torch_cpp_ext.CUDAExtension(
            name="flashinfer.flashinfer_kernels",
            sources=kernel_sources + decode_sources + prefill_sources,
            include_dirs=include_dirs,
            libraries=libraries,
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
            py_limited_api=True,
        )
    ]
    if enable_sm90:
        ext_modules += [
            torch_cpp_ext.CUDAExtension(
                name="flashinfer.flashinfer_kernels_sm90",
                sources=kernel_sm90_sources + prefill_sm90_sources,
                include_dirs=include_dirs,
                libraries=libraries,
                extra_compile_args={
                    "cxx": cxx_flags,
                    "nvcc": nvcc_flags + sm90a_flags,
                },
                py_limited_api=True,
            ),
        ]

setuptools.setup(
    version=get_version(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=install_requires,
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
    distclass=AotDistribution,
)
