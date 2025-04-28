import dataclasses
import logging
import os
import re
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
import torch.utils.cpp_extension as torch_cpp_ext
from filelock import FileLock

from .env import FLASHINFER_CSRC_DIR as FLASHINFER_CSRC_DIR
from .env import FLASHINFER_GEN_SRC_DIR as FLASHINFER_GEN_SRC_DIR
from .env import FLASHINFER_INCLUDE_DIR as FLASHINFER_INCLUDE_DIR
from .env import FLASHINFER_JIT_DIR as FLASHINFER_JIT_DIR
from .env import FLASHINFER_WORKSPACE_DIR as FLASHINFER_WORKSPACE_DIR

os.makedirs(jit_env.FLASHINFER_WORKSPACE_DIR, exist_ok=True)
os.makedirs(jit_env.FLASHINFER_CSRC_DIR, exist_ok=True)


class FlashInferJITLogger(logging.Logger):
    def __init__(self, name):
        super().__init__(name)
        logging_level = os.getenv("FLASHINFER_LOGGING_LEVEL", "info")
        self.setLevel(logging_level.upper())
        self.addHandler(logging.StreamHandler())
        log_path = jit_env.FLASHINFER_WORKSPACE_DIR / "flashinfer_jit.log"
        if not os.path.exists(log_path):
            # create an empty file
            with open(log_path, "w") as f:  # noqa: F841
                pass
        self.addHandler(logging.FileHandler(log_path))
        # set the format of the log
        self.handlers[0].setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - flashinfer.jit: %(message)s"
            )
        )
        self.handlers[1].setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - flashinfer.jit: %(message)s"
            )
        )


logger = FlashInferJITLogger("flashinfer.jit")


def check_cuda_arch():
    # Collect all detected CUDA architectures
    archs = []
    for cuda_arch_flags in torch_cpp_ext._get_cuda_arch_flags():
        arch = int(re.search(r"compute_(\d+)", cuda_arch_flags).group(1))
        archs.append(arch)

    # Raise error only if all detected architectures are lower than sm75
    if all(arch < 75 for arch in archs):
        raise RuntimeError("FlashInfer requires GPUs with sm75 or higher")


def clear_cache_dir():
    if os.path.exists(jit_env.FLASHINFER_JIT_DIR):
        import shutil

        shutil.rmtree(jit_env.FLASHINFER_JIT_DIR)


common_nvcc_flags = [
    "-DFLASHINFER_ENABLE_FP8_E8M0",
    "-DFLASHINFER_ENABLE_FP4_E2M1",
]
sm90a_nvcc_flags = ["-gencode=arch=compute_90a,code=sm_90a"] + common_nvcc_flags
sm100a_nvcc_flags = ["-gencode=arch=compute_100a,code=sm_100a"] + common_nvcc_flags


@dataclasses.dataclass
class JitSpec:
    name: str
    sources: List[Path]
    extra_cflags: Optional[List[str]]
    extra_cuda_cflags: Optional[List[str]]
    extra_ldflags: Optional[List[str]]
    extra_include_dirs: Optional[List[Path]]
    is_class: bool = False
    needs_device_linking: bool = False

    @property
    def ninja_path(self) -> Path:
        return jit_env.FLASHINFER_JIT_DIR / self.name / "build.ninja"

    @property
    def jit_library_path(self) -> Path:
        return jit_env.FLASHINFER_JIT_DIR / self.name / f"{self.name}.so"

    def get_library_path(self) -> Path:
        if self.is_aot:
            return self.aot_path
        return self.jit_library_path

    @property
    def aot_path(self) -> Path:
        return jit_env.FLASHINFER_AOT_DIR / self.name / f"{self.name}.so"

    @property
    def is_aot(self) -> bool:
        return self.aot_path.exists()

    @property
    def lock_path(self) -> Path:
        return get_tmpdir() / f"{self.name}.lock"

    def write_ninja(self) -> None:
        ninja_path = self.ninja_path
        ninja_path.parent.mkdir(parents=True, exist_ok=True)
        content = generate_ninja_build_for_op(
            name=self.name,
            sources=self.sources,
            extra_cflags=self.extra_cflags,
            extra_cuda_cflags=self.extra_cuda_cflags,
            extra_ldflags=self.extra_ldflags,
            extra_include_dirs=self.extra_include_dirs,
            needs_device_linking=self.needs_device_linking,
        )
        write_if_different(ninja_path, content)

    def build(self, verbose: bool, need_lock: bool = True) -> None:
        lock = (
            FileLock(self.lock_path, thread_local=False) if need_lock else nullcontext()
        )
        with lock:
            run_ninja(jit_env.FLASHINFER_JIT_DIR, self.ninja_path, verbose)

    def load(self, so_path: Path, class_name: str = None):
        load_class = class_name is not None
        loader = torch.classes if load_class else torch.ops
        loader.load_library(so_path)
        if load_class:
            cls = torch._C._get_custom_class_python_wrapper(self.name, class_name)
            return cls
        return getattr(loader, self.name)

    def build_and_load(self, class_name: str = None):
        if self.is_aot:
            return self.load(self.aot_path, class_name)

        # Guard both build and load with the same lock to avoid race condition
        # where another process is building the library and removes the .so file.
        with FileLock(self.lock_path, thread_local=False):
            so_path = self.jit_library_path
            verbose = os.environ.get("FLASHINFER_JIT_VERBOSE", "0") == "1"
            self.build(verbose, need_lock=False)
            result = self.load(so_path, class_name)

        return result


def gen_jit_spec(
    name: str,
    sources: Sequence[Union[str, Path]],
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_paths: Optional[List[Union[str, Path]]] = None,
    needs_device_linking: bool = False,
) -> JitSpec:
    check_cuda_arch()
    verbose = os.environ.get("FLASHINFER_JIT_VERBOSE", "0") == "1"

    cflags = ["-O3", "-std=c++17", "-Wno-switch-bool"]
    cuda_cflags = [
        "-O3",
        "-std=c++17",
        f"--threads={os.environ.get('FLASHINFER_NVCC_THREADS', '1')}",
        "-use_fast_math",
        "-DFLASHINFER_ENABLE_F16",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
        "-DFLASHINFER_ENABLE_FP8_E5M2",
    ]
    if verbose:
        cuda_cflags += [
            "-g",
            "-lineinfo",
            "--ptxas-options=-v",
            "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage",
            "-DCUTLASS_DEBUG_TRACE_LEVEL=2",
        ]
    else:
        # non debug mode
        cuda_cflags += ["-DNDEBUG"]

    cflags += extra_cflags
    cuda_cflags += extra_cuda_cflags
    logger.info(f"Loading JIT ops: {name}")
    check_cuda_arch()
    build_directory = FLASHINFER_JIT_DIR / name
    os.makedirs(build_directory, exist_ok=True)
    if extra_include_paths is None:
        extra_include_paths = []
    extra_include_paths += [
        FLASHINFER_INCLUDE_DIR,
        FLASHINFER_CSRC_DIR,
    ]
    lock = FileLock(FLASHINFER_JIT_DIR / f"{name}.lock", thread_local=False)
    with lock:
        torch_cpp_ext.load(
            name,
            list(map(lambda _: str(_), sources)),
            extra_cflags=cflags,
            extra_cuda_cflags=cuda_cflags,
            extra_ldflags=extra_ldflags,
            extra_include_paths=list(map(lambda _: str(_), extra_include_paths)),
            build_directory=build_directory,
            verbose=verbose,
            with_cuda=True,
            # We switched to torch.library, so will be loaded into torch.ops
            # instead of into a separate module.
            is_python_module=False,
        )
    logger.info(f"Finished loading JIT ops: {name}")
    return getattr(torch.ops, name)
