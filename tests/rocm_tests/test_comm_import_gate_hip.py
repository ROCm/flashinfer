# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for the ROCm import gate in flashinfer/comm/__init__.py.
#
# On ROCm the comm package must:
#   1. Import successfully and expose the backend-agnostic re-exports
#      (Mapping, pack_strided_memory).
#   2. Raise ImportError — not AssertionError — for every CUDA-only submodule,
#      via both `import X` and `from X import Y`. vLLM's
#      flashinfer_all_reduce.py guards on ImportError; anything else escapes.

import importlib

import pytest

from flashinfer.device_utils import IS_HIP

pytestmark = pytest.mark.skipif(
    not IS_HIP, reason="comm import gate is only installed on ROCm"
)


_CUDA_ONLY_SUBMODULES = [
    "flashinfer.comm.cuda_ipc",
    "flashinfer.comm.trtllm_ar",
    "flashinfer.comm.trtllm_alltoall",
    "flashinfer.comm.trtllm_mnnvl_ar",
    "flashinfer.comm.vllm_ar",
    "flashinfer.comm.mnnvl",
    "flashinfer.comm.nvshmem",
    "flashinfer.comm.nvshmem_allreduce",
]


def test_comm_package_imports_and_exposes_backend_agnostic_symbols():
    import flashinfer.comm as comm

    assert hasattr(comm, "Mapping")
    assert hasattr(comm, "pack_strided_memory")


@pytest.mark.parametrize("modname", _CUDA_ONLY_SUBMODULES)
def test_cuda_only_submodule_import_raises_importerror(modname):
    with pytest.raises(ImportError):
        importlib.import_module(modname)


@pytest.mark.parametrize("modname", _CUDA_ONLY_SUBMODULES)
def test_cuda_only_submodule_from_import_raises_importerror(modname):
    # `from X import Y` exercises a different code path than plain
    # `import X` — the loader must raise ImportError for both.
    with pytest.raises(ImportError):
        exec(f"from {modname} import _anything", {})
