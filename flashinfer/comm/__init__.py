from ..device_utils import IS_CUDA

# Backend-agnostic exports (pure Python, work on both CUDA and ROCm).
from .dlpack_utils import pack_strided_memory
from .mapping import Mapping

# CUDA-only re-exports: cuda_ipc binds to libcudart at import time, and
# trtllm_ar / vllm_ar wrap kernels and APIs with no ROCm equivalents.
# Load only on CUDA so `import flashinfer.comm` succeeds on ROCm; callers
# that need these features can detect availability via the exported CUDA-only
# names, such as `trtllm_allreduce_fusion`. Matches the IS_CUDA / IS_HIP
# split in flashinfer/__init__.py.
if IS_CUDA:
    from .cuda_ipc import CudaRTLibrary, create_shared_buffer, free_shared_buffer
    from .trtllm_ar import AllReduceFusionOp as AllReduceFusionOp
    from .trtllm_ar import AllReduceFusionPattern as AllReduceFusionPattern
    from .trtllm_ar import AllReduceStrategyConfig as AllReduceStrategyConfig
    from .trtllm_ar import AllReduceStrategyType as AllReduceStrategyType
    from .trtllm_ar import QuantizationSFLayout as QuantizationSFLayout
    from .trtllm_ar import (
        compute_fp4_swizzled_layout_sf_size as compute_fp4_swizzled_layout_sf_size,
    )
    from .trtllm_ar import gen_trtllm_comm_module as gen_trtllm_comm_module
    from .trtllm_ar import trtllm_allreduce_fusion as trtllm_allreduce_fusion
    from .trtllm_ar import (
        trtllm_create_ipc_workspace_for_all_reduce as trtllm_create_ipc_workspace_for_all_reduce,
    )
    from .trtllm_ar import (
        trtllm_create_ipc_workspace_for_all_reduce_fusion as trtllm_create_ipc_workspace_for_all_reduce_fusion,
    )
    from .trtllm_ar import trtllm_custom_all_reduce as trtllm_custom_all_reduce
    from .trtllm_ar import (
        trtllm_destroy_ipc_workspace_for_all_reduce as trtllm_destroy_ipc_workspace_for_all_reduce,
    )
    from .trtllm_ar import (
        trtllm_destroy_ipc_workspace_for_all_reduce_fusion as trtllm_destroy_ipc_workspace_for_all_reduce_fusion,
    )
    from .trtllm_ar import trtllm_lamport_initialize as trtllm_lamport_initialize
    from .trtllm_ar import (
        trtllm_lamport_initialize_all as trtllm_lamport_initialize_all,
    )
    from .trtllm_ar import trtllm_moe_allreduce_fusion as trtllm_moe_allreduce_fusion
    from .trtllm_ar import (
        trtllm_moe_finalize_allreduce_fusion as trtllm_moe_finalize_allreduce_fusion,
    )
    from .vllm_ar import all_reduce as vllm_all_reduce
    from .vllm_ar import dispose as vllm_dispose
    from .vllm_ar import gen_vllm_comm_module as gen_vllm_comm_module
    from .vllm_ar import get_graph_buffer_ipc_meta as vllm_get_graph_buffer_ipc_meta
    from .vllm_ar import init_custom_ar as vllm_init_custom_ar
    from .vllm_ar import meta_size as vllm_meta_size
    from .vllm_ar import register_buffer as vllm_register_buffer
    from .vllm_ar import register_graph_buffers as vllm_register_graph_buffers
else:
    # On ROCm, every other comm submodule is CUDA-only: cuda_ipc and
    # trtllm_ar eagerly bind to libcudart at import time (AssertionError);
    # vllm_ar / nvshmem / nvshmem_allreduce import OK but explode at call
    # time when JIT loading triggers; mnnvl imports pynvml which isn't
    # installed. Install a meta-path finder that intercepts every
    # CUDA-only submodule lookup and raises a uniform catchable
    # ImportError — for both `from X import Y` and plain `import X`.
    import importlib.abc
    import importlib.machinery
    import sys

    _CUDA_ONLY_SUBMODULES = frozenset(
        {
            "flashinfer.comm.cuda_ipc",
            "flashinfer.comm.trtllm_ar",
            "flashinfer.comm.trtllm_alltoall",
            "flashinfer.comm.trtllm_mnnvl_ar",
            "flashinfer.comm.vllm_ar",
            "flashinfer.comm.mnnvl",
            "flashinfer.comm.nvshmem",
            "flashinfer.comm.nvshmem_allreduce",
        }
    )

    class _CudaOnlyLoader(importlib.abc.Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            raise ImportError(
                f"{module.__spec__.name} is CUDA-only and not available on ROCm"
            )

    class _CudaOnlyFinder(importlib.abc.MetaPathFinder):
        # Stable marker so the idempotency check below survives
        # importlib.reload(flashinfer.comm), which redefines the class object
        # and would otherwise make isinstance() return False against the
        # already-installed finder from the previous load.
        _is_flashinfer_cuda_only_finder = True

        def find_spec(self, fullname, path=None, target=None):
            if fullname in _CUDA_ONLY_SUBMODULES:
                return importlib.machinery.ModuleSpec(fullname, _CudaOnlyLoader())
            return None

    if not any(
        getattr(f, "_is_flashinfer_cuda_only_finder", False) for f in sys.meta_path
    ):
        sys.meta_path.insert(0, _CudaOnlyFinder())

# from .mnnvl import MnnvlMemory, MnnvlMoe, MoEAlltoallInfo
