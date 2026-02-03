# SPDX-FileCopyrightText: 2024-2026 FlashInfer team.
# SPDX-FileCopyrightText: 2025-2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0


import ctypes
import functools
import os

from ..device_utils import IS_CUDA, IS_HIP

if IS_CUDA:
    # Re-export
    from . import cubin_loader
    from . import env as env
    from .activation import gen_act_and_mul_module as gen_act_and_mul_module
    from .activation import get_act_and_mul_cu_str as get_act_and_mul_cu_str
    from .attention import cudnn_fmha_gen_module as cudnn_fmha_gen_module
    from .attention import gen_batch_attention_module as gen_batch_attention_module
    from .attention import gen_batch_decode_mla_module as gen_batch_decode_mla_module
    from .attention import gen_batch_decode_module as gen_batch_decode_module
    from .attention import gen_batch_mla_module as gen_batch_mla_module
    from .attention import gen_batch_mla_tvm_binding as gen_batch_mla_tvm_binding
    from .attention import gen_batch_prefill_module as gen_batch_prefill_module
    from .attention import (
        gen_customize_batch_decode_module as gen_customize_batch_decode_module,
    )
    from .attention import (
        gen_customize_batch_decode_tvm_binding as gen_customize_batch_decode_tvm_binding,
    )
    from .attention import (
        gen_customize_batch_prefill_module as gen_customize_batch_prefill_module,
    )
    from .attention import (
        gen_customize_batch_prefill_tvm_binding as gen_customize_batch_prefill_tvm_binding,
    )
    from .attention import (
        gen_customize_single_decode_module as gen_customize_single_decode_module,
    )
    from .attention import (
        gen_customize_single_prefill_module as gen_customize_single_prefill_module,
    )
    from .attention import (
        gen_fmha_cutlass_sm100a_module as gen_fmha_cutlass_sm100a_module,
    )
    from .attention import gen_pod_module as gen_pod_module
    from .attention import gen_sampling_tvm_binding as gen_sampling_tvm_binding
    from .attention import gen_single_decode_module as gen_single_decode_module
    from .attention import gen_single_prefill_module as gen_single_prefill_module
    from .attention import get_batch_attention_uri as get_batch_attention_uri
    from .attention import get_batch_decode_mla_uri as get_batch_decode_mla_uri
    from .attention import get_batch_decode_uri as get_batch_decode_uri
    from .attention import get_batch_mla_uri as get_batch_mla_uri
    from .attention import get_batch_prefill_uri as get_batch_prefill_uri
    from .attention import get_pod_uri as get_pod_uri
    from .attention import get_single_decode_uri as get_single_decode_uri
    from .attention import get_single_prefill_uri as get_single_prefill_uri
    from .attention import trtllm_gen_fmha_module as trtllm_gen_fmha_module
    from .core import JitSpec as JitSpec
    from .core import build_jit_specs as build_jit_specs
    from .core import clear_cache_dir as clear_cache_dir
    from .core import load_cuda_ops as load_cuda_ops
    from .core import gen_jit_spec as gen_jit_spec
    from .core import sm90a_nvcc_flags as sm90a_nvcc_flags
    from .core import sm100a_nvcc_flags as sm100a_nvcc_flags
    from .cubin_loader import setup_cubin_loader

    @functools.cache
    def get_cudnn_fmha_gen_module():
        mod = cudnn_fmha_gen_module()
        op = mod.build_and_load()
        setup_cubin_loader(mod.get_library_path())
        return op

    cuda_lib_path = os.environ.get(
        "CUDA_LIB_PATH", "/usr/local/cuda/targets/x86_64-linux/lib/"
    )
    if os.path.exists(f"{cuda_lib_path}/libcudart.so.12"):
        ctypes.CDLL(f"{cuda_lib_path}/libcudart.so.12", mode=ctypes.RTLD_GLOBAL)
elif IS_HIP:
    # Re-export (HIP/ROCm backend - AMD-ported modules only)
    from . import env as env
    from .activation import gen_act_and_mul_module as gen_act_and_mul_module
    from .activation import get_act_and_mul_cu_str as get_act_and_mul_cu_str
    from .attention import gen_batch_decode_module as gen_batch_decode_module
    from .attention import gen_batch_prefill_module as gen_batch_prefill_module
    from .attention import (
        gen_customize_batch_decode_module as gen_customize_batch_decode_module,
    )
    from .attention import (
        gen_customize_batch_prefill_module as gen_customize_batch_prefill_module,
    )
    from .attention import (
        gen_customize_single_decode_module as gen_customize_single_decode_module,
    )
    from .attention import (
        gen_customize_single_prefill_module as gen_customize_single_prefill_module,
    )
    from .attention import gen_single_decode_module as gen_single_decode_module
    from .attention import (
        gen_single_prefill_module as gen_single_prefill_module,
    )
    from .attention import get_batch_decode_uri as get_batch_decode_uri
    from .attention import get_batch_prefill_uri as get_batch_prefill_uri
    from .attention import get_single_decode_uri as get_single_decode_uri
    from .attention import get_single_prefill_uri as get_single_prefill_uri
    from .core import JitSpec as JitSpec
    from .core import build_jit_specs as build_jit_specs
    from .core import clear_cache_dir as clear_cache_dir
    from .core import load_cuda_ops as load_cuda_ops
    from .core import gen_jit_spec as gen_jit_spec
else:
    # CPU-only torch (no CUDA or HIP)
    raise RuntimeError(
        "FlashInfer requires either CUDA or ROCm/HIP backend. "
        "Detected CPU-only PyTorch installation."
    )
