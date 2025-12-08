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

import ctypes
import functools
import importlib.util
import os
from typing import Optional, Set

# Re-export
from . import cubin_loader
from . import env as env
from .activation import gen_act_and_mul_module as gen_act_and_mul_module
from .activation import get_act_and_mul_cu_str as get_act_and_mul_cu_str
from .attention import (
    gen_batch_decode_mla_module as gen_batch_decode_mla_module,
)
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
from .attention import gen_fmha_cutlass_sm100a_module as gen_fmha_cutlass_sm100a_module
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



def _get_extension_path(name: str) -> Optional[str]:
    """Try to find installed extension module"""
    try:
        spec = importlib.util.find_spec(name)
        if spec and spec.origin:
            return spec.origin
        return None
    except (ImportError, ModuleNotFoundError):
        return None


# noqa: F401
has_prebuilt_ops = False
from .core import logger

# Try and Except to break circular dependencies
try:
    from ..__aot_prebuilt_uris__ import prebuilt_ops_uri
except ImportError:
    prebuilt_ops_uri = None

try:
    from .. import __config__

    if __config__.get_info("aot_torch_exts_cuda"):
        try:
            from .. import flashinfer_kernels

            has_prebuilt_ops = True
            kernels_path = _get_extension_path("flashinfer.flashinfer_kernels")

        except ImportError:
            logger.warning(
                "CUDA kernels were enabled in build but couldn't be imported"
            )

        # Only try to import SM90 kernels if they were enabled during build
        if 90 in __config__.get_info("aot_torch_exts_cuda_archs"):
            try:
                from .. import flashinfer_kernels_sm90  # noqa: F401

                has_prebuilt_ops = True
                kernels_sm90_path = _get_extension_path(
                    "flashinfer.flashinfer_kernels_sm90"
                )

            except ImportError:
                logger.warning(
                    "SM90 kernels were enabled in build but couldn't be imported"
                )

    if prebuilt_ops_uri is not None and __config__.get_info("aot_torch_exts_hip"):
        try:
            from .. import flashinfer_hip_kernels  # noqa: F401

            logger.info("Loading prebuilt HIP kernels")
            has_prebuilt_ops = True

            kernels_hip_path = _get_extension_path("flashinfer.flashinfer_hip_kernels")

        except ImportError as e:
            print(e)
            logger.warning("HIP kernels were enabled in build but couldn't be imported")

except ImportError:
    pass
