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

"""
FlashInfer: Fast Attention Algorithms for LLM Inference
"""
import ctypes
import os

from .jit.core import logger

# ==============================================
# PyTorch ROCm Compatibility Check
# ==============================================


def _get_system_rocm_version():
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
            return ".".join(version.split(".")[:2])
    except (FileNotFoundError, IOError):
        pass

    # Method 2: Try rocminfo command
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=5, check=False
        )
        if result.returncode == 0:
            match = re.search(r"ROCm version:\s*(\d+\.\d+)", result.stdout)
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
            match = re.search(r"rocm-core\s+(\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def _check_torch_rocm_compatibility():
    """
    Verify that PyTorch is installed with compatible ROCm support.

    This function checks:
    1. PyTorch is installed
    2. PyTorch has ROCm/HIP support (not CPU-only)
    3. PyTorch ROCm version matches system ROCm version (if detectable)

    Provides helpful error messages to guide users to correct installation.
    """

    # Check for torch package
    try:
        import torch
    except ImportError:
        raise ImportError(
            "\n" + "=" * 70 + "\n"
            "ERROR: PyTorch is required but not installed.\n\n"
            "amd-flashinfer requires PyTorch compiled with ROCm support.\n\n"
            "Install PyTorch for ROCm with:\n"
            "  pip install torch==2.7.1 --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/\n\n"
            "Or for a different ROCm version (e.g., 7.0):\n"
            "  pip install torch==2.7.1 --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/\n\n"
            "See https://github.com/rocm/flashinfer for detailed installation instructions.\n"
            + "=" * 70
        )

    # Check for torch package with rocm support
    if not hasattr(torch.version, "hip") or torch.version.hip is None:
        raise RuntimeError(
            "\n" + "=" * 70 + "\n"
            "ERROR: PyTorch does NOT have ROCm support.\n\n"
            "You installed the CPU-only version from PyPI.\n"
            "amd-flashinfer requires PyTorch compiled with ROCm support.\n\n"
            "Fix this by:\n"
            "  1. Uninstall current PyTorch:\n"
            "     pip uninstall torch\n\n"
            "  2. Install PyTorch for ROCm:\n"
            "     pip install torch==2.7.1 --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/\n\n"
            "See https://github.com/rocm/flashinfer for detailed installation instructions.\n"
            + "=" * 70
        )

    # Rocm version compatibility warning
    torch_rocm = torch.version.hip  # e.g., "6.4.0" or "7.0.0"
    torch_rocm_major_minor = ".".join(torch_rocm.split(".")[:2])  # "6.4"

    # Try to detect system ROCm version
    system_rocm = _get_system_rocm_version()

    if system_rocm and torch_rocm_major_minor != system_rocm:
        import warnings

        warnings.warn(
            f"\n{'='*70}\n"
            f"WARNING: ROCm version mismatch detected!\n\n"
            f"  System ROCm version: {system_rocm}\n"
            f"  PyTorch ROCm version: {torch_rocm_major_minor}\n\n"
            f"This may cause runtime errors or crashes.\n\n"
            f"To fix, reinstall PyTorch for your ROCm version:\n"
            f"  pip install torch==2.7.1 --index-url "
            f"https://repo.radeon.com/rocm/manylinux/rocm-rel-{system_rocm}/\n\n"
            f"Or if using uv:\n"
            f"  export FLASHINFER_ROCM_VERSION={system_rocm}\n"
            f"  uv pip install torch==2.7.1 --index-url "
            f"https://repo.radeon.com/rocm/manylinux/rocm-rel-{system_rocm}/\n"
            f"{'='*70}",
            RuntimeWarning,
            stacklevel=2,
        )


try:
    from .__aot_prebuilt_uris__ import prebuilt_ops_uri
except ImportError:
    logger.info("Prebuilt AOT kernels not found, using JIT backend.")
    prebuilt_ops_uri = None

try:
    from .__config__ import __version__, get_info, show

    if __config__.get_info("enable_cuda"):
        cuda_lib_path = os.environ.get(
            "CUDA_LIB_PATH", "/usr/local/cuda/targets/x86_64-linux/lib/"
        )
        if os.path.exists(f"{cuda_lib_path}/libcudart.so.12"):
            ctypes.CDLL(f"{cuda_lib_path}/libcudart.so.12", mode=ctypes.RTLD_GLOBAL)
except ImportError as e:
    logger.error(f"Failed to import __config__: {e}")
    raise e

# Run the Rocm check
_check_torch_rocm_compatibility()

from .activation import gelu_and_mul as gelu_and_mul
from .activation import gelu_tanh_and_mul as gelu_tanh_and_mul
from .activation import silu_and_mul as silu_and_mul
from .attention import BatchAttention as BatchAttention
from .autotuner import autotune as autotune
from .cascade import (
    BatchDecodeWithSharedPrefixPagedKVCacheWrapper as BatchDecodeWithSharedPrefixPagedKVCacheWrapper,
)
from .cascade import (
    BatchPrefillWithSharedPrefixPagedKVCacheWrapper as BatchPrefillWithSharedPrefixPagedKVCacheWrapper,
)
from .cascade import (
    MultiLevelCascadeAttentionWrapper as MultiLevelCascadeAttentionWrapper,
)
from .cascade import merge_state as merge_state
from .cascade import merge_state_in_place as merge_state_in_place
from .cascade import merge_states as merge_states
from .decode import (
    BatchDecodeMlaWithPagedKVCacheWrapper as BatchDecodeMlaWithPagedKVCacheWrapper,
)
from .decode import (
    BatchDecodeWithPagedKVCacheWrapper as BatchDecodeWithPagedKVCacheWrapper,
)
from .decode import (
    CUDAGraphBatchDecodeWithPagedKVCacheWrapper as CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
)
from .decode import cudnn_batch_decode_with_kv_cache as cudnn_batch_decode_with_kv_cache
from .decode import single_decode_with_kv_cache as single_decode_with_kv_cache
from .fp4_quantization import (
    SfLayout,
    block_scale_interleave,
    nvfp4_block_scale_interleave,
    e2m1_and_ufp8sf_scale_to_float,
    fp4_quantize,
    mxfp4_dequantize_host,
    mxfp4_dequantize,
    mxfp4_quantize,
    nvfp4_quantize,
    shuffle_matrix_a,
    shuffle_matrix_sf_a,
)
from .fp8_quantization import mxfp8_dequantize_host, mxfp8_quantize
from .fused_moe import (
    RoutingMethodType,
    GatedActType,
    cutlass_fused_moe,
    reorder_rows_for_gated_act_gemm,
    trtllm_fp4_block_scale_moe,
    trtllm_fp4_block_scale_routed_moe,
    trtllm_fp8_block_scale_moe,
    trtllm_fp8_per_tensor_scale_moe,
)
from .gemm import SegmentGEMMWrapper as SegmentGEMMWrapper
from .gemm import bmm_fp8 as bmm_fp8
from .get_include_paths import get_csrc_dir, get_include, get_tvm_binding_dir
from .mla import BatchMLAPagedAttentionWrapper as BatchMLAPagedAttentionWrapper
from .norm import fused_add_rmsnorm as fused_add_rmsnorm
from .norm import gemma_fused_add_rmsnorm as gemma_fused_add_rmsnorm
from .norm import gemma_rmsnorm as gemma_rmsnorm
from .norm import rmsnorm as rmsnorm
from .page import append_paged_kv_cache as append_paged_kv_cache
from .page import append_paged_mla_kv_cache as append_paged_mla_kv_cache
from .page import get_batch_indices_positions as get_batch_indices_positions
from .page import get_seq_lens as get_seq_lens
from .pod import PODWithPagedKVCacheWrapper as PODWithPagedKVCacheWrapper
from .prefill import (
    BatchPrefillWithPagedKVCacheWrapper as BatchPrefillWithPagedKVCacheWrapper,
)
from .prefill import (
    BatchPrefillWithRaggedKVCacheWrapper as BatchPrefillWithRaggedKVCacheWrapper,
)
from .prefill import (
    single_prefill_with_kv_cache as single_prefill_with_kv_cache,
)
from .prefill import (
    single_prefill_with_kv_cache_return_lse as single_prefill_with_kv_cache_return_lse,
)
from .quantization import packbits as packbits
from .quantization import segment_packbits as segment_packbits
from .rope import apply_llama31_rope as apply_llama31_rope
from .rope import apply_llama31_rope_inplace as apply_llama31_rope_inplace
from .rope import apply_llama31_rope_pos_ids as apply_llama31_rope_pos_ids
from .rope import (
    apply_llama31_rope_pos_ids_inplace as apply_llama31_rope_pos_ids_inplace,
)
from .rope import apply_rope as apply_rope
from .rope import apply_rope_inplace as apply_rope_inplace
from .rope import apply_rope_pos_ids as apply_rope_pos_ids
from .rope import apply_rope_pos_ids_inplace as apply_rope_pos_ids_inplace
from .rope import apply_rope_with_cos_sin_cache as apply_rope_with_cos_sin_cache
from .rope import (
    apply_rope_with_cos_sin_cache_inplace as apply_rope_with_cos_sin_cache_inplace,
)
from .sampling import chain_speculative_sampling as chain_speculative_sampling
from .sampling import min_p_sampling_from_probs as min_p_sampling_from_probs
from .sampling import sampling_from_logits as sampling_from_logits
from .sampling import sampling_from_probs as sampling_from_probs
from .sampling import softmax as softmax
from .sampling import top_k_mask_logits as top_k_mask_logits
from .sampling import top_k_renorm_probs as top_k_renorm_probs
from .sampling import top_k_sampling_from_probs as top_k_sampling_from_probs
from .sampling import (
    top_k_top_p_sampling_from_logits as top_k_top_p_sampling_from_logits,
)
from .sampling import (
    top_k_top_p_sampling_from_probs as top_k_top_p_sampling_from_probs,
)
from .sampling import top_p_renorm_probs as top_p_renorm_probs
from .sampling import top_p_sampling_from_probs as top_p_sampling_from_probs
from .sparse import BlockSparseAttentionWrapper as BlockSparseAttentionWrapper

__all__ = [
    "get_csrc_dir",
    "get_include",
    "get_info",
    "get_tvm_binding_dir",
    "show",
    "prebuilt_ops_uri",
]
