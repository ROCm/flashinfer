# Qwen2.5 single sequence prefill.
# ROCm FlashInfer + Aiter Hybrid version
# - Uses FlashInfer for: RoPE, Activation (silu_and_mul), RMSNorm (高度融合)
# - Uses Aiter for: MHA (高性能)
# - Uses record_function for ROCm compatibility (NVTX not available in ROCm)
# - Markers added: qwen2_model, qwen2_decoder_layer, qwen2_attention, qwen2_mlp, qwen2_rmsnorm
import gc
import time
import os
import sys
import argparse
from typing import Callable, Optional
from datetime import datetime

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function, tensorboard_trace_handler

# Try to import nvtx for profiling, but make it optional
try:
    import nvtx
except ImportError:
    # Create a dummy nvtx module if not available (for ROCm environments)
    class DummyNVTX:
        @staticmethod
        def push_range(name):
            pass
        @staticmethod
        def pop_range():
            pass
    nvtx = DummyNVTX()

# Suppress FlashInfer JIT compilation output
# Set environment variables to suppress CMake/ninja output
os.environ.setdefault('CMAKE_MESSAGE_LOG_LEVEL', 'ERROR')
os.environ.setdefault('VERBOSE', '0')

# Import flashinfer with suppressed output
import contextlib
import io

# Suppress CMake/ninja output during flashinfer import and JIT compilation
# These messages are normal: FlashInfer uses JIT compilation if prebuilt kernels are not available
class SuppressFlashInferOutput:
    """Context manager to suppress FlashInfer JIT compilation output"""
    def __init__(self):
        self.original_stderr = None
        self.devnull = None
    
    def __enter__(self):
        self.original_stderr = sys.stderr
        self.devnull = open(os.devnull, 'w')
        sys.stderr = self.devnull
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.devnull:
            self.devnull.close()
        sys.stderr = self.original_stderr
        return False

# Suppress output during import
with SuppressFlashInferOutput():
    import flashinfer

# Import Aiter MHA only (we use FlashInfer for RoPE, Activation, RMSNorm)
USE_AITER_MHA = False
aiter_mha = None

def load_aiter_mha():
    """When loading the Aiter MHA module, the pre-installed version is used first; if that fails, the local version is tried."""
    global USE_AITER_MHA, aiter_mha
    
    try:
        import aiter
        from aiter.ops import mha as aiter_mha_module
        print(f"✓ Using aiter from {aiter.__file__}")
        USE_AITER_MHA = True
        return aiter_mha_module
    except Exception as e:
        print(f"⚠ Aiter not found: {e}")
        print("Please install Aiter by running: pip install amd-aiter")
        print("Falling back to FlashInfer MHA")
    return None


# Load Aiter MHA module
aiter_mha = load_aiter_mha()

num_kv_heads = 2
num_qo_heads = 14
hidden_size = 896
intermediate_size = 4864
rms_norm_eps = 1e-6
pad_token_id = 151643
vocab_size = 151936
num_hidden_layers = 24
max_seq_len = 32 * 1024
rope_theta = 1000000.0
device = 'cuda:0'


# ============================================================================
# Buffer Management Helper Functions
# ============================================================================

def create_cuda_graph_buffers(max_seq_len, num_kv_heads, head_dim, device):
    """Pre-allocated buffer required to create CUDA Graph"""
    page_size = 128
    max_kv_pages = (max_seq_len + page_size - 1) // page_size
    max_padded_len = max_kv_pages * page_size
    
    return {
        'kv_indptr_buf': torch.zeros(2, dtype=torch.int32, device=device),
        'kv_page_indices_buf': torch.zeros(max_kv_pages + 128, dtype=torch.int32, device=device),
        'cu_seqlens_q_buf': torch.zeros(2, dtype=torch.int32, device=device),
        'k_padded_buf': torch.zeros(max_padded_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device),
        'v_padded_buf': torch.zeros(max_padded_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device),
    }

def reset_attention_buffers(model):
    """Reset all attention layer buffers"""
    for module in model.modules():
        if isinstance(module, Qwen2Attention):
            for buf_name in ['kv_indptr_buf', 'kv_page_indices_buf', 'cu_seqlens_q_buf', 'k_padded_buf', 'v_padded_buf']:
                buf = getattr(module, buf_name, None)
                if buf is not None:
                    buf.zero_()

def set_cuda_graph_capture_mode(model, enabled):
    """Set CUDA Graph capture mode"""
    for module in model.modules():
        if isinstance(module, Qwen2Attention):
            module._cuda_graph_capture_mode = enabled

def preset_buffer_values(model, seq_len):
    """Pre-set buffer values (before capture)"""
    if not USE_AITER_MHA:
        return
    
    page_size = 128
    kv_num_used_pages = (seq_len + page_size - 1) // page_size
    
    for module in model.modules():
        if isinstance(module, Qwen2Attention):
            # cu_seqlens_q
            if module.cu_seqlens_q_buf is not None:
                module.cu_seqlens_q_buf[0], module.cu_seqlens_q_buf[1] = 0, seq_len
            
            # kv_indptr
            if module.kv_indptr_buf is not None:
                module.kv_indptr_buf[0], module.kv_indptr_buf[1] = 0, kv_num_used_pages
            
            # kv_page_indices
            if module.kv_page_indices_buf is not None and module._temp_arange_buf is not None:
                if kv_num_used_pages > 0:
                    module.kv_page_indices_buf[:kv_num_used_pages].copy_(module._temp_arange_buf[:kv_num_used_pages])
                    module.kv_page_indices_buf[kv_num_used_pages:].zero_()

def warmup_model(model, input_ids, num_warmup=15, check_jit=True):
    """Warmup model, ensure JIT compilation is completed"""
    sys.stderr.write(f"Warmup: {num_warmup} iterations\n")
    
    for i in range(num_warmup):
        _ = model(input_ids)
        torch.cuda.synchronize()
        if (i + 1) % 3 == 0:
            time.sleep(0.05)
    
    # Additional warmup (ensure JIT is completed)
    if check_jit and USE_AITER_MHA:
        sys.stderr.write("Additional warmup for JIT compilation...\n")
        for i in range(5):
            _ = model(input_ids)
            torch.cuda.synchronize()
            if i < 4:
                time.sleep(0.1)
    
    sys.stderr.write("Warmup completed\n")


class Qwen2RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linti ng for `register_buffer`

    def __init__(self):
        super().__init__()
        self.max_seq_len_cached = max_seq_len
        self.original_max_seq_len = max_seq_len

        rope_init_fn: Callable = self.compute_default_rope_parameters
        inv_freq, self.attention_scaling = rope_init_fn()

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters() -> tuple["torch.Tensor", float]:
        base = rope_theta
        dim = hidden_size // num_qo_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    def forward(self, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(device)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        # emb = torch.cat((freqs, freqs), dim=-1)
        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling

        return torch.concat((cos, sin), dim=-1).to(torch.float32).squeeze(0)


class Qwen2Attention(nn.Module):

    def __init__(self, layer_idx, kv_indptr_buf=None, kv_page_indices_buf=None, cu_seqlens_q_buf=None, k_padded_buf=None, v_padded_buf=None):
        self.layer_idx = layer_idx  # Store layer_idx for debugging
        super().__init__()
        self.layer_idx = layer_idx
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_qo_heads
        self.rope_scale = 1
        self.rope_theta = 1000000.0

        # TODO: load weight from file
        self.qkv_proj_w = torch.randn(hidden_size, (self.num_qo_heads + 2 * self.num_kv_heads) * self.head_dim,
                                      dtype=torch.bfloat16, device=device)
        self.qkv_proj_b = torch.randn((self.num_qo_heads + 2 * self.num_kv_heads) * self.head_dim, dtype=torch.bfloat16,
                                      device=device)
        self.o_proj_w = torch.randn(self.num_qo_heads * self.head_dim, hidden_size, dtype=torch.bfloat16, device=device)
        
        # Pre-allocated buffers for CUDA Graph compatibility (for Aiter MHA)
        # These buffers will be reused during forward pass instead of creating new tensors
        self.kv_indptr_buf = kv_indptr_buf  # [2] buffer for kv_indptr
        self.kv_page_indices_buf = kv_page_indices_buf  # Pre-allocated buffer for kv_page_indices
        self.cu_seqlens_q_buf = cu_seqlens_q_buf  # [2] buffer for cu_seqlens_q
        self.k_padded_buf = k_padded_buf  # Pre-allocated buffer for k_padded
        self.v_padded_buf = v_padded_buf  # Pre-allocated buffer for v_padded
        
        # Pre-allocated temporary buffers for setting values during CUDA Graph capture
        max_kv_pages = (max_seq_len + 127) // 128  # Max pages for max_seq_len
        if kv_page_indices_buf is not None:
            # Pre-create arange buffer for max possible pages
            self._temp_arange_buf = torch.arange(max_kv_pages, dtype=torch.int32, device=device)
        else:
            self._temp_arange_buf = None
        
        # CUDA Graph capture mode flag
        self._cuda_graph_capture_mode = False

        # # create a 1MB workspace buffer
        # workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.bfloat16, device=device)
        # self.qkv_proj_wrapper = flashinfer.SegmentGEMMWrapper(workspace_buffer, backend='auto')
        # self.o_proj_wrapper = flashinfer.SegmentGEMMWrapper(workspace_buffer, backend='auto')

    def _prepare_aiter_inputs(self, q, k, v, seq_len):
        """Prepare input data for Aiter MHA (simplified version)"""
        page_size = 128
        kv_num_used_pages = (seq_len + page_size - 1) // page_size
        padded_len = kv_num_used_pages * page_size
        
        # cu_seqlens_q
        if self.cu_seqlens_q_buf is not None:
            cu_seqlens_q = self.cu_seqlens_q_buf
            if not self._cuda_graph_capture_mode:
                cu_seqlens_q[0], cu_seqlens_q[1] = 0, seq_len
        else:
            cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        
        # k_padded, v_padded
        if self.k_padded_buf is not None and self.v_padded_buf is not None:
            k_padded = self.k_padded_buf[:padded_len]
            v_padded = self.v_padded_buf[:padded_len]
            k_padded[:seq_len] = k.contiguous()
            v_padded[:seq_len] = v.contiguous()
            if seq_len < padded_len:
                k_padded[seq_len:].zero_()
                v_padded[seq_len:].zero_()
        else:
            k_padded = k.contiguous()
            v_padded = v.contiguous()
        
        # Reshape to blocks
        k_blocks = k_padded.reshape(kv_num_used_pages * page_size, num_kv_heads, self.head_dim).contiguous()
        v_blocks = v_padded.reshape(kv_num_used_pages * page_size, num_kv_heads, self.head_dim).contiguous()
        
        # kv_indptr
        if self.kv_indptr_buf is not None:
            kv_indptr = self.kv_indptr_buf
            if not self._cuda_graph_capture_mode:
                kv_indptr[0], kv_indptr[1] = 0, kv_num_used_pages
        else:
            kv_indptr = torch.tensor([0, kv_num_used_pages], dtype=torch.int32, device=device)
        
        # kv_page_indices
        if self.kv_page_indices_buf is not None:
            kv_page_indices = self.kv_page_indices_buf
            if not self._cuda_graph_capture_mode:
                if kv_num_used_pages > 0 and self._temp_arange_buf is not None:
                    kv_page_indices[:kv_num_used_pages].copy_(self._temp_arange_buf[:kv_num_used_pages])
                    kv_page_indices[kv_num_used_pages:].zero_()
                else:
                    kv_page_indices[:kv_num_used_pages] = torch.arange(kv_num_used_pages, device=device, dtype=torch.int32)
                    kv_page_indices[kv_num_used_pages:].zero_()
        else:
            kv_page_indices = torch.arange(kv_num_used_pages, dtype=torch.int32, device=device)
        
        return q.contiguous(), k_blocks, v_blocks, cu_seqlens_q, kv_indptr, kv_page_indices
    
    def _call_aiter_mha_with_retry(self, q_flat, k_blocks, v_blocks, cu_seqlens_q, kv_indptr, kv_page_indices, seq_len):
        """Call Aiter MHA, with JIT compilation retry"""
        max_retries = 5
        retry_delay = 0.2
        
        for retry in range(max_retries):
            result = aiter_mha.mha_batch_prefill_func(
                q=q_flat, k=k_blocks, v=v_blocks,
                cu_seqlens_q=cu_seqlens_q, kv_indptr=kv_indptr, kv_page_indices=kv_page_indices,
                max_seqlen_q=seq_len, max_seqlen_k=seq_len,
                dropout_p=0.0, softmax_scale=None, causal=True,
                window_size=(-1, -1), return_lse=False, return_attn_probs=False,
            )
            
            if result is not None:
                return result[0] if isinstance(result, tuple) else result
            
            if self._cuda_graph_capture_mode:
                raise ValueError("Aiter MHA JIT not compiled in capture mode")
            
            if retry < max_retries - 1:
                torch.cuda.synchronize()
                time.sleep(retry_delay)
                retry_delay *= 1.5
        
        raise ValueError("Aiter MHA JIT not compiled after retries")
    
    def _reshape_aiter_output(self, o_flat, seq_len):
        """Simplified output shape handling"""
        expected_elements = seq_len * self.num_qo_heads * self.head_dim
        if o_flat.numel() != expected_elements:
            raise ValueError(f"Output size mismatch: {o_flat.numel()} != {expected_elements}")
        
        # Unified handling: reshape to target shape regardless of input shape
        o = o_flat.reshape(seq_len, self.num_qo_heads * self.head_dim).contiguous()
        
        if o.shape != (seq_len, self.num_qo_heads * self.head_dim):
            raise ValueError(f"Failed to reshape: {o.shape}")
        
        return o
    
    def _pytorch_attention(self, q, k, v, seq_len):
        """Simplified PyTorch attention implementation (only for warmup fallback)"""
        q_heads = q.reshape(seq_len, self.num_qo_heads, self.head_dim)
        k_heads = k.reshape(seq_len, self.num_kv_heads, self.head_dim)
        v_heads = v.reshape(seq_len, self.num_kv_heads, self.head_dim)
        
        # Simple scaled dot-product attention
        q_heads = q_heads * (self.head_dim ** -0.5)
        kv_group_size = self.num_qo_heads // self.num_kv_heads
        k_heads_expanded = k_heads.repeat_interleave(kv_group_size, dim=1)
        v_heads_expanded = v_heads.repeat_interleave(kv_group_size, dim=1)
        
        attn_scores = torch.einsum('shd,thd->sht', q_heads, k_heads_expanded)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=attn_scores.dtype), diagonal=1)
        causal_mask = causal_mask.unsqueeze(1).expand(seq_len, self.num_qo_heads, seq_len)
        attn_scores = attn_scores.masked_fill(causal_mask.bool(), float('-inf'))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.einsum('sht,thd->shd', attn_probs, v_heads_expanded)
        
        return attn_output.reshape(seq_len, self.num_qo_heads * self.head_dim)
    
    def _fallback_attention(self, q, k, v, seq_len, error_msg=""):
        """Fallback attention实现"""    
        is_warmup = not self._cuda_graph_capture_mode
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        
        # Only record in non-warmup phase or non-JIT error
        if not is_warmup or "JIT" not in error_msg:
            if not hasattr(self, '_fallback_logged'):
                backend = 'PyTorch' if (is_rocm or (is_warmup and "JIT" in error_msg)) else 'FlashInfer'
                sys.stderr.write(f"⚠ Layer {self.layer_idx}: Using {backend} fallback\n")
                self._fallback_logged = True
        
        # ROCm environment or warmup phase: use PyTorch fallback
        if is_rocm or (is_warmup and "JIT" in error_msg):
            return self._pytorch_attention(q, k, v, seq_len)
        else:
            # FlashInfer fallback
            return flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, kv_layout='NHD').reshape(seq_len, -1).contiguous()

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor, position_embeddings: torch.Tensor):
        nvtx.push_range('qwen2_attention')
        with record_function("qwen2_attention"):
            # hidden_states: [seq_len, hidden_size]
            seq_len = hidden_states.shape[0]
            # seq_lens = [hidden_states.shape[0]]
            # seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)

            # qkv proj
            # qkv_proj_w = self.qkv_proj_w.unsqueeze(0)  # [1, 2 * num_kv_heads + num_qo_heads, hidden_size]
            # qkv = self.qkv_proj_wrapper.run(hidden_states, qkv_proj_w, 1, True, seg_lens=seq_lens)
            qkv = torch.matmul(hidden_states, self.qkv_proj_w)  # [seq_len, (2 * num_kv_heads + num_qo_heads)  * head_dim]
            # print(f'qkv.shape: {qkv.shape}')

            # bias
            qkv = qkv + self.qkv_proj_b  # [seq_len, (2 * num_kv_heads + num_qo_heads)  * head_dim]

            # apply rope for qk - USE FLASHINFER RoPE (高度融合)
            q = qkv[..., :self.num_qo_heads * self.head_dim]
            # [seq_len, num_qo_heads * head_dim]
            k = qkv[..., self.num_qo_heads * self.head_dim
                         :(self.num_qo_heads + self.num_kv_heads) * self.head_dim]  # [seq_len, num_kv_heads * head_dim]
            q_rope, k_rope = flashinfer.apply_rope_with_cos_sin_cache(positions, q, k, self.head_dim, position_embeddings)

            # mha - USE AITER MHA (高性能)
            q = q_rope
            k = k_rope
            q = q.reshape(seq_len, self.num_qo_heads, self.head_dim)  # [seq_len, num_qo_heads, head_dim]
            k = k.reshape(seq_len, self.num_kv_heads, self.head_dim)  # [seq_len, num_kv_heads, head_dim]
            v = (qkv[..., (self.num_qo_heads + self.num_kv_heads) * self.head_dim:]
                 .reshape(seq_len, num_kv_heads, self.head_dim))  # [seq_len, num_kv_heads, head_dim]
            
            # Use Aiter MHA if available, otherwise fallback to FlashInfer
            if USE_AITER_MHA and aiter_mha is not None:
                try:
                    q_flat, k_blocks, v_blocks, cu_seqlens_q, kv_indptr, kv_page_indices = \
                        self._prepare_aiter_inputs(q, k, v, seq_len)
                    o_flat = self._call_aiter_mha_with_retry(q_flat, k_blocks, v_blocks, cu_seqlens_q, kv_indptr, kv_page_indices, seq_len)
                    o = self._reshape_aiter_output(o_flat, seq_len)
                    
                    # Log usage (once per layer, skip WARMUP to reduce output)
                    if not hasattr(self, '_aiter_mha_used_logged'):
                        mode = "CAPTURE/REPLAY" if self._cuda_graph_capture_mode else "WARMUP"
                        # Only print for CAPTURE/REPLAY mode, skip WARMUP to reduce output
                        if self._cuda_graph_capture_mode:
                            sys.stderr.write(f"✓ Layer {self.layer_idx}: Using Aiter MHA ({mode})\n")
                        self._aiter_mha_used_logged = True
                except Exception as e:
                    o = self._fallback_attention(q, k, v, seq_len, str(e))
            else:
                # Fallback to FlashInfer MHA
                o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, kv_layout='NHD')  # [seq_len, num_qo_heads * head_dim]
                o = o.reshape(seq_len, -1).contiguous()

            # o proj
            # o_proj_w = self.o_proj_w.unsqueeze(0)  # [1, hidden_size, num_qo_heads * head_dim]
            # o = self.o_proj_wrapper.run(o, o_proj_w, 1, True, seg_lens=seq_lens)  # [seq_len, hidden_size]
            
            # Final shape check before matmul
            expected_o_shape = (seq_len, self.num_qo_heads * self.head_dim)
            if o.shape != expected_o_shape:
                raise RuntimeError(f"Output shape mismatch before o_proj: got {o.shape}, expected {expected_o_shape}. "
                                 f"This indicates Aiter MHA output was not correctly reshaped.")
            
            o = torch.matmul(o, self.o_proj_w)  # [seq_len, hidden_size]

            # print(f'o.shape: {o.shape}')
        nvtx.pop_range()
        return o


class Qwen2MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # TODO: load weight from file
        self.gate_proj = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device=device)
        self.up_proj = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16, device=device)
        self.gate_up_proj = torch.cat([self.gate_proj, self.up_proj], dim=1)
        self.down_proj = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16, device=device)

    def forward(self, x: torch.Tensor):
        nvtx.push_range('qwen2_mlp')
        with record_function("qwen2_mlp"):
            # x: [seq_len, hidden_size]

            # gate proj
            # gate = torch.matmul(x, self.gate_proj)  # [seq_len, intermediate_size]
            # up proj
            # up = torch.matmul(x, self.up_proj)  # [seq_len, intermediate_size]

            # gate and up proj fusion
            gate_up_fuse = torch.matmul(x, self.gate_up_proj)  # [seq_len, 2 * intermediate_size]

            # silu and matmul
            # Try FlashInfer first, fallback to PyTorch if JIT compilation fails
            try:
                up = flashinfer.silu_and_mul(gate_up_fuse)
            except (AttributeError, RuntimeError) as e:
                # FlashInfer JIT compilation may fail, use PyTorch fallback
                # Split gate_up_fuse into gate and up
                gate, up = gate_up_fuse.chunk(2, dim=-1)  # Each: [seq_len, intermediate_size]
                # Apply SiLU to gate: silu(x) = x * sigmoid(x)
                gate = gate * torch.sigmoid(gate)
                # Multiply: up = gate * up
                up = gate * up

            # down proj
            down = torch.matmul(up, self.down_proj)  # [seq_len, hidden_size]

        nvtx.pop_range()
        return down


class Qwen2AddRMSNormFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = rms_norm_eps

        # TODO: load weight from file
        self.weight = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor = None):
        nvtx.push_range('qwen2_rmsnorm')
        with record_function("qwen2_rmsnorm"):
            if residual is None:
                result = flashinfer.rmsnorm(hidden_states, self.weight, self.eps)
                nvtx.pop_range()
                return result
            else:
                flashinfer.fused_add_rmsnorm(hidden_states, residual, self.weight, self.eps)
                nvtx.pop_range()
                return hidden_states, residual


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, layer_idx, kv_indptr_buf=None, kv_page_indices_buf=None, cu_seqlens_q_buf=None, k_padded_buf=None, v_padded_buf=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        # CRITICAL: Each layer needs its own buffer to avoid conflicts during CUDA Graph capture
        # Create separate buffers for each layer
        self.attention = Qwen2Attention(
            layer_idx,
            kv_indptr_buf=torch.zeros(2, dtype=torch.int32, device=device) if kv_indptr_buf is not None else None,
            kv_page_indices_buf=torch.zeros(kv_page_indices_buf.shape[0], dtype=torch.int32, device=device) if kv_page_indices_buf is not None else None,
            cu_seqlens_q_buf=torch.zeros(2, dtype=torch.int32, device=device) if cu_seqlens_q_buf is not None else None,
            k_padded_buf=k_padded_buf,  # k_padded and v_padded can be shared as they're not modified during forward
            v_padded_buf=v_padded_buf
        )
        self.mlp = Qwen2MLP()
        self.input_layernorm = Qwen2AddRMSNormFusion()
        self.post_attention_layernorm = Qwen2AddRMSNormFusion()

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor, position_embeddings: torch.Tensor,
                residual: torch.Tensor):
        nvtx.push_range('qwen2_decoder_layer')
        with record_function("qwen2_decoder_layer"):
            # pre-layer residual add and current layer norm fusion.
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

            # self attention
            hidden_states = self.attention(hidden_states, positions, position_embeddings)

            # post-attention residual add and layer norm fusion.
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

            # ffn
            hidden_states = self.mlp(hidden_states)

        nvtx.pop_range()
        return hidden_states, residual


class Qwen2Model(nn.Module):
    def __init__(self, kv_indptr_buf=None, kv_page_indices_buf=None, cu_seqlens_q_buf=None, k_padded_buf=None, v_padded_buf=None):
        super().__init__()
        self.padding_idx = pad_token_id
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # TODO: load embed weight from file
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, self.padding_idx, dtype=torch.bfloat16,
                                         device=device)
        # CRITICAL: Each layer needs its own buffer to avoid conflicts during CUDA Graph capture
        # Create separate buffers for each layer
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(
                i,
                kv_indptr_buf=torch.zeros(2, dtype=torch.int32, device=device) if kv_indptr_buf is not None else None,
                kv_page_indices_buf=torch.zeros(kv_page_indices_buf.shape[0], dtype=torch.int32, device=device) if kv_page_indices_buf is not None else None,
                cu_seqlens_q_buf=torch.zeros(2, dtype=torch.int32, device=device) if cu_seqlens_q_buf is not None else None,
                k_padded_buf=k_padded_buf,  # k_padded and v_padded can be shared as they're not modified during forward
                v_padded_buf=v_padded_buf
            ) for i in range(num_hidden_layers)
        ])
        self.norm = Qwen2AddRMSNormFusion()

        # TODO: load lm head weight from file
        self.lm_head = torch.randn(self.hidden_size, self.vocab_size, dtype=torch.bfloat16, device=device)

        # prepare rope cos sin for max_seq_len
        self.rotary_emb = Qwen2RotaryEmbedding()
        self.position_embeddings = self.rotary_emb(
            torch.arange(max_seq_len, dtype=torch.int32, device=device).unsqueeze(0))
        print(f'position_embeddings shape: {self.position_embeddings.shape}')

    def forward(self, input_ids: torch.Tensor):
        nvtx.push_range('qwen2_model')
        with record_function("qwen2_model"):
            seq_len = input_ids.shape[0]
            positions = torch.arange(seq_len, dtype=torch.int32, device=device)

            # embedding
            hidden_states = self.embed_tokens(input_ids)  # [seq_len, hidden_size]
            residual = torch.zeros_like(hidden_states)  # [seq_len, hidden_size]

            for decoder_layer in self.layers:
                hidden_states, residual = decoder_layer(hidden_states, positions, self.position_embeddings, residual)
                # print(f'decoder_layer: {decoder_layer.layer_idx}, residual shape: {residual.shape},'
                #       f' hidden_states shape: {hidden_states.shape}')

            # last layer residual add and layer norm fusion.
            hidden_states, residual = self.norm(hidden_states, residual)  # [seq_len, hidden_size]

            # lm head
            lm_head = torch.matmul(hidden_states, self.lm_head)  # [seq_len, vocab_size]

        nvtx.pop_range()
        return lm_head


def test_qwen2_attention():
    qwen2_attention = Qwen2Attention(0)

    hidden_states = torch.randn(500, 896, dtype=torch.bfloat16, device=device)
    positions = torch.arange(500, dtype=torch.int32, device=device)
    # prepare rope cos sin for max_seq_len
    rotary_emb = Qwen2RotaryEmbedding()
    position_embeddings = rotary_emb(torch.arange(max_seq_len, dtype=torch.int32, device=device).unsqueeze(0))

    new_hidden_states = qwen2_attention(hidden_states, positions, position_embeddings)
    print(f'test_qwen2_attention: input shape:  {hidden_states.shape}, positions shape: {positions.shape},'
          f' position_embeddings shape: {position_embeddings.shape}, output shape: {new_hidden_states.shape}')

    for i in range(10):
        torch.cuda.synchronize()
        begin = time.time()
        nvtx.push_range('qwen2_attention')
        new_hidden_states = qwen2_attention(hidden_states, positions, position_embeddings)
        nvtx.pop_range()
        torch.cuda.synchronize()
        end = time.time()
        print(f'test_qwen2_attention: {i}th inference time: {(end - begin) * 1e3} ms')


def test_qwen2_mlp():
    qwen2_mlp = Qwen2MLP()
    hidden_states = torch.randn(500, 896, dtype=torch.bfloat16, device=device)
    new_hidden_states = qwen2_mlp(hidden_states)
    print(f'test_qwen2_mlp: input shape:  {hidden_states.shape}, output shape: {new_hidden_states.shape}')

    for i in range(10):
        torch.cuda.synchronize()
        begin = time.time()
        nvtx.push_range('qwen2_mlp')
        new_hidden_states = qwen2_mlp(hidden_states)
        nvtx.pop_range()
        torch.cuda.synchronize()
        end = time.time()
        print(f'test_qwen2_mlp: {i}th inference time: {(end - begin) * 1e3} ms')


def test_qwen2_rmsnorm():
    qwen2_rmsnorm = Qwen2AddRMSNormFusion()
    hidden_states = torch.randn(500, 896, dtype=torch.bfloat16, device=device)
    residual = torch.zeros_like(hidden_states)
    new_hidden_states, residual = qwen2_rmsnorm(hidden_states, residual)
    print(f'test_qwen2_rmsnorm: input shape:  {hidden_states.shape}, output shape: {new_hidden_states.shape}')

    for i in range(10):
        torch.cuda.synchronize()
        begin = time.time()
        nvtx.push_range('qwen2_rmsnorm')
        new_hidden_states = qwen2_rmsnorm(hidden_states)
        nvtx.pop_range()
        torch.cuda.synchronize()
        end = time.time()
        print(f'test_qwen2_rmsnorm: {i}th inference time: {(end - begin) * 1e3} ms')


def test_qwen2_decoder_layer():
    qwen2_decoder_layer = Qwen2DecoderLayer(0, kv_indptr_buf=None, kv_page_indices_buf=None, cu_seqlens_q_buf=None, k_padded_buf=None, v_padded_buf=None)
    hidden_states = torch.randn(500, 896, dtype=torch.bfloat16, device=device)
    residual = torch.zeros_like(hidden_states)
    positions = torch.arange(500, dtype=torch.int32, device=device)
    # prepare rope cos sin for max_seq_len
    rotary_emb = Qwen2RotaryEmbedding()
    position_embeddings = rotary_emb(torch.arange(max_seq_len, dtype=torch.int32, device=device).unsqueeze(0))

    (new_hidden_states, residual) = qwen2_decoder_layer(hidden_states, positions, position_embeddings, residual)
    print(f'test_qwen2_attention: input shape:  {hidden_states.shape}, positions shape: {positions.shape},'
          f' position_embeddings shape: {position_embeddings.shape}, output shape: {new_hidden_states.shape}')

    for i in range(10):
        torch.cuda.synchronize()
        begin = time.time()
        nvtx.push_range('qwen2_decoder_layer')
        qwen2_decoder_layer(hidden_states, positions, position_embeddings, residual)
        nvtx.pop_range()
        torch.cuda.synchronize()
        end = time.time()
        print(f'test_qwen2_decoder_layer: {i}th inference time: {(end - begin) * 1e3} ms')


def test_qwen2_model():
    qwen2_model = Qwen2Model().to(device)
    input_ids = torch.randint(0, 151936, (500,), dtype=torch.int32, device=device)

    lm_head = qwen2_model(input_ids)
    print(f'test_qwen2_model: input shape:  {input_ids.shape}, output shape: {lm_head.shape}')

    for i in range(10):
        torch.cuda.synchronize()
        begin = time.time()
        nvtx.push_range('qwen2_model')
        lm_head = qwen2_model(input_ids)
        nvtx.pop_range()
        torch.cuda.synchronize()
        end = time.time()
        print(f'test_qwen2_model: {i}th inference time: {(end - begin) * 1e3} ms')


def test():
    test_qwen2_attention()
    test_qwen2_mlp()
    test_qwen2_rmsnorm()
    test_qwen2_decoder_layer()
    test_qwen2_model()


def categorize_events(prof):
    """将 PyTorch Profiler 事件分类 - FlashInfer 版本"""
    categories = {
        'FlashInfer Attention': [],
        'FlashInfer RoPE': [],
        'FlashInfer RMSNorm': [],
        'FlashInfer Activation': [],
        'GEMM (Matrix Multiply)': [],
        'PyTorch Operations': [],
        'Memory Operations': [],
        'Other': []
    }
    
    events_list = list(prof.key_averages())
    
    for event in events_list:
        key = event.key.lower()
        
        # FlashInfer 算子识别
        if ('flashinfer' in key and 'prefill' in key and 'kv' in key) or \
           ('flashinfer' in key and 'attention' in key) or \
           ('prefillwithkvcache' in key):
            categories['FlashInfer Attention'].append(event)
        # Aiter MHA (ck_tile::FmhaBatchPrefillWithPagedKVCacheKernel)
        elif ('ck_tile' in key and 'fmha' in key and 'batchprefill' in key) or \
             ('fmhabatchprefillwithpagedkvcache' in key):
            # Add to FlashInfer Attention category (or create a new "Attention" category)
            # For now, add to FlashInfer Attention to keep compatibility
            categories['FlashInfer Attention'].append(event)
        elif ('flashinfer' in key and 'rope' in key) or \
             ('batchqkapplyrotary' in key) or \
             ('applyrotaryposids' in key):
            categories['FlashInfer RoPE'].append(event)
        elif ('flashinfer' in key and ('rms' in key or 'norm' in key)) or \
             ('fusedaddrmsnorm' in key) or \
             ('rmsnormkernel' in key):
            categories['FlashInfer RMSNorm'].append(event)
        elif ('flashinfer' in key and ('silu' in key or 'activation' in key)) or \
             ('act_and_mul' in key):
            categories['FlashInfer Activation'].append(event)
        elif ('matmul' in key or 'bmm' in key or 'gemm' in key or 'mm' in key or
              'nvjet' in key or 'cutlass' in key or 'cijk' in key):
            categories['GEMM (Matrix Multiply)'].append(event)
        elif 'aten::' in key or 'at::native' in key:
            categories['PyTorch Operations'].append(event)
        elif 'memset' in key or 'memcpy' in key or 'copy' in key or 'rocclr' in key:
            categories['Memory Operations'].append(event)
        else:
            categories['Other'].append(event)
    
    return categories


def get_cuda_time(event):
    """安全地获取 CUDA/Device 时间（兼容 ROCm 和 CUDA）"""
    # 优先使用 device_time_total（ROCm 推荐）
    for attr_name in ['device_time_total', 'self_device_time_total', 
                      'cuda_time_total', 'self_cuda_time_total', 
                      'device_time', 'cuda_time']:
        if hasattr(event, attr_name):
            try:
                val = getattr(event, attr_name)
                if isinstance(val, str):
                    val = val.replace('ms', '').replace('us', '').replace('μs', '').strip()
                    try:
                        val_float = float(val)
                        if 'ms' in str(getattr(event, attr_name, '')):
                            return val_float * 1000.0
                        else:
                            return val_float
                    except:
                        continue
                elif val is not None and val > 0:
                    val_float = float(val)
                    if val_float < 1.0 and val_float > 0:
                        val_float = val_float * 1e6
                    return val_float
            except (AttributeError, ValueError, TypeError):
                continue
    
    # 尝试通过 __dict__ 访问
    if hasattr(event, '__dict__'):
        for key in ['device_time_total', 'self_device_time_total', 
                    'cuda_time_total', 'self_cuda_time_total']:
            if key in event.__dict__:
                val = event.__dict__[key]
                try:
                    if isinstance(val, (int, float)) and val > 0:
                        val_float = float(val)
                        if val_float < 1.0 and val_float > 0:
                            val_float = val_float * 1e6
                        return val_float
                except:
                    continue
    
    return 0


def categorize_by_transformer_structure(event_key):
    """按照 Transformer 结构分类算子 - FlashInfer 版本"""
    key_lower = event_key.lower()
    
    # FlashInfer Attention
    if ('flashinfer' in key_lower and 'prefill' in key_lower and 'kv' in key_lower) or \
       ('flashinfer' in key_lower and 'attention' in key_lower) or \
       ('prefillwithkvcache' in key_lower):
        return "Attention - MHA Kernel"
    
    # Aiter MHA (ck_tile::FmhaBatchPrefillWithPagedKVCacheKernel)
    if ('ck_tile' in key_lower and 'fmha' in key_lower and 'batchprefill' in key_lower) or \
       ('fmhabatchprefillwithpagedkvcache' in key_lower):
        return "Attention - MHA Kernel"
    
    # FlashInfer RoPE
    if ('flashinfer' in key_lower and 'rope' in key_lower) or \
       ('batchqkapplyrotary' in key_lower) or \
       ('applyrotaryposids' in key_lower):
        return "Attention - RoPE"
    
    # GEMM 分类 - 使用统一的 if-elif 结构确保只匹配一次
    if ('nvjet' in key_lower or 'cijk' in key_lower):
        # QKV Projection - 优先匹配精确的 MT 模式
        # MT512x128x32, MT256x128x32 是 QKV Projection (seq_len=500 和 seq_len=2000)
        if 'mt512x128x32' in key_lower or 'mt256x128x32' in key_lower:
            return "Attention - QKV Projection"
        # nvjet 格式: 128x256 (seq_len=500), 256x144 (seq_len=1000+)
        elif 'nvjet' in key_lower and ('128x256' in key_lower or '256x144' in key_lower):
            return "Attention - QKV Projection"
        # 如果包含 qkv 关键字
        elif 'qkv' in key_lower:
            return "Attention - QKV Projection"
        # MLP Gate/Up Projection
        # MT128x48x128 (seq_len=500), MT128x96x128 (seq_len=1000), MT128x160x64 (seq_len=1500), MT128x192x64 (seq_len=2000) 是 MLP Gate/Up
        elif 'mt128x48x128' in key_lower or 'mt128x96x128' in key_lower or 'mt128x160x64' in key_lower or 'mt128x192x64' in key_lower:
            return "MLP - Gate/Up Projection"
        # nvjet 格式: 128x48 (seq_len=500), 128x96 (seq_len=1000+)
        elif 'nvjet' in key_lower and ('128x48' in key_lower or '128x96' in key_lower or '128x192' in key_lower):
            return "MLP - Gate/Up Projection"
        # 如果包含 gate 或 up 关键字
        elif 'gate' in key_lower or 'up' in key_lower:
            return "MLP - Gate/Up Projection"
        # MLP Down Projection
        # MT128x64x128 (seq_len=500), MT128x128x64 (seq_len=1000), MT256x96x64 (seq_len=1500), MT256x128x64 (seq_len=2000), MT64x128x64 是 MLP Down
        elif 'mt128x64x128' in key_lower or 'mt128x128x64' in key_lower or 'mt256x96x64' in key_lower or 'mt256x128x64' in key_lower or 'mt64x128x64' in key_lower:
            return "MLP - Down Projection"
        # nvjet 格式: 64x128 (seq_len=500), 192x80 (seq_len=1000+)
        elif 'nvjet' in key_lower and ('64x128' in key_lower or '192x80' in key_lower):
            return "MLP - Down Projection"
        # 如果包含 down 关键字
        elif 'down' in key_lower:
            return "MLP - Down Projection"
        # Output Projection (Attention 的输出投影)
        # MT256x128x64 也可能是 Output Projection，但通常 MLP Down 优先
        # 这里先不单独分类，如果需要可以通过其他方式区分
        # 其他 GEMM
        else:
            return "Other - GEMM"
    elif ('gemm' in key_lower and 'matmul' in key_lower):
        return "Other - GEMM"
    
    # FlashInfer Activation
    if ('flashinfer' in key_lower and ('silu' in key_lower or 'activation' in key_lower)) or \
       ('act_and_mul' in key_lower):
        return "MLP - Activation"
    
    # FlashInfer RMSNorm
    if ('flashinfer' in key_lower and ('rms' in key_lower or 'norm' in key_lower)) or \
       ('fusedaddrmsnorm' in key_lower) or \
       ('rmsnormkernel' in key_lower):
        return "RMSNorm"
    
    # Embedding
    if 'embed' in key_lower:
        return "Embedding"
    
    # PyTorch Operations
    if 'at::native' in key_lower:
        if 'cos' in key_lower or 'sin' in key_lower:
            return "Attention - RoPE (PyTorch)"
        elif 'elementwise' in key_lower or 'vectorized' in key_lower:
            return "Other - PyTorch Elementwise"
        else:
            return "Other - PyTorch Operations"
    
    # Memory Operations
    if 'memset' in key_lower or 'memcpy' in key_lower or 'copy' in key_lower or 'rocclr' in key_lower:
        return "Other - Memory Operations"
    
    return "Other"


def get_friendly_name_and_category(event_key):
    """根据算子名称获取友好名称和分类 - FlashInfer 版本"""
    key_lower = event_key.lower()
    
    # 缩短算子名称
    short_name = event_key
    if len(short_name) > 60:
        short_name = short_name[:57] + '...'
    
    # 识别分类和友好名称
    friendly_name = ""
    category = "Other"
    
    # FlashInfer Attention
    if ('flashinfer' in key_lower and 'prefill' in key_lower and 'kv' in key_lower) or \
       ('flashinfer' in key_lower and 'attention' in key_lower) or \
       ('prefillwithkvcache' in key_lower):
        category = "Attention"
        friendly_name = "FlashInfer Attention"
    # Aiter MHA (ck_tile::FmhaBatchPrefillWithPagedKVCacheKernel)
    elif ('ck_tile' in key_lower and 'fmha' in key_lower and 'batchprefill' in key_lower) or \
         ('fmhabatchprefillwithpagedkvcache' in key_lower):
        category = "Attention"
        friendly_name = "Aiter MHA"
    # FlashInfer RoPE
    elif ('flashinfer' in key_lower and 'rope' in key_lower) or \
         ('batchqkapplyrotary' in key_lower) or \
         ('applyrotaryposids' in key_lower):
        category = "RoPE"
        friendly_name = "FlashInfer RoPE"
    # FlashInfer RMSNorm
    elif ('flashinfer' in key_lower and ('rms' in key_lower or 'norm' in key_lower)) or \
         ('fusedaddrmsnorm' in key_lower) or \
         ('rmsnormkernel' in key_lower):
        category = "RMSNorm"
        friendly_name = "FlashInfer RMSNorm"
    # FlashInfer Activation
    elif ('flashinfer' in key_lower and ('silu' in key_lower or 'activation' in key_lower)) or \
         ('act_and_mul' in key_lower):
        category = "Activation"
        friendly_name = "FlashInfer Activation"
    # GEMM
    elif 'nvjet' in key_lower or 'cijk' in key_lower or ('gemm' in key_lower and 'matmul' in key_lower):
        category = "GEMM"
        # 优先匹配精确的 MT 模式
        if 'mt512x128x32' in key_lower or 'mt256x128x32' in key_lower:
            friendly_name = "QKV Projection (GEMM)"
        elif 'nvjet' in key_lower and ('128x256' in key_lower or '256x144' in key_lower):
            friendly_name = "QKV Projection (GEMM)"
        elif 'qkv' in key_lower:
            friendly_name = "QKV Projection (GEMM)"
        elif 'mt128x48x128' in key_lower or 'mt128x96x128' in key_lower or 'mt128x160x64' in key_lower or 'mt128x192x64' in key_lower:
            friendly_name = "MLP Gate/Up (GEMM)"
        elif 'nvjet' in key_lower and ('128x48' in key_lower or '128x96' in key_lower or '128x192' in key_lower):
            friendly_name = "MLP Gate/Up (GEMM)"
        elif 'gate' in key_lower or 'up' in key_lower:
            friendly_name = "MLP Gate/Up (GEMM)"
        elif 'mt128x64x128' in key_lower or 'mt128x128x64' in key_lower or 'mt256x96x64' in key_lower or 'mt256x128x64' in key_lower or 'mt64x128x64' in key_lower:
            friendly_name = "MLP Down (GEMM)"
        elif 'nvjet' in key_lower and ('64x128' in key_lower or '192x80' in key_lower):
            friendly_name = "MLP Down (GEMM)"
        elif 'down' in key_lower:
            friendly_name = "MLP Down (GEMM)"
        else:
            friendly_name = "GEMM"
    # PyTorch Operations
    elif 'aten::' in key_lower or 'at::native' in key_lower:
        category = "PyTorch"
        if 'matmul' in key_lower or 'mm' in key_lower:
            friendly_name = "PyTorch MatMul"
        elif 'elementwise' in key_lower:
            friendly_name = "PyTorch Elementwise"
        elif 'vectorized' in key_lower:
            friendly_name = "PyTorch Vectorized"
        else:
            friendly_name = "PyTorch Op"
    # Memory Operations
    elif 'memset' in key_lower or 'memcpy' in key_lower or 'copy' in key_lower or 'rocclr' in key_lower:
        category = "Memory"
        friendly_name = "Memory Copy"
    else:
        category = "Other"
        friendly_name = short_name[:30] + '...' if len(short_name) > 30 else short_name
    
    return friendly_name, category, short_name


def generate_analysis_report(prof, seq_len, num_iterations, latencies, output_file):
    """生成性能分析报告 - FlashInfer 版本"""
    categories = categorize_events(prof)
    
    # 计算总时间（排除 record_function 包装事件）
    total_time_ms = 0
    wrapper_patterns = [
        f'forward_seq_{seq_len}_iter',
        f'cudagraph_replay_seq_{seq_len}_iter',
        'forward_seq_',
        'cudagraph_replay_seq_'
    ]
    
    for events in categories.values():
        for event in events:
            event_key = event.key.lower()
            is_wrapper = any(pattern.lower() in event_key for pattern in wrapper_patterns)
            if is_wrapper:
                continue
            
            cuda_time_us = get_cuda_time(event)
            if cuda_time_us > 0:
                total_time_ms += cuda_time_us / 1000.0
    
    total_calls = sum(len(events) for events in categories.values())
    
    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write(f"ROCm FlashInfer Performance Analysis Summary (Seq Len: {seq_len})\n")
        f.write("=" * 100 + "\n")
        f.write("\n")
        
        if total_time_ms == 0:
            f.write("⚠️  Warning: No CUDA time data found. This might be a ROCm profiling issue.\n")
            return categories, 0
        
        f.write(f"Total Kernel Execution Time: {total_time_ms:.2f} ms\n")
        f.write(f"Total Kernel Calls: {total_calls}\n")
        f.write("\n")
        
        # 按类别统计
        f.write("=" * 100 + "\n")
        f.write("Performance by Category\n")
        f.write("=" * 100 + "\n")
        
        for category, events in categories.items():
            if not events:
                continue
            
            category_total_time_us = 0
            for event in events:
                event_key = event.key.lower()
                is_wrapper = any(pattern.lower() in event_key for pattern in wrapper_patterns)
                if is_wrapper:
                    continue
                
                cuda_time_us = get_cuda_time(event)
                if cuda_time_us > 0:
                    category_total_time_us += cuda_time_us
            category_total_time = category_total_time_us / 1000.0
            category_total_calls = sum(event.count for event in events if not any(pattern.lower() in event.key.lower() for pattern in wrapper_patterns))
            category_percentage = (category_total_time / total_time_ms * 100) if total_time_ms > 0 else 0
            
            f.write(f"\n{category}:\n")
            f.write(f"  Total Time: {category_total_time:.2f} ms ({category_percentage:.1f}%)\n")
            f.write(f"  Total Calls: {category_total_calls}\n")
            f.write("\n")
            
            # 打印该类别中最耗时的算子
            sorted_events = sorted(events, key=lambda x: get_cuda_time(x), reverse=True)
            for event in sorted_events[:5]:
                cuda_time_us = get_cuda_time(event)
                if cuda_time_us > 0 and event.count > 0:
                    avg_time_us = cuda_time_us / event.count
                else:
                    avg_time_us = 0
                total_time_ms_event = cuda_time_us / 1000.0
                
                display_name = event.key[:70] + '...' if len(event.key) > 70 else event.key
                
                f.write(f"    {display_name}\n")
                f.write(f"      Calls: {event.count:6d} | "
                      f"Avg: {avg_time_us:8.2f} μs | "
                      f"Total: {total_time_ms_event:8.2f} ms\n")
        
        # Top 20 最耗时的算子
        f.write("\n")
        f.write("=" * 100 + "\n")
        f.write("Top 20 Most Time-Consuming Operators\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Operator Name':<75} {'Calls':<8} {'Avg (μs)':<12} {'Total (ms)':<12} {'%':<8}\n")
        f.write("-" * 100 + "\n")
        
        all_events = []
        for events in categories.values():
            for event in events:
                event_key = event.key.lower()
                is_wrapper = any(pattern.lower() in event_key for pattern in wrapper_patterns)
                if not is_wrapper:
                    all_events.append(event)
        
        sorted_all = sorted(all_events, key=lambda x: get_cuda_time(x), reverse=True)
        for event in sorted_all[:20]:
            cuda_time_us = get_cuda_time(event)
            if cuda_time_us > 0 and event.count > 0:
                avg_time_us = cuda_time_us / event.count
            else:
                avg_time_us = 0
            total_time_ms_event = cuda_time_us / 1000.0
            percentage = (total_time_ms_event / total_time_ms * 100) if total_time_ms > 0 else 0
            
            display_name = event.key[:70] + '...' if len(event.key) > 70 else event.key
            f.write(f"{display_name:<75} {event.count:<8} {avg_time_us:>11.2f} {total_time_ms_event:>11.2f} {percentage:>7.1f}%\n")
        
        # E2E vs Kernel Time Analysis
        kernel_time_avg = total_time_ms / num_iterations if num_iterations > 0 else total_time_ms
        
        f.write("\n")
        f.write("=" * 100 + "\n")
        f.write("E2E vs Kernel Time Analysis\n")
        f.write("=" * 100 + "\n")
        if latencies:
            avg_e2e = sum(latencies) / len(latencies)
            overhead = avg_e2e - kernel_time_avg
            overhead_percentage = (overhead / avg_e2e * 100) if avg_e2e > 0 else 0
            
            f.write(f"E2E Latency:        {avg_e2e:.3f} ms (average of {num_iterations} runs)\n")
            f.write(f"GPU Kernel Time:    {kernel_time_avg:.3f} ms ({kernel_time_avg/avg_e2e*100:.1f}%) (average of {num_iterations} runs)\n")
            f.write(f"Overhead:           {overhead:.3f} ms ({overhead_percentage:.1f}%)\n")
            f.write("\n")
            f.write("⚠️  Note: PyTorch Profiler only captures GPU kernel execution time.\n")
            f.write("   E2E latency includes synchronization, kernel launch, and other overheads.\n")
            f.write("   For accurate comparison, use E2E latency, not kernel time!\n")
    
    return categories, total_time_ms


def generate_markdown_report(prof, seq_len, num_iterations, latencies, output_file, use_cuda_graph=True):
    """生成 Markdown 格式的性能分析报告 - FlashInfer 版本"""
    categories = categorize_events(prof)
    
    # 计算总时间（排除 record_function 包装事件）
    total_time_ms = 0
    wrapper_patterns = [
        f'forward_seq_{seq_len}_iter',
        f'cudagraph_replay_seq_{seq_len}_iter',
        'forward_seq_',
        'cudagraph_replay_seq_'
    ]
    
    all_events = []
    for events in categories.values():
        for event in events:
            event_key = event.key.lower()
            is_wrapper = any(pattern.lower() in event_key for pattern in wrapper_patterns)
            if not is_wrapper:
                cuda_time_us = get_cuda_time(event)
                if cuda_time_us > 0:
                    total_time_ms += cuda_time_us / 1000.0
                    all_events.append(event)
    
    # 按总时间排序
    sorted_events = sorted(all_events, key=lambda x: get_cuda_time(x), reverse=True)
    
    # 生成 markdown 报告
    with open(output_file, 'w') as f:
        # 标题
        exec_mode = "CUDA Graph" if use_cuda_graph else "Direct Execution"
        f.write(f"# FlashInfer Kernel Time Distribution (Seq Len: {seq_len})\n\n")
        f.write(f"**Execution Mode**: {exec_mode}\n")
        f.write(f"**Total Kernel Time**: {total_time_ms:.2f} ms\n\n")
        
        # E2E Latency
        if latencies:
            avg_e2e = sum(latencies) / len(latencies)
            kernel_time_avg = total_time_ms / num_iterations if num_iterations > 0 else total_time_ms
            f.write(f"**E2E Latency**: {avg_e2e:.3f} ms (average of {num_iterations} runs)\n")
            f.write(f"**GPU Kernel Time (avg)**: {kernel_time_avg:.3f} ms\n\n")
        
        # Top Operators 表格
        f.write("## Top Operators\n\n")
        f.write("| Rank | Category | Friendly Name | Operator Name (Short) | Calls | Avg (μs) | Total (ms) | % |\n")
        f.write("|------|----------|---------------|----------------------|-------|----------|------------|---|\n")
        
        # 创建 event.key -> (rank, category) 的映射
        event_rank_map = {}
        for rank, event in enumerate(sorted_events[:30], 1):  # Top 30
            cuda_time_us = get_cuda_time(event)
            if cuda_time_us > 0 and event.count > 0:
                avg_time_us = cuda_time_us / event.count
            else:
                avg_time_us = 0
            total_time_ms_event = cuda_time_us / 1000.0
            percentage = (total_time_ms_event / total_time_ms * 100) if total_time_ms > 0 else 0
            
            friendly_name, category, short_name = get_friendly_name_and_category(event.key)
            
            # 记录 rank 和 category
            event_rank_map[event.key] = (rank, category)
            
            # 友好名称：如果有友好名称，单独显示；否则显示 "-"
            friendly_display = friendly_name if friendly_name and friendly_name != short_name[:30] else "-"
            
            # 短名称：始终显示
            short_display = f"`{short_name}`"
            
            total_ms_str = f"**{total_time_ms_event:.2f}**" if total_time_ms_event > 1.0 else f"{total_time_ms_event:.2f}"
            f.write(f"| {rank} | {category} | {friendly_display} | {short_display} | {event.count} | {avg_time_us:.2f} | {total_ms_str} | {percentage:.1f}% |\n")
        
        # Category Summary
        f.write("\n## Category Summary\n\n")
        f.write("| Category | Total (ms) | Calls | % |\n")
        f.write("|----------|-----------|-------|---|\n")
        
        category_summary = {}
        for category, events in categories.items():
            category_total_time_us = 0
            category_total_calls = 0
            for event in events:
                event_key = event.key.lower()
                is_wrapper = any(pattern.lower() in event_key for pattern in wrapper_patterns)
                if not is_wrapper:
                    cuda_time_us = get_cuda_time(event)
                    if cuda_time_us > 0:
                        category_total_time_us += cuda_time_us
                        category_total_calls += event.count
            
            category_total_time = category_total_time_us / 1000.0
            category_percentage = (category_total_time / total_time_ms * 100) if total_time_ms > 0 else 0
            
            # 简化分类名称
            cat_display = category.replace('FlashInfer ', '').replace(' (Matrix Multiply)', '')
            if 'GEMM' in category:
                cat_display = "GEMM"
            elif 'Attention' in category:
                cat_display = "Attention"
            elif 'RMSNorm' in category:
                cat_display = "RMSNorm"
            elif 'Activation' in category:
                cat_display = "Activation"
            elif 'RoPE' in category:
                cat_display = "RoPE"
            elif 'PyTorch' in category:
                cat_display = "PyTorch"
            elif 'Memory' in category:
                cat_display = "Memory"
            else:
                cat_display = "Other"
            
            if cat_display not in category_summary:
                category_summary[cat_display] = {'total_ms': 0, 'calls': 0}
            category_summary[cat_display]['total_ms'] += category_total_time
            category_summary[cat_display]['calls'] += category_total_calls
        
        # 按总时间排序并输出
        for cat in sorted(category_summary.keys(), key=lambda x: category_summary[x]['total_ms'], reverse=True):
            total_ms = category_summary[cat]['total_ms']
            calls = category_summary[cat]['calls']
            pct = (total_ms / total_time_ms * 100) if total_time_ms > 0 else 0
            f.write(f"| {cat} | {total_ms:.2f} | {calls} | {pct:.1f}% |\n")
        
        # Transformer Structure Summary
        f.write("\n## Transformer Structure Summary\n\n")
        f.write("| Component | Sub-component | Category | Rank | Total (ms) | Calls | % |\n")
        f.write("|-----------|--------------|----------|------|-----------|-------|---|\n")
        
        transformer_categories = {
            'Embedding': {},
            'Attention': {
                'QKV Projection': [],
                'RoPE': [],
                'MHA Kernel': [],
                'Output Projection': []
            },
            'MLP': {
                'Gate/Up Projection': [],
                'Activation': [],
                'Down Projection': []
            },
            'RMSNorm': {},
            'Other': {}
        }
        
        # 按照 Transformer 结构分类所有事件
        for event in all_events:
            transformer_component = categorize_by_transformer_structure(event.key)
            cuda_time_us = get_cuda_time(event)
            if cuda_time_us > 0:
                rank, category = event_rank_map.get(event.key, (None, None))
                
                if transformer_component.startswith('Attention - '):
                    sub_component = transformer_component.replace('Attention - ', '')
                    if sub_component not in transformer_categories['Attention']:
                        transformer_categories['Attention'][sub_component] = []
                    transformer_categories['Attention'][sub_component].append((event, cuda_time_us, category, rank))
                elif transformer_component.startswith('MLP - '):
                    sub_component = transformer_component.replace('MLP - ', '')
                    if sub_component not in transformer_categories['MLP']:
                        transformer_categories['MLP'][sub_component] = []
                    transformer_categories['MLP'][sub_component].append((event, cuda_time_us, category, rank))
                elif transformer_component == 'RMSNorm':
                    if 'RMSNorm' not in transformer_categories['RMSNorm']:
                        transformer_categories['RMSNorm']['RMSNorm'] = []
                    transformer_categories['RMSNorm']['RMSNorm'].append((event, cuda_time_us, category, rank))
                elif transformer_component == 'Embedding':
                    if 'Embedding' not in transformer_categories['Embedding']:
                        transformer_categories['Embedding']['Embedding'] = []
                    transformer_categories['Embedding']['Embedding'].append((event, cuda_time_us, category, rank))
                else:
                    if transformer_component not in transformer_categories['Other']:
                        transformer_categories['Other'][transformer_component] = []
                    transformer_categories['Other'][transformer_component].append((event, cuda_time_us, category, rank))
        
        # 计算并输出每个组件的统计
        component_totals = {}
        
        # Attention
        attention_total = 0
        attention_calls = 0
        for sub_component, events_list in transformer_categories['Attention'].items():
            sub_total = sum(cuda_time_us for _, cuda_time_us, _, _ in events_list) / 1000.0
            sub_calls = sum(event.count for event, _, _, _ in events_list)
            attention_total += sub_total
            attention_calls += sub_calls
            percentage = (sub_total / total_time_ms * 100) if total_time_ms > 0 else 0
            
            ranks = [rank for _, _, _, rank in events_list if rank is not None]
            categories = list(set([cat for _, _, cat, _ in events_list if cat is not None]))
            rank_str = f"{min(ranks)}" if ranks else "-"
            if len(ranks) > 1:
                rank_str += f"-{max(ranks)}"
            category_str = categories[0] if len(categories) == 1 else ", ".join(categories) if categories else "-"
            
            f.write(f"| Attention | {sub_component} | {category_str} | {rank_str} | {sub_total:.2f} | {sub_calls} | {percentage:.1f}% |\n")
        if attention_total > 0:
            percentage = (attention_total / total_time_ms * 100) if total_time_ms > 0 else 0
            f.write(f"| **Attention** | **Total** | - | - | **{attention_total:.2f}** | **{attention_calls}** | **{percentage:.1f}%** |\n")
            component_totals['Attention'] = attention_total
        
        # MLP
        mlp_total = 0
        mlp_calls = 0
        for sub_component, events_list in transformer_categories['MLP'].items():
            sub_total = sum(cuda_time_us for _, cuda_time_us, _, _ in events_list) / 1000.0
            sub_calls = sum(event.count for event, _, _, _ in events_list)
            mlp_total += sub_total
            mlp_calls += sub_calls
            percentage = (sub_total / total_time_ms * 100) if total_time_ms > 0 else 0
            
            ranks = [rank for _, _, _, rank in events_list if rank is not None]
            categories = list(set([cat for _, _, cat, _ in events_list if cat is not None]))
            rank_str = f"{min(ranks)}" if ranks else "-"
            if len(ranks) > 1:
                rank_str += f"-{max(ranks)}"
            category_str = categories[0] if len(categories) == 1 else ", ".join(categories) if categories else "-"
            
            f.write(f"| MLP | {sub_component} | {category_str} | {rank_str} | {sub_total:.2f} | {sub_calls} | {percentage:.1f}% |\n")
        if mlp_total > 0:
            percentage = (mlp_total / total_time_ms * 100) if total_time_ms > 0 else 0
            f.write(f"| **MLP** | **Total** | - | - | **{mlp_total:.2f}** | **{mlp_calls}** | **{percentage:.1f}%** |\n")
            component_totals['MLP'] = mlp_total
        
        # RMSNorm
        if 'RMSNorm' in transformer_categories['RMSNorm']:
            events_list = transformer_categories['RMSNorm']['RMSNorm']
            rmsnorm_total = sum(cuda_time_us for _, cuda_time_us, _, _ in events_list) / 1000.0
            rmsnorm_calls = sum(event.count for event, _, _, _ in events_list)
            percentage = (rmsnorm_total / total_time_ms * 100) if total_time_ms > 0 else 0
            
            ranks = [rank for _, _, _, rank in events_list if rank is not None]
            categories = list(set([cat for _, _, cat, _ in events_list if cat is not None]))
            rank_str = f"{min(ranks)}" if ranks else "-"
            if len(ranks) > 1:
                rank_str += f"-{max(ranks)}"
            category_str = categories[0] if len(categories) == 1 else ", ".join(categories) if categories else "-"
            
            f.write(f"| RMSNorm | RMSNorm | {category_str} | {rank_str} | {rmsnorm_total:.2f} | {rmsnorm_calls} | {percentage:.1f}% |\n")
            component_totals['RMSNorm'] = rmsnorm_total
        
        # Embedding
        if 'Embedding' in transformer_categories['Embedding']:
            events_list = transformer_categories['Embedding']['Embedding']
            embedding_total = sum(cuda_time_us for _, cuda_time_us, _, _ in events_list) / 1000.0
            embedding_calls = sum(event.count for event, _, _, _ in events_list)
            percentage = (embedding_total / total_time_ms * 100) if total_time_ms > 0 else 0
            
            ranks = [rank for _, _, _, rank in events_list if rank is not None]
            categories = list(set([cat for _, _, cat, _ in events_list if cat is not None]))
            rank_str = f"{min(ranks)}" if ranks else "-"
            if len(ranks) > 1:
                rank_str += f"-{max(ranks)}"
            category_str = categories[0] if len(categories) == 1 else ", ".join(categories) if categories else "-"
            
            f.write(f"| Embedding | Embedding | {category_str} | {rank_str} | {embedding_total:.2f} | {embedding_calls} | {percentage:.1f}% |\n")
            component_totals['Embedding'] = embedding_total
        
        # Other
        other_total = 0
        other_calls = 0
        for component, events_list in transformer_categories['Other'].items():
            sub_total = sum(cuda_time_us for _, cuda_time_us, _, _ in events_list) / 1000.0
            sub_calls = sum(event.count for event, _, _, _ in events_list)
            other_total += sub_total
            other_calls += sub_calls
            percentage = (sub_total / total_time_ms * 100) if total_time_ms > 0 else 0
            
            ranks = [rank for _, _, _, rank in events_list if rank is not None]
            categories = list(set([cat for _, _, cat, _ in events_list if cat is not None]))
            rank_str = f"{min(ranks)}" if ranks else "-"
            if len(ranks) > 1:
                rank_str += f"-{max(ranks)}"
            category_str = categories[0] if len(categories) == 1 else ", ".join(categories) if categories else "-"
            
            f.write(f"| Other | {component} | {category_str} | {rank_str} | {sub_total:.2f} | {sub_calls} | {percentage:.1f}% |\n")
        if other_total > 0:
            percentage = (other_total / total_time_ms * 100) if total_time_ms > 0 else 0
            f.write(f"| **Other** | **Total** | - | - | **{other_total:.2f}** | **{other_calls}** | **{percentage:.1f}%** |\n")
            component_totals['Other'] = other_total
        
        # GEMM Operations Detail
        gemm_events = [e for e in sorted_events if 'nvjet' in e.key.lower() or 'cijk' in e.key.lower() or ('gemm' in e.key.lower() and 'matmul' in e.key.lower())]
        if gemm_events:
            f.write("\n## GEMM Operations Detail\n\n")
            f.write("| GEMM Type | Calls | Avg (μs) | Total (ms) | % | 说明 |\n")
            f.write("|-----------|-------|----------|------------|---|------|\n")
            
            for event in gemm_events[:10]:  # Top 10 GEMM
                cuda_time_us = get_cuda_time(event)
                if cuda_time_us > 0 and event.count > 0:
                    avg_time_us = cuda_time_us / event.count
                else:
                    avg_time_us = 0
                total_time_ms_event = cuda_time_us / 1000.0
                percentage = (total_time_ms_event / total_time_ms * 100) if total_time_ms > 0 else 0
                
                friendly_name, _, short_name = get_friendly_name_and_category(event.key)
                description = friendly_name.replace(" (GEMM)", "") if friendly_name else "GEMM"
                
                f.write(f"| `{short_name}` | {event.count} | {avg_time_us:.2f} | **{total_time_ms_event:.2f}** | {percentage:.1f}% | {description} |\n")
            
            gemm_total = sum(get_cuda_time(e) for e in gemm_events) / 1000.0
            gemm_calls = sum(e.count for e in gemm_events)
            f.write(f"\n**GEMM 总计**: {gemm_total:.2f} ms ({gemm_total/total_time_ms*100:.1f}%), {gemm_calls} calls\n")
    
    return categories, total_time_ms


def benchmark(seq_lens=None, use_cuda_graph=True, enable_debug=False, num_warmup=5, num_iterations=10, 
              enable_profiling=False, profile_output_dir=None, profile_activities=None):
    """
    Benchmark Qwen2 model with CUDA Graph support and optional PyTorch Profiler.
    
    Args:
        seq_lens: List of sequence lengths to test. Default: [500, 1000, 1500, 2000]
        use_cuda_graph: Whether to use CUDA Graph. Default: True
        enable_debug: Enable debug logging. Default: False
        num_warmup: Number of warmup iterations. Default: 5
        num_iterations: Number of benchmark iterations. Default: 10
        enable_profiling: Enable PyTorch Profiler. Default: False
        profile_output_dir: Directory to save profiling results. Default: None (profiling_results)
        profile_activities: List of ProfilerActivity to profile. Default: None (auto-detect)
    """
    sys.stderr.write("=" * 80 + "\n")
    sys.stderr.write("=== ROCm FlashInfer + Aiter Hybrid Benchmark Started ===\n")
    sys.stderr.write("=" * 80 + "\n")
    sys.stderr.write(f"Backend: FlashInfer (RoPE, Activation, RMSNorm) + Aiter (MHA)\n")
    sys.stderr.write(f"Aiter MHA: {'ENABLED' if USE_AITER_MHA else 'DISABLED (fallback to FlashInfer)'}\n")
    sys.stderr.write(f"CUDA Graph: {'ENABLED' if use_cuda_graph else 'DISABLED'}\n")
    sys.stderr.write(f"Profiling: {'ENABLED' if enable_profiling else 'DISABLED'}\n")
    sys.stderr.write(f"Arguments: seq_lens={seq_lens}, use_cuda_graph={use_cuda_graph}, enable_debug={enable_debug}, num_warmup={num_warmup}, num_iterations={num_iterations}\n")
    sys.stderr.write("=" * 80 + "\n")
    
    # Setup profiling output directory
    if enable_profiling:
        if profile_output_dir is None:
            profile_output_dir = "profiling_results"
        os.makedirs(profile_output_dir, exist_ok=True)
        sys.stderr.write(f"Profiling results will be saved to: {profile_output_dir}\n")
        
        # Setup profile activities
        if profile_activities is None:
            profile_activities = []
            if torch.cuda.is_available():
                profile_activities.append(ProfilerActivity.CUDA)
            profile_activities.append(ProfilerActivity.CPU)
        sys.stderr.write(f"Profiling activities: {profile_activities}\n")
    
    # Store performance results for summary
    performance_results = []
    
    # Default sequence lengths if not provided
    if seq_lens is None:
        seq_lens = [500, 1000, 1500, 2000]
    
    # Pre-allocate buffers for CUDA Graph compatibility (for Aiter MHA)
    max_seq_len_for_buf = max(max(seq_lens), max_seq_len)
    head_dim = hidden_size // num_qo_heads
    buffers = create_cuda_graph_buffers(max_seq_len_for_buf, num_kv_heads, head_dim, device)
    
    for seq_len in seq_lens:
        sys.stderr.write("\n" + "=" * 80 + "\n")
        sys.stderr.write(f"Testing seq_len={seq_len} (CUDA Graph: {'ENABLED' if use_cuda_graph else 'DISABLED'})\n")
        sys.stderr.write("=" * 80 + "\n")
        
        input_ids = torch.randint(0, 151936, (seq_len,), dtype=torch.int32, device=device)
        # Create model with pre-allocated buffers for CUDA Graph compatibility
        qwen2_model = Qwen2Model(**buffers).to(device)
        
        # Warmup
        warmup_model(qwen2_model, input_ids, num_warmup=max(num_warmup, 15), check_jit=USE_AITER_MHA)
        
        if use_cuda_graph:
            # Reset buffers
            reset_attention_buffers(qwen2_model)
            
            # Set capture mode
            set_cuda_graph_capture_mode(qwen2_model, True)
            
            # Preset values
            preset_buffer_values(qwen2_model, seq_len)
            
            torch.cuda.synchronize()

        # Create and capture CUDA Graph for performance optimization
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            lm_head = qwen2_model(input_ids)
        sys.stderr.write(f'Warmup for seq len: {seq_len}, inp shape: {input_ids.shape}, output shape: {lm_head.shape}\n')
        
        # Reset capture mode after capture (for next iteration)
        if USE_AITER_MHA and use_cuda_graph:
            set_cuda_graph_capture_mode(qwen2_model, False)
            # Reset the logged flag so we can log again in replay phase
            for module in qwen2_model.modules():
                if isinstance(module, Qwen2Attention):
                    if hasattr(module, '_aiter_mha_used_logged'):
                        delattr(module, '_aiter_mha_used_logged')

        # Benchmark using CUDA Graph replay
        latencies = []
        
        if enable_profiling:
            sys.stderr.write(f"Starting profiling for seq_len={seq_len} (CUDA Graph replay)...\n")
            # Warmup iteration (BEFORE profiling, to avoid counting it)
            # Reset the logged flag so we can log Aiter MHA usage during replay
            if USE_AITER_MHA:
                for module in qwen2_model.modules():
                    if isinstance(module, Qwen2Attention):
                        if hasattr(module, '_aiter_mha_used_logged'):
                            delattr(module, '_aiter_mha_used_logged')
            g.replay()
            torch.cuda.synchronize()
            
            with profile(
                activities=profile_activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                on_trace_ready=tensorboard_trace_handler(
                    os.path.join(profile_output_dir, f"seq_{seq_len}_cudagraph")
                )
            ) as prof:
                # Profiled iterations
                for i in range(num_iterations):
                    start = time.time()
                    with record_function(f"cudagraph_replay_seq_{seq_len}_iter_{i}"):
                        g.replay()
                    torch.cuda.synchronize()
                    end = time.time()
                    latencies.append((end - start) * 1000)
            
            # Export profiling results after context manager exits
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                profile_prefix = f"seq_{seq_len}_cudagraph_{timestamp}"
                
                # Export table
                table_file = os.path.join(profile_output_dir, f"{profile_prefix}_table.txt")
                with open(table_file, 'w') as f:
                    # Write the standard profiling table
                    table_str = prof.key_averages().table(
                        sort_by="cuda_time_total" if ProfilerActivity.CUDA in profile_activities else "cpu_time_total",
                        row_limit=100
                    )
                    f.write(table_str)
                    
                    # Add per-iteration summary
                    f.write("\n" + "=" * 100 + "\n")
                    f.write("Per-Iteration Summary (for comparison with E2E latency)\n")
                    f.write("=" * 100 + "\n")
                    f.write(f"Number of iterations: {num_iterations}\n")
                    
                    # Extract total times from profiler
                    key_averages = prof.key_averages()
                    cpu_time_total_ms = sum(event.self_cpu_time_total for event in key_averages) / 1000.0
                    device_time_total_ms = sum(getattr(event, 'self_device_time_total', 0) for event in key_averages) / 1000.0
                    
                    cpu_time_per_iteration = cpu_time_total_ms / num_iterations
                    device_time_per_iteration = device_time_total_ms / num_iterations
                    
                    f.write(f"CPU time total: {cpu_time_total_ms:.3f} ms\n")
                    f.write(f"Device time total: {device_time_total_ms:.3f} ms\n")
                    f.write(f"CPU time per iteration: {cpu_time_per_iteration:.3f} ms\n")
                    f.write(f"Device time per iteration: {device_time_per_iteration:.3f} ms\n")
                    
                    # Add E2E latency if available
                    if latencies:
                        avg_e2e = sum(latencies) / len(latencies)
                        f.write(f"\nE2E latency (from benchmark): {avg_e2e:.3f} ms\n")
                        f.write(f"Difference (CPU per iter - E2E): {cpu_time_per_iteration - avg_e2e:.3f} ms\n")
                        f.write(f"Difference (Device per iter - E2E): {device_time_per_iteration - avg_e2e:.3f} ms\n")
                    
                    f.write("\nNote: Profiling times include synchronization overhead. E2E latency is measured end-to-end.\n")
                sys.stderr.write(f"Profiling table saved to: {table_file}\n")
                
                # Generate detailed analysis report
                analysis_file = os.path.join(profile_output_dir, f"{profile_prefix}_analysis.txt")
                try:
                    generate_analysis_report(prof, seq_len, num_iterations, latencies, analysis_file)
                    sys.stderr.write(f"Detailed analysis report saved to: {analysis_file}\n")
                except Exception as analysis_error:
                    sys.stderr.write(f"WARNING: Failed to generate analysis report: {analysis_error}\n")
                
                # Generate markdown report
                markdown_file = os.path.join(profile_output_dir, f"{profile_prefix}_analysis_report.md")
                try:
                    generate_markdown_report(prof, seq_len, num_iterations, latencies, markdown_file, use_cuda_graph=True)
                    sys.stderr.write(f"Markdown analysis report saved to: {markdown_file}\n")
                except Exception as markdown_error:
                    sys.stderr.write(f"WARNING: Failed to generate markdown report: {markdown_error}\n")
                
                # Export JSON (skip if tensorboard_trace_handler already saved it)
                try:
                    json_file = os.path.join(profile_output_dir, f"{profile_prefix}_events.json")
                    prof.export_chrome_trace(json_file)
                    sys.stderr.write(f"Chrome trace saved to: {json_file}\n")
                except RuntimeError as e:
                    if "Trace is already saved" in str(e):
                        sys.stderr.write(f"Chrome trace already saved by tensorboard handler\n")
                    else:
                        raise
            except Exception as export_error:
                sys.stderr.write(f"WARNING: Failed to export profiling results: {export_error}\n")
                import traceback
                sys.stderr.write(traceback.format_exc() + "\n")
        else:
            # No profiling, just benchmark
            for i in range(num_iterations):
                torch.cuda.synchronize()
                begin = time.time()
                nvtx.push_range('qwen2_model')
                g.replay()
                torch.cuda.synchronize()
                nvtx.pop_range()
                end = time.time()
                latencies.append((end - begin) * 1000)

        # Calculate average latency
        avg_latency = sum(latencies) / len(latencies)
        result_msg = f"qwen2_model: seq len: {seq_len}, inference latency: {avg_latency:.15f} ms"
        sys.stderr.write(result_msg + "\n")
        performance_results.append((seq_len, avg_latency))
    
    # Generate performance summary at the end
    if performance_results:
        sys.stderr.write("\n" + "=" * 80 + "\n")
        sys.stderr.write("Performance Summary\n")
        sys.stderr.write("=" * 80 + "\n")
        sys.stderr.write(f"Backend: FlashInfer (RoPE, Activation, RMSNorm) + Aiter (MHA)\n")
        sys.stderr.write(f"Aiter MHA: {'ENABLED' if USE_AITER_MHA else 'DISABLED'}\n")
        sys.stderr.write(f"CUDA Graph: {'ENABLED' if use_cuda_graph else 'DISABLED'}\n")
        sys.stderr.write("=" * 80 + "\n")
        
        for seq_len, avg_latency in performance_results:
            summary_line = f"qwen2_model: seq len: {seq_len}, inference latency: {avg_latency:.15f} ms"
            sys.stderr.write(summary_line + "\n")
        
        sys.stderr.write("=" * 80 + "\n\n")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Qwen2 ROCm FlashInfer CUDA Graph Benchmark")
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[500, 1000, 1500, 2000],
        help="Sequence lengths to test (default: [500, 1000, 1500, 2000])"
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Disable CUDA Graph and use direct execution"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations (default: 10)"
    )
    parser.add_argument(
        "--enable-profiling",
        action="store_true",
        help="Enable PyTorch Profiler"
    )
    parser.add_argument(
        "--profile-output-dir",
        type=str,
        default=None,
        help="Directory to save profiling results (default: profiling_results)"
    )
    parser.add_argument(
        "--profile-activities",
        type=str,
        nargs="+",
        choices=["cpu", "cuda"],
        default=None,
        help="Profiling activities: 'cpu', 'cuda', or both (default: auto-detect based on CUDA availability)"
    )
    
    args = parser.parse_args()
    
    # Convert profile activities string to ProfilerActivity enum
    profile_activities = None
    if args.enable_profiling and args.profile_activities:
        profile_activities = []
        if "cuda" in args.profile_activities:
            profile_activities.append(ProfilerActivity.CUDA)
        if "cpu" in args.profile_activities:
            profile_activities.append(ProfilerActivity.CPU)
    
    benchmark(
        seq_lens=args.seq_lens,
        use_cuda_graph=not args.no_cuda_graph,
        enable_debug=args.debug,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        enable_profiling=args.enable_profiling,
        profile_output_dir=args.profile_output_dir,
        profile_activities=profile_activities
    )
