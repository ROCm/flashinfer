# Qwen2.5 single sequence prefill.
# ROCm FlashInfer version with profiling markers for operator fusion analysis
# - Uses record_function for ROCm compatibility (NVTX not available in ROCm)
# - Markers added: qwen2_model, qwen2_decoder_layer, qwen2_attention, qwen2_mlp, qwen2_rmsnorm
import argparse
import gc
import os
import sys
import time
from datetime import datetime
from typing import Callable, Optional
import flashinfer

from flashinfer.jit.core import logger
import logging

logger.setLevel(logging.ERROR)

import torch
import torch.nn as nn
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    tensorboard_trace_handler,
)

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
# class SuppressFlashInferOutput:
#     """Context manager to suppress FlashInfer JIT compilation output"""
#     def __init__(self):
#         self.original_stderr = None
#         self.devnull = None

#     def __enter__(self):
#         self.original_stderr = sys.stderr
#         self.devnull = open(os.devnull, 'w')
#         sys.stderr = self.devnull
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         if self.devnull:
#             self.devnull.close()
#         sys.stderr = self.original_stderr
#         return False

# Suppress output during import
# with SuppressFlashInferOutput():
#     import flashinfer

# Reset any logging handlers that cached the now-closed devnull file descriptor.
# flashinfer/aiter set up StreamHandlers during import; those handlers hold a
# reference to the devnull file that SuppressFlashInferOutput just closed.
# Without this reset they raise "ValueError: I/O operation on closed file."
def _fix_closed_log_handlers():
    import logging
    for candidate in logging.Logger.manager.loggerDict.values():
        if isinstance(candidate, logging.Logger):
            for handler in list(candidate.handlers):
                if isinstance(handler, logging.StreamHandler):
                    stream = handler.stream
                    if stream is None or getattr(stream, 'closed', False):
                        handler.stream = sys.stderr
                    else:
                        try:
                            stream.write('')
                        except (ValueError, OSError):
                            # Stream is closed; bypass setStream() (which would
                            # try to flush the dead stream) and assign directly.
                            handler.stream = sys.stderr

_fix_closed_log_handlers()
del _fix_closed_log_handlers

# Check if flashinfer has the required functions
def _check_flashinfer_api():
    """Check if flashinfer has the required API functions and raise error if not"""
    missing_functions = []
    if not hasattr(flashinfer, 'rmsnorm'):
        missing_functions.append('rmsnorm')
    if not hasattr(flashinfer, 'fused_add_rmsnorm'):
        missing_functions.append('fused_add_rmsnorm')

    if missing_functions:
        raise AttributeError(
            f"flashinfer module is missing required functions: {', '.join(missing_functions)}. "
            f"Please ensure flashinfer is properly installed and compiled with ROCm support. "
            f"Available flashinfer attributes: {[x for x in dir(flashinfer) if not x.startswith('_')]}"
        )

# Verify flashinfer API at import time
_check_flashinfer_api()

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


class Qwen2RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

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

    def __init__(self, layer_idx):
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

        # # create a 1MB workspace buffer
        # workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.bfloat16, device=device)
        # self.qkv_proj_wrapper = flashinfer.SegmentGEMMWrapper(workspace_buffer, backend='auto')
        # self.o_proj_wrapper = flashinfer.SegmentGEMMWrapper(workspace_buffer, backend='auto')

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

            # apply rope for qk
            q = qkv[..., :self.num_qo_heads * self.head_dim]
            # [seq_len, num_qo_heads * head_dim]
            k = qkv[..., self.num_qo_heads * self.head_dim
                         :(self.num_qo_heads + self.num_kv_heads) * self.head_dim]  # [seq_len, num_kv_heads * head_dim]
            q_rope, k_rope = flashinfer.apply_rope_with_cos_sin_cache(positions, q, k, self.head_dim, position_embeddings)

            # mha
            q = q_rope
            k = k_rope
            q = q.reshape(seq_len, self.num_qo_heads, self.head_dim)  # [seq_len, num_qo_heads, head_dim]
            k = k.reshape(seq_len, self.num_kv_heads, self.head_dim)  # [seq_len, num_kv_heads, head_dim]
            v = (qkv[..., (self.num_qo_heads + self.num_kv_heads) * self.head_dim:]
                 .reshape(seq_len, num_kv_heads, self.head_dim))  # [seq_len, num_kv_heads, head_dim]
            o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, kv_layout='NHD', backend="aiter")  # [seq_len, num_qo_heads * head_dim]
            o = o.reshape(seq_len, -1).contiguous()
            # print(f'q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}, o.shape: {o.shape}')

            # o proj
            # o_proj_w = self.o_proj_w.unsqueeze(0)  # [1, hidden_size, num_qo_heads * head_dim]
            # o = self.o_proj_wrapper.run(o, o_proj_w, 1, True, seg_lens=seq_lens)  # [seq_len, hidden_size]
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
            up = flashinfer.silu_and_mul(gate_up_fuse)

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
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.attention = Qwen2Attention(layer_idx)
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
    def __init__(self):
        super().__init__()
        self.padding_idx = pad_token_id
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # TODO: load embed weight from file
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, self.padding_idx, dtype=torch.bfloat16,
                                         device=device)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(i) for i in range(num_hidden_layers)])
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
    qwen2_decoder_layer = Qwen2DecoderLayer(0)
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


# Optional: Import detailed profiling utilities if needed
# from profiling_utils import categorize_events, get_cuda_time, generate_analysis_report, generate_markdown_report


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
    sys.stderr.write("=== ROCm FlashInfer Benchmark Started ===\n")
    sys.stderr.write("=" * 80 + "\n")
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

    for seq_len in seq_lens:
        sys.stderr.write("\n" + "=" * 80 + "\n")
        sys.stderr.write(f"Testing seq_len={seq_len} (CUDA Graph: {'ENABLED' if use_cuda_graph else 'DISABLED'})\n")
        sys.stderr.write("=" * 80 + "\n")

        input_ids = torch.randint(0, 151936, (seq_len,), dtype=torch.int32, device=device)
        qwen2_model = Qwen2Model().to(device)

        # Warmup: run once to fill caches before graph capture
        _ = qwen2_model(input_ids)
        torch.cuda.synchronize()

        # Create and capture CUDA Graph for performance optimization
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            lm_head = qwen2_model(input_ids)
        sys.stderr.write(f'Warmup for seq len: {seq_len}, inp shape: {input_ids.shape}, output shape: {lm_head.shape}\n')

        # Benchmark using CUDA Graph replay
        latencies = []

        if enable_profiling:
            sys.stderr.write(f"Starting profiling for seq_len={seq_len} (CUDA Graph replay)...\n")
            # Warmup iteration (BEFORE profiling, to avoid counting it)
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

                # Optional: Generate detailed analysis reports (requires profiling_utils)
                try:
                    from profiling_utils import (
                        generate_analysis_report,
                        generate_markdown_report,
                    )
                    analysis_file = os.path.join(profile_output_dir, f"{profile_prefix}_analysis.txt")
                    generate_analysis_report(prof, seq_len, num_iterations, latencies, analysis_file)
                    sys.stderr.write(f"Detailed analysis report saved to: {analysis_file}\n")

                    markdown_file = os.path.join(profile_output_dir, f"{profile_prefix}_analysis_report.md")
                    generate_markdown_report(prof, seq_len, num_iterations, latencies, markdown_file, use_cuda_graph=True)
                    sys.stderr.write(f"Markdown analysis report saved to: {markdown_file}\n")
                except ImportError:
                    sys.stderr.write("Note: Detailed analysis reports require profiling_utils.py (optional)\n")
                except Exception as e:
                    sys.stderr.write(f"WARNING: Failed to generate detailed reports: {e}\n")

                # Export JSON (skip if tensorboard_trace_handler already saved it)
                try:
                    json_file = os.path.join(profile_output_dir, f"{profile_prefix}_events.json")
                    prof.export_chrome_trace(json_file)
                    sys.stderr.write(f"Chrome trace saved to: {json_file}\n")
                except RuntimeError as e:
                    if "Trace is already saved" in str(e):
                        sys.stderr.write("Chrome trace already saved by tensorboard handler\n")
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
        # result_msg = f"qwen2_model: seq len: {seq_len}, inference latency: {avg_latency:.15f} ms"
        result_msg = f"qwen2_model: seq len: {seq_len}, inference latency: {avg_latency:.2f} ms"
        sys.stderr.write(result_msg + "\n")
        performance_results.append((seq_len, avg_latency))

    # Generate performance summary at the end
    if performance_results:
        sys.stderr.write("\n" + "=" * 80 + "\n")
        sys.stderr.write("Performance Summary\n")
        sys.stderr.write("=" * 80 + "\n")
        sys.stderr.write(f"CUDA Graph: {'ENABLED' if use_cuda_graph else 'DISABLED'}\n")
        sys.stderr.write("=" * 80 + "\n")

        for seq_len, avg_latency in performance_results:
            # summary_line = f"qwen2_model: seq len: {seq_len}, inference latency: {avg_latency:.15f} ms"
            summary_line = f"qwen2_model: seq len: {seq_len}, inference latency: {avg_latency:.2f} ms"
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
