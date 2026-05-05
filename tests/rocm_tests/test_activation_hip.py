"""
Copyright (c) 2026 Advanced Micro Devices, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

HIP/ROCm tests for fused activation kernels: silu_and_mul, gelu_tanh_and_mul,
gelu_and_mul.

Each test generates input of shape (num_tokens, 2*d), runs the flashinfer
fused kernel, and compares against a pure-PyTorch reference computed in fp32
and cast back to the target dtype.

Shapes cover:
  - Small d values (< 64 elements / vec_size, partial last wavefront)
  - Typical LLM FFN sizes (LLaMA-2 11008, Mistral 14336, Phi-3 14336 / 2)
  - Large d values that hit the 1024-thread blockDim cap
  - Non-power-of-two d values (e.g., 5504 = LLaMA-2 11008 // 2)
"""

import pytest
import torch

import flashinfer


def _silu_and_mul_ref(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate, up = x.float()[..., :d], x.float()[..., d:]
    return (gate / (1.0 + torch.exp(-gate)) * up).to(x.dtype)


def _gelu_tanh_and_mul_ref(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate, up = x.float()[..., :d], x.float()[..., d:]
    y = torch.nn.functional.gelu(gate, approximate="tanh") * up
    return y.to(x.dtype)


def _gelu_and_mul_ref(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate, up = x.float()[..., :d], x.float()[..., d:]
    y = torch.nn.functional.gelu(gate, approximate="none") * up
    return y.to(x.dtype)


# (num_tokens, d) — input tensor is (num_tokens, 2*d)
# d values chosen to cover: < blockDim threshold, wave64-unaligned (5504),
# typical LLM dims, blockDim-cap boundary (8192→1024 threads).
_SHAPES = [
    (1, 64),
    (1, 128),
    (4, 256),
    (8, 512),
    (1, 2752),  # 5504 // 2 — from LLaMA-2 intermediate with GQA
    (4, 4096),
    (8, 5504),  # LLaMA-2 ffn_dim // 2  (non-multiple-of-64-threads)
    (16, 7168),  # Mistral-7B ffn_dim // 2
    (1, 8192),  # hits 1024-thread cap
    (4, 14336),  # Llama-3-70B ffn_dim // 2, multiple stride iterations
]


@pytest.mark.parametrize("num_tokens,d", _SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_silu_and_mul(num_tokens, d, dtype):
    torch.manual_seed(42)
    x = torch.randn(num_tokens, 2 * d, device="cuda", dtype=dtype)
    ref = _silu_and_mul_ref(x)
    out = flashinfer.activation.silu_and_mul(x)
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("num_tokens,d", _SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gelu_tanh_and_mul(num_tokens, d, dtype):
    torch.manual_seed(42)
    x = torch.randn(num_tokens, 2 * d, device="cuda", dtype=dtype)
    ref = _gelu_tanh_and_mul_ref(x)
    out = flashinfer.activation.gelu_tanh_and_mul(x)
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("num_tokens,d", _SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gelu_and_mul(num_tokens, d, dtype):
    torch.manual_seed(42)
    x = torch.randn(num_tokens, 2 * d, device="cuda", dtype=dtype)
    ref = _gelu_and_mul_ref(x)
    out = flashinfer.activation.gelu_and_mul(x)
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


if __name__ == "__main__":
    test_silu_and_mul(8, 5504, torch.float16)
    test_gelu_tanh_and_mul(8, 7168, torch.float16)
    test_gelu_and_mul(8, 7168, torch.bfloat16)
