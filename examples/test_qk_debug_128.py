#!/usr/bin/env python3
"""
Test script to compare raw QK scores between FlashInfer and reference for HEAD_DIM=128.
Uses a very small problem size to make debug output manageable.
"""
import torch

import flashinfer
from tests.attention_reference import naive_attention

# Very small config for easy debugging
qo_len = 128
kv_len = 128
num_qo_heads = 1  # Single head for simplicity
num_kv_heads = 1
head_dim = 64  # Testing HEAD_DIM=128"""  """

print("=" * 80)
print(f"Testing QK computation: HEAD_DIM={head_dim}")
print("=" * 80)
print(
    f"Config: qo_len={qo_len}, kv_len={kv_len}, num_heads={num_qo_heads}, head_dim={head_dim}"
)
print("=" * 80)

# Set seed for reproducibility
torch.manual_seed(42)

# Create inputs
q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=torch.float16, device="cuda:0")
k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0")
v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0")

print(f"\nCalling FlashInfer (will print raw QK scores from kernel)...")
print("-" * 80)

# Call FlashInfer - this will print QK scores from the kernel
o = flashinfer.single_prefill_with_kv_cache(
    q, k, v, causal=False, kv_layout="NHD", pos_encoding_mode="NONE"
)

print("-" * 80)
print(f"\nCalling reference implementation (will print raw QK scores)...")
print("-" * 80)

# Call reference - this will print QK scores from PyTorch
o_ref, _ = naive_attention(
    q.float(), k.float(), v.float(), causal=False, pos_encoding_mode="NONE"
)
o_ref = o_ref.half()

print("-" * 80)
print(f"\nOutput comparison:")
diff = (o - o_ref).abs()
print(f"  Max absolute difference: {diff.max().item():.6f}")
print(f"  Mean absolute difference: {diff.mean().item():.6f}")

if diff.max() > 0.01:
    print(f"  ❌ FAIL")
else:
    print(f"  ✓ PASS")
