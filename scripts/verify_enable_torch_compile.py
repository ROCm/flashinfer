"""Benchmark ``append_paged_kv_cache``: eager vs ``torch.compile`` on ROCm/AMD.

Opaque ``torch.library`` custom ops (see ``FLASHINFER_USE_TORCH_CUSTOM_OPS`` in
``flashinfer.utils``) must be enabled before importing ``flashinfer`` so Dynamo
does not trace into the HIP extension.
"""

from __future__ import annotations

import os
import statistics
import time
from collections.abc import Callable

# Default ON for this example so ``torch.compile`` works without extra env setup.
# Override with ``FLASHINFER_USE_TORCH_CUSTOM_OPS=0`` to measure eager without
# custom-op dispatch overhead (``torch.compile`` will fail in that mode).
os.environ.setdefault("FLASHINFER_USE_TORCH_CUSTOM_OPS", "1")

import flashinfer
import torch
from flashinfer.page import append_paged_kv_cache

print(f"torch:      {torch.__version__}")
print(f"flashinfer: {flashinfer.__version__}")
print(f"GPU:        {torch.cuda.get_device_name(0)}")
print(f"backend:    ROCm {torch.version.hip}")
print(
    f"FLASHINFER_USE_TORCH_CUSTOM_OPS (effective): {flashinfer.use_torch_custom_ops_enabled()}"
)
print()

B, PAGE_SIZE, KV_HEADS, HEAD_DIM, NUM_TOKENS = 2, 16, 1, 128, 8
DEVICE = "cuda"

# Paged KV cache (NHD layout)
pages = B * 2  # enough pages
k_cache = torch.zeros(
    pages, PAGE_SIZE, KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16
)
v_cache = torch.zeros_like(k_cache)

# CSR metadata
indptr = torch.arange(0, B + 1, dtype=torch.int32, device=DEVICE) * 2  # 2 pages/seq
indices = torch.arange(pages, dtype=torch.int32, device=DEVICE)
last_page_len = torch.full((B,), PAGE_SIZE, dtype=torch.int32, device=DEVICE)

# Tokens to append
batch_idx = torch.arange(B, device=DEVICE, dtype=torch.int32).repeat_interleave(
    NUM_TOKENS
)
positions = torch.arange(NUM_TOKENS, device=DEVICE, dtype=torch.int32).repeat(B)


def append(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    append_paged_kv_cache(
        k, v, batch_idx, positions, (k_cache, v_cache), indices, indptr, last_page_len
    )
    return k


def bench_microseconds(
    fn: Callable[[], None], *, warmup: int = 10, iters: int = 50
) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.mean(samples), statistics.stdev(samples) if len(
        samples
    ) > 1 else 0.0


k = torch.randn(B * NUM_TOKENS, KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
v = torch.randn_like(k)

append(k, v)
print("OK eager append_paged_kv_cache")

eager_mean, eager_std = bench_microseconds(lambda: append(k, v))
print(
    f"eager:        {eager_mean:.2f} ± {eager_std:.2f} µs / iter (warmup={10}, iters={50})"
)

if not flashinfer.use_torch_custom_ops_enabled():
    print(
        "Skipping torch.compile: set FLASHINFER_USE_TORCH_CUSTOM_OPS=1 before "
        "import (this script uses setdefault('1') unless you override the env)."
    )
else:
    compiled = torch.compile(append, dynamic=True)
    compiled(k, v)
    print("OK torch.compile(append, dynamic=True)")

    c_mean, c_std = bench_microseconds(lambda: compiled(k, v))
    print(
        f"torch.compile: {c_mean:.2f} ± {c_std:.2f} µs / iter (warmup={10}, iters={50})"
    )
    if c_mean > 0:
        print(f"ratio compile/eager: {c_mean / eager_mean:.3f}x")
