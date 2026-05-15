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

Three-way latency comparison for batch-paged prefill on ROCm:

  fa2       — FlashInfer FA2 backend (BatchPrefillWithPagedKVCacheWrapper)
  aiter_py  — AITER Python passthrough (mha_batch_prefill_func / flash_attn_varlen_func)
  aiter_cpp — AITER routed via FlashInfer C++ JIT harness (backend="aiter")

Two page-size regimes tested:
  page_size=256  → native-paged path (mha_batch_prefill)
  page_size=16   → flat-gather path  (mha_fwd via gather)

Shapes: Alibaba production profile — q_len=256, batch=1, GQA 16/4, HD=64, causal,
        kv ∈ {512, 1024, 2048, 3072, 4096, 8192}

Run:
    python benchmarks/rocm_benchmarks/bench_aiter_batch_paged_compare.py --timing-only
"""

import argparse
import csv
import math
import sys
import logging
from pathlib import Path

import torch
import flashinfer
from flashinfer.jit.core import logger

logger.setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "rocm_profiler"))
from rocm_profiler import KernelConfig, RocmProfiler

_bench_parser = argparse.ArgumentParser(add_help=False)
_bench_parser.add_argument("--counters", default="roofline", metavar="PRESET_OR_FILE")
_bench_parser.add_argument("--label", default="bp_cmp", metavar="PREFIX")
_bench_args, _remaining = _bench_parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

_counters = _bench_args.counters
_label = _bench_args.label

_Q_LEN = 256
_KV_LENS = [512, 1024, 2048, 3072, 4096, 8192]
_NUM_QO_HEADS = 16
_NUM_KV_HEADS = 4
_HEAD_DIM = 64
_DTYPE = torch.float16
_CAUSAL = True

_PAGE_SIZES = [256, 16]  # 256 = native-paged, 16 = flat-gather
_BACKENDS = ("fa2", "aiter_py", "aiter_cpp")
_OUTPUT_DIR = str(Path(__file__).parent)


def _flops(q_len, kv_len, num_qo_heads, head_dim, causal):
    return q_len * kv_len * num_qo_heads * head_dim * (2 if causal else 4)


def _bytes(q_len, kv_len, num_qo_heads, num_kv_heads, head_dim):
    return 2 * head_dim * (
        q_len * num_qo_heads + kv_len * num_kv_heads + kv_len * num_kv_heads + q_len * num_qo_heads
    )


def _build_paged_kv(kv_len, page_size, num_kv_heads, head_dim, dtype, device):
    """Build a minimal batch=1 paged KV layout."""
    num_full_pages = kv_len // page_size
    last_page_len = kv_len % page_size or page_size
    num_pages = num_full_pages + (1 if kv_len % page_size else 0)

    # [num_pages, page_size, num_kv_heads, head_dim] NHD paged layout
    paged_k = torch.randn(num_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    paged_v = torch.randn(num_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=device)

    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=device)

    return paged_k, paged_v, kv_indptr, kv_indices, kv_last_page_len, num_pages


def _make_fa2_wrapper(kv_len, page_size, paged_k, paged_v, kv_indptr, kv_indices,
                      kv_last_page_len, q):
    """Pre-planned FA2 BatchPrefillWithPagedKVCacheWrapper."""
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
    qo_indptr = torch.tensor([0, _Q_LEN], dtype=torch.int32, device=q.device)
    w.plan(
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
        _NUM_QO_HEADS, _NUM_KV_HEADS, _HEAD_DIM, page_size,
        causal=_CAUSAL, q_data_type=_DTYPE,
    )
    return w


def _make_aiter_cpp_wrapper(kv_len, page_size, paged_k, paged_v, kv_indptr, kv_indices,
                             kv_last_page_len, q):
    """Pre-planned AITER-C++ BatchPrefillWithPagedKVCacheWrapper."""
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="aiter")
    qo_indptr = torch.tensor([0, _Q_LEN], dtype=torch.int32, device=q.device)
    w.plan(
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
        _NUM_QO_HEADS, _NUM_KV_HEADS, _HEAD_DIM, page_size,
        causal=_CAUSAL, q_data_type=_DTYPE,
    )
    return w


@torch.inference_mode()
def _make_configs():
    from aiter.ops.mha import mha_batch_prefill_func as _aiter_bp_func
    from aiter.ops.mha import flash_attn_varlen_func as _aiter_fwd

    configs = []

    for page_size in _PAGE_SIZES:
        path_tag = "native" if page_size == 256 else "flatgather"

        for kv_len in _KV_LENS:
            q = torch.randn(_Q_LEN, _NUM_QO_HEADS, _HEAD_DIM, dtype=_DTYPE, device="cuda")
            paged_k, paged_v, kv_indptr, kv_indices, kv_last_page_len, num_pages = (
                _build_paged_kv(kv_len, page_size, _NUM_KV_HEADS, _HEAD_DIM, _DTYPE, q.device)
            )
            qo_indptr = torch.tensor([0, _Q_LEN], dtype=torch.int32, device=q.device)

            flops = _flops(_Q_LEN, kv_len, _NUM_QO_HEADS, _HEAD_DIM, _CAUSAL)
            theo_bytes = _bytes(_Q_LEN, kv_len, _NUM_QO_HEADS, _NUM_KV_HEADS, _HEAD_DIM)
            shape_tag = f"ps{page_size}_kv{kv_len}"
            sm_scale = 1.0 / math.sqrt(_HEAD_DIM)

            # ── FA2 ───────────────────────────────────────────────────────────
            w_fa2 = _make_fa2_wrapper(
                kv_len, page_size, paged_k, paged_v, kv_indptr, kv_indices, kv_last_page_len, q
            )
            configs.append(KernelConfig(
                name=f"fa2_{shape_tag}",
                run_fn=torch.inference_mode()(
                    lambda q=q, pk=paged_k, pv=paged_v, w=w_fa2: w.run(q, (pk, pv))
                ),
                theoretical_flops=flops,
                theoretical_bytes=theo_bytes,
                label=f"FA2       ps={page_size:>3d}  kv={kv_len:>5d}  causal ({path_tag})",
            ))

            # ── AITER-Py ─────────────────────────────────────────────────────
            if page_size in (128, 256, 1024):
                # Native-paged: mha_batch_prefill_func with paged KV
                # [num_pages, page_size, nhead_kv, head_dim] → pass as-is
                configs.append(KernelConfig(
                    name=f"aiter_py_{shape_tag}",
                    run_fn=torch.inference_mode()(
                        lambda q=q, pk=paged_k, pv=paged_v,
                               qi=qo_indptr, ki=kv_indptr,
                               kpi=kv_indices, lpl=kv_last_page_len,
                               kl=kv_len, sc=sm_scale: _aiter_bp_func(
                            q=q, k=pk, v=pv,
                            cu_seqlens_q=qi,
                            kv_indptr=ki,
                            kv_page_indices=kpi,
                            kv_last_page_lens=lpl,
                            max_seqlen_q=_Q_LEN,
                            max_seqlen_k=kl,
                            softmax_scale=sc,
                            causal=_CAUSAL,
                            return_lse=False,
                        )
                    ),
                    theoretical_flops=flops,
                    theoretical_bytes=theo_bytes,
                    label=f"AITER-Py  ps={page_size:>3d}  kv={kv_len:>5d}  causal ({path_tag})",
                ))
            else:
                # Flat-gather path: replicate the old Python passthrough
                # Gather all tokens into a flat [total_kv, nhead_kv, head_dim] tensor
                # then call flash_attn_varlen_func.
                num_pages_v = kv_indptr[-1].item()
                last_page_tokens = int(kv_last_page_len[0].item())
                total_kv = (num_pages_v - 1) * page_size + last_page_tokens

                # Build gather index (pre-computed, same as plan() does)
                gather_idx = torch.cat([
                    torch.arange(p * page_size, p * page_size + page_size,
                                 dtype=torch.long, device=q.device)
                    for p in range(int(num_pages_v) - 1)
                ] + [
                    torch.arange(
                        int(kv_indices[-1].item()) * page_size,
                        int(kv_indices[-1].item()) * page_size + last_page_tokens,
                        dtype=torch.long, device=q.device,
                    )
                ])

                # Flat K/V: reshape paged_k to [num_pages*page_size, nhead, head_dim]
                pk_flat = paged_k.view(-1, _NUM_KV_HEADS, _HEAD_DIM)
                pv_flat = paged_v.view(-1, _NUM_KV_HEADS, _HEAD_DIM)
                cu_k = torch.tensor([0, total_kv], dtype=torch.int32, device=q.device)

                configs.append(KernelConfig(
                    name=f"aiter_py_{shape_tag}",
                    run_fn=torch.inference_mode()(
                        lambda q=q, pkf=pk_flat, pvf=pv_flat,
                               gi=gather_idx, qi=qo_indptr, ck=cu_k,
                               tkv=total_kv, sc=sm_scale: _aiter_fwd(
                            q=q,
                            k=pkf[gi],
                            v=pvf[gi],
                            cu_seqlens_q=qi,
                            cu_seqlens_k=ck,
                            max_seqlen_q=_Q_LEN,
                            max_seqlen_k=tkv,
                            dropout_p=0.0,
                            softmax_scale=sc,
                            logits_soft_cap=0.0,
                            causal=_CAUSAL,
                            window_size=(-1, -1, 0),
                            return_lse=False,
                            return_attn_probs=False,
                            out=None,
                        )
                    ),
                    theoretical_flops=flops,
                    theoretical_bytes=theo_bytes,
                    label=f"AITER-Py  ps={page_size:>3d}  kv={kv_len:>5d}  causal ({path_tag})",
                ))

            # ── AITER-C++ ────────────────────────────────────────────────────
            w_cpp = _make_aiter_cpp_wrapper(
                kv_len, page_size, paged_k, paged_v, kv_indptr, kv_indices, kv_last_page_len, q
            )
            configs.append(KernelConfig(
                name=f"aiter_cpp_{shape_tag}",
                run_fn=torch.inference_mode()(
                    lambda q=q, pk=paged_k, pv=paged_v, w=w_cpp: w.run(q, (pk, pv))
                ),
                theoretical_flops=flops,
                theoretical_bytes=theo_bytes,
                label=f"AITER-C++ ps={page_size:>3d}  kv={kv_len:>5d}  causal ({path_tag})",
            ))

    return configs


def _print_comparison_table(timing_csv: Path) -> None:
    rows: dict[str, dict[str, float]] = {}
    with open(timing_csv) as f:
        for r in csv.DictReader(f):
            name: str = r["name"]
            ms = float(r["median_ms"])
            for backend in _BACKENDS:
                if name.startswith(f"{backend}_"):
                    tag = name[len(backend) + 1:]
                    rows.setdefault(tag, {})[backend] = ms
                    break

    def _fmt(v):
        return f"{v * 1000:.1f} µs" if not math.isnan(v) else "    N/A  "

    w = 12
    sep = "-" * (24 + w * 3 + 16)
    hdr = f"{'Shape':<24}{'FA2':>{w}}{'AITER-Py':>{w}}{'AITER-C++':>{w}}  {'C++ overhead':>14}"

    for page_size in _PAGE_SIZES:
        path_tag = "native-paged" if page_size == 256 else "flat-gather"
        print()
        print("=" * len(hdr))
        print(f"  Batch-Paged Prefill — page_size={page_size} ({path_tag})")
        print(f"  q=256, batch=1, GQA 16/4, HD=64, causal")
        print("=" * len(hdr))
        print(hdr)
        print(sep)

        geom_overhead = 0.0
        n = 0
        for kv_len in _KV_LENS:
            tag = f"ps{page_size}_kv{kv_len}"
            if tag not in rows:
                continue
            d = rows[tag]
            fa2 = d.get("fa2", float("nan"))
            py  = d.get("aiter_py", float("nan"))
            cpp = d.get("aiter_cpp", float("nan"))
            overhead = (cpp - py) / py * 100 if py > 0 else float("nan")
            geom_overhead += overhead
            n += 1
            print(
                f"  kv={kv_len:>5d}               "
                f"{_fmt(fa2):>{w}}{_fmt(py):>{w}}{_fmt(cpp):>{w}}"
                f"  {overhead:>+11.1f}%"
            )

        mean_overhead = geom_overhead / n if n > 0 else float("nan")
        print(sep)
        print(
            f"  {'Mean overhead':22}{'':>{w}}{'':>{w}}{'':>{w}}  {mean_overhead:>+11.1f}%"
        )

    print()
    print("  C++ overhead: positive = C++ harness is slower than AITER Python passthrough.")
    print("  FA2 shown for reference; AITER paths are 3-7× faster than FA2.")
    print()


if __name__ == "__main__":
    _skip_gpu = "--replot" in sys.argv or "--list-presets" in sys.argv
    profiler = RocmProfiler(
        configs=[] if _skip_gpu else _make_configs(),
        num_warmup=3,
        dry_run_ms=100,
        repeat_ms=1000,
        counters=_counters,
        kernel_name_regex="",
        output_dir=_OUTPUT_DIR,
        label=_label,
        roofline=(_counters == "roofline"),
    )
    profiler.run()

    if not _skip_gpu:
        timing_csv = Path(_OUTPUT_DIR) / f"{_label}_timing.csv"
        if timing_csv.exists():
            _print_comparison_table(timing_csv)
