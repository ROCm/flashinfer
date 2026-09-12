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

Block-sparse attention against dense prefill: where sparsity starts paying.

Both ROCm wrappers are here -- ``VariableBlockSparseAttentionWrapper`` (per-head
block sizes, the shape SGLang's hybrid attention uses) and the fixed-block
``BlockSparseAttentionWrapper``. Unlike the upstream
``benchmarks/bench_block_sparse_attention.py`` there is no fa3 arm:
``determine_attention_backend`` never returns aiter for block_sparse, and fa3
raises "FA3 backend not currently supported for ROCm" from the JIT generator.

``VariableBlockSparseAttentionWrapper.run`` rearranges q, k and v on every
call, and there is no way through that API to avoid it, so those copies are
inside its timed region and its rows read lower than the kernel alone. The
fixed-block rows carry no such copy; compare across kinds with that in mind.

The sweep is over block *density*, because that is the only axis that decides
whether sparsity wins: the dense baseline is flat in it and the sparse kernel is
linear, so the crossover is a property of the pair, not of either one. A
density-1.0 row is deliberately included -- sparse must not beat dense there, and
if it does the dense baseline is set up wrong.

Run:
    python benchmarks/rocm/bench_block_sparse_attention.py
    python benchmarks/rocm/bench_block_sparse_attention.py --csv bsa-gfx942.csv
    python benchmarks/rocm/bench_block_sparse_attention.py --accuracy
"""

import argparse
import csv
import logging
import statistics
import subprocess
from pathlib import Path

import torch

import flashinfer
from flashinfer.jit.core import logger as _jit_logger
from flashinfer.testing import bench_gpu_time

_jit_logger.setLevel(logging.WARNING)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SEQ_LENS = [2048, 4096, 8192]
_DENSITIES = [0.1, 0.25, 0.5, 1.0]
_HEAD_DIM = 128
# MHA and GQA. The variable-block wrapper requires equal head counts, so the
# GQA row exercises the fixed-block wrapper only.
_HEAD_PAIRS = [(32, 32), (32, 8)]
_DTYPE = torch.float16
_WORKSPACE_BYTES = 128 * 1024 * 1024


def _assert_import_provenance() -> Path:
    """The editable install and the pinned worktree are easy to confuse, and the
    wrong one rebuilds cleanly -- so fail loudly rather than measure it."""
    got = Path(flashinfer.__file__).resolve()
    if _REPO_ROOT not in got.parents:
        raise SystemExit(
            f"flashinfer resolved to {got}, which is outside this checkout "
            f"({_REPO_ROOT}). Set PYTHONPATH to the tree you mean to measure."
        )
    return got


def _git_describe() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _provenance() -> dict:
    props = torch.cuda.get_device_properties(0)
    return {
        "flashinfer": str(_assert_import_provenance()),
        "git": _git_describe(),
        "arch": props.gcnArchName,
        "torch": torch.__version__,
    }


def _time_us(fn, dry_run_iters: int, repeat_iters: int) -> tuple[float, float]:
    """Median and p95-p05 spread in microseconds."""
    times = bench_gpu_time(fn, dry_run_iters=dry_run_iters, repeat_iters=repeat_iters)
    times = sorted(float(t) * 1000.0 for t in times)
    lo = times[max(0, int(0.05 * len(times)) - 1)]
    hi = times[min(len(times) - 1, int(0.95 * len(times)))]
    return statistics.median(times), hi - lo


def _block_mask(num_heads: int, nb_row: int, nb_col: int, density: float, seed: int):
    """A per-head block mask with at least one live block per row.

    An all-dead row makes the kernel skip the row entirely, so a naive Bernoulli
    draw at low density measures a shrinking problem rather than a sparse one.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    mask = torch.rand(num_heads, nb_row, nb_col, device="cuda", generator=gen) < density
    diag = torch.arange(min(nb_row, nb_col), device="cuda")
    mask[:, diag, diag] = True
    return mask


def _to_csr(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-block wrapper takes one shared CSR layout, so collapse over heads."""
    flat = mask.any(dim=0)
    counts = flat.sum(dim=1)
    indptr = torch.zeros(flat.shape[0] + 1, dtype=torch.int32, device="cuda")
    indptr[1:] = torch.cumsum(counts, dim=0)
    indices = flat.nonzero()[:, 1].to(torch.int32)
    return indptr, indices


def _variable_case(
    seq_len: int, num_qo: int, num_kv: int, density: float, nb: int, seed: int
):
    """Zero-arg closures for the variable-block wrapper and its dense baseline.

    The mask and block sizes are shaped on num_kv_heads -- a group of qo heads
    shares one kv head's sparsity pattern -- so GQA is served, not skipped.
    """
    mask = _block_mask(num_kv, nb, nb, density, seed)
    block_sz = torch.full((num_kv, nb), seq_len // nb, dtype=torch.int32, device="cuda")
    q = torch.randn(num_qo, seq_len, _HEAD_DIM, dtype=_DTYPE, device="cuda")
    k = torch.randn(num_kv, seq_len, _HEAD_DIM, dtype=_DTYPE, device="cuda")
    v = torch.randn(num_kv, seq_len, _HEAD_DIM, dtype=_DTYPE, device="cuda")

    ws = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.sparse.VariableBlockSparseAttentionWrapper(ws, backend="fa2")
    wrapper.plan(
        block_mask_map=mask,
        block_row_sz=block_sz,
        block_col_sz=block_sz,
        num_qo_heads=num_qo,
        num_kv_heads=num_kv,
        head_dim=_HEAD_DIM,
        q_data_type=_DTYPE,
    )
    # NHD for the dense arm: single_prefill takes [seq, heads, dim], the sparse
    # wrapper [heads, seq, dim]. Transposing inside the timed closure would
    # measure the permute.
    qd, kd, vd = (t.transpose(0, 1).contiguous() for t in (q, k, v))
    return {
        "sparse": lambda: wrapper.run(q, k, v),
        "dense": lambda: flashinfer.single_prefill_with_kv_cache(
            qd, kd, vd, causal=False, backend="fa2"
        ),
    }


def _fixed_case(
    seq_len: int, num_qo: int, num_kv: int, density: float, nb: int, seed: int
):
    """Zero-arg closures for the fixed-block wrapper and its dense baseline."""
    block = seq_len // nb
    indptr, indices = _to_csr(_block_mask(1, nb, nb, density, seed))
    q = torch.randn(seq_len, num_qo, _HEAD_DIM, dtype=_DTYPE, device="cuda")
    k = torch.randn(seq_len, num_kv, _HEAD_DIM, dtype=_DTYPE, device="cuda")
    v = torch.randn(seq_len, num_kv, _HEAD_DIM, dtype=_DTYPE, device="cuda")

    ws = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BlockSparseAttentionWrapper(ws, backend="fa2")
    wrapper.plan(
        indptr,
        indices,
        seq_len,
        seq_len,
        block,
        block,
        num_qo,
        num_kv,
        _HEAD_DIM,
        q_data_type=_DTYPE,
    )
    return {
        "sparse": lambda: wrapper.run(q, k, v),
        "dense": lambda: flashinfer.single_prefill_with_kv_cache(
            q, k, v, causal=False, backend="fa2"
        ),
    }


def _sweep(kinds, dry_run_iters: int, repeat_iters: int, nb: int, seed: int) -> list:
    rows = []
    for kind in kinds:
        for seq_len in _SEQ_LENS:
            for num_qo, num_kv in _HEAD_PAIRS:
                for density in _DENSITIES:
                    rec = {
                        "kind": kind,
                        "seq_len": seq_len,
                        "num_qo_heads": num_qo,
                        "num_kv_heads": num_kv,
                        "density": density,
                        "num_blocks": nb,
                    }
                    try:
                        if kind == "variable":
                            fns = _variable_case(
                                seq_len, num_qo, num_kv, density, nb, seed
                            )
                        else:
                            fns = _fixed_case(
                                seq_len, num_qo, num_kv, density, nb, seed
                            )
                        for arm in ("sparse", "dense"):
                            med, spread = _time_us(
                                fns[arm], dry_run_iters, repeat_iters
                            )
                            rec[f"{arm}_us"] = round(med, 3)
                            rec[f"{arm}_spread_us"] = round(spread, 3)
                    except Exception as exc:  # noqa: BLE001 - a refusal is a result
                        rec["err"] = f"{type(exc).__name__}: {exc}"[:160]
                    finally:
                        # A hard fault surfaces here, outside the try above, and
                        # would otherwise discard every row measured so far.
                        try:
                            torch.cuda.synchronize()
                        except Exception as exc:  # noqa: BLE001
                            rec.setdefault("err", f"{type(exc).__name__}: {exc}"[:160])
                        torch.cuda.empty_cache()
                    s, d = rec.get("sparse_us"), rec.get("dense_us")
                    rec["speedup"] = round(d / s, 4) if s and d else None
                    rows.append(rec)
                    note = f"  !! {rec['err']}" if "err" in rec else ""
                    print(
                        f"{kind:9s} s={seq_len:<6d} h={num_qo}/{num_kv:<3d} "
                        f"density={density:<5.2f} sparse={s} dense={d} "
                        f"speedup={rec['speedup']}{note}",
                        flush=True,
                    )
    return rows


def _accuracy(nb: int, seed: int) -> None:
    """Fixed-block output against a masked float32 reference, per density."""
    seq_len, num_heads = 2048, 8
    block = seq_len // nb
    for density in _DENSITIES:
        mask = _block_mask(1, nb, nb, density, seed)
        indptr, indices = _to_csr(mask)
        q = torch.randn(seq_len, num_heads, _HEAD_DIM, dtype=_DTYPE, device="cuda")
        k = torch.randn(seq_len, num_heads, _HEAD_DIM, dtype=_DTYPE, device="cuda")
        v = torch.randn(seq_len, num_heads, _HEAD_DIM, dtype=_DTYPE, device="cuda")
        ws = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.BlockSparseAttentionWrapper(ws, backend="fa2")
        wrapper.plan(
            indptr,
            indices,
            seq_len,
            seq_len,
            block,
            block,
            num_heads,
            num_heads,
            _HEAD_DIM,
            q_data_type=_DTYPE,
        )
        got = wrapper.run(q, k, v)

        elem = mask[0].repeat_interleave(block, 0).repeat_interleave(block, 1)
        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float())
        scores /= _HEAD_DIM**0.5
        scores = scores.masked_fill(~elem.unsqueeze(0), float("-inf"))
        ref = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float())
        err = (got.float() - ref).abs().max().item()
        print(f"fixed    density={density:<5.2f} max_abs_err={err:.5f}")
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--kinds",
        nargs="+",
        choices=["variable", "fixed"],
        default=["variable", "fixed"],
    )
    ap.add_argument("--dry-run-iters", type=int, default=25)
    ap.add_argument("--repeat-iters", type=int, default=100)
    ap.add_argument("--num-blocks", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--accuracy", action="store_true")
    ap.add_argument("--csv", type=str, default="")
    args = ap.parse_args()

    bad = [n for n in _SEQ_LENS if n % args.num_blocks]
    if bad:
        raise SystemExit(
            f"--num-blocks {args.num_blocks} does not divide {bad}; the truncated "
            "block size would shrink the sparse problem below the dense baseline."
        )

    for key, value in _provenance().items():
        print(f"# {key}: {value}")

    if args.accuracy:
        _accuracy(args.num_blocks, args.seed)
        return

    rows = _sweep(
        args.kinds, args.dry_run_iters, args.repeat_iters, args.num_blocks, args.seed
    )

    if args.csv:
        fields = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"# wrote {args.csv} ({len(rows)} rows)")

    # Density 1.0 is the control: sparse has no work to skip, so a speedup there
    # means the dense arm is mis-specified and every other row is suspect.
    for row in rows:
        if row["density"] == 1.0 and row.get("speedup") and row["speedup"] > 1.05:
            print(
                f"# WARNING dense-equivalent row is {row['speedup']}x faster sparse "
                f"({row['kind']} s={row['seq_len']}): check the dense baseline"
            )
    wins = [r["speedup"] for r in rows if r.get("speedup") and r["density"] < 1.0]
    if wins:
        print(
            f"# sparse/dense speedup at density<1: n={len(wins)} "
            f"median={statistics.median(wins):.4f} min={min(wins):.4f} "
            f"max={max(wins):.4f}"
        )


if __name__ == "__main__":
    main()
