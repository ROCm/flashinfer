# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run a FlashInfer single-prefill example with the attention backend forced.

This monkey-patches ``flashinfer.single_prefill_with_kv_cache`` (and its
``_return_lse`` partial) so that every call uses the backend selected on the
command line, regardless of what the example passes. Useful for verifying that
a specific backend (e.g. ``aiter`` on ROCm gfx942/gfx950) actually works for
the example's configurations.

By default this runs the sibling ``single_prefill_example.py`` script located
in the same directory as this file, with ``backend='aiter'``.

Usage:
    python single_prefill_force_backend.py
    python single_prefill_force_backend.py --backend fa2
    python single_prefill_force_backend.py --backend aiter path/to/other_example.py
"""

from __future__ import annotations

import argparse
import logging
import os
import runpy
import sys
from collections import Counter

import flashinfer
import flashinfer.prefill_rocm as pr


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_example = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "single_prefill_example.py"
    )
    parser.add_argument(
        "--backend",
        default="aiter",
        help="Backend to force for every single_prefill call (default: %(default)s)",
    )
    parser.add_argument(
        "example",
        nargs="?",
        default=default_example,
        help="Path to the example script to run (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    backend_calls: Counter[str] = Counter()
    orig_get_module = pr.get_single_prefill_module
    orig_single = flashinfer.single_prefill_with_kv_cache
    orig_single_lse = flashinfer.single_prefill_with_kv_cache_return_lse

    def traced_get_module(backend, *a, **kw):
        backend_calls[backend] += 1
        print(f"  [trace] get_single_prefill_module backend={backend!r}", flush=True)
        return orig_get_module(backend, *a, **kw)

    pr.get_single_prefill_module = traced_get_module

    def force(*a, **kw):
        kw["backend"] = args.backend
        return orig_single(*a, **kw)

    def force_lse(*a, **kw):
        kw["backend"] = args.backend
        return orig_single_lse(*a, **kw)

    flashinfer.single_prefill_with_kv_cache = force
    flashinfer.single_prefill_with_kv_cache_return_lse = force_lse

    print(f"Forcing backend={args.backend!r} for every single_prefill call", flush=True)
    sys.argv = [args.example]
    try:
        runpy.run_path(args.example, run_name="__main__")
    finally:
        print("\n" + "=" * 60, flush=True)
        print("Backend usage summary (get_single_prefill_module calls):", flush=True)
        if not backend_calls:
            print("  (no calls captured)", flush=True)
        for name, n in backend_calls.most_common():
            print(f"  {name}: {n}", flush=True)
        used = backend_calls.get(args.backend, 0) > 0
        print(
            f"\n{args.backend.upper()} backend used: {'YES' if used else 'NO'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
