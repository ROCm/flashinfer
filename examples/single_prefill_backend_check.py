# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run a FlashInfer single-prefill example while tracing which attention backend
(fa2 / fa3 / aiter) is actually used for each call.

By default this runs the sibling ``single_prefill_example.py`` script located in
the same directory as this file. A different example path can be passed as the
first positional argument.

Usage:
    python single_prefill_backend_check.py
    python single_prefill_backend_check.py path/to/other_example.py
"""

from __future__ import annotations

import argparse
import logging
import os
import runpy
import sys
from collections import Counter

import flashinfer.prefill_rocm as pr


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_example = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "single_prefill_example.py"
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

    def traced_get_module(backend, *a, **kw):
        backend_calls[backend] += 1
        print(f"  [trace] get_single_prefill_module backend={backend!r}", flush=True)
        return orig_get_module(backend, *a, **kw)

    pr.get_single_prefill_module = traced_get_module

    if hasattr(pr, "determine_attention_backend"):
        orig_determine = pr.determine_attention_backend

        def traced_determine(*a, **kw):
            sel = orig_determine(*a, **kw)
            print(f"  [trace] determine_attention_backend -> {sel!r}", flush=True)
            return sel

        pr.determine_attention_backend = traced_determine

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
        aiter_used = backend_calls.get("aiter", 0) > 0
        print(f"\nAITER backend used: {'YES' if aiter_used else 'NO'}", flush=True)


if __name__ == "__main__":
    main()
