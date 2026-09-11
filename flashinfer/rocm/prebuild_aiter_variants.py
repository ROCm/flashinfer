# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build the AITER attention variants ahead of time, into FlashInfer's store.

Without this, the first ``plan()`` for each shape pays an AITER CK-tile compile
-- 74-360s on gfx942 depending on the family, per shape, and again after every
container restart that does not persist site-packages. On a read-only install it
does not stall, it fails, and ``backend="auto"`` silently drops to fa2.

    python -m flashinfer.rocm.prebuild_aiter_variants --list
    python -m flashinfer.rocm.prebuild_aiter_variants --arch gfx942
    python -m flashinfer.rocm.prebuild_aiter_variants --only mha_batch_prefill

**This needs a GPU.** The only supported way to make AITER emit a variant is to
call the op, which launches a kernel -- so it cannot be a ``docker build`` step
and has to run as a GPU-attached job whose output the image copies in.

One build at a time, deliberately. ``_aiter_env_scope`` mutates process-global
environment that AITER reads, so two concurrent builds in one process have the
first to finish restore the environment under the second. Parallelism belongs
across *processes*, each with its own environment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from ..jit.rocm.aiter_source import (
    _aiter_env_scope,
    _BUILD_LOCK,
    resolve_aiter_build_arch,
)
from ..jit.rocm.aiter_variants import (
    MANIFEST_NAME,
    BuildSpec,
    Family,
    VariantKey,
    builds,
    reachable_variants,
    so_name,
    variant_store_dir,
)


def _dtype(name: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16}[name]


def _run_build(spec: BuildSpec, device_idx: int, head_dim: int) -> None:
    """Invoke the bootstrap that produces ``spec``.

    Calls the real bootstraps rather than reimplementing AITER's naming: they
    already encode which (dtype, mask, lse, logits) maps to which call, they are
    what plan() uses, and they are covered by the existing tests. Only the output
    directory differs, and that comes from the env scope the caller opened.
    """
    from . import prefill as _prefill

    dtype = _dtype(spec.dtype)
    if spec.family is Family.MHA_FWD:
        _prefill._aiter_bootstrap_single_prefill_mha_fwd(
            dtype, spec.needs_mask, bool(spec.has_lse), head_dim, device_idx
        )
    elif spec.family is Family.MHA_VARLEN_FWD:
        # Loops return_lse internally and emits both arms.
        _prefill._aiter_bootstrap_batch_ragged_prefill(
            dtype, spec.has_logits_cap, spec.needs_mask, head_dim, device_idx
        )
    elif spec.family is Family.MHA_BATCH_PREFILL:
        page_size = _native_page_size()
        _prefill._aiter_bootstrap_batch_prefill(
            dtype,
            spec.has_logits_cap,
            spec.needs_mask,
            bool(spec.has_lse),
            page_size,
            head_dim,
            device_idx,
        )
    else:  # pragma: no cover - Family is closed
        raise AssertionError(f"unhandled family {spec.family}")


def _native_page_size() -> int:
    """A page size this AITER build actually has a paged kernel for.

    The version predicate can name sizes the installed build rejects with "no
    matching kernel found", so try each and let the caller see the failure only
    when none work.
    """
    from .prefill import _aiter_native_page_sizes

    sizes = sorted(_aiter_native_page_sizes())
    return sizes[0] if sizes else 16


def _publish(produced: Iterable[Path], store: Path) -> List[str]:
    """Copy artifacts into the store, atomically. Returns the names published."""
    store.mkdir(parents=True, exist_ok=True)
    names = []
    for src in produced:
        dst = store / src.name
        tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        names.append(src.name)
    return names


def _locate(keys: Sequence[VariantKey], search: Sequence[Path]) -> List[Path]:
    found = []
    for key in keys:
        name = so_name(key)
        for root in search:
            candidate = root / name
            if candidate.is_file():
                found.append(candidate)
                break
            hits = sorted(root.rglob(name)) if root.exists() else []
            if hits:
                found.append(hits[0])
                break
    return found


def prebuild(
    specs: Sequence[BuildSpec],
    *,
    arch: Optional[str] = None,
    device_idx: int = 0,
    head_dim: int = 128,
    force: bool = False,
) -> Tuple[int, int]:
    """Build ``specs`` into the variant store. Returns (built, skipped).

    A spec whose every output is already in the store is skipped, so a rerun
    after a partial failure resumes rather than restarting.
    """
    from . import prefill as _prefill

    # The varlen bootstraps reach torch.ops.aiter.mha_varlen_fwd, which is not
    # registered until aiter.ops is imported. plan() always passes through this
    # probe first; the driver does not, so call it explicitly rather than rely
    # on an import another path happened to do. It also applies the ABI floor.
    if not _prefill._aiter_ops_importable():
        raise RuntimeError(
            "aiter.ops is not importable, or amd-aiter is below the supported "
            "floor; nothing can be prebuilt."
        )

    store = variant_store_dir(arch)
    store.mkdir(parents=True, exist_ok=True)

    built = skipped = 0
    for index, spec in enumerate(specs, start=1):
        wanted = [so_name(key) for key in spec.produces]
        if not force and all((store / name).is_file() for name in wanted):
            skipped += 1
            continue

        label = ", ".join(wanted)
        print(f"[{index}/{len(specs)}] building {label}", flush=True)
        started = time.time()
        # build_dir=None on purpose: AITER imports the variant it just built,
        # and only puts AITER_JIT_DIR on sys.path at its own import time, so
        # redirecting it here yields ModuleNotFoundError for the new module.
        # Let AITER build where it wants and copy the result into the store.
        with _BUILD_LOCK, _aiter_env_scope(None, symbol_visible=False):
            _run_build(spec, device_idx, head_dim)
            from aiter.jit import core as aiter_core

            search = [Path(aiter_core.get_user_jit_dir())]
            produced = _locate(spec.produces, search)
        if not produced:
            raise RuntimeError(
                f"AITER produced none of {label}. Searched {[str(p) for p in search]}."
            )
        names = _publish(produced, store)
        built += 1
        print(
            f"    -> {len(names)} file(s) in {time.time() - started:.1f}s", flush=True
        )

    _write_manifest(store, arch)
    return built, skipped


def _write_manifest(store: Path, arch: Optional[str]) -> None:
    try:
        import importlib.metadata as _md

        aiter_version = _md.version("amd-aiter")
    except Exception:
        aiter_version = "unknown"
    try:
        import torch

        rocm = torch.version.hip or "unknown"
    except Exception:
        rocm = "unknown"
    present = sorted(p.name for p in store.glob("*.so"))
    (store / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "rocm_arch_list": arch or resolve_aiter_build_arch(),
                "aiter_version": aiter_version,
                "rocm_version": rocm,
                "variants": present,
            },
            indent=2,
        )
        + "\n"
    )


def prune(*, apply: bool = False) -> List[Path]:
    """Stores whose tag does not match this install. Deletes only when ``apply``.

    Each (arch, aiter, rocm) tuple gets its own directory and nothing removes
    the old ones, so every upgrade leaves ~165 MB behind -- on shared nodes,
    indefinitely.

    Dry-run by default, and it deletes the enumerated paths it printed rather
    than sweeping a glob: the store lives under a shared cache root, and "every
    directory except the current one" is how somebody else's arch gets deleted.
    """
    root = variant_store_dir().parent
    current = variant_store_dir().name
    stale = (
        sorted(d for d in root.glob("*") if d.is_dir() and d.name != current)
        if root.is_dir()
        else []
    )

    for path in stale:
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        print(f"{'removing' if apply else 'stale'}   {path}  ({size / 1e6:.0f} MB)")
        if apply:
            shutil.rmtree(path)
    if not stale:
        print(f"no stale stores under {root}")
    elif not apply:
        print(f"{len(stale)} stale store(s); re-run with --prune --yes to delete")
    return stale


def _select(only: Optional[str]) -> List[BuildSpec]:
    specs = list(builds())
    if only:
        wanted = {name.strip() for name in only.split(",") if name.strip()}
        known = {f.name.lower() for f in Family}
        unknown = wanted - known
        if unknown:
            raise SystemExit(
                f"unknown family {sorted(unknown)}; expected {sorted(known)}"
            )
        specs = [s for s in specs if s.family.name.lower() in wanted]
    return specs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", help="override the store arch (default: this device)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--head-dim",
        type=int,
        default=128,
        help="a cache key only; one .so serves every head dim",
    )
    parser.add_argument("--only", help="comma-separated families, e.g. mha_fwd")
    parser.add_argument("--force", action="store_true", help="rebuild present variants")
    parser.add_argument("--list", action="store_true", help="print the plan and exit")
    parser.add_argument(
        "--prune", action="store_true", help="report stores from other arch/aiter/rocm"
    )
    parser.add_argument("--yes", action="store_true", help="with --prune, delete them")
    args = parser.parse_args(argv)

    if args.prune:
        prune(apply=args.yes)
        return 0

    specs = _select(args.only)
    store = variant_store_dir(args.arch)
    if args.list:
        print(f"store: {store}")
        print(f"{len(reachable_variants())} variants from {len(specs)} builds")
        for spec in specs:
            state = (
                "present"
                if all((store / so_name(k)).is_file() for k in spec.produces)
                else "missing"
            )
            print(f"  {state:8s} {', '.join(so_name(k) for k in spec.produces)}")
        return 0

    built, skipped = prebuild(
        specs,
        arch=args.arch,
        device_idx=args.device,
        head_dim=args.head_dim,
        force=args.force,
    )
    print(f"built {built}, skipped {skipped} already present -> {store}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
