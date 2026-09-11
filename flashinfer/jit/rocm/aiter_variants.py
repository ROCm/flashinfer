# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The AITER attention variants FlashInfer ``dlopen``s, and where they live.

``csrc/rocm/aiter_loader.cc`` composes a ``.so`` filename from a variant key and
loads it out of AITER's JIT directory. The amd-aiter wheel prebuilds almost none
of them -- measured on 0.1.20: zero ``mha_fwd_*``, zero ``mha_batch_prefill_*``,
and of five shipped ``mha_varlen_fwd_*`` only two carry the ``_nskip_`` spelling
the loader asks for. Everything else is built by AITER's lazy JIT at ``plan()``
time, which costs 74-360s per file depending on the family.

This module enumerates that set so it can be built ahead of time instead.

**Variants and builds are not one-to-one.** Two of the bootstraps in
``flashinfer/rocm/prefill.py`` loop over ``return_lse`` internally and so emit
the ``_lse`` and ``_nlse`` files from a single call, while the other two take
``has_lse`` as a parameter and emit one. Driving this table by filename would
run those builds twice, concurrently, into one output directory -- hence
:func:`builds` alongside :func:`reachable_variants`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

from .. import env as jit_env
from .aiter_source import resolve_aiter_build_arch

__all__ = [
    "Family",
    "VariantKey",
    "builds",
    "reachable_variants",
    "so_name",
    "variant_store_dir",
]


class Family(Enum):
    """The three ``.so`` families, keyed by their loader in aiter_loader.cc.

    ``prefix``/``suffix``/``include_logits`` mirror the ``build_so_name`` call
    sites there exactly; ``tests/rocm/test_aiter_variants.py`` asserts the
    spelling against the C++ source rather than trusting this copy.
    """

    MHA_FWD = ("mha_fwd_", "_ndropout_nqscale.so", False)
    MHA_VARLEN_FWD = ("mha_varlen_fwd_", "_ndropout_nskip_nqscale.so", True)
    MHA_BATCH_PREFILL = ("mha_batch_prefill_", "_ndropout_nqscale_nsink.so", True)

    def __init__(self, prefix: str, suffix: str, include_logits: bool) -> None:
        self.prefix = prefix
        self.suffix = suffix
        self.include_logits = include_logits


# Deliberately not order=True: `family` is an Enum, so the generated __lt__
# raises rather than sorting. Sort by so_name() when an order is needed.
@dataclass(frozen=True)
class VariantKey:
    """One ``.so``. Mirrors ``VariantKey`` / ``BatchPrefillVariantKey`` in C++.

    ``has_alibi`` is absent on purpose: all three construction sites in
    ``include/flashinfer/rocm/attention/aiter/`` hard-code it to false, so only
    the ``_nbias`` arm is reachable and enumerating the other doubles the set
    for nothing.
    """

    family: Family
    dtype: str  # "fp16" or "bf16"
    has_logits_cap: bool
    needs_mask: bool
    has_lse: bool


def so_name(key: VariantKey) -> str:
    """The filename ``aiter_loader.cc`` will ask ``dlopen`` for."""
    name = key.family.prefix + key.dtype
    if key.family.include_logits:
        name += "_logits" if key.has_logits_cap else "_nlogits"
    name += "_nbias"
    name += "_mask" if key.needs_mask else "_nmask"
    name += "_lse" if key.has_lse else "_nlse"
    return name + key.family.suffix


_DTYPES = ("bf16", "fp16")


def reachable_variants() -> Tuple[VariantKey, ...]:
    """Every ``.so`` the loader can ask for: 8 + 16 + 16 = 40 per architecture.

    ``mha_fwd`` contributes only 8 because ``get_aiter_mha_fwd_handle`` refuses
    ``has_logits_cap=true`` -- that template has no ``_logits`` arm, and a
    soft-capped call is routed to the varlen family instead.
    """
    out = []
    for family in Family:
        logits_values = (False, True) if family.include_logits else (False,)
        for dtype in _DTYPES:
            for has_logits_cap in logits_values:
                for needs_mask in (False, True):
                    for has_lse in (False, True):
                        out.append(
                            VariantKey(
                                family, dtype, has_logits_cap, needs_mask, has_lse
                            )
                        )
    return tuple(out)


@dataclass(frozen=True)
class BuildSpec:
    """One AITER build, and the variants it produces.

    ``emits_both_lse`` records the asymmetry: the varlen bootstraps loop over
    ``return_lse`` and yield two files per call, so a driver must not schedule
    the ``_lse`` and ``_nlse`` variants as separate builds.
    """

    family: Family
    dtype: str
    has_logits_cap: bool
    needs_mask: bool
    has_lse: Optional[bool]  # None when the build emits both arms
    produces: Tuple[VariantKey, ...]

    @property
    def emits_both_lse(self) -> bool:
        return self.has_lse is None


def builds() -> Tuple[BuildSpec, ...]:
    """The 32 AITER builds that cover all 40 variants.

    ``mha_varlen_fwd`` needs 8 builds for 16 files because
    ``_aiter_bootstrap_batch_ragged_prefill`` loops ``return_lse`` internally.
    """
    by_key = {
        (v.family, v.dtype, v.has_logits_cap, v.needs_mask, v.has_lse): v
        for v in reachable_variants()
    }
    out = []
    for family in Family:
        logits_values = (False, True) if family.include_logits else (False,)
        for dtype in _DTYPES:
            for has_logits_cap in logits_values:
                for needs_mask in (False, True):
                    if family is Family.MHA_VARLEN_FWD:
                        produces = tuple(
                            by_key[(family, dtype, has_logits_cap, needs_mask, lse)]
                            for lse in (False, True)
                        )
                        out.append(
                            BuildSpec(
                                family,
                                dtype,
                                has_logits_cap,
                                needs_mask,
                                None,
                                produces,
                            )
                        )
                        continue
                    for has_lse in (False, True):
                        key = by_key[
                            (family, dtype, has_logits_cap, needs_mask, has_lse)
                        ]
                        out.append(
                            BuildSpec(
                                family,
                                dtype,
                                has_logits_cap,
                                needs_mask,
                                has_lse,
                                (key,),
                            )
                        )
    return tuple(out)


MANIFEST_NAME = "variants_manifest.json"


def _rocm_version() -> str:
    """The ROCm/HIP version the artifacts were compiled against.

    Part of the store tag because these are CK-tile objects that ship between
    machines in a wheel, unlike the ``aiter_libs`` cache, which never leaves the
    box that built it.
    """
    try:
        import torch

        return torch.version.hip or "unknown"
    except Exception:
        return "unknown"


def variant_store_dir(arch: Optional[str] = None) -> Path:
    """``<cache>/aiter_variants/<arch>__aiter-<ver>__rocm-<ver>/``.

    Keyed hard enough that staleness is structural rather than something to
    detect: bump AITER or ROCm and the tag names a different directory, so the
    old contents are simply never found and the lookup misses into a rebuild.
    A mismatched artifact is never loaded.
    """
    try:
        import importlib.metadata as _md

        aiter_version = _md.version("amd-aiter")
    except Exception:
        aiter_version = "unknown"
    tag = f"{arch or resolve_aiter_build_arch()}__aiter-{aiter_version}__rocm-{_rocm_version()}"
    # The tag becomes a directory name; refuse anything that is not one
    # component, the same guard _aiter_cache_tag applies.
    if not tag or tag != Path(tag).name or tag.startswith("."):
        raise ValueError(f"refusing to build a cache directory name from {tag!r}")
    return jit_env.FLASHINFER_CACHE_DIR / "aiter_variants" / tag


def store_override() -> Optional[Path]:
    """``FLASHINFER_AITER_VARIANT_DIR``, for a store shipped outside the cache.

    Set by the container image and by the jit-cache wheel, both of which place
    the store somewhere the running user can read but not necessarily write.
    """
    raw = os.environ.get("FLASHINFER_AITER_VARIANT_DIR")
    return Path(raw) if raw else None


def find_variant(key: VariantKey) -> Optional[Path]:
    """The prebuilt ``.so`` for ``key``, or None if this install has none.

    The override is searched first so an image-supplied store wins over a
    half-populated cache dir from an earlier run.
    """
    for root in (store_override(), variant_store_dir()):
        if root is None:
            continue
        candidate = root / so_name(key)
        if candidate.is_file():
            return candidate
    return None


def dtype_tag(dtype) -> Optional[str]:
    """``torch.bfloat16`` -> ``"bf16"``. None for a dtype with no variant."""
    import torch

    return {torch.bfloat16: "bf16", torch.float16: "fp16"}.get(dtype)


def prebuilt(
    family: Family,
    dtype,
    *,
    has_logits_cap: bool = False,
    needs_mask: bool,
    has_lse: Optional[bool] = None,
) -> bool:
    """Is every ``.so`` this bootstrap would produce already in the store?

    ``has_lse=None`` asks about a varlen bootstrap, which emits both arms from
    one call -- so both must be present, or the call still has work to do.

    Only the store is consulted, not AITER's own JIT directory: a variant
    already there makes the bootstrap a no-op anyway (measured at 0.0s), so
    checking it would buy nothing and would couple this to AITER's layout.
    """
    tag = dtype_tag(dtype)
    if tag is None:
        return False
    lse_values = (False, True) if has_lse is None else (has_lse,)
    return all(
        find_variant(VariantKey(family, tag, has_logits_cap, needs_mask, lse))
        is not None
        for lse in lse_values
    )


def export_variant_store() -> Optional[Path]:
    """Publish the store to ``FLASHINFER_AITER_VARIANT_DIR`` for the C++ loader.

    Resolved here and passed through the environment rather than baked in as a
    ``-D``: an AOT-packaged module carries whatever path its *build* machine
    had, which does not exist on the consumer's. ``aiter_loader.cc`` reads this
    at call time.

    An operator-set value wins -- that is how a store shipped in an image or a
    jit-cache wheel is pointed at. A store that does not exist is not exported,
    so the loader's candidate list stays as short as the install warrants.
    """
    existing = store_override()
    if existing is not None:
        return existing
    try:
        store = variant_store_dir()
    except Exception:
        return None
    if not store.is_dir():
        return None
    os.environ["FLASHINFER_AITER_VARIANT_DIR"] = str(store)
    return store


def read_manifest(root: Path) -> Optional[dict]:
    """The store's manifest, or None when it is absent or unreadable.

    Unreadable resolves to None rather than raising, matching ``_aot_arch_is_usable``:
    a half-written manifest must not make the package unimportable.
    """
    try:
        return json.loads((root / MANIFEST_NAME).read_text())
    except Exception:
        return None
