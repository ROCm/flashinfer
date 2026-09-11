# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The variant table must agree with the C++ that ``dlopen``s the files.

``csrc/rocm/aiter_loader.cc`` builds the ``.so`` name, and
``flashinfer/jit/rocm/aiter_variants.py`` builds the same name in order to
prebuild it. Nothing links the two, so a prefix or suffix edited on one side
produces a store full of files the loader never asks for -- and the symptom is
not a crash but a silent return to the 74-360s first-call build.

These read the C++ source instead of trusting the Python copy. No torch, no GPU.
"""

import re
from pathlib import Path

import pytest

from flashinfer.jit.rocm import aiter_variants as av

_LOADER = Path(__file__).resolve().parents[2] / "csrc" / "rocm" / "aiter_loader.cc"

# build_so_name(key, "<prefix>", "<suffix>", /*include_logits=*/<bool>)
_CALL = re.compile(
    r'build_so_name\(\s*key,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*'
    r"(?:/\*\s*include_logits\s*=\s*\*/)?\s*(true|false)",
    re.DOTALL,
)


@pytest.fixture(scope="module")
def loader_src() -> str:
    assert _LOADER.is_file(), f"{_LOADER} is missing"
    return _LOADER.read_text()


def test_every_family_matches_a_cpp_call_site(loader_src):
    found = {
        (prefix, suffix, flag == "true")
        for prefix, suffix, flag in _CALL.findall(loader_src)
    }
    assert found, "no build_so_name call sites parsed; the regex or the C++ moved"

    declared = {(f.prefix, f.suffix, f.include_logits) for f in av.Family}
    assert declared == found, (
        "the variant table and aiter_loader.cc disagree on .so naming.\n"
        f"  python: {sorted(declared)}\n"
        f"  c++   : {sorted(found)}"
    )


@pytest.mark.parametrize(
    "segment",
    [
        '"fp16"',
        '"bf16"',
        '"_logits"',
        '"_nlogits"',
        '"_nbias"',
        '"_mask"',
        '"_nmask"',
        '"_lse"',
        '"_nlse"',
    ],
)
def test_the_segment_spellings_are_still_in_the_cpp(loader_src, segment):
    """so_name() reproduces build_so_name's segments by hand; if one is renamed
    upstream the Python spelling silently stops matching any real file."""
    assert segment in loader_src


def test_alibi_is_hardcoded_false_at_every_construction_site():
    """The table drops the alibi axis. That is only sound while every caller
    pins it, so check the callers rather than the loader."""
    root = Path(__file__).resolve().parents[2]
    sites = [
        root / "include/flashinfer/rocm/attention/aiter/single_prefill.cuh",
        root / "include/flashinfer/rocm/attention/aiter/batch_prefill.cuh",
    ]
    seen = 0
    for path in sites:
        for line in path.read_text().splitlines():
            if ".has_alibi" in line:
                seen += 1
                assert "false" in line, (
                    f"{path.name}: has_alibi is no longer false: {line.strip()}"
                )
    assert seen == 3, f"expected 3 has_alibi assignments, found {seen}"


def test_the_reachable_set_is_the_expected_shape():
    variants = av.reachable_variants()
    assert len(variants) == len(set(variants)), "duplicate variant keys"
    assert len(variants) == 40

    per_family = {f: sum(1 for v in variants if v.family is f) for f in av.Family}
    assert per_family == {
        av.Family.MHA_FWD: 8,
        av.Family.MHA_VARLEN_FWD: 16,
        av.Family.MHA_BATCH_PREFILL: 16,
    }


def test_mha_fwd_has_no_logits_arm():
    """get_aiter_mha_fwd_handle raises on has_logits_cap; enumerating that arm
    would put files in the store that nothing can ever ask for."""
    assert not any(
        v.has_logits_cap
        for v in av.reachable_variants()
        if v.family is av.Family.MHA_FWD
    )


def test_so_names_are_unique_and_well_formed():
    names = [av.so_name(v) for v in av.reachable_variants()]
    assert len(set(names)) == len(names)
    for name in names:
        assert name.endswith(".so")
        assert "_nbias" in name, "alibi is unreachable, so every name carries _nbias"


def test_a_known_name_is_reproduced_exactly():
    """One spelled-out case, so a refactor of so_name cannot pass by agreeing
    with itself. This file ships in the 0.1.20 wheel."""
    key = av.VariantKey(
        av.Family.MHA_VARLEN_FWD,
        dtype="bf16",
        has_logits_cap=False,
        needs_mask=True,
        has_lse=False,
    )
    assert (
        av.so_name(key)
        == "mha_varlen_fwd_bf16_nlogits_nbias_mask_nlse_ndropout_nskip_nqscale.so"
    )


def test_builds_cover_every_variant_exactly_once():
    produced = [v for b in av.builds() for v in b.produces]
    assert sorted(av.so_name(v) for v in produced) == sorted(
        av.so_name(v) for v in av.reachable_variants()
    )
    assert len(produced) == len(set(produced)), "two builds claim the same variant"


def test_only_varlen_emits_both_lse_arms():
    """The two varlen bootstraps loop return_lse internally; the other two take
    has_lse as a parameter. Scheduling per-file would run those builds twice."""
    both = {b.family for b in av.builds() if b.emits_both_lse}
    assert both == {av.Family.MHA_VARLEN_FWD}
    assert len(av.builds()) == 32


class TestPrebuildDriver:
    """The driver maps a BuildSpec to one of prefill.py's bootstraps.

    No GPU and no build: what is worth guarding is the mapping being total and
    the family filter rejecting typos, not AITER's compiler.
    """

    def test_every_family_has_a_bootstrap(self):
        """_run_build dispatches on Family and asserts on anything unhandled.
        A new family must fail loudly rather than silently build nothing."""
        import inspect

        from flashinfer.rocm import prebuild_aiter_variants as drv

        source = inspect.getsource(drv._run_build)
        for family in av.Family:
            assert f"Family.{family.name}" in source, (
                f"{family.name} has no arm in _run_build"
            )

    def test_the_family_filter_rejects_a_typo(self):
        from flashinfer.rocm import prebuild_aiter_variants as drv

        with pytest.raises(SystemExit, match="unknown family"):
            drv._select("mha_fwdd")

    def test_the_family_filter_selects_one_family(self):
        from flashinfer.rocm import prebuild_aiter_variants as drv

        selected = drv._select("mha_fwd")
        assert selected
        assert {s.family for s in selected} == {av.Family.MHA_FWD}

    def test_no_filter_selects_everything(self):
        from flashinfer.rocm import prebuild_aiter_variants as drv

        assert len(drv._select(None)) == len(av.builds())

    def test_the_store_tag_carries_arch_aiter_and_rocm(self):
        """All three are in the directory name, so a bump makes the lookup miss
        into a rebuild rather than load a mismatched artifact."""
        tag = av.variant_store_dir("gfx942").name
        assert tag.startswith("gfx942__")
        assert "__aiter-" in tag
        assert "__rocm-" in tag
