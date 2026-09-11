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


class TestStoreLookup:
    """find_variant / prebuilt / export_variant_store, and the C++ candidate order."""

    def _store(self, tmp_path, monkeypatch):
        monkeypatch.setattr(av, "variant_store_dir", lambda arch=None: tmp_path)
        monkeypatch.delenv("FLASHINFER_AITER_VARIANT_DIR", raising=False)
        return tmp_path

    def test_a_missing_variant_is_not_found(self, tmp_path, monkeypatch):
        self._store(tmp_path, monkeypatch)
        key = av.reachable_variants()[0]
        assert av.find_variant(key) is None

    def test_a_present_variant_is_found(self, tmp_path, monkeypatch):
        store = self._store(tmp_path, monkeypatch)
        key = av.reachable_variants()[0]
        (store / av.so_name(key)).write_bytes(b"\x7fELF")
        assert av.find_variant(key) == store / av.so_name(key)

    def test_the_override_is_searched_first(self, tmp_path, monkeypatch):
        """An image- or wheel-supplied store must win over a half-populated
        cache dir left by an earlier run."""
        cache, shipped = tmp_path / "cache", tmp_path / "shipped"
        cache.mkdir()
        shipped.mkdir()
        monkeypatch.setattr(av, "variant_store_dir", lambda arch=None: cache)
        monkeypatch.setenv("FLASHINFER_AITER_VARIANT_DIR", str(shipped))
        key = av.reachable_variants()[0]
        (cache / av.so_name(key)).write_bytes(b"\x7fELF")
        (shipped / av.so_name(key)).write_bytes(b"\x7fELF")
        assert av.find_variant(key) == shipped / av.so_name(key)

    def test_prebuilt_needs_both_lse_arms_for_varlen(self, tmp_path, monkeypatch):
        """The varlen bootstrap emits both arms, so one present file does not
        make the call a no-op."""
        torch = pytest.importorskip("torch")
        store = self._store(tmp_path, monkeypatch)
        kw = dict(has_logits_cap=True, needs_mask=True)
        one = av.VariantKey(av.Family.MHA_VARLEN_FWD, "bf16", True, True, False)
        (store / av.so_name(one)).write_bytes(b"\x7fELF")
        assert not av.prebuilt(
            av.Family.MHA_VARLEN_FWD, torch.bfloat16, has_lse=None, **kw
        )

        other = av.VariantKey(av.Family.MHA_VARLEN_FWD, "bf16", True, True, True)
        (store / av.so_name(other)).write_bytes(b"\x7fELF")
        assert av.prebuilt(av.Family.MHA_VARLEN_FWD, torch.bfloat16, has_lse=None, **kw)

    def test_prebuilt_is_false_for_an_unsupported_dtype(self, tmp_path, monkeypatch):
        torch = pytest.importorskip("torch")
        self._store(tmp_path, monkeypatch)
        assert not av.prebuilt(
            av.Family.MHA_FWD, torch.float32, needs_mask=True, has_lse=False
        )

    def test_export_skips_a_store_that_does_not_exist(self, tmp_path, monkeypatch):
        """Exporting a nonexistent directory would only lengthen the loader's
        candidate list and its failure message."""
        monkeypatch.setattr(
            av, "variant_store_dir", lambda arch=None: tmp_path / "nope"
        )
        monkeypatch.delenv("FLASHINFER_AITER_VARIANT_DIR", raising=False)
        assert av.export_variant_store() is None
        import os

        assert "FLASHINFER_AITER_VARIANT_DIR" not in os.environ

    def test_export_publishes_an_existing_store(self, tmp_path, monkeypatch):
        import os

        self._store(tmp_path, monkeypatch)
        assert av.export_variant_store() == tmp_path
        assert os.environ["FLASHINFER_AITER_VARIANT_DIR"] == str(tmp_path)


def test_the_cpp_tries_the_variant_dir_and_keeps_aiter_jit_dir_first(loader_src):
    """Order is load-bearing: an operator-set AITER_JIT_DIR must still win,
    because that is what the loader's own failure message tells them to set."""
    assert "FLASHINFER_AITER_VARIANT_DIR" in loader_src
    aiter_at = loader_src.index('getenv("AITER_JIT_DIR")')
    variant_at = loader_src.index('getenv("FLASHINFER_AITER_VARIANT_DIR")')
    assert aiter_at < variant_at


class TestPrune:
    """Dry-run by default, and it must never touch the live store."""

    def _roots(self, tmp_path, monkeypatch):
        from flashinfer.rocm import prebuild_aiter_variants as drv

        current = tmp_path / "gfx942__aiter-1__rocm-2"
        stale = tmp_path / "gfx942__aiter-0__rocm-2"
        other_arch = tmp_path / "gfx950__aiter-1__rocm-2"
        for d in (current, stale, other_arch):
            d.mkdir(parents=True)
            (d / "x.so").write_bytes(b"\x7fELF")
        # The driver imported variant_store_dir by name, so patching it on
        # aiter_variants would not reach the reference prune() actually uses.
        monkeypatch.setattr(drv, "variant_store_dir", lambda arch=None: current)
        return current, stale, other_arch

    def test_dry_run_deletes_nothing(self, tmp_path, monkeypatch, capsys):
        from flashinfer.rocm import prebuild_aiter_variants as drv

        current, stale, other = self._roots(tmp_path, monkeypatch)
        found = drv.prune()
        assert set(found) == {stale, other}
        assert current.is_dir() and stale.is_dir() and other.is_dir()
        assert "--prune --yes" in capsys.readouterr().out

    def test_apply_removes_only_the_stale_ones(self, tmp_path, monkeypatch):
        from flashinfer.rocm import prebuild_aiter_variants as drv

        current, stale, other = self._roots(tmp_path, monkeypatch)
        drv.prune(apply=True)
        assert current.is_dir(), "the live store must survive"
        assert not stale.exists()
        assert not other.exists()

    def test_a_missing_root_is_not_an_error(self, tmp_path, monkeypatch):
        from flashinfer.rocm import prebuild_aiter_variants as drv

        monkeypatch.setattr(
            drv, "variant_store_dir", lambda arch=None: tmp_path / "gone" / "tag"
        )
        assert drv.prune() == []
