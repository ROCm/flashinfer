# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The variant table must agree with the C++ that ``dlopen``s the files.

``csrc/rocm/aiter_loader.cc`` builds the ``.so`` name, and
``flashinfer/jit/rocm/aiter_variants.py`` builds the same name in order to
prebuild it. Nothing links the two, so a prefix or suffix edited on one side
produces a store full of files the loader never asks for -- and the symptom is
not a crash but a silent return to the 74-360s first-call build.

These read the C++ source instead of trusting the Python copy rather than
introspecting a built module, so they catch drift without a GPU. They do
still import torch, transitively through ``flashinfer.jit``.
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


def _isolate_variant_env(monkeypatch):
    """Make FLASHINFER_AITER_VARIANT_DIR restorable even when it starts unset.

    monkeypatch.delenv(raising=False) records nothing to undo for an absent
    variable, so a test that then *creates* it leaks a now-deleted tmp path into
    every later test in the worker -- which find_variant searches and the C++
    loader would add as a candidate directory.
    """
    monkeypatch.setenv("FLASHINFER_AITER_VARIANT_DIR", "")
    monkeypatch.delenv("FLASHINFER_AITER_VARIANT_DIR")


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
        _isolate_variant_env(monkeypatch)
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
        _isolate_variant_env(monkeypatch)
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
    aiter_at = loader_src.index('env_dir("AITER_JIT_DIR")')
    variant_at = loader_src.index('env_dir("FLASHINFER_AITER_VARIANT_DIR")')
    assert aiter_at < variant_at


def test_an_operator_aiter_jit_dir_replaces_the_baked_default(loader_src):
    """Appending the baked path as a fallback would let a custom AITER build
    silently fall through to the pinned install -- the mangled symbol still
    resolves, so the kernel would come from a different build with no error."""
    assert "if (!aiter_dir) {" in loader_src


def test_an_empty_env_var_is_not_a_directory(loader_src):
    """`export FLASHINFER_AITER_VARIANT_DIR=` is a common way to clear one;
    taking it would dlopen "/<name>.so" and bury the real diagnostic."""
    assert "(value && *value) ? value : nullptr" in loader_src


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
        assert set(found) == {stale}
        assert current.is_dir() and stale.is_dir() and other.is_dir()
        assert "--prune --yes" in capsys.readouterr().out

    def test_apply_removes_only_this_architectures_stale_stores(
        self, tmp_path, monkeypatch
    ):
        """FLASHINFER_CACHE_DIR is shared and both arch stores are expected to
        exist, so a gfx942 run must not delete the gfx950 one."""
        from flashinfer.rocm import prebuild_aiter_variants as drv

        current, stale, other = self._roots(tmp_path, monkeypatch)
        drv.prune(apply=True)
        assert current.is_dir(), "the live store must survive"
        assert not stale.exists()
        assert other.is_dir(), "another architecture's store is not ours to delete"

    def test_a_missing_root_is_not_an_error(self, tmp_path, monkeypatch):
        from flashinfer.rocm import prebuild_aiter_variants as drv

        monkeypatch.setattr(
            drv, "variant_store_dir", lambda arch=None: tmp_path / "gone" / "tag"
        )
        assert drv.prune() == []


class TestReviewRegressions:
    """Cases found by review of the first cut of the store, each a real defect."""

    def test_batch_prefill_bootstrap_is_not_short_circuited(self):
        """It doubles as _aiter_native_paging_available's page-size probe, and
        the variant filename has no page-size axis -- so a store built at one
        page size would answer for every other, turning a warned flat-gather
        fallback into "no matching kernel found" inside run()."""
        import inspect

        from flashinfer.rocm import prefill

        src = inspect.getsource(prefill._aiter_bootstrap_batch_prefill)
        assert "_variants.prebuilt(" not in src

    def test_the_other_three_bootstraps_are_short_circuited(self):
        """Guard the guard: if these lost their skip, the store would silently
        stop being a latency win and every test here would still pass."""
        import inspect

        from flashinfer.rocm import prefill

        for fn in (
            prefill._aiter_bootstrap_single_prefill_varlen,
            prefill._aiter_bootstrap_single_prefill_mha_fwd,
            prefill._aiter_bootstrap_batch_ragged_prefill,
        ):
            assert "_variants.prebuilt(" in inspect.getsource(fn), fn.__name__

    def test_symbol_visible_false_clears_an_ambient_flag(self, tmp_path, monkeypatch):
        """Only skipping the set would let an operator's AITER_SYMBOL_VISIBLE=1
        compile a dlopen'd variant with the linkable flags -- same filename,
        different build."""
        import os

        from flashinfer.jit.rocm import aiter_source

        monkeypatch.setenv("AITER_SYMBOL_VISIBLE", "1")
        monkeypatch.setattr(aiter_source, "resolve_aiter_build_arch", lambda: "gfx942")
        with aiter_source._aiter_env_scope(tmp_path, symbol_visible=False):
            assert "AITER_SYMBOL_VISIBLE" not in os.environ
        assert os.environ["AITER_SYMBOL_VISIBLE"] == "1"

    def test_find_variant_survives_an_unusable_store_tag(self, monkeypatch):
        """find_variant is now on the plan() path; letting variant_store_dir's
        ValueError escape would demote AITER for the whole process."""

        def boom(arch=None):
            raise ValueError("refusing to build a cache directory name")

        monkeypatch.setattr(av, "variant_store_dir", boom)
        _isolate_variant_env(monkeypatch)
        assert av.find_variant(av.reachable_variants()[0]) is None

    def test_a_mismatched_arch_is_refused(self, monkeypatch):
        """--arch only renames the store; GPU_ARCHS and the launching device
        come from elsewhere, so it would file this box's objects under another
        architecture's tag."""
        from flashinfer.rocm import prebuild_aiter_variants as drv

        monkeypatch.setattr(drv, "resolve_aiter_build_arch", lambda: "gfx942")
        with pytest.raises(SystemExit, match="does not match the resolved build arch"):
            drv.main(["--arch", "gfx950", "--list"])

    def test_a_matching_arch_is_allowed(self, monkeypatch, capsys):
        from flashinfer.rocm import prebuild_aiter_variants as drv

        monkeypatch.setattr(drv, "resolve_aiter_build_arch", lambda: "gfx942")
        assert drv.main(["--arch", "gfx942", "--list"]) == 0
        assert "40 variants from 32 builds" in capsys.readouterr().out

    def test_list_counts_only_the_selected_variants(self, capsys):
        from flashinfer.rocm import prebuild_aiter_variants as drv

        drv.main(["--only", "mha_fwd", "--list"])
        assert "8 variants from 8 builds" in capsys.readouterr().out

    def test_locate_prefers_the_installed_copy_over_the_staged_one(self, tmp_path):
        """AITER stages under <jit_dir>/build/<md_name>/ before installing; the
        first hit in path order is the staged copy."""
        from flashinfer.rocm import prebuild_aiter_variants as drv

        key = av.reachable_variants()[0]
        name = av.so_name(key)
        staged = tmp_path / "build" / "mod"
        staged.mkdir(parents=True)
        (staged / name).write_bytes(b"staged")
        (tmp_path / name).write_bytes(b"installed")
        assert drv._locate([key], [tmp_path]) == [tmp_path / name]

    def test_locate_falls_back_to_the_shallowest_nested_hit(self, tmp_path):
        from flashinfer.rocm import prebuild_aiter_variants as drv

        key = av.reachable_variants()[0]
        name = av.so_name(key)
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / name).write_bytes(b"deep")
        shallow = tmp_path / "a"
        (shallow / name).write_bytes(b"shallow")
        assert drv._locate([key], [tmp_path]) == [shallow / name]


class TestWheelStore:
    """A store shipped inside the amd-flashinfer-jit-cache wheel."""

    def _fake_wheel(self, tmp_path, monkeypatch, *, tag, accessor=True):
        """Stand in for the jit-cache package, with or without the accessor."""
        import sys
        import types

        root = tmp_path / "wheel" / "aiter_variants"
        (root / tag).mkdir(parents=True)
        mod = types.ModuleType("amd_flashinfer_jit_cache")
        if accessor:
            mod.get_aiter_variant_dir = lambda: str(root)
        monkeypatch.setitem(sys.modules, "amd_flashinfer_jit_cache", mod)
        _isolate_variant_env(monkeypatch)
        return root / tag

    def test_the_matching_tag_is_used(self, tmp_path, monkeypatch):
        tag = "gfx942__aiter-1__rocm-2"
        shipped = self._fake_wheel(tmp_path, monkeypatch, tag=tag)
        monkeypatch.setattr(av, "variant_store_dir", lambda arch=None: tmp_path / tag)
        assert av.store_override() == shipped

    def test_a_wheel_for_another_tag_is_ignored(self, tmp_path, monkeypatch):
        """A multi-arch wheel carries several tags; one that is not ours must
        not be loaded, since the tag is the only compatibility check."""
        self._fake_wheel(tmp_path, monkeypatch, tag="gfx950__aiter-1__rocm-2")
        monkeypatch.setattr(
            av,
            "variant_store_dir",
            lambda arch=None: tmp_path / "gfx942__aiter-1__rocm-2",
        )
        assert av.store_override() is None

    def test_an_older_wheel_without_the_accessor_degrades(self, tmp_path, monkeypatch):
        """FLASHINFER_DISABLE_VERSION_CHECK can get an older wheel past the
        version gate, so a bare attribute access would be an import-time crash."""
        self._fake_wheel(
            tmp_path, monkeypatch, tag="gfx942__aiter-1__rocm-2", accessor=False
        )
        monkeypatch.setattr(
            av,
            "variant_store_dir",
            lambda arch=None: tmp_path / "gfx942__aiter-1__rocm-2",
        )
        assert av.store_override() is None

    def test_no_wheel_at_all_is_fine(self, tmp_path, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "amd_flashinfer_jit_cache", None)
        _isolate_variant_env(monkeypatch)
        monkeypatch.setattr(av, "variant_store_dir", lambda arch=None: tmp_path / "x")
        assert av.store_override() is None

    def test_the_env_var_still_wins(self, tmp_path, monkeypatch):
        tag = "gfx942__aiter-1__rocm-2"
        self._fake_wheel(tmp_path, monkeypatch, tag=tag)
        monkeypatch.setattr(av, "variant_store_dir", lambda arch=None: tmp_path / tag)
        monkeypatch.setenv("FLASHINFER_AITER_VARIANT_DIR", "/operator/choice")
        assert av.store_override() == Path("/operator/choice")

    def test_a_wheel_store_is_exported_to_the_cpp(self, tmp_path, monkeypatch):
        """The C++ reads only the env var, so a wheel-shipped store that is
        merely *resolved* would never be loaded. Returning early on any override
        was exactly that bug."""
        import os

        tag = "gfx942__aiter-1__rocm-2"
        shipped = self._fake_wheel(tmp_path, monkeypatch, tag=tag)
        monkeypatch.setattr(av, "variant_store_dir", lambda arch=None: tmp_path / tag)
        assert av.export_variant_store() == shipped
        assert os.environ["FLASHINFER_AITER_VARIANT_DIR"] == str(shipped)

    def test_an_operator_value_needs_no_export(self, tmp_path, monkeypatch):
        import os

        monkeypatch.setenv("FLASHINFER_AITER_VARIANT_DIR", "/operator/choice")
        assert av.export_variant_store() == Path("/operator/choice")
        assert os.environ["FLASHINFER_AITER_VARIANT_DIR"] == "/operator/choice"


class TestPackagingTheStore:
    def test_a_store_is_copied_under_its_own_tag(self, tmp_path, monkeypatch):
        from flashinfer.rocm import aot as aot_hip

        store = tmp_path / "cache" / "gfx942__aiter-1__rocm-2"
        store.mkdir(parents=True)
        (store / "mha_fwd_bf16_nbias_mask_nlse_ndropout_nqscale.so").write_bytes(
            b"\x7fELF"
        )
        monkeypatch.setattr(
            "flashinfer.jit.rocm.aiter_variants.variant_store_dir",
            lambda arch=None: store,
        )
        out = tmp_path / "out"
        out.mkdir()
        aot_hip._copy_aiter_variant_store(out)
        assert (
            out
            / "aiter_variants"
            / store.name
            / "mha_fwd_bf16_nbias_mask_nlse_ndropout_nqscale.so"
        ).is_file()

    def test_no_store_packages_nothing(self, tmp_path, monkeypatch):
        """A wheel built without running the prebuild must be exactly as before."""
        from flashinfer.rocm import aot as aot_hip

        monkeypatch.setattr(
            "flashinfer.jit.rocm.aiter_variants.variant_store_dir",
            lambda arch=None: tmp_path / "never",
        )
        out = tmp_path / "out"
        out.mkdir()
        aot_hip._copy_aiter_variant_store(out)
        assert not (out / "aiter_variants").exists()

    def test_an_empty_store_packages_nothing(self, tmp_path, monkeypatch):
        from flashinfer.rocm import aot as aot_hip

        store = tmp_path / "cache" / "gfx942__aiter-1__rocm-2"
        store.mkdir(parents=True)
        (store / "variants_manifest.json").write_text("{}")
        monkeypatch.setattr(
            "flashinfer.jit.rocm.aiter_variants.variant_store_dir",
            lambda arch=None: store,
        )
        out = tmp_path / "out"
        out.mkdir()
        aot_hip._copy_aiter_variant_store(out)
        assert not (out / "aiter_variants").exists()


class TestDriverStoreBypass:
    """The driver must build even when a *foreign* store would satisfy the
    bootstraps -- a jit-cache wheel or an operator FLASHINFER_AITER_VARIANT_DIR.
    Without the bypass every build returns immediately and then fails
    "AITER produced 0 of N"."""

    def test_lookup_is_disabled_while_the_driver_builds(self, tmp_path, monkeypatch):
        shipped = tmp_path / "shipped"
        shipped.mkdir()
        key = av.reachable_variants()[0]
        (shipped / av.so_name(key)).write_bytes(b"\x7fELF")
        monkeypatch.setenv("FLASHINFER_AITER_VARIANT_DIR", str(shipped))

        assert av.find_variant(key) is not None
        monkeypatch.setattr(av, "_SKIP_STORE_LOOKUP", True)
        assert av.find_variant(key) is None

    def test_the_flag_is_cleared_even_when_a_build_raises(self, tmp_path, monkeypatch):
        from flashinfer.rocm import prebuild_aiter_variants as drv

        monkeypatch.setattr(drv, "variant_store_dir", lambda arch=None: tmp_path)
        monkeypatch.setattr(
            drv,
            "_prebuild_specs",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        monkeypatch.setattr(
            "flashinfer.rocm.prefill._aiter_ops_importable", lambda: True
        )
        with pytest.raises(RuntimeError, match="boom"):
            drv.prebuild([])
        assert av._SKIP_STORE_LOOKUP is False


class TestLookupAndExportAgree:
    """find_variant deciding the bootstrap is unnecessary while the C++ is told
    to look somewhere else is a skip-then-fail-to-load."""

    def test_both_resolve_the_same_directory(self, tmp_path, monkeypatch):
        import os

        _isolate_variant_env(monkeypatch)
        store = tmp_path / "cache"
        store.mkdir()
        key = av.reachable_variants()[0]
        (store / av.so_name(key)).write_bytes(b"\x7fELF")
        monkeypatch.setattr(av, "variant_store_dir", lambda arch=None: store)
        monkeypatch.setattr(av, "_wheel_store", lambda: None)

        found = av.find_variant(key)
        exported = av.export_variant_store()
        assert found is not None
        assert found.parent == exported
        assert os.environ["FLASHINFER_AITER_VARIANT_DIR"] == str(exported)


class TestDriverExitCode:
    def test_all_failures_exit_non_zero(self, tmp_path, monkeypatch):
        """An image build that produced an empty store must not look like a
        success, or every consumer silently pays the per-shape compile."""
        from flashinfer.rocm import prebuild_aiter_variants as drv

        monkeypatch.setattr(drv, "variant_store_dir", lambda arch=None: tmp_path)
        monkeypatch.setattr(drv, "resolve_aiter_build_arch", lambda: "gfx942")
        monkeypatch.setattr(
            drv, "prebuild", lambda *a, **k: (0, 0, ["everything: RuntimeError: boom"])
        )
        assert drv.main([]) == 1

    def test_a_clean_run_exits_zero(self, tmp_path, monkeypatch):
        from flashinfer.rocm import prebuild_aiter_variants as drv

        monkeypatch.setattr(drv, "variant_store_dir", lambda arch=None: tmp_path)
        monkeypatch.setattr(drv, "resolve_aiter_build_arch", lambda: "gfx942")
        monkeypatch.setattr(drv, "prebuild", lambda *a, **k: (32, 0, []))
        assert drv.main([]) == 0


class TestPruneHonoursArch:
    def test_a_mismatched_arch_is_refused_before_pruning(self, tmp_path, monkeypatch):
        """--arch gfx950 --prune --yes from a gfx942 box would otherwise delete
        a colleague's store."""
        from flashinfer.rocm import prebuild_aiter_variants as drv

        monkeypatch.setattr(drv, "resolve_aiter_build_arch", lambda: "gfx942")
        called = []
        monkeypatch.setattr(drv, "prune", lambda **kw: called.append(kw))
        with pytest.raises(SystemExit, match="does not match the resolved build arch"):
            drv.main(["--arch", "gfx950", "--prune", "--yes"])
        assert not called

    def test_an_unusable_tag_is_a_message_not_a_traceback(self, monkeypatch):
        from flashinfer.rocm import prebuild_aiter_variants as drv

        def boom(arch=None):
            raise ValueError("refusing to build a cache directory name")

        monkeypatch.setattr(drv, "variant_store_dir", boom)
        with pytest.raises(SystemExit, match="refusing to build a cache directory"):
            drv.main(["--prune"])
