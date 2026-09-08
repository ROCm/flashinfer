# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Every op that names a kernel source must have that source on ROCm.

A module whose ``gen_*`` JIT spec lists a file absent from ``csrc/rocm`` imports
fine and dies in ninja on first call -- the failure upstream syncs keep
introducing, and the one ``CUDA_ONLY_MODULES`` exists to convert into a
statement about the backend. ``test_comm_import_gate.py`` asserts every gated
module raises; this asserts the complement, so a newly vendored op cannot land
unclassified.

Pure AST, no torch and no GPU: it has to run in the hardware-less lane, which
installs neither, and under ``--noconftest``.
"""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PKG = _REPO_ROOT / "flashinfer"
# The JIT resolves FLASHINFER_CSRC_DIR here on ROCm (get_include_paths.py).
_CSRC = _REPO_ROOT / "csrc" / "rocm"

# Ops with no ROCm kernel that are deliberately left importable, because
# gating them would break a module that does work: topk_varlen imports topk at
# module scope, sampling reaches it for the top-k-first path, and
# tests/trace/template_registry.py registers four of the five. Their ninja
# error already names the missing file, which is a fair report for an op
# nobody has ported.
_UNPORTED = {
    "flashinfer.concat_ops": "no concat_mla kernel",
    "flashinfer.mhc": "no mhc kernel",
    "flashinfer.nvfp4_attention_sm120": "SM120 NVFP4 attention",
    "flashinfer.topk": "no topk kernel",
    "flashinfer.xqa": "SM90+ XQA kernels",
    # Two logging side paths, both lazily built and neither reached by any
    # in-tree caller: set_log_level() in utils, and the opt-in GPU stats
    # counter in api_logging. Porting them is a separate change.
    "flashinfer.utils": "spdlog logging.cc",
    "flashinfer.api_logging": "api_log_stats.cu",
    # comm submodules that are not themselves gated but whose only JIT spec is
    # the CUDA comm module; importing one already fails transitively through a
    # gated dependency.
    "flashinfer.comm.dcp_alltoall": "CUDA comm kernels",
    "flashinfer.comm.trtllm_moe_alltoall": "CUDA comm kernels",
    "flashinfer.comm.ulysses": "CUDA comm kernels",
    # Mamba/SSM kernels arrived with v0.6.18 and are unported.
    "flashinfer.mamba.checkpointing_ssu": "Mamba SSM kernels",
    "flashinfer.mamba.ssd_combined": "Mamba SSM kernels",
    # TensorRT-LLM host utilities (nv_internal).
    "flashinfer.tllm_utils": "nv_internal TensorRT-LLM sources",
}


def _load_registry():
    """flashinfer/rocm/__init__.py without importing flashinfer (needs torch)."""
    name = "_fi_rocm_registry"
    spec = importlib.util.spec_from_file_location(name, _PKG / "rocm" / "__init__.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_registry = _load_registry()


def _dotted(path: Path) -> str:
    parts = list(path.relative_to(_REPO_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve(module: str | None, level: int, package: str) -> str:
    """Absolute name for a relative import, as importlib would resolve it."""
    base = package.rsplit(".", level - 1)[0] if level > 1 else package
    return f"{base}.{module}" if module else base


def _csrc_names(jit_file: Path) -> set[str]:
    """Every "..." in a `FLASHINFER_CSRC_DIR / "..."` expression."""
    names = set()
    for node in ast.walk(ast.parse(jit_file.read_text())):
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and isinstance(node.right, ast.Constant)
            and isinstance(node.right.value, str)
            and ast.unparse(node.left).endswith("FLASHINFER_CSRC_DIR")
        ):
            names.add(node.right.value)
    return names


def _exempt(dotted: str) -> bool:
    """True when the module, or a package above it, is gated or shadowed."""
    parts = dotted.split(".")
    prefixes = {".".join(parts[: i + 1]) for i in range(len(parts))}
    return bool(
        prefixes & set(_registry.CUDA_ONLY_MODULES)
        or prefixes & set(_registry.SHADOW_MODULES)
    )


def _gen_importers():
    """(importing module, jit file) for each `import gen_*` an op makes.

    Function-scope imports count -- concat_ops defers its generator, and a
    deferred import fails just as hard on the first call. Generators importing
    each other inside flashinfer.jit are skipped: they describe kernels rather
    than offering an op.
    """
    for path in sorted(_PKG.rglob("*.py")):
        dotted = _dotted(path)
        if dotted == "flashinfer.jit" or dotted.startswith("flashinfer.jit."):
            continue
        package = dotted if path.name == "__init__.py" else dotted.rsplit(".", 1)[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            if not any(alias.name.startswith("gen_") for alias in node.names):
                continue
            target = _resolve(node.module, node.level, package)
            if not target.startswith("flashinfer.jit."):
                continue
            jit_file = _REPO_ROOT / (target.replace(".", "/") + ".py")
            if jit_file.exists():
                yield dotted, jit_file


_CASES = sorted({(dotted, str(jit)) for dotted, jit in _gen_importers()})


@pytest.mark.parametrize("dotted,jit_file", _CASES, ids=[c[0] for c in _CASES])
def test_kernel_sources_present_or_module_classified(dotted, jit_file):
    if _exempt(dotted) or dotted in _UNPORTED:
        pytest.skip(f"{dotted} is gated, shadowed, or a known unported op")
    missing = sorted(n for n in _csrc_names(Path(jit_file)) if not (_CSRC / n).exists())
    assert not missing, (
        f"{dotted} builds kernels that do not exist under csrc/rocm: "
        f"{', '.join(missing)}. Port them, add the module to CUDA_ONLY_MODULES "
        f"in flashinfer/rocm/__init__.py, or record it in _UNPORTED here."
    )


def test_unported_allowlist_has_no_stale_entries():
    """An entry that now builds, or is gated, must leave the allowlist."""
    covered = {dotted for dotted, _ in _CASES}
    stale = sorted(name for name in _UNPORTED if name not in covered or _exempt(name))
    assert not stale, f"remove from _UNPORTED: {', '.join(stale)}"


def test_api_parity_covers_every_shadowed_module():
    """A shadow with no PAIRS entry is a twin whose signatures nothing audits."""
    pairs = ast.literal_eval(
        next(
            ast.unparse(node.value)
            for node in ast.parse(
                (_REPO_ROOT / "scripts" / "rocm_api_parity.py").read_text()
            ).body
            if isinstance(node, ast.AnnAssign)
            and getattr(node.target, "id", "") == "PAIRS"
        )
    )
    audited = {rocm_path for rocm_path, _ in pairs}
    missing = sorted(
        rocm
        for rocm in _registry.SHADOW_MODULES.values()
        if rocm.replace(".", "/") + ".py" not in audited
        and (rocm.replace(".", "/") + "/_core.py") not in audited
    )
    assert not missing, (
        f"shadowed but absent from scripts/rocm_api_parity.py PAIRS: "
        f"{', '.join(missing)}"
    )
