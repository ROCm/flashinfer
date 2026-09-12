# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Routing to AITER's asm prefill arm, and the numerics once it is reached.

The arm was unreachable before this test existed: the variant .so the loader
opens is built -DFAV2_ON=1 with no -DFAV3_ON, so `args.use_asm_v3` selected code
that was not in the binary. Nothing here is a regression test for that -- it is
first coverage.

`return_lse` is the case worth watching. The asm kernel writes LSE from a
different code path than CK Tile, and single_prefill_aiter.cu divides whatever
comes back by log(2) unconditionally, so a differing base would corrupt every
LSE silently rather than failing loudly.
"""

import logging
import pathlib
import subprocess
import sys

import pytest
import torch
from attention_reference import naive_attention

import flashinfer
from flashinfer.jit.core import logger
from flashinfer.rocm.aiter_utils import is_aiter_supported
from flashinfer.rocm.arch_caps import (
    _AITER_ASM_PREFILL_MIN_QO_LEN,
    _device_arch,
    aiter_asm_prefill_min_qo_len,
)
from flashinfer.rocm.prefill import _aiter_ops_importable

logger.setLevel(logging.ERROR)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CUH = _REPO_ROOT / "include/flashinfer/rocm/attention/aiter/single_prefill.cuh"

HEAD_DIM = 128
NUM_QO_HEADS = 32
NUM_KV_HEADS = 8


def _skip_unless_aiter(device: torch.device) -> None:
    if not is_aiter_supported(device) or not _aiter_ops_importable():
        pytest.skip("AITER requires a gfx942/gfx950 GPU and the aiter package")
    from flashinfer.rocm.arch_caps import capability_available, capability_reason

    if not capability_available(device, "single_prefill", "aiter"):
        pytest.skip(capability_reason(device, "single_prefill", "aiter"))


# --------------------------------------------------------------------------
# Policy. No GPU needed.
# --------------------------------------------------------------------------


def test_cuh_mirrors_arch_caps():
    """The C++ gate carries its own copy of the table; drift would be silent.

    A substring check rather than a parse of the C++ literal, matching
    test_aiter_version_gate.py's header check -- parsing would fail on
    reformatting rather than on a real divergence.
    """
    expected = " ".join(
        f"{arch}={'None' if v is None else v}"
        for arch, v in sorted(_AITER_ASM_PREFILL_MIN_QO_LEN.items())
    )
    text = _CUH.read_text()
    assert f"arch_caps: {expected}" in text, (
        f"single_prefill.cuh does not record 'arch_caps: {expected}'. "
        "Update the marker comment and AiterAsmPrefillMinQoLen together."
    )


@pytest.mark.parametrize(
    "arch,expected",
    [
        ("gfx942", None),
        ("gfx950", 2048),
        ("gfx950:sramecc+:xnack-", 2048),
        ("gfx90a", None),
        ("unknown", None),
    ],
)
def test_accessor(arch, expected):
    assert aiter_asm_prefill_min_qo_len(arch) == expected


def test_gfx942_never_routes_to_asm():
    """Not an oversight: asm wins and loses non-monotonically on CDNA3, so no
    threshold holds there. Pinning it stops a later edit reintroducing one
    without a fresh sweep."""
    assert _AITER_ASM_PREFILL_MIN_QO_LEN["gfx942"] is None


# --------------------------------------------------------------------------
# Numerics. Needs a GPU.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "qo_len,kv_len", [(512, 512), (2048, 2048), (4096, 4096), (512, 4096)]
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
def test_asm_gate_numerics(qo_len, kv_len, causal, return_lse):
    """Both sides of the threshold against an fp32 reference.

    Shapes are chosen to straddle it: 512 is always CK Tile, 2048 and 4096 take
    asm on gfx950, and (512, 4096) checks the gate reads qo_len rather than
    kv_len -- that shape measured 0.73x, so routing it to asm would be a
    regression.
    """
    device = torch.device("cuda:0")
    _skip_unless_aiter(device)
    if causal and qo_len > kv_len:
        pytest.skip("causal requires qo_len <= kv_len")

    torch.manual_seed(7)
    q = torch.randn(qo_len, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k = torch.randn(kv_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v = torch.randn(kv_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)

    ref_o, ref_lse = naive_attention(q, k, v, causal=causal, return_lse=return_lse)
    res = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=causal, backend="aiter", return_lse=return_lse
    )
    out, lse = res if return_lse else (res, None)

    torch.testing.assert_close(out.float(), ref_o.float(), rtol=2e-2, atol=2e-2)
    if return_lse:
        assert torch.isfinite(lse).all(), "asm arm returned a non-finite LSE"
        torch.testing.assert_close(lse.float(), ref_lse.float(), rtol=5e-2, atol=5e-2)


_KILL_SWITCH_PROBE = """
import torch, flashinfer
d = torch.device("cuda:0")
torch.manual_seed(7)
q = torch.randn(2048, 32, 128, dtype=torch.bfloat16, device=d)
k = torch.randn(2048, 8, 128, dtype=torch.bfloat16, device=d)
v = torch.randn(2048, 8, 128, dtype=torch.bfloat16, device=d)
o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, backend="aiter")
print(float(o.float().sum()))
"""


def test_kill_switch_pins_ck_tile():
    """FLASHINFER_AITER_ASM_PREFILL=0 has to actually change which kernel runs.

    Needs subprocesses: the switch is read once into a function-local static on
    the C++ side, so it cannot be toggled within a process. On gfx942 the gate
    never fires and both runs are the same kernel, which is itself the assertion
    that the switch is safe to leave set.
    """
    device = torch.device("cuda:0")
    _skip_unless_aiter(device)

    def run(env_value):
        import os

        env = dict(os.environ)
        if env_value is None:
            env.pop("FLASHINFER_AITER_ASM_PREFILL", None)
        else:
            env["FLASHINFER_AITER_ASM_PREFILL"] = env_value
        proc = subprocess.run(
            [sys.executable, "-c", _KILL_SWITCH_PROBE],
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,
        )
        assert proc.returncode == 0, f"probe failed: {proc.stderr[-2000:]}"
        return float(proc.stdout.strip().splitlines()[-1])

    on, off = run(None), run("0")
    # bf16 accumulation order differs between the two kernels, so this is a
    # "same answer", not a "same bits", comparison.
    assert abs(on - off) <= 1e-2 * max(1.0, abs(off)), (
        f"asm and CK Tile disagree beyond tolerance: {on} vs {off}"
    )

    if _device_arch(device) == "gfx942":
        assert on == off, "gfx942 must never route to asm, so both runs are one kernel"
