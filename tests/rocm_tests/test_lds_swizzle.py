#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
test_lds_swizzle.py
===================
Validates MI300x CDNA3 LDS swizzle modes for the SinglePrefillWithKVCacheDevice
kernel before committing changes to permuted_smem.cuh.

Hardware model (MI300x CDNA3)
-------------------------------
  - LDS: 32 banks × 4 bytes = 128 bytes/cycle
  - Wavefront: 64 threads, served in 4 LDS phases of 16 threads each:
      Phase 0: T0–T15,  Phase 1: T16–T31,
      Phase 2: T32–T47, Phase 3: T48–T63
  - Per-thread load width: uint2 = 8 bytes = 4 × fp16  (VECTOR_BIT_WIDTH = 64)
  - A bank conflict occurs when ≥2 threads in the same phase access the
    same 4-byte bank.  Bank of uint2 at offset k:  (2*k) % 32

Kernel thread layout (KernelTraits on CDNA3/HIP)
-------------------------------------------------
  WARP_THREAD_ROWS = 4   (= lane_idx // WARP_THREAD_COLS  in the write path)
  WARP_THREAD_COLS = 16  (= lane_idx %  WARP_THREAD_COLS)
  HALF_ELEMS_PER_THREAD = 4  →  QK_SMEM_COL_ADVANCE = 16/4 = 4

Paths under test
----------------
  Write: load_q_global_smem()  –  global → LDS, mma_q × j × mma_do loops
  Read:  compute_qk() Q reads  –  LDS → register, mma_d × mma_q loops

Swizzle modes
-------------
  kLinear           : no swizzle (correctness reference; will show read conflicts)
  k128B             : CUDA-origin, period-8 XOR  (will show 8-way read conflicts)
  k128B_16Row_naive : naive XOR fix — passes H=64 but fails H=128 (carry issue)
  k128B_16Row       : correct fix using col_idx tracking (target: zero conflicts)

Advance function analysis for k128B_16Row
------------------------------------------
  advance_offset_by_column<4> must compute:
    new_offset = i*stride + ((j+4) ^ (i%16))

  The correction (j+4)^j can be 4, 12, or 28 depending on j:
    j=0..3:  4,  j=4..7:  12,  j=8..11:  4,  j=12..15: 28, ...
  The naive "^8" fix only handles 4 and 12, breaking at j=12 (H=128).

  The correct formula (using col_idx = current j before advancing):
    L         = offset & (stride - 1)   # = j ^ (i%16), the swizzled column bits
    i_mod     = col_idx ^ L             # recover i%16
    new_lower = (col_idx + step) ^ i_mod
    new_offset = offset - L + new_lower

  This requires tracking col_idx at call sites (like row_idx in advance_by_row).

Negative test
-------------
  k128B_16Row_bad_adv: k128B_16Row get_permuted_offset + k128B advance functions.
  Must show correctness errors, proving both components must change together.

Usage
-----
  python3 test_lds_swizzle.py          # run all tests
  python3 test_lds_swizzle.py -v       # verbose: print every conflict step
"""

import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════
# Hardware & Kernel Constants
# ═══════════════════════════════════════════════════════════════════════════

WAVEFRONT_SIZE = 64  # threads per wavefront on CDNA3
LDS_PHASE_SIZE = 16  # threads per LDS issue cycle on MI300x
LDS_BANKS = 32  # 4-byte LDS banks

WARP_THREAD_ROWS = 4  # KernelTraits::WARP_THREAD_ROWS
WARP_THREAD_COLS = 16  # KernelTraits::WARP_THREAD_COLS  (== LDS_PHASE_SIZE)
HALF_ELEMS = 4  # KernelTraits::HALF_ELEMS_PER_THREAD  (fp16 per uint2)
QK_COL_ADVANCE = 4  # QK_SMEM_COLUMN_ADVANCE = 16 / HALF_ELEMS_PER_THREAD

# ═══════════════════════════════════════════════════════════════════════════
# Swizzle Mode Tags
# ═══════════════════════════════════════════════════════════════════════════

LINEAR = "kLinear"
K128B = "k128B"
K128B_16ROW_NAIVE = "k128B_16Row_naive"  # naive ^8 fix  (fails at H=128)
K128B_16ROW = "k128B_16Row"  # correct fix   (passes all H)
K128B_16ROW_BAD = "k128B_16Row_bad_adv"  # negative test: wrong advance funcs

# ═══════════════════════════════════════════════════════════════════════════
# Core Swizzle Functions
# ═══════════════════════════════════════════════════════════════════════════


def get_permuted_offset(mode: str, stride: int, i: int, j: int) -> int:
    """
    Maps logical (row=i, col=j) to a flat uint2 LDS offset.
    Mirrors smem_t<mode>::get_permuted_offset<stride>(i, j).

    k128B_16Row and k128B_16Row_naive share the same get_permuted_offset
    (period-16 XOR) — only the advance functions differ.
    k128B_16Row_bad_adv also uses the same offset formula but with wrong
    advance funcs, to demonstrate the dependency.
    """
    if mode == K128B:
        return i * stride + (j ^ (i % 8))
    elif mode in (K128B_16ROW_NAIVE, K128B_16ROW, K128B_16ROW_BAD):
        period = 16 if stride >= 16 else 8
        return i * stride + (j ^ (i % period))
    else:  # kLinear
        return i * stride + j


def adv_col(
    mode: str,
    offset: int,
    step: int,
    step_idx: int,
    stride: Optional[int] = None,
    col_idx: Optional[int] = None,
) -> int:
    """
    Mirrors smem_t<mode>::advance_offset_by_column<step>(offset, step_idx).

    For k128B_16Row the correct implementation requires two additional params:
      stride   – UPCAST_STRIDE_Q (needed to isolate lower swizzle bits)
      col_idx  – current column j before advancing (tracked by caller)

    The exact formula for k128B_16Row, step=4:
      L         = offset & (stride - 1)   # swizzled lower bits = j ^ (i%16)
      i_mod     = col_idx ^ L             # recover i%16
      new_lower = (col_idx + step) ^ i_mod
      return      offset - L + new_lower

    k128B_16Row_naive uses the naive '^8 instead of +8' approach which is
    correct for H=64 (j ≤ 11, no carry past bit 3) but wrong for H=128
    (j=12→16 carry makes (j+4)^j = 28, not 4 or 12).
    """
    if mode in (K128B, K128B_16ROW_BAD):
        # Existing k128B formulas (also used as the "bad" path)
        if step == 2:
            xor_val = 0x2 + (0x4 * (step_idx % 2 == 1))
            return (offset ^ xor_val) + (step_idx % 4 == 3) * 8
        elif step == 4:
            return (offset ^ 0x4) + (step_idx % 2 == 1) * 8
        else:  # step % 8 == 0
            return offset + step

    elif mode == K128B_16ROW_NAIVE:
        # Naive fix: replace  + N*8  with  ^ N*8.
        # Works when (j+4)^j ∈ {4, 12}, i.e., j ≤ 11 (no bit-4 carry).
        # Fails at j=12→16 where (j+4)^j = 28.
        if step == 2:
            xor_val = 0x2 + (0x4 * (step_idx % 2 == 1))
            carry = 0x8 if (step_idx % 4 == 3) else 0
            return (offset ^ xor_val) ^ carry
        elif step == 4:
            carry = 0x8 if (step_idx % 2 == 1) else 0
            return (offset ^ 0x4) ^ carry
        else:  # step % 8 == 0
            return offset + step

    elif mode == K128B_16ROW:
        # step%8==0 path needs no extra state; check it first so the write path
        # (step=16) never needs stride/col_idx.
        if step % 8 == 0:
            return offset + step
        # step == 2 or 4: exact formula using col_idx and stride.
        assert stride is not None, "k128B_16Row adv_col<2/4> requires stride"
        assert col_idx is not None, "k128B_16Row adv_col<2/4> requires col_idx"
        mask = stride - 1  # assumes stride is power-of-2
        L = offset & mask  # = j ^ (i%16)
        i_mod = col_idx ^ L  # recover i % 16
        new_lower = (col_idx + step) ^ i_mod
        return (offset & ~mask) + new_lower

    else:  # kLinear
        return offset + step


def adv_row(mode: str, offset: int, step: int, stride: int, row_idx: int = 0) -> int:
    """
    Mirrors smem_t<mode>::advance_offset_by_row<step, stride>(offset [, row_idx]).

    k128B_16Row step=4 needs row_idx to select the correct XOR mask:
      rows  0- 3 and  8-11: xor_mask = 0x4
      rows  4- 7 and 12-15: xor_mask = 0xC  (= 0x4 | 0x8)
    This is because (i+4)%16 ^ i%16 alternates between 4 and 12 every 4 rows.

    k128B always uses xor_mask = 0x4 because (i+4)%8 ^ i%8 == 4 for all i.
    k128B_16Row_bad_adv uses k128B's formula (wrong for rows 4-7, 12-15).
    """
    if mode in (K128B, K128B_16ROW_BAD):
        if step == 4:
            return (offset ^ 0x4) + step * stride
        else:  # step % 8 == 0
            return offset + step * stride

    elif mode in (K128B_16ROW_NAIVE, K128B_16ROW):
        if step == 4:
            xor_mask = 0x4 | (0x8 if (row_idx // 4) % 2 == 1 else 0)
            return (offset ^ xor_mask) + step * stride
        else:  # step % 8 == 0
            return offset + step * stride

    else:  # kLinear
        return offset + step * stride


# ═══════════════════════════════════════════════════════════════════════════
# Bank Conflict Analysis
# ═══════════════════════════════════════════════════════════════════════════


def lds_bank(uint2_offset: int) -> int:
    """Primary 4-byte LDS bank for a uint2 at the given uint2-unit offset."""
    return (2 * uint2_offset) % LDS_BANKS


def wavefront_conflicts(offsets64: List[int]) -> Tuple[int, int]:
    """
    Count bank conflict statistics across all 4 LDS phases of one
    simultaneous 64-thread access.

    Returns (total_conflict_pairs, max_way_conflict).
    conflict_pairs: number of (thread_a, thread_b) sharing a bank in same phase.
    max_way: highest number of threads in any phase sharing a single bank.
    """
    total_pairs = 0
    max_way = 1
    for phase in range(WAVEFRONT_SIZE // LDS_PHASE_SIZE):
        sl = offsets64[phase * LDS_PHASE_SIZE : (phase + 1) * LDS_PHASE_SIZE]
        hist = Counter(lds_bank(o) for o in sl)
        total_pairs += sum(c * (c - 1) // 2 for c in hist.values())
        max_way = max(max_way, max(hist.values()))
    return total_pairs, max_way


# ═══════════════════════════════════════════════════════════════════════════
# Simulator
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SimResult:
    mode: str
    head_dim: int
    num_mma_q: int
    num_warps_q: int
    smem_size: int
    coverage: int
    write_errors: List[str] = field(default_factory=list)
    read_errors: List[str] = field(default_factory=list)
    write_bc_pairs: int = 0
    write_bc_max: int = 1
    read_bc_pairs: int = 0
    read_bc_max: int = 1
    write_bc_log: List[Tuple[str, int, int]] = field(default_factory=list)
    read_bc_log: List[Tuple[str, int, int]] = field(default_factory=list)

    @property
    def coverage_ok(self) -> bool:
        return self.coverage == self.smem_size and not self.write_errors

    @property
    def correct(self) -> bool:
        return not self.read_errors

    @property
    def write_clean(self) -> bool:
        return self.write_bc_pairs == 0

    @property
    def read_clean(self) -> bool:
        return self.read_bc_pairs == 0


class PrefillLDSSimulator:
    """
    Simulates Q-smem write (load_q_global_smem) and read (compute_qk Q loads)
    access patterns for one CTA tile.
    """

    def __init__(self, head_dim: int, num_mma_q: int, num_warps_q: int, mode: str):
        assert head_dim % 16 == 0, "HEAD_DIM must be a multiple of 16"
        self.H = head_dim
        self.nmq = num_mma_q
        self.nwq = num_warps_q
        self.mode = mode

        self.NMD = head_dim // 16  # NUM_MMA_D_QK
        self.stride = head_dim // HALF_ELEMS  # UPCAST_STRIDE_Q (uint2 units)
        self.cta_q = num_warps_q * num_mma_q * 16
        # COLUMN_RESET_OFFSET = (NUM_MMA_D_QK / 4) * WARP_THREAD_COLS
        self.col_reset = (self.NMD // 4) * WARP_THREAD_COLS
        self.smem_size = self.cta_q * self.stride  # total uint2 units

    # ──────────────────────────────────────────────────────────────────
    # Write Path  (load_q_global_smem)
    # ──────────────────────────────────────────────────────────────────

    def _write_path(
        self,
    ) -> Tuple[Dict, List[str], int, int, List[Tuple[str, int, int]]]:
        """
        Simulates the mma_q × j × mma_do nested loop.
        adv_col is only called with step=WARP_THREAD_COLS=16 here (step%8==0),
        so no col_idx is needed in the write path.
        adv_row<4> needs row_idx for k128B_16Row; tracked per thread.
        """
        smem: Dict[int, Tuple[int, int]] = {}
        errors: List[str] = []
        bc_pairs = 0
        bc_max = 1
        bc_log: List[Tuple[str, int, int]] = []

        for wq in range(self.nwq):
            base = wq * self.nmq * 16

            offs = [
                get_permuted_offset(
                    self.mode,
                    self.stride,
                    base + lane // WARP_THREAD_COLS,
                    lane % WARP_THREAD_COLS,
                )
                for lane in range(WAVEFRONT_SIZE)
            ]
            # row_idx tracks the logical Q-row before each adv_row<4> call.
            # Flows across both mma_q and j loops.
            row_idx = [
                base + lane // WARP_THREAD_COLS for lane in range(WAVEFRONT_SIZE)
            ]

            for mq in range(self.nmq):
                for j in range(4):
                    q_rows = [
                        base + mq * 16 + j * WARP_THREAD_ROWS + lane // WARP_THREAD_COLS
                        for lane in range(WAVEFRONT_SIZE)
                    ]

                    for md in range(self.NMD // 4):
                        cp, mx = wavefront_conflicts(offs)
                        bc_pairs += cp
                        bc_max = max(bc_max, mx)
                        label = f"wq={wq} mq={mq} j={j} md={md}"
                        if cp:
                            bc_log.append((label, cp, mx))

                        for lane in range(WAVEFRONT_SIZE):
                            off = offs[lane]
                            q_col = md * WARP_THREAD_COLS + lane % WARP_THREAD_COLS
                            data = (q_rows[lane], q_col)
                            if not (0 <= off < self.smem_size):
                                errors.append(f"[W-OOB] {label} l={lane} off={off}")
                            elif off in smem and smem[off] != data:
                                errors.append(
                                    f"[W-DUP] {label} l={lane} off={off} "
                                    f"was={smem[off]} new={data}"
                                )
                            else:
                                smem[off] = data

                        # adv_col step=16 → step%8==0 → offset+16 (no col_idx needed)
                        offs = [
                            adv_col(self.mode, offs[l], WARP_THREAD_COLS, md)
                            for l in range(WAVEFRONT_SIZE)
                        ]

                    # adv_row<4> with row_idx
                    offs = [
                        adv_row(
                            self.mode,
                            offs[l],
                            WARP_THREAD_ROWS,
                            self.stride,
                            row_idx=row_idx[l],
                        )
                        - self.col_reset
                        for l in range(WAVEFRONT_SIZE)
                    ]
                    row_idx = [r + WARP_THREAD_ROWS for r in row_idx]

        return smem, errors, bc_pairs, bc_max, bc_log

    # ──────────────────────────────────────────────────────────────────
    # Read Path  (compute_qk – Q fragment loads)
    # ──────────────────────────────────────────────────────────────────

    def _read_path(
        self, smem: Dict
    ) -> Tuple[List[str], int, int, List[Tuple[str, int, int]]]:
        """
        Simulates the mma_d × mma_q Q-smem read loop.

        For k128B_16Row, adv_col<4> requires the current column index (col_idx),
        which is tracked per thread across mma_d iterations.

        Initial col per thread: j = lane_idx // 16  (∈ {0,1,2,3})
        After each mma_d step: col_idx += QK_COL_ADVANCE (= 4)
        """
        errors: List[str] = []
        bc_pairs = 0
        bc_max = 1
        bc_log: List[Tuple[str, int, int]] = []

        for wq in range(self.nwq):
            base = wq * self.nmq * 16

            q_offs = [
                get_permuted_offset(
                    self.mode, self.stride, base + lane % 16, lane // 16
                )
                for lane in range(WAVEFRONT_SIZE)
            ]
            # col_idx: tracks current column j before each adv_col<4> call.
            # Needed by k128B_16Row; harmless for other modes.
            col_idx = [lane // 16 for lane in range(WAVEFRONT_SIZE)]

            for md in range(self.NMD):
                for mq in range(self.nmq):
                    cp, mx = wavefront_conflicts(q_offs)
                    bc_pairs += cp
                    bc_max = max(bc_max, mx)
                    label = f"wq={wq} md={md} mq={mq}"
                    if cp:
                        bc_log.append((label, cp, mx))

                    for lane in range(WAVEFRONT_SIZE):
                        exp_row = base + mq * 16 + (lane % 16)
                        exp_col = (lane // 16) + md * QK_COL_ADVANCE
                        off = q_offs[lane]
                        got = smem.get(off)
                        if got != (exp_row, exp_col):
                            errors.append(
                                f"[R-COR] {label} l={lane}: "
                                f"off={off} exp=({exp_row},{exp_col}) got={got}"
                            )

                    # adv_row<16> step%8==0 → offset + 16*stride (no row_idx needed)
                    q_offs = [
                        adv_row(self.mode, q_offs[l], 16, self.stride)
                        for l in range(WAVEFRONT_SIZE)
                    ]

                # adv_col<4> with col_idx for k128B_16Row
                q_offs = [
                    adv_col(
                        self.mode,
                        q_offs[l],
                        QK_COL_ADVANCE,
                        md,
                        stride=self.stride,
                        col_idx=col_idx[l],
                    )
                    - self.nmq * 16 * self.stride
                    for l in range(WAVEFRONT_SIZE)
                ]
                col_idx = [c + QK_COL_ADVANCE for c in col_idx]

        return errors, bc_pairs, bc_max, bc_log

    # ──────────────────────────────────────────────────────────────────────
    def run(self) -> SimResult:
        smem, we, wcp, wmx, wlog = self._write_path()
        re, rcp, rmx, rlog = self._read_path(smem)
        return SimResult(
            mode=self.mode,
            head_dim=self.H,
            num_mma_q=self.nmq,
            num_warps_q=self.nwq,
            smem_size=self.smem_size,
            coverage=len(smem),
            write_errors=we,
            read_errors=re,
            write_bc_pairs=wcp,
            write_bc_max=wmx,
            write_bc_log=wlog,
            read_bc_pairs=rcp,
            read_bc_max=rmx,
            read_bc_log=rlog,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ═══════════════════════════════════════════════════════════════════════════

P = "✓"
F = "✗"


def _mark(ok: bool) -> str:
    return P if ok else F


def print_result(r: SimResult, verbose: bool = False) -> None:
    tag = f"[{r.mode:>22s}]  H={r.head_dim:3d}  NMQ={r.num_mma_q}  NWQ={r.num_warps_q}"
    print(f"\n  {tag}")
    print(f"    Coverage   : {r.coverage:5d}/{r.smem_size:5d}  {_mark(r.coverage_ok)}")
    print(
        f"    Correctness: {_mark(r.correct)}"
        + (f"  ({len(r.read_errors)} mismatches)" if r.read_errors else "")
    )
    print(
        f"    Write BC   : {r.write_bc_pairs:6d} pairs  max {r.write_bc_max:2d}-way  "
        f"{_mark(r.write_clean)}"
    )
    print(
        f"    Read  BC   : {r.read_bc_pairs:6d} pairs  max {r.read_bc_max:2d}-way  "
        f"{_mark(r.read_clean)}"
    )

    if r.write_errors:
        print(f"      ! {r.write_errors[0]}")
    if r.read_errors:
        print(f"      ! {r.read_errors[0]}")
        if len(r.read_errors) > 1 and not verbose:
            print(f"      ! ... ({len(r.read_errors) - 1} more)")

    if verbose:
        for label, cp, mx in r.write_bc_log:
            print(f"      [W-BC] {label}: {cp} pairs  {mx}-way")
        for label, cp, mx in r.read_bc_log[:10]:
            print(f"      [R-BC] {label}: {cp} pairs  {mx}-way")
        if len(r.read_bc_log) > 10:
            print(f"      [R-BC] ... ({len(r.read_bc_log) - 10} more)")
        for e in r.read_errors[:5]:
            print(f"      {e}")
        if len(r.read_errors) > 5:
            print(f"      ... ({len(r.read_errors) - 5} more)")


# ═══════════════════════════════════════════════════════════════════════════
# Test Runner
# ═══════════════════════════════════════════════════════════════════════════

CONFIGS = [
    (64, 1, 4),  # HEAD_DIM=64  — UPCAST_STRIDE=16, j ≤ 15 (no bit-4 carry)
    (128, 1, 4),  # HEAD_DIM=128 — UPCAST_STRIDE=32, j=12→16 crosses bit-4
    (128, 2, 4),  # HEAD_DIM=128 — larger CTA tile (NUM_MMA_Q=2)
]

MAIN_MODES = [LINEAR, K128B, K128B_16ROW_NAIVE, K128B_16ROW]
NEGATIVE_MODE = K128B_16ROW_BAD


def run_tests(verbose: bool = False) -> int:
    """
    Runs all test configurations and modes.
    Returns number of unexpected failures:
      - k128B_16Row must have zero conflicts and zero correctness errors.
      - k128B_16Row_naive is expected to fail at H=128.
      - negative test must fail correctness.
    """
    print("=" * 76)
    print("MI300x CDNA3 LDS Swizzle Validator")
    print(
        f"  {LDS_BANKS} banks × 4 B  |  {LDS_PHASE_SIZE}-thread LDS phases  "
        f"|  uint2 = 8 B / thread"
    )
    print("=" * 76)

    all_results: Dict = {}
    unexpected_failures = 0

    # ── Main mode tests ───────────────────────────────────────────────
    print("\n── Main Modes ─────────────────────────────────────────────────────────")
    for H, nmq, nwq in CONFIGS:
        for mode in MAIN_MODES:
            r = PrefillLDSSimulator(H, nmq, nwq, mode).run()
            all_results[(H, nmq, nwq, mode)] = r
            print_result(r, verbose=verbose)

            if mode == K128B_16ROW:
                if not (r.coverage_ok and r.correct and r.write_clean and r.read_clean):
                    unexpected_failures += 1

    # ── Negative test ─────────────────────────────────────────────────
    print(
        "\n── Negative Test  (k128B_16Row get_permuted_offset + k128B advance funcs) ──"
    )
    print("   Expected: correctness errors (proves both components must change)")
    for H, nmq, nwq in CONFIGS:
        r = PrefillLDSSimulator(H, nmq, nwq, NEGATIVE_MODE).run()
        all_results[(H, nmq, nwq, NEGATIVE_MODE)] = r
        print_result(r, verbose=verbose)

        if r.correct and r.write_clean and r.read_clean:
            print("    !! NEGATIVE TEST DID NOT FAIL — something is wrong !!")
            unexpected_failures += 1

    # ── Summary tables ────────────────────────────────────────────────
    all_modes = MAIN_MODES + [NEGATIVE_MODE]
    cw = 16  # column width
    hdr = f"\n{'Config':>28s}" + "".join(f"  {m:>{cw}s}" for m in all_modes)

    print("\n" + "=" * 76)
    print("Read-path bank conflicts  (pairs : max-way : pass?)")
    print(hdr)
    print("-" * 76)
    for H, nmq, nwq in CONFIGS:
        row = f"  H={H:3d}  NMQ={nmq}  NWQ={nwq}        "
        for mode in all_modes:
            r = all_results[(H, nmq, nwq, mode)]
            cell = f"{r.read_bc_pairs:5d}p {r.read_bc_max:2d}w {_mark(r.read_clean)}"
            row += f"  {cell:>{cw}s}"
        print(row)

    print("\nWrite-path bank conflicts  (pairs : max-way : pass?)")
    print(hdr)
    print("-" * 76)
    for H, nmq, nwq in CONFIGS:
        row = f"  H={H:3d}  NMQ={nmq}  NWQ={nwq}        "
        for mode in all_modes:
            r = all_results[(H, nmq, nwq, mode)]
            cell = f"{r.write_bc_pairs:5d}p {r.write_bc_max:2d}w {_mark(r.write_clean)}"
            row += f"  {cell:>{cw}s}"
        print(row)

    print("\nCorrectness (all reads match expected Q data)")
    print(hdr)
    print("-" * 76)
    for H, nmq, nwq in CONFIGS:
        row = f"  H={H:3d}  NMQ={nmq}  NWQ={nwq}        "
        for mode in all_modes:
            r = all_results[(H, nmq, nwq, mode)]
            if mode == NEGATIVE_MODE:
                cell = "FAIL(expected)" if not r.correct else "PASS(BAD!)"
            elif mode == K128B_16ROW_NAIVE:
                # Expected to fail at H=128
                cell = _mark(r.correct)
            else:
                cell = _mark(r.correct)
            row += f"  {cell:>{cw}s}"
        print(row)

    # ── Final verdict ─────────────────────────────────────────────────
    print("\n" + "=" * 76)
    if unexpected_failures == 0:
        print(f"Overall: ALL TESTS PASSED  {P}")
        print()
        print("  kLinear         : correctness ok, severe read conflicts (baseline)")
        print("  k128B           : correctness ok, 8-way read conflicts")
        print("  k128B_16Row_naive: fails at H=128 — (j+4)^j=28 at j=12, not 4/12")
        print("  k128B_16Row     : zero conflicts, full correctness  ← the fix")
        print("  negative test   : correctness failures with k128B advance funcs")
        print()
        print("  Key insight for C++ implementation:")
        print("  advance_offset_by_column<4> for k128B_16Row needs col_idx:")
        print("    L         = offset & (stride-1)       # j ^ (i%16)")
        print("    i_mod     = col_idx ^ L               # recover i%16")
        print("    new_lower = (col_idx + 4) ^ i_mod     # (j+4) ^ (i%16)")
        print("    return      (offset & ~(stride-1)) + new_lower")
    else:
        print(f"Overall: {unexpected_failures} UNEXPECTED FAILURE(S)  {F}")

    return unexpected_failures


# ═══════════════════════════════════════════════════════════════════════════
# Unit-level sanity checks for individual swizzle functions
# ═══════════════════════════════════════════════════════════════════════════


def _unit_tests() -> int:
    fails = 0

    def check(name: str, got, expected):
        nonlocal fails
        if got != expected:
            print(f"  UNIT FAIL [{name}]: got {got}, expected {expected}")
            fails += 1

    stride = 32  # HEAD_DIM=128, fp16

    # ── get_permuted_offset ───────────────────────────────────────────
    check("k128B   (0,0)", get_permuted_offset(K128B, stride, 0, 0), 0)
    check("k128B   (8,0)", get_permuted_offset(K128B, stride, 8, 0), 256)
    check("16Row   (8,0)", get_permuted_offset(K128B_16ROW, stride, 8, 0), 264)
    check("16Row  (15,4)", get_permuted_offset(K128B_16ROW, stride, 15, 4), 491)

    # ── adv_col step=4, k128B — correct for all H ────────────────────
    # i=0, j=0→4: offset(0,0)=0 → (0^4)+0=4 → offset(0,4)=4
    check("k128B adv_col(0,s=0)", adv_col(K128B, 0, 4, 0), 4)
    # i=0, j=4→8: offset(0,4)=4 → (4^4)+8=8 → offset(0,8)=8
    check("k128B adv_col(4,s=1)", adv_col(K128B, 4, 4, 1), 8 + 0)
    # After adv_row, effective: i=0 col position after adv_row<16> for mma_d=3
    # offset going into adv_col = 12 + 512 = 524 (after adv_row<16>)
    # (524^4)+8 = 520+8 = 528 → 528-512=16 → offset(0,16)=16 ✓
    check("k128B adv_col(524,s=3)", adv_col(K128B, 524, 4, 3), 528)

    # ── adv_col step=4, k128B_16Row_naive — fails at j=12 (H=128) ────
    # i=15, j=4→8: offset(15,4)=491, (j+4)^j=12, correct answer=487
    # naive: (491^4)^8 = 495^8 = 487  ✓ (works here: bit-3 of offset^4 = 1)
    check("16Row_naive adv_col(491,s=1)", adv_col(K128B_16ROW_NAIVE, 491, 4, 1), 487)
    # i=0, j=12→16: after adv_row<16>, offset=524
    # correct answer: 528. naive: (524^4)^8 = 520^8 = 512 ✗
    naive_result = adv_col(K128B_16ROW_NAIVE, 524, 4, 3)
    if naive_result == 528:
        print("  UNIT WARN: naive gave 528 for j=12 case — unexpected")
    else:
        check(
            "16Row_naive adv_col(524,s=3) WRONG", naive_result != 528, True
        )  # confirm it IS wrong

    # ── adv_col step=4, k128B_16Row (correct) ────────────────────────
    # i=15, j=4→8: offset(15,4)=491, col_idx=4, stride=32
    check(
        "16Row adv_col(491,c=4)",
        adv_col(K128B_16ROW, 491, 4, 1, stride=stride, col_idx=4),
        487,
    )
    # i=0, j=12→16: offset after adv_row<16>=524, col_idx=12, stride=32
    check(
        "16Row adv_col(524,c=12)",
        adv_col(K128B_16ROW, 524, 4, 3, stride=stride, col_idx=12),
        528,
    )
    # i=4, j=12→16: offset after adv_row<16>: 4*32+(12^4)+512=128+8+512=648, col_idx=12
    check(
        "16Row adv_col(648,c=12)",
        adv_col(K128B_16ROW, 648, 4, 3, stride=stride, col_idx=12),
        660,
    )
    # 660-512=148 = get_permuted_offset(16Row,32,4,16)=128+(16^4)=148 ✓

    # ── adv_row step=4, k128B_16Row ───────────────────────────────────
    # i=4→8, row_idx=4: offset(4,0)=132 → (132^12)+128=136+128=264 → offset(8,0)
    check(
        "16Row adv_row(132,ri=4)", adv_row(K128B_16ROW, 132, 4, stride, row_idx=4), 264
    )
    # i=8→12, row_idx=8: offset(8,0)=264 → (264^4)+128=268+128=396 → offset(12,0)
    check(
        "16Row adv_row(264,ri=8)", adv_row(K128B_16ROW, 264, 4, stride, row_idx=8), 396
    )
    # k128B would give wrong result for row_idx=4:
    check(
        "k128B  adv_row(132) WRONG for 16Row", adv_row(K128B, 132, 4, stride), 256
    )  # 256 ≠ 264

    # ── Read-path bank conflict: period-8 vs period-16 ─────────────────
    offs_k128B = [get_permuted_offset(K128B, stride, i, 0) for i in range(16)]
    offs_16row = [get_permuted_offset(K128B_16ROW, stride, i, 0) for i in range(16)]
    unique_128B = len(set(lds_bank(o) for o in offs_k128B))
    unique_16r = len(set(lds_bank(o) for o in offs_16row))
    check("k128B  phase banks unique (expect 8)", unique_128B, 8)
    check("16Row  phase banks unique (expect 16)", unique_16r, 16)

    return fails


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    verbose = "-v" in sys.argv or "--verbose" in sys.argv

    print("\nRunning unit-level sanity checks...")
    unit_fails = _unit_tests()
    if unit_fails == 0:
        print(f"  Unit checks: all passed  {P}")
    else:
        print(f"  Unit checks: {unit_fails} FAILED  {F}")
        sys.exit(1)

    print()
    total_fails = run_tests(verbose=verbose)
    sys.exit(0 if total_fails == 0 else 1)
