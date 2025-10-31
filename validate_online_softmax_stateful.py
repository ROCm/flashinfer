#!/usr/bin/env python3
"""
Stateful validation of online softmax across multiple iterations.
Maintains running m (maximum) state across KV chunks.

Usage:
    ./validate_online_softmax_stateful.py [LOG_FILE]

Examples:
    ./validate_online_softmax_stateful.py prefill.log
    ./validate_online_softmax_stateful.py my_debug.log
    ./validate_online_softmax_stateful.py  # defaults to prefill.log
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

# ============================================================================
# PARSING FUNCTIONS
# ============================================================================


def parse_sm_scale(lines):
    """Extract sm_scale from log file."""
    for line in lines:
        if "sm_scale" in line:
            match = re.search(r"sm_scale\s*:\s*([\d.]+)", line)
            if match:
                return float(match.group(1))
    raise ValueError("Could not find sm_scale in log file")


def parse_matrix(lines, start_line):
    """Parse matrix starting from start_line. Returns 128×64 matrix."""
    data = []
    for i in range(start_line, len(lines)):
        line = lines[i]
        if "frag" in line or "DEBUG" in line or line.strip().startswith("num_"):
            break
        if line.strip():
            nums = re.findall(r"-?\d+\.\d+", line)
            if nums:
                data.extend([float(x) for x in nums])

    if len(data) == 0:
        return None

    expected_size = 128 * 64
    if len(data) != expected_size:
        print(f"Warning: Expected {expected_size} values, got {len(data)}")
        return None

    return np.array(data).reshape(128, 64)


def find_iteration_data(lines, iteration_num):
    """
    Find before and after matrices for a given iteration.
    Returns (before_matrix, after_matrix).
    """
    # Find before data
    iter_count = 0
    before_line = None
    for i, line in enumerate(lines):
        if "S frag before update_mdo for iteration" in line:
            if iter_count == iteration_num:
                before_line = i + 2
            iter_count += 1

    # Find after data
    iter_count = 0
    after_line = None
    for i, line in enumerate(lines):
        if "S frag after update_mdo for iteration" in line:
            if iter_count == iteration_num:
                after_line = i + 2
                break
            iter_count += 1

    if before_line is None or after_line is None:
        return None, None

    before = parse_matrix(lines, before_line)
    after = parse_matrix(lines, after_line)

    return before, after


def validate_with_state(before_row, after_row, m_prev, sm_scale):
    """
    Validate online softmax transformation with stateful m.

    Args:
        before_row: Raw scores for this chunk (64 values)
        after_row: Transformed scores (64 values)
        m_prev: Maximum from previous chunks (scalar)
        sm_scale: Softmax scale factor (scalar)

    Returns:
        (is_valid, m_new, max_error)
    """
    # Step 1: Find maximum in current chunk
    m_chunk = before_row.max()

    # Step 2: Update running maximum
    m_new = max(m_prev, m_chunk)

    # Step 3: Apply transformation to each element
    expected = np.exp2((before_row - m_new) * sm_scale)

    # Step 4: Compare with actual
    diff = np.abs(after_row - expected)
    max_error = diff.max()

    # Tolerance for floating point comparison
    tolerance = 5e-3  # 0.005
    is_valid = max_error < tolerance

    return is_valid, m_new, max_error


# ============================================================================
# VALIDATION ORCHESTRATION
# ============================================================================


def validate_all_iterations(lines, sm_scale, num_iterations=2):
    """
    Validate all iterations with proper state management.

    Returns:
        (total_passed, total_rows, max_error_overall)
    """
    print(f"{'='*80}")
    print(f"STATEFUL ONLINE SOFTMAX VALIDATION")
    print(f"{'='*80}")
    print(f"sm_scale: {sm_scale}")
    print(f"Formula: s_after = exp2((s_before - m_new) * sm_scale)")
    print(f"  where m_new = max(m_prev, max(s_before_chunk))")
    print(f"{'='*80}\n")

    total_passed = 0
    total_rows = 0
    max_error_overall = 0.0

    # Initialize m_prev to -inf for first iteration
    m_prev_per_row = np.full(128, -np.inf)

    for iteration in range(num_iterations):
        print(f"\n{'─'*80}")
        print(f"ITERATION {iteration}")
        print(f"{'─'*80}")

        before, after = find_iteration_data(lines, iteration)

        if before is None or after is None:
            print(f"❌ Could not find data for iteration {iteration}")
            continue

        print(f"Matrix shape: {before.shape}")

        passed = 0
        failed = 0
        max_error_iter = 0.0

        # Process each row with its own m_prev
        for row_idx in range(128):
            before_row = before[row_idx, :]
            after_row = after[row_idx, :]
            m_prev = m_prev_per_row[row_idx]

            is_valid, m_new, max_error = validate_with_state(
                before_row, after_row, m_prev, sm_scale
            )

            # Update running m for this row
            m_prev_per_row[row_idx] = m_new

            max_error_iter = max(max_error_iter, max_error)
            max_error_overall = max(max_error_overall, max_error)

            if is_valid:
                passed += 1
            else:
                failed += 1
                if failed <= 3:  # Show first 3 failures
                    print(
                        f"  ❌ Row {row_idx}: m_prev={m_prev:.6f}, "
                        f"m_chunk={before_row.max():.6f}, "
                        f"m_new={m_new:.6f}, max_error={max_error:.6e}"
                    )

        total_passed += passed
        total_rows += 128

        print(f"\nIteration {iteration} Results:")
        print(f"  ✓ Passed: {passed}/128 rows")
        print(f"  ✗ Failed: {failed}/128 rows")
        print(f"  📊 Max error: {max_error_iter:.6e}")

        if failed == 0:
            print(f"  🎉 ITERATION {iteration} VALIDATED SUCCESSFULLY!")

        # Debug: Show sample row state
        sample_row = 0
        print(f"\n  Sample (Row {sample_row}):")
        print(f"    m_prev: {m_prev_per_row[sample_row]:.6f}")
        print(f"    m_chunk: {before[sample_row, :].max():.6f}")
        print(f"    m_new: {m_prev_per_row[sample_row]:.6f}")

    print(f"\n{'='*80}")
    print(f"OVERALL RESULTS")
    print(f"{'='*80}")
    print(
        f"Total rows validated: {total_passed}/{total_rows} ({100*total_passed/total_rows:.1f}%)"
    )
    print(f"Max error across all iterations: {max_error_overall:.6e}")

    if total_passed == total_rows:
        print(f"\n🎉 ALL ROWS VALIDATED SUCCESSFULLY!")
        print(f"✅ Online softmax is correctly implemented with stateful m propagation")
        return True
    else:
        print(f"\n⚠️  VALIDATION INCOMPLETE: {total_rows - total_passed} rows failed")
        return False


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Validate online softmax with stateful m propagation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s prefill.log        Validate using prefill.log
  %(prog)s my_debug.log       Validate using my_debug.log
  %(prog)s                    Validate using prefill.log (default)
        """,
    )
    parser.add_argument(
        "logfile",
        nargs="?",
        default="prefill.log",
        help="Path to log file (default: prefill.log)",
    )
    parser.add_argument(
        "-n",
        "--num-iterations",
        type=int,
        default=2,
        help="Number of iterations to validate (default: 2)",
    )

    args = parser.parse_args()

    logfile = Path(args.logfile)

    if not logfile.exists():
        print(f"❌ Error: Log file '{logfile}' not found")
        sys.exit(1)

    print(f"Reading log file: {logfile}")

    with open(logfile, "r") as f:
        lines = f.readlines()

    print(f"Loaded {len(lines)} lines from {logfile}\n")

    try:
        sm_scale = parse_sm_scale(lines)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    success = validate_all_iterations(lines, sm_scale, args.num_iterations)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
