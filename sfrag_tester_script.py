#!/usr/bin/env python3
import re
import sys

import numpy as np
import pandas as pd


def parse_sfrag_log(log_file_path, num_threads=64, num_mma_q=2, num_mma_kv=4):
    """
    Parse s_frag debug output from multiple thread runs into a 32x128 DataFrame.

    Args:
        log_file_path: Path to the concatenated log file
        num_threads: Number of threads (default 64)
        num_mma_q: NUM_MMA_Q value (default 2)
        num_mma_kv: NUM_MMA_KV value (default 4)

    Returns:
        DataFrame with shape (32, 128) containing the s_frag values
    """

    # Initialize the result matrix
    matrix = np.zeros((32, 128))

    # Read the log file
    with open(log_file_path, "r") as f:
        lines = f.readlines()

    # Track current thread and value position
    current_thread = -1
    value_idx = 0
    values = []

    for line in lines:
        line = line.strip()

        # Check if this is a thread ID line
        if line.startswith("Debug thread ID set to:"):
            if current_thread >= 0 and values:
                # Process the previous thread's data
                populate_matrix(matrix, current_thread, values)

            # Extract thread ID
            current_thread = int(line.split(":")[-1].strip())
            values = []
            value_idx = 0

        # Otherwise, it should be a line of float values
        elif line and current_thread >= 0:
            # Parse the float values from the line
            line_values = [float(x) for x in line.split()]
            values.extend(line_values)

    # Don't forget to process the last thread
    if current_thread >= 0 and values:
        populate_matrix(matrix, current_thread, values)

    # Create DataFrame with appropriate column and row labels
    df = pd.DataFrame(matrix)
    df.index = [f"Row_{i}" for i in range(32)]
    df.columns = [f"Col_{i}" for i in range(128)]

    return df


def populate_matrix(matrix, thread_id, values):
    """
    Populate the matrix with values from a specific thread according to the
    pattern the values are printed.

    Args:
        matrix: The 32x128 numpy array to populate
        thread_id: The thread ID (0-63)
        values: List of 64 float values from this thread
    """

    if len(values) != 64:
        print(f"Warning: Thread {thread_id} has {len(values)} values instead of 64")
        return

    # Calculate base row and column for this thread
    row_base = (thread_id // 16) * 4
    col_base = thread_id % 16

    # Split values into two calls (32 values each)
    first_call = values[:32]
    second_call = values[32:]

    # Process first call (columns 0-63)
    process_call_values(matrix, first_call, row_base, col_base, col_offset=0)

    # Process second call (columns 64-127)
    process_call_values(matrix, second_call, row_base, col_base, col_offset=64)


def process_call_values(matrix, values, row_base, col_base, col_offset):
    """
    Process 32 values from one call according to the nested loop pattern.

    Args:
        matrix: The matrix to populate
        values: 32 values from one call
        row_base: Base row for this thread
        col_base: Base column for this thread
        col_offset: Column offset (0 for first call, 64 for second call)
    """

    value_idx = 0
    current_row = row_base
    current_col = col_base + col_offset

    # Outer loop: 2 iterations (NUM_MMA_Q)
    for mma_q in range(2):
        # Middle loop: 4 iterations (NUM_MMA_KV)
        for mma_kv in range(4):
            # Inner loop: 4 values
            for i in range(4):
                if value_idx < len(values):
                    # Place values in consecutive rows, same column
                    matrix[current_row + i, current_col] = values[value_idx]
                    value_idx += 1

            # After inner loop, move to next column set
            current_col += 16

        # After middle loop, reset column and move to next row set
        current_col = col_base + col_offset
        current_row += 16


def print_matrix_info(df):
    """Print summary information about the populated matrix."""
    print(f"Matrix shape: {df.shape}")
    print(f"Non-zero elements: {(df != 0).sum().sum()}")
    print(f"Matrix statistics:")
    print(f"  Min: {df.min().min():.6f}")
    print(f"  Max: {df.max().max():.6f}")
    print(f"  Mean: {df.mean().mean():.6f}")
    print(f"  Std: {df.values.std():.6f}")


def save_results(df, output_prefix="sfrag_matrix"):
    """Save the DataFrame in multiple formats."""
    # Save as CSV
    csv_file = f"{output_prefix}.csv"
    df.to_csv(csv_file)
    print(f"Saved matrix to {csv_file}")

    # Save as pickle for exact preservation
    pickle_file = f"{output_prefix}.pkl"
    df.to_pickle(pickle_file)
    print(f"Saved matrix to {pickle_file}")

    # Save a heatmap visualization
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(20, 6))
        sns.heatmap(
            df.iloc[:32, :64], cmap="RdBu_r", center=0, cbar_kws={"label": "Value"}
        )
        plt.title("S_FRAG Matrix Heatmap (First 32x64 block)")
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_heatmap.png", dpi=150)
        plt.close()
        print(f"Saved heatmap to {output_prefix}_heatmap.png")
    except ImportError:
        print("Matplotlib/Seaborn not available, skipping heatmap")


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_sfrag.py <log_file> [output_prefix]")
        sys.exit(1)

    log_file = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else "sfrag_matrix"

    print(f"Parsing log file: {log_file}")

    # Parse the log file
    df = parse_sfrag_log(log_file)

    # Print summary information
    print_matrix_info(df)

    # Save results
    save_results(df, output_prefix)

    # Print a sample of the matrix
    print("\nSample of the matrix (first 8x8 block):")
    print(df.iloc[:8, :8])


if __name__ == "__main__":
    main()
