#!/usr/bin/env python3
"""
Update the coverage include list in pyproject.toml based on AMD/HIP modified files.

This script compares the current branch against the merge base with upstream/main
to find modified Python files in the flashinfer/ module and updates the coverage include list.
"""

import re
import subprocess
import sys
from pathlib import Path


def get_merge_base(upstream_ref="upstream/main"):
    """Get the merge base (fork point) between current branch and upstream."""
    try:
        result = subprocess.run(
            ["git", "merge-base", upstream_ref, "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        merge_base = result.stdout.strip()
        if not merge_base:
            raise ValueError("Empty merge base")
        return merge_base
    except (subprocess.CalledProcessError, ValueError) as e:
        print(
            f"Error: Failed to find merge base with {upstream_ref}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def get_modified_files(upstream_ref="upstream/main"):
    """Get list of modified Python files in flashinfer/ module."""
    # Find the fork point
    merge_base = get_merge_base(upstream_ref)
    print(f"  Merge base: {merge_base[:8]}")

    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", merge_base, "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to get git diff: {e}", file=sys.stderr)
        sys.exit(1)

    modified_files = [
        line.strip()
        for line in result.stdout.strip().split("\n")
        if line.strip() and line.startswith("flashinfer/") and line.endswith(".py")
    ]

    return sorted(modified_files)


def update_pyproject_toml(modified_files, dry_run=False):
    """Update the include list in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")

    if not pyproject_path.exists():
        print("Error: pyproject.toml not found", file=sys.stderr)
        sys.exit(1)

    # Read current content
    with open(pyproject_path, "r") as f:
        content = f.read()

    # Build the new include list
    include_lines = ["include = ["]
    for file in modified_files:
        include_lines.append(f'    "{file}",')
    include_lines.append("]")
    new_include = "\n".join(include_lines)

    # Pattern to match the include section
    # Matches from "include = [" to the closing "]"
    pattern = r"(# AMD/HIP modified files.*\n)include = \[[^\]]*\]"

    # Check if pattern exists
    if not re.search(pattern, content, flags=re.DOTALL):
        print(
            "Error: Could not find coverage include section in pyproject.toml",
            file=sys.stderr,
        )
        print(
            "Make sure the file has the marker comment: '# AMD/HIP modified files'",
            file=sys.stderr,
        )
        sys.exit(1)

    # Replace the include section
    new_content = re.sub(pattern, r"\1" + new_include, content, flags=re.DOTALL)

    # Check if content changed
    if new_content == content:
        print("✓ Coverage include list is already up to date")
        return False

    if dry_run:
        print("Would update pyproject.toml with the following files:")
        for file in modified_files:
            print(f"  - {file}")
        return True

    # Write updated content
    with open(pyproject_path, "w") as f:
        f.write(new_content)

    print("✓ Updated coverage include list in pyproject.toml")
    print(f"  Modified files: {len(modified_files)}")
    for file in modified_files:
        print(f"  - {file}")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Update coverage include list based on AMD/HIP modified files"
    )
    parser.add_argument(
        "--upstream",
        default="upstream/main",
        help="Upstream reference to compare against (default: upstream/main)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )

    args = parser.parse_args()

    print(f"Comparing against {args.upstream}...")
    modified_files = get_modified_files(args.upstream)

    if not modified_files:
        print("No modified Python files found in flashinfer/ module")
        sys.exit(0)

    print(f"Found {len(modified_files)} modified Python files in flashinfer/")

    changed = update_pyproject_toml(modified_files, dry_run=args.dry_run)

    if changed and not args.dry_run:
        print("\nNext steps:")
        print("  1. Review changes: git diff pyproject.toml")
        print("  2. Test coverage: pytest --cov --cov-report=term-missing")
        print(
            "  3. Commit: git add pyproject.toml && git commit -m 'Update coverage include list'"
        )


if __name__ == "__main__":
    main()
