#!/usr/bin/env python3
"""
Update the coverage include list in pyproject.toml based on:
1. AMD/HIP modified files (git diff)
2. Modules imported/used by tests in testpaths

This script combines two sources:
- Modified files: Tracks active development work
- Test usage: Ensures tested modules are covered even if unmodified

Smart resolution features:
- Parses __init__.py to understand HIP/ROCm module aliases (decode -> decode_rocm)
- Builds function→module map to resolve re-exported functions (apply_rope -> rope.py)
- Tracks attribute access patterns (flashinfer.norm.rmsnorm)
- No build required - pure static analysis via AST parsing
"""

import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Set, Dict, List


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
    merge_base = get_merge_base(upstream_ref)
    print(f"  Merge base: {merge_base[:8]}")

    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=AM", merge_base, "HEAD"],
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

    return sorted(set(modified_files))


def parse_testpaths_from_pyproject():
    """Extract testpaths from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("Error: pyproject.toml not found", file=sys.stderr)
        sys.exit(1)

    with open(pyproject_path, "r") as f:
        content = f.read()

    # Find testpaths section
    pattern = r"\[tool\.pytest\.ini_options\].*?testpaths\s*=\s*\[(.*?)\]"
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        print("Warning: Could not find testpaths in pyproject.toml", file=sys.stderr)
        return []

    # Extract test file paths
    testpaths_str = match.group(1)
    testpaths = re.findall(r'"([^"]+)"', testpaths_str)

    return testpaths


def parse_flashinfer_init_aliases() -> Dict[str, str]:
    """
    Parse flashinfer/__init__.py to extract module aliases for HIP/ROCm.

    Returns a dict mapping generic module names to their ROCm variants:
        {'decode': 'decode_rocm', 'prefill': 'prefill_rocm'}
    """
    aliases: Dict[str, str] = {}
    init_file = Path("flashinfer/__init__.py")

    if not init_file.exists():
        return aliases

    try:
        with open(init_file, "r") as f:
            tree = ast.parse(f.read(), filename=str(init_file))
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse flashinfer/__init__.py: {e}", file=sys.stderr)
        return aliases

    def check_if_node(node):
        """Recursively check If nodes including elif (orelse)"""
        if not isinstance(node, ast.If):
            return

        # Check if this is an IS_HIP condition
        is_hip = hasattr(node.test, "id") and node.test.id == "IS_HIP"

        if is_hip:
            # Parse imports in this block
            for item in node.body:
                if isinstance(item, ast.ImportFrom):
                    # Check if it's a relative import (level > 0)
                    if item.level > 0 and item.module:
                        module = item.module

                        # Check if this is a _rocm variant
                        if module.endswith("_rocm"):
                            # Map generic name to rocm variant
                            base = module[:-5]  # Remove '_rocm' suffix
                            aliases[base] = module

        # Check elif/else chains
        for orelse in node.orelse:
            check_if_node(orelse)

    # Walk top-level nodes
    for node in tree.body:
        check_if_node(node)

    return aliases


def parse_flashinfer_init_function_map() -> Dict[str, str]:
    """
    Parse flashinfer/__init__.py to build function → source module map.

    Returns dict mapping function names to their source modules:
        {'apply_rope': 'rope', 'rmsnorm': 'norm', ...}

    This resolves re-exported functions so when tests call flashinfer.apply_rope(),
    we know it comes from rope.py without needing to import/build flashinfer.
    """
    function_to_module: Dict[str, str] = {}
    init_file = Path("flashinfer/__init__.py")

    if not init_file.exists():
        return function_to_module

    try:
        with open(init_file, "r") as f:
            tree = ast.parse(f.read(), filename=str(init_file))
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse flashinfer/__init__.py: {e}", file=sys.stderr)
        return function_to_module

    def check_if_node(node):
        """Recursively check If nodes including elif (orelse)"""
        if not isinstance(node, ast.If):
            return

        # Check for IS_HIP block
        is_hip = hasattr(node.test, "id") and node.test.id == "IS_HIP"

        if is_hip:
            # Parse from imports in this block
            for item in node.body:
                if isinstance(item, ast.ImportFrom) and item.level > 0 and item.module:
                    source_module = item.module
                    for alias in item.names:
                        if alias.name != "*":
                            func_name = alias.name
                            function_to_module[func_name] = source_module

        for orelse in node.orelse:
            check_if_node(orelse)

    for node in tree.body:
        check_if_node(node)

    return function_to_module


class FlashinferUsageExtractor(ast.NodeVisitor):
    """Extract flashinfer module usage including attribute access and function calls."""

    def __init__(self, aliases: Dict[str, str], function_map: Dict[str, str]):
        self.imports: set[str] = set()  # Direct imports: import flashinfer
        self.from_imports: set[str] = set()  # From imports: from flashinfer.x import y
        self.flashinfer_attrs: set[str] = set()  # Attribute access: flashinfer.norm
        self.flashinfer_calls: set[str] = (
            set()
        )  # Function calls: flashinfer.apply_rope()
        self.imported_names: Dict[str, str] = {}  # Track what names refer to flashinfer
        self.aliases = aliases  # Module aliases (e.g., decode -> decode_rocm)
        self.function_map = function_map  # Function → module map

    def _apply_aliases(self, module_name: str) -> str:
        """Apply known aliases to module name."""
        if not module_name.startswith("flashinfer"):
            return module_name

        parts = module_name.split(".")
        if len(parts) >= 2:
            # Check if second part needs aliasing
            if parts[1] in self.aliases:
                parts[1] = self.aliases[parts[1]]
                return ".".join(parts)

        return module_name

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.startswith("flashinfer"):
                resolved = self._apply_aliases(alias.name)
                self.imports.add(resolved)
                # Track the name it's imported as
                name = alias.asname if alias.asname else alias.name
                self.imported_names[name] = resolved
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module and node.module.startswith("flashinfer"):
            resolved = self._apply_aliases(node.module)
            self.from_imports.add(resolved)
            for alias in node.names:
                if alias.name != "*":
                    # Track imported names
                    name = alias.asname if alias.asname else alias.name
                    self.imported_names[name] = f"{resolved}.{alias.name}"
        self.generic_visit(node)

    def visit_Call(self, node):
        """Track function calls like flashinfer.apply_rope()"""
        if isinstance(node.func, ast.Attribute):
            # Get the attribute chain
            obj = node.func.value
            func_name = node.func.attr

            # Check if calling through flashinfer
            if isinstance(obj, ast.Name) and obj.id in self.imported_names:
                base = self.imported_names[obj.id]
                if base == "flashinfer":
                    # Look up which module this function comes from
                    if func_name in self.function_map:
                        source_module = self.function_map[func_name]
                        # Apply aliases to source module
                        resolved = self._apply_aliases(f"flashinfer.{source_module}")
                        self.flashinfer_calls.add(resolved)

        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Track attribute access like flashinfer.norm or flashinfer.activation.silu_and_mul"""
        # Get the full attribute chain
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value

        # Check if it starts with flashinfer or an alias
        if isinstance(current, ast.Name):
            if current.id in self.imported_names:
                base = self.imported_names[current.id]
                if base.startswith("flashinfer"):
                    # Build the full path
                    parts.reverse()
                    full_path = base
                    for part in parts:
                        full_path = f"{full_path}.{part}"
                        resolved = self._apply_aliases(full_path)
                        self.flashinfer_attrs.add(resolved)
                        # Also add intermediate paths
                        if resolved.count(".") <= 3:  # Limit depth
                            base_path = resolved.rsplit(".", 1)[0]
                            self.flashinfer_attrs.add(base_path)

        self.generic_visit(node)


def extract_flashinfer_usage_from_file(
    filepath: Path, aliases: Dict[str, str], function_map: Dict[str, str]
) -> Set[str]:
    """Extract flashinfer module usage from a Python file using AST parsing."""
    try:
        with open(filepath, "r") as f:
            tree = ast.parse(f.read(), filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {filepath}: {e}", file=sys.stderr)
        return set()

    extractor = FlashinferUsageExtractor(aliases, function_map)
    extractor.visit(tree)

    # Combine all discovered modules
    all_modules = set()
    all_modules.update(extractor.imports)
    all_modules.update(extractor.from_imports)
    all_modules.update(extractor.flashinfer_attrs)
    all_modules.update(extractor.flashinfer_calls)

    return all_modules


def module_to_filepath(module_name: str) -> Set[str]:
    """
    Convert a module import to potential file paths.

    Examples:
        flashinfer -> flashinfer/__init__.py
        flashinfer.norm -> flashinfer/norm.py
        flashinfer.jit.core -> flashinfer/jit/core.py
        flashinfer.decode_rocm -> flashinfer/decode_rocm.py
    """
    filepaths = set()

    # Handle base module
    if module_name == "flashinfer":
        filepaths.add("flashinfer/__init__.py")
        return filepaths

    # Split module path
    parts = module_name.split(".")
    if parts[0] != "flashinfer":
        return filepaths

    # Try different combinations (module could be a file or package)
    # Start from the longest path and work backwards
    for i in range(len(parts), 0, -1):
        # Try as a file: flashinfer/a/b.py
        module_path = "/".join(parts[:i]) + ".py"
        if Path(module_path).exists():
            filepaths.add(module_path)
            break  # Found it, stop searching

        # Try as a package: flashinfer/a/b/__init__.py
        package_path = "/".join(parts[:i]) + "/__init__.py"
        if Path(package_path).exists():
            filepaths.add(package_path)
            break  # Found it, stop searching

    return filepaths


def get_files_from_test_usage(
    testpaths, aliases: Dict[str, str], function_map: Dict[str, str]
) -> List[str]:
    """Extract flashinfer module files used by test files."""
    all_modules = set()
    all_files = set()

    for testpath in testpaths:
        test_file = Path(testpath)
        if not test_file.exists():
            print(f"Warning: Test file not found: {testpath}", file=sys.stderr)
            continue

        modules = extract_flashinfer_usage_from_file(test_file, aliases, function_map)
        all_modules.update(modules)

    # Convert modules to file paths
    for module_name in all_modules:
        filepaths = module_to_filepath(module_name)
        all_files.update(filepaths)

    return sorted(all_files)


def update_pyproject_toml(coverage_files, dry_run=False):
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
    for file in coverage_files:
        include_lines.append(f'    "{file}",')
    include_lines.append("]")
    new_include = "\n".join(include_lines)

    # Pattern to match the include section
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
        for file in coverage_files:
            print(f"  - {file}")
        return True

    # Write updated content
    with open(pyproject_path, "w") as f:
        f.write(new_content)

    print("✓ Updated coverage include list in pyproject.toml")
    print(f"  Total files: {len(coverage_files)}")
    for file in coverage_files:
        print(f"  - {file}")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Update coverage include list based on git diff + test usage"
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information about sources",
    )

    args = parser.parse_args()

    # Parse flashinfer/__init__.py for HIP aliases and function map
    print("1. Parsing flashinfer/__init__.py...")
    aliases = parse_flashinfer_init_aliases()
    function_map = parse_flashinfer_init_function_map()

    if aliases:
        print(f"   Found {len(aliases)} module aliases:")
        for generic, rocm in sorted(aliases.items()):
            print(f"     {generic} -> {rocm}")

    if function_map:
        print(f"   Found {len(function_map)} function mappings")
        if args.verbose:
            for func, mod in sorted(function_map.items())[:10]:
                print(f"     {func}() -> {mod}.py")
            if len(function_map) > 10:
                print(f"     ... and {len(function_map) - 10} more")

    # Get modified files from git
    print(f"\n2. Checking git diff against {args.upstream}...")
    modified_files = get_modified_files(args.upstream)
    print(f"   Found {len(modified_files)} modified files")
    if args.verbose:
        for f in modified_files:
            print(f"     - {f}")

    # Get files from test usage
    print("\n3. Analyzing test usage from pyproject.toml testpaths...")
    testpaths = parse_testpaths_from_pyproject()
    print(f"   Found {len(testpaths)} test files")

    test_used_files = get_files_from_test_usage(testpaths, aliases, function_map)
    print(f"   Found {len(test_used_files)} used modules")
    if args.verbose:
        for f in test_used_files:
            print(f"     - {f}")

    # Combine both sources
    print("\n4. Combining sources...")
    all_files = sorted(set(modified_files) | set(test_used_files))

    git_only = set(modified_files) - set(test_used_files)
    test_only = set(test_used_files) - set(modified_files)
    both = set(modified_files) & set(test_used_files)

    print(f"   Git diff only: {len(git_only)}")
    if args.verbose and git_only:
        for f in sorted(git_only):
            print(f"     - {f}")

    print(f"   Test usage only: {len(test_only)}")
    if args.verbose and test_only:
        for f in sorted(test_only):
            print(f"     - {f}")

    print(f"   Both sources: {len(both)}")
    if args.verbose and both:
        for f in sorted(both):
            print(f"     - {f}")

    print(f"   Total unique: {len(all_files)}")

    if not all_files:
        print("\n⚠ No files found from either source")
        sys.exit(0)

    # Update pyproject.toml
    print("\n5. Updating pyproject.toml...")
    changed = update_pyproject_toml(all_files, dry_run=args.dry_run)

    if changed and not args.dry_run:
        print("\n✓ Done! Next steps:")
        print("  1. Review changes: git diff pyproject.toml")
        print("  2. Test coverage: pytest --cov --cov-report=term-missing")
        print(
            "  3. Commit: git add pyproject.toml && git commit -m 'Update coverage include list'"
        )
    elif not changed:
        print("\n✓ No changes needed")


if __name__ == "__main__":
    main()
