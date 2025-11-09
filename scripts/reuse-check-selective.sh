#!/bin/bash
# Check only specified files for REUSE compliance
# This is much faster than running 'reuse lint' on the whole repo

exit_code=0

for file in "$@"; do
    # Check if file has proper SPDX headers
    if ! grep -q "^// SPDX-FileCopyrightText:" "$file" 2>/dev/null; then
        echo "✗ $file: Missing SPDX-FileCopyrightText header"
        exit_code=1
        continue
    fi

    if ! grep -q "^// SPDX-License-Identifier:" "$file" 2>/dev/null; then
        echo "✗ $file: Missing SPDX-License-Identifier header"
        exit_code=1
        continue
    fi

    # Check for malformed SPDX tag names - specifically spaces around the first hyphen after SPDX
    # Bad patterns: "SPDX - FileCopyrightText", "SPDX -FileCopyrightText", "SPDX- FileCopyrightText"
    # Good pattern: "SPDX-FileCopyrightText" (no spaces around the hyphen right after SPDX)
    if grep -q "^// SPDX \+-\|^// SPDX- " "$file" 2>/dev/null; then
        echo "✗ $file: Malformed SPDX header (spaces around hyphen after SPDX)"
        exit_code=1
        continue
    fi

    # Check for spaces before the colon in SPDX tags
    # Match patterns like "SPDX-FileCopyrightText :" or "SPDX-License-Identifier :"
    if grep -q "^// SPDX-\(FileCopyrightText\|License-Identifier\) \+:" "$file" 2>/dev/null; then
        echo "✗ $file: Malformed SPDX header (space before colon)"
        exit_code=1
        continue
    fi
done

if [ $exit_code -eq 0 ]; then
    echo "✓ All checked files are REUSE compliant"
fi

exit $exit_code
