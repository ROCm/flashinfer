---
name: pr-workflow
description: How to create and edit PRs on the amd-flashinfer repo — gh CLI quirks and the project's PR-description conventions.
---

# PR Workflow (amd-flashinfer)

## GitHub CLI

`gh pr edit` fails with a "Projects (classic) is being deprecated" GraphQL error on this repo. Use the REST API instead:

```bash
# Update PR description
gh api repos/ROCm/flashinfer/pulls/<number> --method PATCH --field body="<body>"

# Or from a file
gh api repos/ROCm/flashinfer/pulls/<number> --method PATCH --field body="$(cat /tmp/pr_body.md)"
```

Ask to push to remote.

## PR Description

**Body** — include sections that apply, skip the rest:

- `## Summary` — 1–3 sentences on what and why.
- `### What changed` with `####` per component when the PR spans multiple
  subsystems. Bullet by file: ``- **`path`** — one-line purpose``. Call out
  non-obvious design choices.
- `### Architecture / design notes` — only when there's a real choice to record.
  Tables for routing/dispatch logic; explain *why*.
- `## Benchmark results` — for perf-touching PRs. Shape line + table per entry
  point + mean overhead/speedup row.
- `## Test plan` — checklist of what was actually run (not aspirational), ending
  with `pre-commit run -a`.

Don't restate the diff and commits. Explain non-obvious decisions and surprising behaviors.
