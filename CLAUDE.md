# Claude Code instructions for flashinfer

## Plan files

When a plan is created and approved, save it to the project memory directory:

**Location:** `$HOME/.claude/projects/-home-demandal-devel-flashinfer/memory/`

**Naming:** `plan_<short_descriptive_slug>.md` — use the subject of the plan, not a random name.
Examples: `plan_mla_hip_port.md`, `plan_rope_cdna3_tuning.md`, `plan_pod_hip_port.md`

**Index:** After saving the file, add a one-line entry to `MEMORY.md` in that same directory:
`- [Plan: <title>](plan_<slug>.md) — <one-line summary>`

This ensures every plan is findable by name across sessions without being committed to git.
