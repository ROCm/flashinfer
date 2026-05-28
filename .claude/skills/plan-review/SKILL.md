---
name: plan-review
description: Interactive plan-mode review workflow for amd-flashinfer changes — engineering preferences, the four review stages (Architecture / Code Quality / Tests / Performance), per-issue option format, and the BIG/SMALL CHANGE interaction model. Load when entering plan mode or when the user asks for a structured plan review.
---

# Plan Review Workflow

Review this plan thoroughly before making any code changes. For every issue or
recommendation, explain the concrete tradeoffs, give an opinionated
recommendation, and ask for the user's input before assuming a direction.

## Engineering preferences (use these to guide recommendations)

- **DRY is important** — tag repetition aggressively.
- **Well-tested code is non-negotiable** — err toward too many tests, not too few.
- Code should be *engineered enough* — not under-engineered (fragile, hacky)
  and not over-engineered (premature abstraction, unnecessary complexity).
- Err on the side of **readable code** over edge cases, but never thoughtlessness
  or speed.

## Review stages

### 1. Architecture review

- Overall system design and component boundaries.
- Dependency graph and coupling concerns.
- Data flow patterns and potential bottlenecks.
- Scaling characteristics and single points of failure.
- Security architecture (auth, data access, API boundaries).

### 2. Code quality review

- Code organization and module structure.
- DRY violations — be aggressive here.
- Error handling patterns and missing edge cases (call these out explicitly).
- Technical debt hotspots.
- Areas that are over- or under-engineered relative to the preferences above.

### 3. Test review

- Test coverage gaps (unit, integration, e2e).
- Test quality and assertion strength.
- Missing edge case coverage — be thorough.
- Untested failure modes and error paths.

### 4. Performance review

- N+1 queries and database access patterns.
- Memory usage concerns.
- Caching opportunities.
- Slow or high-complexity code paths.

## Per-issue format

For every specific issue (bug, smell, design concern, or risk):

- Describe the problem clearly, with file and line references.
- Present **2–3 options**, including a **"do nothing"** option where reasonable.
- For each option, specify:
  - Implementation effort
  - Risk
  - Impact on behavior, code, and maintenance burden
- Give an explicit **recommended option** and why.
- Then **explicitly ask whether the user wants a different direction before
  proceeding**.

## Workflow and interaction

- Do **not** assume the user's priorities on timeline or scale.
- After each section, pause and ask for feedback before moving on.

## Before you start — ask which mode

- **BIG CHANGE** — Work through interactively, one section at a time
  (Architecture → Code Quality → Tests → Performance) with at most **4 top
  issues** per section.
- **SMALL CHANGE** — Work through interactively, **one question per review
  section**.

## For each stage

Output the explanation and pros/cons of each stage's questions **and** an
opinionated recommendation with reasoning.

Then use **`AskUserQuestion`**:

- Use **NUMBERED issues**.
- Use **LETTERED options**.
- Label each option with the issue **NUMBER** and option **LETTER** so the user
  can reference recommendations precisely (e.g., "1A", "2C").
