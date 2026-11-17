# Working in shapiq-probly

This is the shared instruction file for Codex and Claude Code. Read
[the contributor guide](.github/CONTRIBUTING.md) for setup, commands and the code map.

## Collaboration

- Work only in `smtyyy/shapiq-probly`. `origin`, pushes and PR base repositories must refer to this
  fork. Do not open PRs against `mmschlk/shapiq` unless the user explicitly requests it.
- Work on `feature/<topic>` branches. Prefer a separate Git worktree per human or agent task;
  one writer owns each branch/worktree. Never develop or commit directly on `main`.
- PRs target this fork's `main`, require passing checks and one human approval, and merge by rebase.
  An agent review supplements the required human review. Do not bypass rules or merge unless
  explicitly asked for that action. The owner's bypass is an exceptional manual escape hatch.
- Before fetching/rebasing a published branch, reject non-`feature/*` branches and staged,
  unstaged or untracked changes. Never overwrite a collaborator's commits; use the contributor
  guide's guarded explicit-OID lease procedure when updating a published rebased branch.
- Keep commits focused. Prefer English `type(scope): concrete change` subjects; split unrelated
  work and include implementation, relevant tests and documentation together.

## Implementation

- Search existing shapiq and probly code, public APIs and tests before adding functionality.
  Reuse implementations; add an adapter only where necessary. Use the `probly-reuse` skill for
  uncertainty work. Do not silently add or change a probly dependency or source revision.
- Keep the existing shapiq Ruff, formatter and `ty` settings authoritative: 100-column formatting,
  Python 3.12 minimum, typed library APIs, Google-style English docstrings. Do not copy probly's
  different formatting rules or weaken checks to make a change pass.
- Preserve package boundaries: core algorithms in `shapiq`, problem definitions in `shapiq_games`,
  dataset/model loading and benchmark orchestration in `shapiq_benchmark`.
- Preserve scalar game-value/coalition shape contracts, baseline/index semantics and deterministic
  seeded evaluation independent of batching, call order and cache state. Keep optional backends
  optional at import time; reuse lazy dispatch and informative missing-extra errors.
- Add meaningful tests for features and regression tests for fixes. Prefer small analytical or
  exhaustive numerical oracles, explicit tolerances and existing fixtures. New regular tests must be
  offline; use temporary caches and fakes instead of downloading data or model weights.
- Commit only durable tests of supported behavior or regressions. Keep disposable harnesses and
  one-off setup probes in ignored `.local-checks/` or a temporary directory. Do not commit them or
  tests that merely mirror configuration or implementation details.
- Public API changes include exports, reference documentation and runnable small examples. Read the
  contributor guide's documentation checklist. Do not hand-edit generated API pages or `_version.py`.
- Keep data, credentials, generated output and training artifacts out of source control. Preserve
  native build flags and OpenMP safeguards unless the task specifically changes those mechanisms.

## Verification and tools

- Initialize each checkout/worktree using the contributor guide's bootstrap command before
  development. Use a maintained main-checkout environment for installed Git hooks; do not remove it while using them.
- Before finishing code changes, run relevant focused tests and
  `uv run --no-sync python build_tools/dev.py check`. Report exact commands, failures and skipped optional-backend coverage. Never claim an unrun check
  passed. Hooks and required CI are the quality gate; do not disable them to complete ordinary work.
- For library/framework/SDK/API/CLI/service usage, fetch current official documentation. Prefer
  Context7: resolve the library ID first, then query one concept at a time. Use official docs if
  Context7 has no suitable source. This is unnecessary for ordinary refactoring or business logic.
- Use project-scoped Context7 and existing GitHub tools or `gh` when available. Authentication
  stays user-local; never commit tokens. A missing MCP must not prevent local code search, testing or manual setup.
- Use `fork-workflow` for branch/worktree/PR operations and `probly-reuse` for uncertainty changes.
  Keep shared policy here; `CLAUDE.md` imports it. Avoid duplicating rules in separate agent files.
