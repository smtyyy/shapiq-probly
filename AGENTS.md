# Working in shapiq-probly

This is the shared instruction file for Codex and Claude Code. Read
[the contributor guide](.github/CONTRIBUTING.md) for setup, commands and the code map.
Read [the experiment instructions](experiments/AGENTS.md) when working under `experiments/`.

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
- Follow the package's Ruff and `ty` settings, including 100-column formatting, Python 3.12
  minimum, typed public APIs, and Google-style English docstrings.
- Preserve package boundaries: core algorithms in `shapiq`, problem definitions in `shapiq_games`,
  dataset/model loading and benchmark orchestration in `shapiq_benchmark`.
  Exploratory workflows and their shared utilities live under `experiments/`. Packages and
  package tests must be independent of experiment code.
- Preserve scalar game-value/coalition shape contracts, baseline/index semantics and deterministic
  seeded evaluation independent of batching, call order and cache state. Keep optional backends
  optional at import time; reuse lazy dispatch and informative missing-extra errors.
- Add meaningful tests for supported package features and regression tests for package fixes under
  `tests/`. Prefer small analytical or exhaustive numerical oracles, explicit tolerances and
  existing fixtures. New package tests must run offline with temporary caches and fakes.
- Commit only durable tests of supported behavior or regressions. Keep disposable harnesses and
  one-off setup probes in ignored `.local-checks/` or a temporary directory. Do not commit them or
  tests that merely mirror configuration or implementation details.
- Public API changes include exports, reference documentation and runnable small examples. Read the
  contributor guide's documentation checklist. Do not hand-edit generated API pages or `_version.py`.
- Keep data, credentials, generated output and training artifacts out of source control. Preserve
  native build flags and OpenMP safeguards unless the task specifically changes those mechanisms.

## Verification and tools

- Initialize each checkout/worktree using the contributor guide's bootstrap command. Keep the
  main checkout's environment available for the installed Git hooks.
- For package changes, run relevant focused tests and `uv run --no-sync python build_tools/dev.py check`.
  Report commands, failures, and skipped optional backends. Use the required hooks and CI for package
  verification. Follow the experiment instructions for opt-in checks under `experiments/`.
- For library/framework/SDK/API/CLI/service usage, fetch current official documentation. Prefer
  Context7: resolve the library ID first, then query one concept at a time. Use official docs if
  Context7 has no suitable source. This is unnecessary for ordinary refactoring or business logic.
- Use project-scoped Context7 and existing GitHub tools or `gh` when available. Authenticate
  locally and keep credentials out of Git.
- Use `fork-workflow` for branch, worktree, and PR operations. Keep shared policy here so the
  `CLAUDE.md` import gives both agents the same instructions.
