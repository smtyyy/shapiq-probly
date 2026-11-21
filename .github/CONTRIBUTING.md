# Contributing to shapiq-probly

This fork extends shapiq with uncertainty representations and quantification supplied by probly.
Humans, Codex and Claude Code share [AGENTS.md](../AGENTS.md), the existing shapiq tools and this
workflow. The [code of conduct](CODE_OF_CONDUCT.md) still applies.

## Set up a checkout

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git, then clone the fork:

```sh
git clone https://github.com/smtyyy/shapiq-probly.git
cd shapiq-probly
gh repo set-default smtyyy/shapiq-probly
uv run --no-project --python 3.12 python build_tools/dev.py setup
```

`gh` is optional; if used, authenticate personally and set its default to this fork. GitHub can
otherwise infer a fork's parent as the PR base. Always check the base repository in a browser PR.
An `upstream` remote is unnecessary for our current work.

The setup helper creates a local Python 3.12 environment, installs the locked development
dependencies and configures local Git defaults and hooks. Run it in every checkout/worktree.
Hooks use the main checkout's durable environment: keep that environment available even while
developing in a worktree. Nothing installs globally, and environments/credentials are not committed.

Building shapiq requires a C++17 compiler. On macOS, install Xcode Command Line Tools and a static
OpenMP runtime (`libomp.a`, e.g. Homebrew `libomp`); `LIBOMP_PREFIX` can select another prefix with
`include/omp.h` and `lib/libomp.a`. On Linux use the distribution's compiler/OpenMP packages; on
Windows use the Visual C++ build tools. See `setup.py` for the existing platform behavior.

## One task, one branch, one worktree

Keep the main checkout on `main`; update it with `git pull --ff-only`. Create each task from the
fork's current main without tracking `origin/main` as the feature branch's upstream:

```sh
git fetch origin
git worktree add --no-track -b feature/my-topic .worktrees/my-topic origin/main
cd .worktrees/my-topic
uv run --no-project --python 3.12 python build_tools/dev.py setup
```

Use different names for concurrent tasks. Only one person/agent writes to a branch/worktree at a
time. Do not stage, reset or clean another task's files. The setup helper must run inside the new
worktree so it gets its own environment. Do not place one task's build/test artifacts in another.

Use focused commits with concrete English subjects such as `feat(uncertainty): adapt probly samples`
or `test(games): cover coalition batch invariance`. This is a convention, not a commitlint service.
Include tests and documentation with the behavior they explain; avoid unrelated mass formatting.

## Local gates and pull requests

Run these commands from your active worktree after setup:

```sh
uv run --no-sync python build_tools/dev.py quality
uv run --no-sync python build_tools/dev.py test
uv run --no-sync python build_tools/dev.py check
```

`quality` runs the shared formatting, lint, and type checks. `test` runs the package suite under
`tests/`. Some inherited tests may download small datasets. `check` combines quality checks and the
package suite for the pre-push gate. For a fast edit loop use
`uv run --no-sync pytest path/to/test.py`. Run `check` before publishing. Inspect formatting hook
changes before staging them and rerunning the check. Set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
and `MPLBACKEND=Agg` when invoking tests directly to match the test helper.

The shared hooks and CI exclude `experiments/` from formatting, lint, type checks, and tests.
Experiments choose their own local checks and keep optional tests under `experiments/<topic>/tests/`.
The [experiment guide](../experiments/README.md#run-local-tests) shows the local commands and
separate VS Code workspace.

Push a new feature branch and open the PR explicitly in this fork:

```sh
git push --set-upstream origin HEAD
gh pr create --repo smtyyy/shapiq-probly --base main
```

The PR must explain the concrete behavior, reuse decisions and verification. `main` requires a PR,
passing required checks and at least one approval by another human. New commits invalidate stale
approval. Resolve review conversations. Only rebase merging is enabled so main stays linear;
do not use a merge commit to update a feature branch. The owner's bypass is available for an
explicit exceptional decision, not for routine agent automation or a failing local check.

CI runs on PR changes, not ordinary branch pushes. Follow-up commits on an open PR must rerun CI
because an earlier green check does not establish that the new code passes. Concurrency cancels
superseded runs. Heavy downloads, full documentation/example execution and release jobs are
opt-in rather than part of every feature push. Review a PR's reported skips: an absent optional
backend is not evidence that backend passed.

### Safely update a published branch

Only rebase a published `feature/*` branch with a clean worktree. The guarded subshell below
rejects `main`, detached HEAD and other branches, as well as staged, unstaged or untracked changes,
before fetching or rebasing. It captures the exact remote revision and verifies local HEAD
contains it; if it does not, coordinate with the other writer instead of overwriting their work:

```sh
(
    set -eu
    branch=$(git branch --show-current)
    case "$branch" in
        feature/*) ;;
        *) printf '%s\n' 'Refusing: switch to a feature/* branch first.' >&2; exit 1 ;;
    esac
    if [ -n "$(git status --porcelain)" ]; then
        printf '%s\n' 'Refusing: resolve staged, unstaged and untracked changes first.' >&2
        exit 1
    fi
    git fetch origin
    expected=$(git rev-parse "refs/remotes/origin/$branch")
    git merge-base --is-ancestor "$expected" HEAD
    git rebase origin/main
    uv run --no-sync python build_tools/dev.py check
    git push --force-with-lease="refs/heads/$branch:$expected" origin "HEAD:refs/heads/$branch"
)
```

The subshell stops at any error without closing your terminal. Resolve a rebase conflict or abort
it deliberately before retrying. The explicit OID lease rejects an update if someone pushes after
the fetch, including when background fetches update local remote-tracking refs. Never replace it
with `--force` or an unqualified lease. Resolve remote changes by fetching and coordinating; do not
retry with a fresh lease merely to make the rejection disappear.

## Reuse map and scientific contracts

| Concern | Existing entry points | Preserve |
| --- | --- | --- |
| Core game/explanation machinery | `src/shapiq/game.py`, `src/shapiq/explainer`, `src/shapiq/imputer` | Coalition evaluation, masking, normalization, index/baseline semantics |
| Problem definitions | `src/shapiq_games` | Caller supplies data/models; do not load benchmark datasets here |
| Existing uncertainty game | `src/shapiq_games/uncertainty.py` | Random-forest total/aleatoric/epistemic entropy; current base is 2 |
| Generic local game | `src/shapiq_games/local_xai.py` | Scalar prediction callables, seeded imputation, batch/call-order invariance |
| Shapes | `src/shapiq/typing.py` | Boolean `(n_coalitions, n_players)` coalitions, float `(n_coalitions,)` values |
| Benchmark/data orchestration | `src/shapiq_benchmark` | Cached dataset/model setup is separate from games |
| Optional backends | `src/lazy_dispatch`, `shapiq.utils.modules` | Import core without installing unrelated ML backends |

Before implementing uncertainty logic, search the selected probly revision for public
`predictor.predict`, `representation.sample.Sample` / `create_sample`, distribution factories,
`representer.Sampler`, and `quantification.decompose` / `measure` / `quantify`. Probly already
provides entropy/variance and total/aleatoric/epistemic decompositions; shapiq already has coalition
algorithms. Prefer a thin integration between them. Document why new logic is required.

The optional `probly` extra uses the Git revision pinned in [pyproject.toml](../pyproject.toml) and
resolved in [uv.lock](../uv.lock). Agree a reproducible revision before changing the pin. Use
portable dependency sources in committed configuration. Shapiq's 100-column formatting applies
here. Keep Torch and JAX optional at import time. A public package upload requires a dependency
that the package index can resolve.

For package changes, test uncertainty sample axes, weights, log base, deterministic seeds, finite
scalar outputs and decomposition identities. Add new package games to
`tests/shapiq_games/test_contract.py` so repeated calls, fresh instances, coalition batches/orders
and empty-coalition normalization are checked. Use small closed-form or brute-force oracles as in
`tests/shapiq_benchmark/test_chain_of_trust.py`, with justified floating-point tolerances. Commit
durable package tests of supported behavior and regressions. Keep one-off setup probes and
disposable harnesses in ignored `.local-checks/` or a temporary directory.

Keep new regular tests offline; inherited tests are not all offline yet. Reuse fakes/fixtures from
`tests/shapiq_games/helpers.py` and `tests/shapiq/fixtures`, `tmp_path` and temporary `SHAPIQ_DATA_DIR` caches. Download tests in
`tests/shapiq_benchmark/test_heavy.py` explicitly require `SHAPIQ_RUN_HEAVY_TESTS=1`.
Never write downloaded datasets, model weights or training output under `src/`.

## Documentation and packaging

- Public functions/classes have typed signatures and Google-style English docstrings with small
  runnable examples. Add their public exports and an entry in the relevant API reference page.
  Existing game/benchmark tests execute docstrings and check documented `__all__` names.
- Explain user-visible changes in `CHANGELOG.md` when relevant. Cite scientific methods with
  `:cite:` and `docs/source/references.bib`; the existing bibliography hook maintains its format.
- Use the existing Sphinx reference/tutorial/example structure. Generated API pages, galleries,
  build output and `src/shapiq/_version.py` are not hand-maintained source files.
- Keep native C++/OpenMP flags and NaN routing safeguards intact. Changes to native code/build
  configuration require relevant platform build/import checks; ordinary Python adapters do not
  require rebuilding a release-wheel matrix.
- Inspect package contents when changing packaging. Development configuration, credentials,
  caches and test/download artifacts must not leak into distributed wheels/sdists.

Choose the maintained source location rather than editing generated Sphinx output:

| Change | Documentation source |
| --- | --- |
| Core public API/export | `docs/source/api_reference.rst` and the implementation docstring |
| Game public API/export | `docs/source/api_reference_games.rst` and the implementation docstring |
| Benchmark/setup public API/export | `docs/source/api_reference_benchmark.rst` and the implementation docstring |
| Concept or method explanation | Relevant chapter in `docs/source/introduction/` |
| Runnable worked example | Relevant script under `examples/`, exposed by `docs/source/examples.rst` |
| Scientific citation | `docs/source/references.bib` plus the citation at its use |
| Contributor/tooling policy | Root `AGENTS.md` and this contributor guide |

For the full existing documentation build:

```sh
uv run --no-sync python build_tools/dev.py setup --docs
uv run --no-sync python build_tools/dev.py docs
```

This is the full Sphinx HTML build with warnings treated as errors. It executes examples and may
download models/data, so run it deliberately when documentation/example behavior needs validation;
it is not a cheap pre-commit hook. Inspect the rendered pages for layout and working links.

## Codex, Claude Code and MCP

Open the repository/worktree itself as the project, rather than its parent course directory.
In VS Code, install the workspace's recommended Python, Ruff and ty extensions and select this
worktree's `.venv` interpreter. Shared settings enable formatting and import fixes on save and use
the environment's pinned tools. The `shapiq: quality`, `test`, `check before push` and `docs` tasks
call the same helper as the terminal/CI. Bootstrap a new environment from the terminal first.
The workspace uses the Python extension's environment manager. The separate Python Environments
extension can currently pass a `.venv` directory as an executable or incomplete Python-version
metadata to the pinned language tools, so it is disabled for this project only. Reassess this
workaround when updating those integrations. The portable interpreter default selects this
checkout's `.venv`; if an older selection persists, use **Python: Select Interpreter** to select
the actual executable (`.venv/bin/python` on macOS/Linux, `.venv/Scripts/python.exe` on Windows),
then reload the window.
Test Explorer loads `.vscode/test.env` to use the same native-thread limits and headless plotting
as the test helper.
Other editors should use this worktree's interpreter and the existing `pyproject.toml` rules.

`AGENTS.md` is the shared repository policy. Codex discovers it and the skills in `.agents/skills`.
Claude Code reads the same policy through `CLAUDE.md` and uses the wrappers in `.claude/skills`.
For work under `experiments/`, read the [scoped experiment rules](../experiments/AGENTS.md). The
scoped `CLAUDE.md` imports those rules when Claude Code works in that directory.

Trust the project configuration in Codex and authenticate Context7 with `codex mcp login context7`.
In Claude Code, approve the project MCP server from `.mcp.json` and authenticate in the MCP
interface. Keep credentials, personal preferences, and machine paths in local settings.

The shared MCP configuration provides Context7 for current documentation. Use an existing
GitHub connector or authenticated `gh` for fork operations; no extra GitHub MCP is installed.
Authenticate individually; tool availability does not grant authority to merge, bypass, publish or change remote policy. Read/search only the repository needed for the task.
Use the local tools/official docs if an MCP connection is unavailable.

For library/framework/SDK/API/CLI/service questions, resolve the Context7 library ID first, then
query current documentation one concept at a time. Source retrieval is not needed for a routine
refactor or general programming concept. Refer to official product guidance when maintaining
the integrations:

- [Codex AGENTS.md discovery](https://developers.openai.com/codex/guides/agents-md)
- [Codex skills](https://developers.openai.com/codex/skills)
- [Codex MCP](https://developers.openai.com/codex/mcp)
- [Claude Code memory/imports](https://code.claude.com/docs/en/memory)
- [Claude Code skills](https://code.claude.com/docs/en/skills)
- [Claude Code MCP](https://code.claude.com/docs/en/mcp)

Keep instructions concise and executable. Add a new skill/MCP only for a demonstrated workflow;
do not duplicate policy across tools or install a broad collection of unrelated services.
