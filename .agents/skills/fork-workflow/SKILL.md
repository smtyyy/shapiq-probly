---
name: fork-workflow
description: Work with feature branches, Git worktrees and reviewed pull requests inside the shapiq-probly fork; use for publishing or updating a task branch.
---

Read root `AGENTS.md` and the workflow sections of `.github/CONTRIBUTING.md` before Git mutations.

1. Inspect the worktree, branch, remotes and uncommitted files. The repository for all pushes/PRs is
   `smtyyy/shapiq-probly`. Preserve another task's changes. A task uses `feature/<topic>` with one
   writer; create a separate worktree when concurrent work would share a branch or files.
2. Set up the active worktree with the contributor guide's bootstrap command. Keep the main
   checkout's environment available for installed hooks. Do not develop on main or track origin/main as a feature branch's push upstream.
3. Run focused tests, then `uv run --no-sync python build_tools/dev.py check`. Use concrete English commits and
   include meaningful tests/docs with behavior changes. Report failures and skips accurately.
4. Publish a new branch with `git push --set-upstream origin HEAD`. Explicitly target the fork with
   `gh pr create --repo smtyyy/shapiq-probly --base main`; inspect the base repository before
   submitting through a browser. Creating a PR does not authorize merging or bypassing rules.
5. For a published branch rebase, follow the guide's guarded explicit-OID lease procedure. Reject
   a non-`feature/*` branch or staged/unstaged/untracked changes before fetching or rebasing. Fetch,
   capture the remote branch OID, verify it is an ancestor of HEAD, rebase on origin/main, rerun
   checks and push with that exact expected OID. Stop if ancestry/lease validation fails; do not refresh the
   expected OID merely to overwrite a collaborator's work.

Required human approval and CI apply before a rebase merge. An owner bypass requires an explicit
request for that exception; never use it as an automatic recovery from a failing check.
