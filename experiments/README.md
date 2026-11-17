# Experiments

Use this directory for exploratory work on uncertainty-aware Shapley values and interactions.
Small Python scripts and Jupyter notebooks are both welcome. Keep supported library behavior in
`src/`, regression tests in `tests/`, and the existing upstream examples in `examples/`.

Create a topic directory with a snake_case name, for example `uncertainty_decomposition/`. Include
the experiment source and a short README describing the question, how to run it, and its findings.
There is no requirement to create a notebook, test, or Python package for an empty topic.

## Run an experiment

Start in your own feature branch and worktree. Run the project setup there:

```bash
python build_tools/dev.py setup
```

The development environment includes Jupyter tooling. In VS Code, open that worktree as a folder and
select its `.venv` interpreter and notebook kernel. The recommended Ruff and ty extensions use the
project environment. If an interpreter was selected before setup, use **Python: Select Interpreter**
again; the shared default does not replace an existing personal selection.

Run scripts from the repository root so relative paths are reproducible:

```bash
uv run --no-sync python experiments/<topic>/<script>.py
uv run --no-sync jupyter lab
```

Document any additional dependency and its version before using it. Reuse the project environment
when possible. A topic needing a conflicting stack should have its own documented environment;
keep experiments out of shapiq's runtime dependencies.

## Keep research reproducible

- Record the question, relevant package versions, random seeds, dataset source, parameters, and
  important hardware assumptions. Distinguish observed results from interpretations.
- Use existing shapiq and probly implementations wherever they fit; explain any necessary new
  algorithm rather than copying existing implementations.
- Use repository-relative paths. Keep downloaded data in `data/`, model weights in `models/`, and
  generated plots or results in `outputs/` inside the topic. These local artifacts stay out of Git.
- Commit readable `.py`, `.ipynb`, and Markdown source only. Do not commit secrets, private data,
  embedded images, widgets, downloaded datasets, or generated results.
- Clear all notebook outputs and execution counts before committing. The local quality checks
  validate notebooks without executing their cells; they do not establish scientific correctness.

## Quality and promotion to the library

Run the shared local checks before pushing:

```bash
python build_tools/dev.py quality
```

Ruff still checks and formats Python scripts and notebook code. Only production documentation,
annotation, package-layout, and console-print requirements are relaxed here. Experimental cells do
not each need unit tests. When a result becomes a supported feature, move the reusable behavior into
`src/`, add its public documentation and meaningful deterministic tests, and run the full check:

```bash
python build_tools/dev.py check
```

Routine CI never executes notebooks or starts research training runs. Discuss expensive experiments
and external downloads explicitly in the pull request so another person can reproduce the result.
