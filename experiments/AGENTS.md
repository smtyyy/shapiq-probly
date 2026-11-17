# Experiments

These rules extend the root AGENTS.md for exploratory Python scripts and notebooks.

- Use a descriptive snake_case topic directory, such as `uncertainty_decomposition/`.
- Reuse shapiq and probly APIs before adding exploratory helpers. Do not copy production algorithms.
- Keep the research question, environment, random seeds, data source, and limitations in the topic's
  README or notebook Markdown cells. Use repository-relative paths, not machine-specific paths.
- Exploratory code does not require production docstrings, type annotations, or a new test for every
  cell. Ruff formatting, correctness checks, and secret checks still apply. Do not disable them.
- Do not execute notebooks as part of routine formatting, linting, tests, or CI. A requested run should
  state its data, dependencies, runtime, and external downloads before execution when these matter.
- Clear notebook outputs and execution counts before committing. Do not commit embedded images,
  widget state, credentials, private data, downloaded datasets, model weights, or generated results.
  Store local artifacts in the topic's ignored `data/`, `models/`, or `outputs/` directory.
- Run the existing local quality gate before pushing. Keep experimental dependencies explicit; do
  not add them to shapiq's runtime dependencies just to run a notebook.
- Move reusable, supported behavior into the appropriate production package with typed APIs,
  documentation, and meaningful tests. An experiment is not a substitute for a regression test.
