# Experiments

These rules extend the root `AGENTS.md` for exploratory Python scripts and notebooks.
See the [experiment guide](README.md) for commands and a cache example.

- Use a descriptive snake_case topic directory, such as `uncertainty_decomposition/`.
- Reuse public shapiq and probly APIs and existing dependencies before adding helpers. Keep
  scientific choices in the topic. Share small, proven mechanisms under `_shared/` and keep
  artifact paths, cache manifests and scientific validation with the calling experiment.
- Keep the research question, environment, random seeds, data source, and limitations in the topic's
  README or notebook Markdown cells. Use repository-relative paths. Keep the parameters, stages,
  and interpretation visible in the notebook. Put long plotting or export routines in named
  topic modules when that makes the notebook easier to follow.
- Choose local tests and static checks that serve the experiment. Put tests under `<topic>/tests/`
  and validate shared utilities through the topic that uses them. Run experiment checks explicitly.
  The package suite, shared hooks, and CI exclude `experiments/`.
- Run a notebook explicitly when its results help verify the change. Before execution, state the
  material data, dependencies, expected runtime, and downloads. Reuse the project environment when
  possible and document additional dependencies in the topic.
- Commit source notebooks with outputs, execution counts, and widget state cleared. Keep credentials,
  private data, and generated artifacts local. Use the topic's ignored `data/`, `models/`, and
  `outputs/` directories for data, weights, and executed notebook copies.
- Move reusable, supported behavior into the appropriate production package with typed APIs,
  documentation, and package tests for that behavior.
