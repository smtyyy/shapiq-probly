# Experiments

Use this directory for exploratory work on uncertainty-aware Shapley values and interactions.
Small Python scripts and Jupyter notebooks are both welcome. Supported library behavior lives in
`src/` and its tests in the repository's `tests/` directory.

Create a topic directory with a snake_case name, for example `uncertainty_decomposition/`. Include
the experiment source and a short README describing the question, how to run it, and its findings.
Add a notebook, helper module, or local tests when they serve the topic.

## Run an experiment

Start in your own feature branch and worktree. Run the project setup there:

```bash
uv run --no-project --python 3.12 python build_tools/dev.py setup
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
when possible. Use a separate documented environment for a topic that needs a conflicting stack.

## Keep research reproducible

- Record the question, relevant package versions, random seeds, dataset source, parameters, and
  important hardware assumptions. Distinguish observed results from interpretations.
- Use existing shapiq and probly implementations wherever they fit. Document the reason for any
  new algorithm. Keep the experimental choices visible in the notebook and use named modules for
  long plotting and export routines.
- Use repository-relative paths. Keep downloaded data in `data/`, model weights in `models/`, and
  generated plots or results in `outputs/` inside the topic. These local artifacts stay out of Git.
- Commit the experiment source and documentation. Keep secrets, private data and generated
  artifacts local. Clear notebook outputs, execution counts and widget state before committing.

## Reuse a Torch cache

Use [`load_or_compute`](_shared/torch_cache.py) to cache tensors and primitive metadata under the
topic's `outputs/` or `models/` directory. For example, a topic that has computed an input digest
and selected an encoder can use this pattern.

```python
from experiments._shared.torch_cache import load_or_compute

payload, cache_hit = load_or_compute(
    topic / "outputs" / "features.pt",
    {"schema": 1, "inputs": input_digest, "encoder": encoder_name},
    lambda: {"features": compute_features()},
)
```

The helper loads a matching payload on the CPU with `weights_only=True`. It computes and atomically
writes a payload when the file is missing or its manifest differs. Define the manifest in the topic
using inputs, settings and versions that affect the result, and check the loaded values against the
experiment's expectations. Keep artifact paths and scientific validation with the calling topic.

## Run local tests

Put optional tests in `experiments/<topic>/tests/` and run them explicitly from the repository root.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg uv run --no-sync pytest experiments/<topic>/tests
```

In VS Code, open the [Experiments workspace](../.vscode/experiments.code-workspace) to discover
tests in each topic's `tests/` directory. The Test Explorer refreshes when you save a test file.
The workspace uses the worktree's `.venv` and the same test environment as the standard workspace.
The regular VS Code workspace and root pytest discovery run the package suite. Shared hooks and CI
exclude the experiment files.

## Optional local checks

Choose the checks that help the topic. To opt into static checks, pass the topic directory to Ruff
or the relevant source files to ty. For example, from the repository root, run the following for an
experiment located at `experiments/my_topic`.

```bash
uv run --no-sync ruff check experiments/my_topic
uv run --no-sync ruff format --check experiments/my_topic
uv run --no-sync ty check experiments/my_topic/analysis.py
```

Execute notebooks explicitly when the results are needed. Before committing, save any executed
copy you want to keep under the topic's ignored `outputs/` directory, then strip the source notebook.

```bash
uvx nbstripout@0.9.1 experiments/my_topic/analysis.ipynb
```

Record local verification and any data, weights, or downloads needed to reproduce it in the PR.

## Promotion to the library

When a result becomes a supported feature, move the reusable behavior into `src/`, add public
documentation and deterministic tests under `tests/`, and run
`uv run --no-sync python build_tools/dev.py check`.
