## Behavior and motivation

Describe the concrete problem and resulting behavior. Include public API changes, if any.

## Reuse

Which existing shapiq/probly APIs were reused? Explain any new implementation or dependency choice.

## Verification

For package changes, list the commands and their results, including relevant regression tests and
any skipped backends. For experiments, record any opt-in local verification and the data or
downloads needed to reproduce it. Shared hooks and CI exclude experiment files.

- [ ] The PR base is `smtyyy/shapiq-probly:main`.
- [ ] Package changes pass `uv run --no-sync python build_tools/dev.py check` locally.
- [ ] Package behavior changes have durable tests. Numerical tests use an independent oracle.
- [ ] Disposable harnesses/setup probes remain local; committed tests do not merely mirror config.
- [ ] Public API/reference/examples and `CHANGELOG.md` are updated where relevant.
- [ ] Optional dependencies, deterministic game contracts and data/cache isolation are preserved.

Request approval from another human before merging. Agent reviews supplement that review.
