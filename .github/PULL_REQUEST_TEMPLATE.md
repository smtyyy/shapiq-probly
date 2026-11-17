## Behavior and motivation

Describe the concrete problem and resulting behavior. Include public API changes, if any.

## Reuse

Which existing shapiq/probly APIs were reused? Explain any new implementation or dependency choice.

## Verification

List commands run and results, including relevant new/regression tests and any skipped backends.

- [ ] The PR base is `smtyyy/shapiq-probly:main`.
- [ ] `uv run --no-sync python build_tools/dev.py check` passes locally.
- [ ] Behavior changes have durable tests; numerical behavior has an independent oracle.
- [ ] Disposable harnesses/setup probes remain local; committed tests do not merely mirror config.
- [ ] Public API/reference/examples and `CHANGELOG.md` are updated where relevant.
- [ ] Optional dependencies, deterministic game contracts and data/cache isolation are preserved.

Request approval from another human before merging. Agent reviews supplement that review.
