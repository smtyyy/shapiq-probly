---
name: probly-reuse
description: Find existing shapiq and probly uncertainty implementations before designing or changing an uncertainty integration, representation or decomposition.
---

Read root `AGENTS.md` and the reuse/scientific-contract sections of `.github/CONTRIBUTING.md`.

- Search existing public APIs, tests and examples in the selected dependency revision before
  adding code. Do not infer APIs from memory or silently add a probly dependency/version/path.
- Start in shapiq with `src/shapiq_games/uncertainty.py` (existing forest entropy game),
  `local_xai.py` (scalar callable and imputation), `src/shapiq/game.py` and `src/shapiq/typing.py`.
  Reuse games, masks, approximators and baseline/index conventions.
- In probly inspect `predictor.predict`, distribution factories, `representation.sample.Sample`
  and `create_sample`, `representer.Sampler`, and `quantification.decompose` / `measure` / `quantify`.
  Existing measures/decompositions cover entropy, variance and total/aleatoric/epistemic notions.
  Prefer an adapter connecting public surfaces over copied uncertainty algorithms.
- Check sample/class/row axes, weights, log base (shapiq's forest game uses base 2), seeded sampling
  and scalar game-value conversion explicitly. Keep optional Torch/JAX backends optional at import.
- For package changes, test a small independent numerical oracle plus decomposition identities,
  shape validation, repeated calls, coalition batching/order and fresh instances. Add new package
  games to the existing contract registry and use offline fixtures and temporary caches. For
  exploratory workflows, follow `experiments/AGENTS.md` for optional topic-local tests.
- Record reused entry points, justified new logic and dependency choice in the PR. Public API
  additions include exports, reference entries and runnable examples under shapiq formatting.

If a needed dependency source is unavailable, report what was inspected and what remains
unverified; do not substitute an unrelated local checkout or fabricate compatibility evidence.
