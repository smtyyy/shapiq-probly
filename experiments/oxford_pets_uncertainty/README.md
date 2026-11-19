# Oxford Pets: uncertainty games with probly

The [notebook](oxford_pets_uncertainty.ipynb) explains which image regions change a classifier's
prediction and uncertainty. It uses the supported `ProblyPredictor`, `ProblyGame`, and
`ImageProblyGame` APIs rather than defining uncertainty measures or cooperative games locally.
It runs independently of the original notebooks in the probly fork.

## Run

Use the feature worktree's Python 3.12 environment and follow the repository
[contributor setup](../../.github/CONTRIBUTING.md). The setup helper installs the development
environment and optional extras, including the pinned `probly` source, Torch, torchvision,
scikit-learn, scikit-image, Pillow, Matplotlib, NumPy, and Jupyter. No installation runs inside
the notebook. Superpixel segmentation uses scikit-image from the existing `games` extra.
The experiment uses the official `pwhofman/probly` revision pinned in
[pyproject.toml](../../pyproject.toml).

```bash
uv run --no-project --python 3.12 python build_tools/dev.py setup
uv run --no-sync jupyter lab
```

Open this notebook with the worktree's `.venv` kernel. Execute cells in order. A fresh run downloads
about 0.8 GB of Oxford-IIIT Pet image archives and 0.5 GB of ImageNet VGG16 weights. Extracting
features for all 7,349 original images and up to 512 coalitions per selected image can take minutes to
hours depending on hardware. Allow several GB for unpacked data, weights, and feature caches.
Only six small classifier heads are trained; VGG is frozen throughout.

The configuration cell exposes these choices:

- With `COHORT_MODE = "demo"`, the notebook selects three contrasting predictions from the official test split.
  Set `COHORT_MODE = "seeded"` to select images by breed label and seed. In this mode,
  `Config(n_images=3)` shows three images and `n_images=37` shows one image per breed.
- `MASK_FILL` sets the reference for missing regions. The default `"blur"` uses a blurred copy.
- `REGION_MODE = "trimap"` (default) uses the official Oxford annotations. `"grid"` uses nine
  fixed rectangles, and `"superpixels"` asks shapiq's SLIC helper for up to nine regions.
- With `Config(feature_device="auto")`, the notebook tries CUDA, then Apple MPS, then CPU.
  It checks availability and a real encoder forward pass,
  without a performance benchmark. Explicit `"cpu"`, `"mps"`, and `"cuda:0"` choices remain
  available. Recognized backend-initialization failures in automatic mode fall back to CPU with
  an explanation; explicit requests fail clearly instead of silently selecting another device.
- Frozen extraction keeps float32 and an overridable batch size of four. The notebook reports
  available backends, requested and selected devices, dtypes, and CPU threads. Existing thread
  settings are respected. Training and uncertainty calculations use CPU; inference uses float64
  head outputs. Different feature devices can produce small numerical differences.
- `DOWNLOAD_DATA`: whether torchvision may download missing Oxford data. VGG weights are downloaded
  when absent from its selected cache.
- `DATA_ROOT` and `WEIGHTS_ROOT`: optional explicit local data and Torch Hub cache roots. Defaults
  are this topic's `data/` and `models/torch_hub/`. Loading VGG restores the previous Torch Hub
  cache path afterward.
- `EXTERNAL_FEATURES`: optional explicit `{"trainval": Path(...), "test": Path(...)}` frozen-feature
  files. This acceleration accepts the original Oxford tensor format only after checking image
  identities, classes, weights, preprocessing, labels, shapes, and finite float32 values. Their manifests
  and tensor digests are recorded in the result provenance. No external feature file is required.

Outputs stay in the topic's ignored `outputs/`, models in `models/`, and downloaded data in `data/`.
Grid results retain their paths. Trimap and superpixel coalition caches and results use
`outputs/trimap/` and `outputs/superpixels/`, respectively.
The [shared Torch cache](../_shared/torch_cache.py) reads and writes the `.pt` payloads. Oxford's
feature and head manifests determine when each cache is valid. Coalition feature caches also depend
on the pixels of the masked reference image. Head caches depend on scientific
training settings and feature content. Runtime information and source fingerprints are recorded
separately in the result provenance. Clear notebook outputs and execution counts before committing.

## Local verification

The notebook is the executable experiment. [`helpers.py`](helpers.py) prepares data, encoders and
heads. [`selection.py`](selection.py) chooses the images from the test split.
[`reporting.py`](reporting.py) renders figures and checks saved explanations. The topic's
[local tests](tests/) and optional lint run from the repository root.

```bash
uv run --no-sync ruff check experiments/oxford_pets_uncertainty
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg uv run --no-sync pytest experiments/oxford_pets_uncertainty/tests
```

The tests cover demo selection, device selection, and cache behavior with small CPU fixtures.
See the [experiment guide](../README.md#run-local-tests) for VS Code discovery. Run the notebook
explicitly to inspect figures, tables and exports. Keep the executed copy in `outputs/`.

## Experiment

Oxford-IIIT Pet supplies 37 breed labels. The official trainval split is stratified into 80%
training and 20% validation with seed 0. The official test split is used for the diagnosis after
model fitting. In `seeded` mode, `select_cases` chooses one candidate per breed in seed order
using the labels. The fitted predictions supply the displayed scores.

VGG16 with fixed `IMAGENET1K_V1` weights yields 4,096 features after its final classifier is removed.
The weights' resize, 224-by-224 center crop, and normalization are used without random augmentation.
All backbone layers remain frozen and in evaluation mode.

Public probly `subensemble(..., reset_params=True)` initializes three independent
`Dropout(0.5) -> Linear(4096, 37)` heads with initialization seed 0. CPU training seeds 0, 1, and 2
control head dropout and minibatch order. Each head uses AdamW, learning rate 0.001, weight decay
0.0001, batch size 64, at most 30 epochs, and patience 5. The lowest validation cross-entropy selects
its checkpoint. The second condition uses 20 images per breed sampled from training only with
subset seed 10; it retains the same validation split and settings.

The equivalent public probly head ensemble evaluates cached features. This avoids the normal
subensemble forward path recomputing the shared VGG for every head. MC Dropout uses probly's public
`Sampler` with 64 draws from head 0, activating its existing trained dropout without inserting a
second layer. Both methods compute uncertainty from CPU float64 head outputs. Seeded CPU evaluation
contexts make coalition calls independent of their order, batches, and cache state.

In `demo` mode, the full-training ensemble scores the cached features of all 3,669 original test
crops. Model confidence is the mean probability of the ensemble's predicted breed. Among correct
predictions, the demo shows the highest and lowest confidence. Among incorrect predictions, it
shows the highest confidence. Ties use ascending image ID. The overview names any missing cases,
including a missing low-confidence case when the correct predictions have no confidence contrast.
The labels *Sicher richtig*, *Unsicher richtig*, and *Sicher falsch* describe these predictions
from the reference ensemble. The notebook displays the true and predicted breeds and the model
confidence beside each crop. The original three-case design used the same ranking criterion within
a fixed 37-image cohort. See the [original shapiq notebook](https://github.com/sep-trustworthy-gruppe-d/probly/blob/dan/oxford-pets-vgg16/experiments/group-d_experiments/dan_02-2_oxford-pets-shapiq.ipynb).

In trimap mode, the selected image's annotation is resized and cropped with the same geometry
as the RGB image. Nearest-neighbor interpolation preserves Oxford's labels for pet, background,
and unclassified pixels. The unclassified label can cover the transition around the pet. Only
regions present in the crop become players. For example, the trimap of `wheaten_terrier_195`
marks the entire crop as background, yielding two coalitions and one Shapley value. SLIC instead
groups nearby pixels by similarity and can cross the pet boundary. All three modes use the same
selected images, predictions, and default `blur` reference for missing regions.
`ImageProblyGame.masked_image` creates coalition images for the selected crops. Their VGG features
are cached once and shared across all models and outputs. The full ensemble's unmasked prediction
fixes the probability target across coalitions and conditions. Raw games preserve the empty reference value:

- P: mean target-class probability.
- TU: entropy of the mean categorical distribution, in nats.
- AU: mean member entropy, in nats.
- EU: TU minus AU, in nats.

The notebook computes exact Shapley values and order-two k-SII when at least two players are
present. It checks efficiency and linearity of TU = AU + EU. It compares the direct image API
against the cached-feature game at the empty and full coalitions. The figures use shapiq's
`region_attribution_plot` for single-region contributions and `interaction_heatmap_plot` for
pairwise k-SII. The EU attribution scale is shared across displayed images and model runs within
each region mode. TU and AU share a separate scale, and probability contributions and pairwise EU
interactions each have their own scale. Each EU comparison reports the value at
the masked reference and at the fully visible image. The color scales follow shapiq's blue,
white, and red convention for negative, zero, and positive contributions. The signed EU labels
make small contributions readable on the shared scale.

`summary.json` records seeds, settings, package/source versions, feature provenance, the selection
mode and rule, case labels, image IDs, true and predicted breeds, model confidence, and scalar
results. It records the region mode and ordered player names. Superpixel runs also record the
requested number of regions and the scikit-image version. The notebook checks agreement between
selection confidence and the full ensemble's P game value for the fully visible image. NumPy
archives retain game values and the coalition/region definitions, including mode and player
names. Canonical `InteractionValues` JSON exports preserve explanation indices, baselines, and
interaction lookup metadata. The notebook reloads its exports to verify the saved results.

## Interpretation and limits

Positive uncertainty contributions increase uncertainty relative to this reference game. They do
not indicate useful regions. The entropy decomposition describes these fitted models; the data
does not supply ground-truth aleatoric or epistemic uncertainty. Shared-backbone ensembles cannot
express backbone variation, and agreement can coexist with confident errors. A center crop can omit
parts of the animal. Blur retains coarse information; another fill defines another game.
Different region modes also define different games. With the same image, model and fill, the
empty and full coalition values agree across modes. Region contributions refer to the players
of their own mode.

The three diagnostic cases are deliberately selected for contrast, and model probabilities are
uncalibrated. The comparison describes those individual predictions and their masked variants.
Less training data need not increase measured EU. Decomposition identities are consistency checks,
not empirical discoveries. Calibration, conformal prediction, deletion
studies, and approximation-budget experiments are outside this first integrated notebook.

For the design and interpretation of these figures, see
[Crameri et al. on scientific color scales](https://doi.org/10.1038/s41467-020-19160-7),
[Sundararajan and Najmi on the effect of the reference on Shapley explanations](https://proceedings.mlr.press/v119/sundararajan20b.html),
and [Wimmer et al. on interpreting entropy-based uncertainty components](https://proceedings.mlr.press/v216/wimmer23a.html).

The original final experiment was
`dan_02-3_oxford-pets-probly-shapiq.ipynb` at probly fork commit
`ac678f315452be0682e4f2404d1d2fa12e938b18`. This notebook adds MC Dropout and chooses its fixed P
target from the full ensemble; the original used the separate baseline head. Historical results
are therefore context, not a numerical reference for this newly fitted pipeline.

Sources: [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/),
[probly pipeline](https://pwhofman.github.io/probly/latest/guide/core_pillars.html),
[VGG16 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html),
[PyTorch MPS](https://docs.pytorch.org/docs/2.9/notes/mps.html),
[PyTorch reproducibility](https://docs.pytorch.org/docs/2.9/notes/randomness.html),
[Matplotlib constrained layout](https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html),
and [ty notebook support](https://docs.astral.sh/ty/features/language-server/#notebook-support).
