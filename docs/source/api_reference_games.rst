============
shapiq_games
============

.. grid:: 2 2 4 4
   :gutter: 2
   :class-container: shapiq-cards shapiq-cards-compact

   .. grid-item-card:: :octicon:`beaker` Synthetic
      :link: games-synthetic
      :link-type: ref
      :class-card: shapiq-card

      Values known in closed form

   .. grid-item-card:: :octicon:`workflow` Model-Specific
      :link: games-model-specific
      :link-type: ref
      :class-card: shapiq-card

      Trees, neighbors, kernels

   .. grid-item-card:: :octicon:`search` Explanation
      :link: games-explanation
      :link-type: ref
      :class-card: shapiq-card

      Local, global, uncertainty

   .. grid-item-card:: :octicon:`law` Valuation and Selection
      :link: games-valuation
      :link-type: ref
      :class-card: shapiq-card

      Data, features, ensembles

   .. grid-item-card:: :octicon:`dot-fill` Unsupervised
      :link: games-unsupervised
      :link-type: ref
      :class-card: shapiq-card

      Clusters and data

   .. grid-item-card:: :octicon:`git-branch` Causal
      :link: games-causal
      :link-type: ref
      :class-card: shapiq-card

      Confounding

   .. grid-item-card:: :octicon:`image` Image and Language
      :link: games-image-language
      :link-type: ref
      :class-card: shapiq-card

      Vision and text models

   .. grid-item-card:: :octicon:`tools` Utilities
      :link: games-utilities
      :link-type: ref
      :class-card: shapiq-card

      Helpers, constants, and types

.. automodule:: shapiq_games
   :no-members:

.. currentmodule:: shapiq_games

.. _games-synthetic:

Synthetic Games
---------------

Games whose interaction values are known in closed form.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   DummyGame
   UnanimityGame
   SOUM
   RandomTableGame

.. _games-model-specific:

Model-Specific Games
--------------------

Games of models with an exact explainer in ``shapiq``, so their values are known for many
players.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   PathDependentTreeGame
   InterventionalTreeGame
   KNNGame
   WeightedKNNGame
   ThresholdNNGame
   ProductKernelGame
   GraphGame

.. _games-explanation:

Explanation Games
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   TabularLocalExplanation
   TabularGlobalExplanation
   TabularUncertaintyExplanation

To explain a scalar derived from a probly representation, supply a :class:`ProblyPredictor`
as the model for :class:`TabularLocalExplanation`. See :ref:`games-probly`.

.. _games-probly:

Probly Uncertainty Games
------------------------

The optional ``probly`` extra connects probly representations to scalar games. See the
:doc:`installation instructions <introduction/installation>` for installing from source.

* :class:`ProblyPredictor` wraps a caller-supplied representer and optional preprocessing. Use
  ``represent`` to get the representation with its member or sample structure intact, and
  ``evaluate_outputs`` to get one scalar per input row for each available output. Model fitting,
  sampler construction, and input preparation are the caller's responsibility.
* :class:`ProblyGame` passes boolean coalition batches to the predictor. Supply ``preprocess``
  when the representer needs model inputs derived from the coalitions.
* :class:`ImageProblyGame` uses the image regions as players. It replaces absent regions with a
  fixed fill and passes masked RGB image batches to the predictor. The caller supplies the image,
  region labels, fill choice, and model preprocessing. See :ref:`probly-image-example` for a
  runnable example.

The built-in quantifier makes these outputs available.

.. list-table::
   :header-rows: 1
   :widths: 22 78
   :class: shapiq-table

   * - Output name
     - Meaning and availability
   * - ``"total"``
     - Entropy of a single categorical distribution or of the mean of an unweighted categorical
       sample, such as an ensemble's predictions.
   * - ``"aleatoric"``
     - Mean member entropy for an unweighted categorical sample.
   * - ``"epistemic"``
     - Total entropy minus mean member entropy for an unweighted categorical sample.
   * - ``"probability"``
     - Probability of a fixed ``class_index`` in the distribution or the sample mean. Supply
       ``class_index`` explicitly to enable it.

Entropy uses natural logarithms with ``base=None`` (nats). A custom quantifier can supply other
named scalar outputs. The built-in quantifier rejects weighted samples because the pinned
probly revision does not apply weights consistently to the entropy components.

Use ``for_output(name)`` to get a scalar game view compatible with the existing exact computers
and approximators. Each view has its own empty-coalition baseline. The default ``normalize=True``
subtracts that baseline from the named output. Use ``normalize=False`` for raw scores. In an
image game, the empty coalition replaces every region with the fill.

Both games use a bounded in-memory cache that shares scalar outputs across views of the same
game. Set ``cache_size=0`` to disable this memoization. Callers can cache model inputs or extracted
features separately in their preprocessing. For example, an experiment can load saved feature
vectors using a coalition as the lookup key for ``ProblyGame``.

For CPU MC dropout, pass a probly ``Sampler`` and
``evaluation_context=torch_cpu_context(model)``. Scalar evaluation uses the same seed for each
singleton input. The built-in context restores the CPU random state and the model's training
flags. Supply an explicit context factory for other backends.

These games explain a scalar uncertainty function. They do not yet estimate the
uncertainty of Shapley values themselves. That future task needs fixed model-member
or sample identities across all coalitions; model sampling, missing-feature
imputation, and Shapley approximation are distinct sources of randomness.
Keep supplied models and preprocessing fixed while a game or its cached views are
in use; construct a new game after training or changing the pipeline.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ProblyPredictor
   ProblyGame
   ImageProblyGame
   probly.torch_cpu_context

.. _games-valuation:

Valuation and Selection Games
-----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   FeatureSelection
   DataValuation
   DatasetValuation
   EnsembleSelection
   RandomForestEnsembleSelection

.. _games-unsupervised:

Unsupervised Games
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ClusterExplanation
   UnsupervisedData

.. _games-causal:

Causal Games
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   GlobalConfoundingXAI
   LocalConfoundingXAI

.. _games-image-language:

Image and Language Games
------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ImageClassifier
   ImageTextSimilarity
   SentimentAnalysis

To explain predictions or uncertainty over image regions using a probly representer, see
:class:`ImageProblyGame` under :ref:`games-probly` and the :ref:`worked image example
<probly-image-example>`.

.. _games-utilities:

Utilities
---------

The helpers and constants the games use, and the types of their arguments.

Helpers
~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   causal.tabpfn_regressor
   language.SENTIMENT_MODEL_ID
   unsupervised.total_correlation
   vision.get_superpixels
   vision.grid_regions

.. _games-types:

Types
~~~~~

.. automodule:: shapiq_games.typing
   :no-members:

.. currentmodule:: shapiq_games.typing

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Task
   PredictFunction
   Metric
   MetricName
   Loss
   LossName
   ImputerName
   Uncertainty
   PlayerSizes
   ClusterMethod
   ClusterScore
   ConfoundingMode
   MaskStrategy
   ImageModel
   ImageTextModel
   Fill
