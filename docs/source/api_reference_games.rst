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

.. _games-probly:

Probly Uncertainty Games
------------------------

The optional ``shapiq[probly]`` extra connects probly representations to scalar games.
``ProblyPredictor`` separates representation building from quantification: its
``represent`` method preserves member and sample structure, while ``evaluate_outputs``
produces named scalar quantities. ``ProblyGame`` accepts coalition inputs;
``ImageProblyGame`` reuses the image games' region masks and fill images.
The same predictor can also be supplied to ``TabularLocalExplanation``.

Use ``for_output(name)`` to explain several quantities with shared model evaluations.
Each view has its own empty-coalition baseline and works with the existing exact
computers, approximators, explainers, and ``InteractionValues``.

The default quantifier returns total entropy for a single categorical distribution.
An unweighted sample of categorical distributions also provides the aleatoric and
epistemic components. Entropy uses natural logarithms unless ``base`` is specified.
An explicit, fixed ``class_index`` additionally exposes the probability of that class
under the categorical distribution or the mean of the sample. A custom quantifier
returning a mapping from names to scalar vectors supports other probly measures.
The selected probly revision does not consistently apply sample weights to entropy
components, so the default quantifier rejects weighted samples.

The ``shapiq[probly-torch]`` extra also installs Torch. For MC dropout, pass a probly
``Sampler`` and ``evaluation_context=torch_cpu_context(model)``. Scalar evaluation
uses the same seed for each singleton input and restores the CPU random state and
the model's training flags. Context factories for other backends can be supplied
explicitly; the built-in context supports CPU models.

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
