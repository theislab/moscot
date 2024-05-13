# User guide

moscot is a toolbox which can solve a wide range of tasks in single-cell genomics building upon the concept of [Optimal Transport (OT)](https://en.wikipedia.org/wiki/Transportation_theory_(mathematics)).

moscot builds upon three principles:

- moscot applications are scalable. While traditional OT implementations are computationally expensive, moscot implements a wide range of solvers which can handle hunreds of thousands of cells.
- moscot supports OT applications across [multiple modalities](#multimodality)
- moscot offers a unified [user interface](#user-interface) and provides flexible implementations

## Problems

### Biological problems

#### Temporal data

```{eval-rst}
.. list-table::
   :widths: 15 100
   :header-rows: 1

   * - Problem
     - Description
   * - :mod:`moscot.problems.time.TemporalProblem`
     - Class for analyzing time-series single cell data based on :cite:`schiebinger:19`.
   * - :mod:`moscot.problems.time.LineageProblem`
     - Estimator for modelling time series single cell data based on :cite:`lange-moslin:23`.
```

#### Spatial data

```{eval-rst}
.. list-table::
   :widths: 15 100
   :header-rows: 1

   * - Problem
     - Description
   * - :mod:`moscot.problems.space.AlignmentProblem`
     - Class for aligning spatial omics data, based on :cite:`zeira:22`.
   * - :mod:`moscot.problems.space.MappingProblem`
     - Class for mapping single cell omics data onto spatial data, based on :cite:`nitzan:19`.
```

#### Spatiotemporal data

```{eval-rst}
.. list-table::
   :widths: 15 100
   :header-rows: 1

   * - Problem
     - Description
   * - :mod:`moscot.problems.cross_modality.TranslationProblem`
     - Class for analyzing time series spatial single-cell data.
```

#### Multimodal data

```{eval-rst}
.. list-table::
   :widths: 15 100
   :header-rows: 1

   * - Problem
     - Description
   * - :mod:`moscot.problems.spatiotemporal.SpatioTemporalProblem`
     - Class for integrating single-cell multi-omics data, based on :cite:`demetci-scot:22`.
```

### Generic problems

```{eval-rst}
.. list-table::
   :widths: 15 100
   :header-rows: 1

   * - Problem
     - Description
   * - :mod:`moscot.problems.generic.SinkhornProblem`
     - Class for solving a :term:`linear problem`.
   * - :mod:`moscot.problems.generic.GWProblem`
     - Class for solving a :term:`Gromov-Wasserstein` problem.
   * - :mod:`moscot.problems.generic.FGWProblem`
     - Class for solving a :term:`fused Gromov-Wasserstein` problem.
```

## Scalability

In their original formulation, OT algorithms don't scale to large datasets du to their high computational complexity. Moscot overcomes this limitation by allowing for the use of low-rank solvers. in each `solve` method we have the rank parameter, by default $-1$. Whenever possible,it's best to start with the full rank, but when needed, the rank should be set to a positive integer. The higher the rank, the better the full-rank approximation. Hence, one should start with a reasonable high rank, e.g. $5000$. Consecutively decrease the rank if needed due to memory constraints. Note that the scale of $\tau_a$ and $\tau_b$ change whenever we are in the low-rank setting. while they should be still between $0$ and $1$, empirically they should be set in the range between $0.1$ and $0.5$. See [below](#hyperparameters) for a more detailed discussion and {doc}`/notebooks/examples/solvers/100_linear_problems_basic` and {doc}`/notebooks/examples/solvers/300_quad_problems_basic` on how to use low-rank solutions.

## Multimodality

All moscot problems are in general applicable to any modality, as the solution of the moscot problem only depends on pairwise distances of cells. Yet, it is up to the users to apply the preprocessing. We recommend using embeddings, e.g. [scVI-tools](https://docs.scvi-tools.org/en/stable/index.html) based or linear embeddings of dimension $10-100$. On how to pass certain embeddings please have a look at {doc}`/notebooks/tutorials/600_tutorial_translation`.
When working with multiple modalities, we can construct a joint space, e.g. by using VAEs incorporating multiple modalities ([MultiVI](https://docs.scvi-tools.org/en/stable/user_guide/models/multivi.html)), or by concatenating linear embeddings (e.g. concatenate PCA and LSI space of GEX and ATAC, respectively)

## User interface

moscot problems implement problem-specific downstream methods, so we recommend to use task-specific moscot problems. Yet, we also offer [generic solvers](#generic-problems) with a limited range of downstream applications for more advanced users, which allow for more flexiblity.

## Hyperparameters

moscot problems' `solve` methods have the following parameters that can be set depending on the specific task:

- $\alpha$ - Parameter in $(0, 1]$ that interpolates between the {term}`quadratic term` and the {term}`linear term`. $\alpha = 1$ corresponds to the pure {term}`Gromov-Wasserstein` problem while $\alpha \to 0$ corresponds to the pure {term}`linear problem`.
- $\tau_a$ and $\tau_b$ - Parameters in $(0, 1]$ that define how {term}`unbalanced <unbalanced OT problem>` is the problem on the source and target {term}`marginals`. If $1$, the problem is {term}`balanced <balanced OT problem>`.
- $\varepsilon$ - {term}`Entropic regularization`.
- `rank` - Rank of the {term}`low-rank OT` solver {cite}`scetbon:21b`. If $-1$, full-rank solver {cite}`peyre:2016` is used.
