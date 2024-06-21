# User guide

Moscot is a toolbox which can solve a wide range of tasks in single-cell genomics building upon the concept of [Optimal Transport (OT)](<https://en.wikipedia.org/wiki/Transportation_theory_(mathematics)>).

Moscot builds upon three principles:

- moscot applications are scalable. While traditional OT implementations are computationally expensive, moscot implements a wide range of solvers which can handle hundreds of thousands of cells
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
   * - :mod:`moscot.problems.spatiotemporal.SpatioTemporalProblem`
     - Class for analyzing time series spatial single-cell data.
```

#### Multimodal data

```{eval-rst}
.. list-table::
   :widths: 15 100
   :header-rows: 1

   * - Problem
     - Description
   * - :mod:`moscot.problems.cross_modality.TranslationProblem`
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

In their original formulation, OT algorithms don't scale to large datasets due to their high computational complexity. Moscot provides several options to overcome this limitation.\
For {term}`linear problem`s we can specify the `batch_size` parameter of the `solve` method. It determines the number of rows or columns of the cost matrix to materialize during the {term}`Sinkhorn` iterations. Smaller `batch_size` reduces memory complexity, but slightly increases time complexity.\
Whenever time complexity in a {term}`linear problem` (e.g. {class}`moscot.problems.time.TemporalProblem`) should be reduced, or memory/time complexity in a {term}`quadratic problem` should be reduced, we use {term}`low-rank OT`.
In each `solve` method we have the `rank` parameter, by default $-1$ -- the full rank.
Whenever possible, it's best to start with the full rank, but when needed, the rank should be set to a positive integer. The higher the rank, the better the full-rank approximation. Hence, one should start with a reasonable high rank, e.g. $5000$. Consecutively decrease the rank if needed due to memory constraints. Note that the scale of $\tau_a$ and $\tau_b$ changes whenever we are in the low-rank setting. While they should be still between $0$ and $1$, empirically they should be set in the range between $0.1$ and $0.5$. See {doc}`/notebooks/examples/solvers/100_linear_problems_basic` and {doc}`/notebooks/examples/solvers/300_quad_problems_basic` on how to use low-rank solutions.\
See [below](#hyperparameters) for a more detailed discussion.

## Multimodality

All moscot problems are in general applicable to any modality, as the solution of the moscot problem only depends on pairwise distances of cells. Yet, it is up to the users to apply the preprocessing. We recommend using embeddings, e.g. [scVI-tools](https://docs.scvi-tools.org/en/stable/index.html) based or linear embeddings ([PCA for GEX](https://muon-tutorials.readthedocs.io/en/latest/single-cell-rna-atac/pbmc10k/1-Gene-Expression-Processing.html) and [LSI for ATAC-seq data](https://muon-tutorials.readthedocs.io/en/latest/single-cell-rna-atac/pbmc10k/2-Chromatin-Accessibility-Processing.html)) of dimension $10-100$.
When working with multiple modalities, we can construct a joint space, e.g. by using VAEs incorporating multiple modalities ([MultiVI](https://docs.scvi-tools.org/en/stable/user_guide/models/multivi.html)), or by concatenating linear embeddings (e.g. concatenate PCA and LSI space of GEX and ATAC, respectively)

## User interface

Moscot problems implement problem-specific downstream methods, so we recommend to use task-specific moscot problems. Yet, we also offer [generic solvers](#generic-problems) with a limited range of downstream applications for more advanced users, which allow for more flexibility.

## Hyperparameters

The `solve` method of moscot problems has a wide range of parameters. In the following, we discuss the most relevant ones:

- $\varepsilon$ - {term}`Entropic regularization`. This determines the stochasticity of the map. The higher the $\varepsilon$, the more stochastic the map is.
- $\tau_a$ and $\tau_b$ - Parameters in $(0, 1]$ that define how {term}`unbalanced <unbalanced OT problem>` is the problem on the source and target {term}`marginals`. The lower the $\tau$, the more {term}`unbalanced <unbalanced OT problem>` the problem. Unbalancedness allows to automatically discard outliers, compensate for undesired distributional shifts, and model cell proliferation and apoptosis. If $\tau = 1$, the problem is {term}`balanced <balanced OT problem>`.
- $\alpha$ (only in problems building upon {term}`fused Gromov-Wasserstein`) - Parameter in $(0, 1]$ that interpolates between the {term}`quadratic term` and the {term}`linear term`. $\alpha = 1$ corresponds to the pure {term}`Gromov-Wasserstein` problem while $\alpha \to 0$ corresponds to the pure {term}`linear problem`.
- `batch_size` - Number of rows/columns of the cost matrix to materialize during the solver iterations. Larger value will require more memory. See above the [](#scalability).
- `rank` - Rank of the {term}`low-rank OT` solver {cite}`scetbon:21b`. If $-1$, full-rank solver {cite}`peyre:2016` is used. See above the [](#scalability).

For more hyperparameters and their usage please refer to {doc}`/notebooks/examples/solvers/100_linear_problems_basic`, {doc}`/notebooks/examples/solvers/200_linear_problems_advanced`, {doc}`/notebooks/examples/solvers/300_quad_problems_basic` and {doc}`/notebooks/examples/solvers/400_quad_problems_advanced`.

## Further links

For tutorials showcasing use cases for data analysis please see {doc}`/notebooks/tutorials/index`.\
For short examples showcasing core moscot functionality please refer to {doc}`/notebooks/examples/index`.
