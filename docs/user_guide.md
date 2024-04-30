# User guide

moscot is a toolbox which can solve a wide range of tasks in single-cell genomics building upon the concept of Optimal Transport.

moscot builds upon three principles:

- moscot applications are scalable. While traditional OT implementations are computationally expensive, moscot implements a wide range of solvers which can handle hunreds of thousands of cells.
- moscot supports OT applications across [multiple modalities](#multimodality)
- moscot offers a unified [user interface](#user-interface) and provides flexible implementations

## Problems

### Biological problems

```{eval-rst}
.. list-table::
   :widths: 15 100 25
   :header-rows: 1

   * - Problem
     - Description
     - Module
   * - :mod:`moscot.problems.time.TemporalProblem`
     - Class for analyzing time-series single cell data based on :cite:p:`schiebinger:19`
     - :module:`moscot.problems.time`
```

### Generic problems

```{eval-rst}
.. list-table::
   :widths: 15 100 25
   :header-rows: 1

   * - Problem
     - Description
     - Module
   * - :mod:`moscot.problems.generic.SinkhornProblem`
     - Class for solving a :term:`linear problem`.
     - :module:`moscot.problems.generic`
```

## Scalability

In their original formulation, OT algorithms don't scale to large datasets du to their high computational complexity. Moscot overcomes this limitation by allowing for the use of low-rank solvers. in each solve method we have the rank parameter, by default -1. Whenever possible,it's best to start with the full rank, but when needed, the rank should be set to a positive integer. The higher the rank, the better the full-rank approximation. Hence, one should start with a reasonable high rank, e.g. 5000. Consecutively decrease the rank if needed due to memory constraints. Note that the scale of $\tau_a$ and $\tau_b$ change whenever we are in the low-rank setting. while they should be still between 0 and 1, empirically they should be set in the range between 0.1 and 0.5. See [below](#hyperparameters) for a more detailed discussion.

## Multimodality

All moscot problems are in general aplciable to any modality, as the solution of the moscot problem only depends on pairwise distances of cells. Yet, it is up to the usre to apply the preprocessing. We recommend using embeddings, e.g. [scVI-tools](https://docs.scvi-tools.org/en/stable/index.html) based or linear embeddings of dimension 10-100. When wokring with multiple modalities, we can construct a joint space, e.g. by using VAEs incorporating mulitple modalities ([MultiVI](https://docs.scvi-tools.org/en/stable/user_guide/models/multivi.html)), or by concatenating linear embeddings (e.g. concatenate PCA and LSI space of GEX and ATAC, respectively)

## User interface

moscot problems implement problem-specific downstream methods, so we recommend the user to use those. We also offer [generic solvers](#generic-problems) with a limited range of downstream applications for more advanced users, which allow for more flexiblity.

## Hyperparameters

- $\tau_a$ and $\tau_b$
- $\varepsilon$
