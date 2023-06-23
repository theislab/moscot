References
==========

Bibliography
------------
.. bibliography::
    :cited:

Glossary
--------
.. glossary::

    OT
        An `optimal transport <https://en.wikipedia.org/wiki/Transportation_theory_(mathematics)>_` problem is defined as a matching task between distributions, e.g. sets of cells.

    low-rank OT
        `low-rank <https://en.wikipedia.org/wiki/Low-rank_approximation>`_` OT approximates full-rank :term:`OT`, which allows for faster computations and lower memory complexity :cite:`scetbon:21a,scetbon:21b,scetbon:22b,scetbon:23`. The :term:`transport matrix` will have a low rank.

    balanced OT problem
        :term:`OT` problem where the :term:`marginals` are fixed. Each data point (cell) of the source distribution emits a certain amount of mass given by the source :term:`marginals`, and each data point (cell) of the target distribution receives a certain amount of mass given by the target :term:`marginals`.

    unbalanced OT problem
        :term:`OT` problem where the :term:`marginals` are not fixed. If beneficial, a data point might emit or receive more or less mass than prescribed by the :term:`marginals`. The larger the unbalancedness parameters ``tau_a`` and ``tau_b``, the more the mass emitted, and received, respectively, can deviate from the :term:`marginals` :cite:`chizat:18`.

    linear problem
        :term:`OT` problem only containing a :term:`linear term` and no :term:`quadratic term`.

    linear term
        Term of the cost function on the shared space, e.g. gene expression space.

    quadratic problem
        :term:`OT` problem containing a :term:`quadratic term` and possibly a :term:`linear term`.

    quadratic term
        Term of the cost function comparing two different spaces.

    Gromov-Wasserstein
        :term:`OT` problem between two distributions where a data point, e.g. a cell. in the source distribution does not live in the same space as a data point in the target distribution. Such a problem is a :term:`quadratic problem`.

    fused Gromov-Wasserstein
        :term:`OT` problem between two distributions where a data point, e.g. a cell, of the source distribution has both features in the same space as the target distribution (:term:`linear term`) and features in a different space than a data point in the target distribution (:term:`quadratic term`). Such a problem is a :term:`quadratic problem`.

    dual potentials
        Potentials obtained by the :term:`Sinkhorn` algorithm which define the solution of a :term:`linear problem` :cite:`cuturi:2013`.

    marginals
        An :term:`OT` problem matches distributions, e.g. set of cells. The distribution is defined by the location of a cell, e.g. in gene expression space, and the weight assigned to one cell. These weights are refered to as `marginals`.

    Sinkhorn
        The Sinkhorn algorithm :cite:`cuturi:2013` is used for solving a :term:`linear problem`, and is also used in inner iterations for solving a :term:`quadratic problem`.

    low-rank
        If the OT problem is solved with a low-rank solver, the :term:`transport matrix` is the product of several matrices with low rank (i.e. lower than the number of data points in the source distribution and the target distribution), and hence the :term:`transport matrix`` is low-rank.

    transport matrix
        The output of a discrete :term:`OT` problem indicating how much mass from data point :math:`x_i` in row :math:`i` is transported to data point :math:`y_j` in column :math:`j`.

    entropic regularization
        Entropy regularization of :term:`OT` problems :cite:`cuturi:2013` reduces the time complexity and allows for more desirable statistical properties. The higher the entropy regularization, the more diffused the OT solution.
