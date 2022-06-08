API
===

Biological Problems
~~~~~~~~~~~~~~~~~~~

.. module:: moscot.problems
.. currentmodule:: moscot.problems

.. autosummary::
    :toctree: api

    time.TemporalProblem
    time.LineageProblem
    space.AlignmentProblem
    space.MappingProblem
    spatio_temporal.SpatioTemporalProblem


Generic Problems
~~~~~~~~~~~~~~~~
.. module:: moscot.problems.generic
.. currentmodule:: moscot.problems.generic

.. autosummary::
    :toctree: api

    SinkhornProblem
    GWProblem
    FGWProblem


Solvers
~~~~~~~

.. module:: moscot.backends.ott
.. currentmodule:: moscot.backends.ott

.. autosummary::
    :toctree: api

    moscot.backends.ott.SinkhornSolver
    moscot.backends.ott.GWSolver
    moscot.backends.ott.FGWSolver
