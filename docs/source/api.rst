API
===

Biological Problems
~~~~~~~~~~~~~~~~~~~

.. module:: moscot.problems
.. currentmodule:: moscot.problems

.. autosummary::
    :toctree: api

    moscot.problems.time.TemporalProblem
    moscot.problems.time.LineageProblem
    moscot.problems.space.AlignmentProblem
    moscot.problems.space.MappingProblem
    moscot.problems.spatio_temporal.SpatioTemporalProblem


Generic Problems
~~~~~~~~~~~~~~~~
.. module:: moscot.problems.generic
.. currentmodule:: moscot.problems.generic

.. autosummary::
    :toctree: api

    moscot.problems.generic.SinkhornProblem
    moscot.problems.generic.GWProblem
    moscot.problems.generic.FGWProblem


Solvers
~~~~~~~

.. module:: moscot.backends.ott
.. currentmodule:: moscot.backends.ott

.. autosummary::
    :toctree: api

    moscot.backends.ott.SinkhornSolver
    moscot.backends.ott.GWSolver
    moscot.backends.ott.FGWSolver
