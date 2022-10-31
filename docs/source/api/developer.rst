Developer
#########


OTT Backend
~~~~~~~~~~~~

.. autosummary::
    :toctree: genapi

    moscot.backends.ott.SinkhornSolver
    moscot.backends.ott.GWSolver
    moscot.backends.ott.FGWSolver
    moscot.backends.ott.OTTOutput
    moscot.backends.ott.OTTCost

Costs
~~~~~

.. autosummary::
    :toctree: genapi

    moscot.costs.LeafDistance
    moscot.costs.BarcodeDistance


Translation Classes
~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: genapi

    moscot.problems.base.CompoundProblem


Optimal Transport classes
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: genapi

    moscot.problems.base.OTProblem
    moscot.problems.base.BirthDeathProblem


Mixin classes
~~~~~~~~~~~~~
.. autosummary::
    :toctree: genapi

    moscot.problems.base._birth_death.BirthDeathMixin

Base classes
~~~~~~~~~~~~
.. autosummary::
    :toctree: genapi

    moscot.problems.base.BaseProblem
    moscot.problems.base.OTProblem
    moscot.solvers.BaseSolver
    moscot.solvers.OTSolver
    moscot.solvers.BaseSolverOutput
    moscot.solvers.MatrixSolverOutput
    moscot.costs.BaseCost
    moscot.solvers.TaggedArray
    moscot.solvers.Tag
