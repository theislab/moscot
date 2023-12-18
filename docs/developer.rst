Developer API
#############

Backends
~~~~~~~~
.. module:: moscot.backends
.. currentmodule:: moscot
.. autosummary::
    :toctree: genapi

    backends.ott.SinkhornSolver
    backends.ott.GWSolver
    backends.ott.OTTOutput
    backends.ott.GraphOTTOutput
    backends.utils.get_solver
    backends.utils.get_available_backends

Costs
~~~~~
.. module:: moscot.costs
.. currentmodule:: moscot
.. autosummary::
    :toctree: genapi

    costs.BarcodeDistance
    costs.LeafDistance
    costs.get_cost
    costs.get_available_costs
    costs.register_cost

Base
~~~~
.. module:: moscob.base
.. currentmodule:: moscot.base

Problems
^^^^^^^^
.. autosummary::
    :toctree: genapi

    problems.BaseProblem
    problems.OTProblem
    problems.BirthDeathProblem
    problems.BaseCompoundProblem
    problems.CompoundProblem
    cost.BaseCost

Mixins
^^^^^^
.. autosummary::
    :toctree: genapi

    problems.AnalysisMixin
    problems.BirthDeathMixin

Solvers
^^^^^^^
.. module:: moscot.solvers
.. currentmodule:: moscot.base
.. autosummary::
    :toctree: genapi

    solver.BaseSolver
    solver.OTSolver
    output.BaseSolverOutput

Output
^^^^^^
.. autosummary::
    :toctree: genapi

    output.BaseSolverOutput
    output.MatrixSolverOutput

Utils
~~~~~
.. module:: moscot.utils
.. currentmodule:: moscot.utils

Policies
^^^^^^^^
.. autosummary::
    :toctree: genapi

    subset_policy.SubsetPolicy
    subset_policy.OrderedPolicy
    subset_policy.StarPolicy
    subset_policy.ExternalStarPolicy
    subset_policy.SequentialPolicy
    subset_policy.TriangularPolicy
    subset_policy.ExplicitPolicy

Miscellaneous
^^^^^^^^^^^^^
.. autosummary::
    :toctree: genapi

    data.transcription_factors
    data.proliferation_markers
    data.apoptosis_markers
    tagged_array.TaggedArray
    tagged_array.Tag

.. currentmodule:: moscot.base.problems
.. autosummary::
    :toctree: genapi

    birth_death.beta
    birth_death.delta
