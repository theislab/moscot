User API
########
Import :mod:`moscot` as::

    import moscot as mt

Biological Problems
~~~~~~~~~~~~~~~~~~~
.. module:: moscot.problems
.. currentmodule:: moscot.problems
.. autosummary::
    :toctree: genapi

    time.TemporalProblem
    time.LineageProblem
    space.AlignmentProblem
    space.MappingProblem
    spatiotemporal.SpatioTemporalProblem
    cross_modality.TranslationProblem


Generic Problems
~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: genapi

    generic.SinkhornProblem
    generic.GWProblem
    generic.FGWProblem
    generic.GENOTLinProblem

Plotting
~~~~~~~~
.. module:: moscot.plotting
.. currentmodule:: moscot
.. autosummary::
    :toctree: genapi

    plotting.cell_transition
    plotting.sankey
    plotting.push
    plotting.pull

Datasets
~~~~~~~~
.. module:: moscot.datasets
.. currentmodule:: moscot
.. autosummary::
    :toctree: genapi

    datasets.bone_marrow
    datasets.c_elegans
    datasets.drosophila
    datasets.hspc
    datasets.mosta
    datasets.sciplex
    datasets.sim_align
    datasets.simulate_data
    datasets.tedsim
    datasets.zebrafish
