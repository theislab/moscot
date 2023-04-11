User API
########
.. module:: moscot

Import ``moscot`` as::

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


Generic Problems
~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: genapi

    generic.SinkhornProblem
    generic.GWProblem

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

    datasets.mosta
    datasets.hspc
    datasets.drosophila
    datasets.c_elegans
    datasets.zebrafish
    datasets.tedsim
    datasets.sim_align
    datasets.simulate_data
