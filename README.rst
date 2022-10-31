|Codecov|

moscot - multi-omic single-cell optimal transport tools
=======================================================

**moscot** is a general framework to apply tools from
optimal transport to time-course single-cell data. It supports:

- single-cell RNA-seq and ATAC-seq data (paired and unpaired)
- single-cell lineage-traced data (prospective and retrospective)

while scaling to large cell numbers. In the backend, moscot is powered by
`OTT <https://ott-jax.readthedocs.io/en/latest/>`_ which is a Jax-based optimal
transport toolkit that supports just-in-time compilation, automatic
differentiation and linear memory complexity for OT problems.

Installation
------------
In order to install **moscot**, run::

    git clone https://github.com/theislab/moscot
    cd moscot
    pip install -e.'[dev]'
    pre-commit install

for ``pre-commit`` you might have to install ``prettier`` with conda: ``conda install -c conda-forge prettier``.

If used with GPU, additionally run::

    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


.. |Codecov| image:: https://codecov.io/gh/theislab/moscot/branch/master/graph/badge.svg?token=Rgtm5Tsblo
    :target: https://codecov.io/gh/theislab/moscot
    :alt: Coverage
