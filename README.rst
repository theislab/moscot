|PyPI| |Downloads| |CI| |Pre-commit| |Codecov| |Docs|

moscot - multi-omic single-cell optimal transport tools
=======================================================

**moscot** is a scalable framework for Optimal Transport (OT) applications in
single-cell genomics. It can be used for

- trajectory inference (incorporating spatial and lineage information)
- mapping cells to their spatial organisation
- aligning spatial transcriptomics slides
- translating modalities
- prototyping of new OT models in single-cell genomics

**moscot** is powered by
`OTT <https://ott-jax.readthedocs.io>`_ which is a JAX-based Optimal
Transport toolkit that supports just-in-time compilation, GPU acceleration, automatic
differentiation and linear memory complexity for OT problems.

Installation
------------
You can install **moscot** via::

    pip install moscot

In order to install **moscot** from in editable mode, run::

    git clone https://github.com/theislab/moscot
    cd moscot
    pip install -e .

For further instructions how to install jax, please refer to https://github.com/google/jax.

Resources
---------

Please have a look at our `documentation <https://moscot.readthedocs.io>`_

Reference
---------

Our preprint "Mapping cells through time and space with moscot" can be found `here <https://www.biorxiv.org/content/10.1101/2023.05.11.540374v1>`_.

.. |Codecov| image:: https://codecov.io/gh/theislab/moscot/branch/master/graph/badge.svg?token=Rgtm5Tsblo
    :target: https://codecov.io/gh/theislab/moscot
    :alt: Coverage

.. |PyPI| image:: https://img.shields.io/pypi/v/moscot.svg
    :target: https://pypi.org/project/moscot/
    :alt: PyPI

.. |CI| image:: https://img.shields.io/github/actions/workflow/status/theislab/moscot/test.yml?branch=main
    :target: https://github.com/theislab/moscot/actions
    :alt: CI

.. |Pre-commit| image:: https://results.pre-commit.ci/badge/github/theislab/moscot/main.svg
   :target: https://results.pre-commit.ci/latest/github/theislab/moscot/main
   :alt: pre-commit.ci status

.. |Docs| image:: https://img.shields.io/readthedocs/moscot
    :target: https://moscot.readthedocs.io/en/stable/
    :alt: Documentation

.. |Downloads| image:: https://pepy.tech/badge/moscot
    :target: https://pepy.tech/project/moscot
    :alt: Downloads
