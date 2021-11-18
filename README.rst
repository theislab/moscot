moscot - multi-omic single-cell optimal transport tools
=======================================================

.. image:: https://raw.githubusercontent.com/theislab/moscot/dev/resources/images/logo.png?token=ALENVBTTXMZ2MH2RPENXLX3BT5PQI
    :width: 600px
    :align: center
    :alt: Logo

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
