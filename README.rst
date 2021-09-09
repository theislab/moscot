moscot - Multi-Omics Single-Cell Optimal Transport
==================================================

**moscot** is intended to be a general framework to apply tools from
optimal transport to time-course single-cell data. It should support:

- multimodal data (esp. ATAC & RNA)
- lineage-tracing data (prospective & retrospective)

while scaling to large (~10k cells per time point) samples.

In the backend, it will be based on either `OTT <https://ott-jax.readthedocs.io/en/latest/index.html>`_ or
`GeomLoss <https://www.kernel-operations.io/geomloss/index.html>`_ for fast and memory-efficient computations.

Installation
------------
In order to install **moscot** run::

    git clone https://github.com/theislab/moscot
    cd scott
    pip install -e.'[dev]'
    pre-commit install
