|Codecov|

moscot - multi-omic single-cell optimal transport tools
=======================================================

**moscot** is a scalable framework for Optimal Transport (OT) applications in
single-cell genomics. It can be used for
- temporal and spatio-temporal trajectory inference
- spatial mapping
- spatial alignment
- prototyping of new OT models in single-cell genomics

**moscot** is powered by
`OTT <https://ott-jax.readthedocs.io>`_ which is a JAX-based Optimal
Transport toolkit that supports just-in-time compilation, GPU acceleration, automatic
differentiation and linear memory complexity for OT problems.

Installation
------------
You can install **moscot** via::

    pip install moscot

In order to install **moscot** from source, run::

    git clone https://github.com/theislab/moscot
    cd moscot
    pip install -e .'[dev]'

If used with GPU, additionally run::

    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


.. |Codecov| image:: https://codecov.io/gh/theislab/moscot/branch/master/graph/badge.svg?token=Rgtm5Tsblo
    :target: https://codecov.io/gh/theislab/moscot
    :alt: Coverage

Resources
---------

Please have a look at our `documentation <https://moscot.readthedocs.io>`_

Reference
---------

Our preprint "Mapping cells through time and space with moscot" can be found `here <https://www.biorxiv.org/content/10.1101/2023.05.11.540374v1>`_.
