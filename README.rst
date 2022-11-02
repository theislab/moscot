|Codecov|

moscot - multi-omic single-cell optimal transport tools
=======================================================

**moscot** is a scalable framework for Optimal Transport (OT) applications in
single-cell genomics. It aims at comprising the most relevant OT models while
providing an intuitive user-interface.

In the backend, moscot is powered by
`OTT <https://ott-jax.readthedocs.io/en/latest/>`_ which is a JAX-based Optimal
Transport toolkit that supports just-in-time compilation, GPU acceleration, automatic
differentiation and linear memory complexity for OT problems.

Installation
------------
You can install **moscot** via::

    pip install moscot

In order to install **moscot** from source, run::

    git clone https://github.com/theislab/moscot
    cd moscot
    pip install -e.'[dev]'

If used with GPU, additionally run::

    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


.. |Codecov| image:: https://codecov.io/gh/theislab/moscot/branch/master/graph/badge.svg?token=Rgtm5Tsblo
    :target: https://codecov.io/gh/theislab/moscot
    :alt: Coverage
