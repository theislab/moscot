|PyPI| |Downloads| |CI| |Pre-commit| |Codecov| |Docs|

Moscot - Multiomics Single-cell Optimal Transport
=======================================================

.. image:: _static/img/light_mode_concept_revised.png
    :width: 800px
    :align: center
    :class: only-light

.. image:: _static/img/dark_mode_concept_revised.png
    :width: 800px
    :align: center
    :class: only-dark


**moscot** is a framework for Optimal Transport (OT) applications in
single-cell genomics. It scales to large datasets and can be used for a
variety of applications across different modalities.

moscot's key applications
---------------------------
- :doc:`Trajectory inference <notebooks/tutorials/200_temporal_problem>`
  (incorporating :doc:`spatial <notebooks/tutorials/500_spatiotemporal>`
  and :doc:`lineage <notebooks/tutorials/100_lineage>` information).
- :doc:`Mapping <notebooks/tutorials/400_spatial_mapping>` cells to their spatial organisation.
- :doc:`Aligning <notebooks/tutorials/300_spatial_alignment>` spatial transcriptomics slides.
- :doc:`Translating <notebooks/tutorials/600_tutorial_translation>` modalities.
- prototyping of new OT models in single-cell genomics.
- ... and more, check out the :doc:`user guide <user_guide>` for more information.


**moscot** is powered by
`OTT <https://ott-jax.readthedocs.io>`_ which is a JAX-based Optimal
Transport toolkit that supports just-in-time compilation, GPU acceleration, automatic
differentiation and linear memory complexity for OT problems.

Installation
------------
Install :mod:`moscot` by running::

    pip install moscot

In order to install **moscot** from in editable mode, run::

    git clone https://github.com/theislab/moscot
    cd moscot
    pip install -e .

For further instructions how to install jax, please refer to https://github.com/google/jax.

Resources
---------

Please have a look at our `documentation <https://moscot.readthedocs.io>`_

Citing moscot
-------------
If you find a model useful for your research, please consider citing the `Klein et al., 2025`_ manuscript as
well as the publication introducing the model, which can be found in the corresponding documentation.

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

.. |Downloads| image:: https://static.pepy.tech/badge/moscot
    :target: https://pepy.tech/project/moscot
    :alt: Downloads
