|PyPI| |Downloads| |CI| |Pre-commit| |Codecov| |Docs|

Moscot - Multiomics Single-cell Optimal Transport
=================================================
.. module:: moscot

:mod:`moscot` is a framework for optimal transport applications in single cell genomics.

.. image:: _static/img/light_mode_logo.png
    :width: 600px
    :align: center
    :class: only-light

.. image:: _static/img/dark_mode_logo.png
    :width: 600px
    :align: center
    :class: only-dark

If you find a model useful for your research, please consider citing the ``moscot`` manuscript [`Klein et al., 2023 <https://www.biorxiv.org/content/10.1101/2023.05.11.540374v2>`_] as
well as the publication introducing the model, which can be found in the corresponding documentation.

.. grid:: 3
    :gutter: 1

    .. grid-item-card:: Installation
        :link: installation
        :link-type: doc

        Learn how to install :mod:`moscot`.

    .. grid-item-card:: User API
        :link: user
        :link-type: doc

        Find a detailed documentation of :mod:`moscot`.

    .. grid-item-card:: Contributing
        :link: contributing
        :link-type: doc

        Add a functionality or report a bug.

    .. grid-item-card:: Examples
        :link: notebooks/examples/index
        :link-type: doc

        Find brief and concise examples of :mod:`moscot`.

    .. grid-item-card:: Tutorials
        :link: notebooks/tutorials/index
        :link-type: doc

        Check out how to use :mod:`moscot` for data analysis.

    .. grid-item-card:: Manuscript

        Please have a look at our manuscript [`Klein et al., 2023 <https://www.biorxiv.org/content/10.1101/2023.05.11.540374v2>`_] to learn more.

.. toctree::
    :maxdepth: 2
    :hidden:

    installation
    user
    developer
    contributing
    notebooks/tutorials/index
    notebooks/examples/index
    references


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
