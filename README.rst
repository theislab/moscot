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

Development (temporary)
-----------------------
For generating the UML proceed as follows (reference: https://medium.com/@ganesh.alalasundaram/uml-diagram-using-pyreverse-for-python-repository-dd68cdf9e7e1)::

    pip install pylint
    pip install graphviz
    pip install pydot

Grahpviz might have to be installed via brew on MAC (brew install graphviz).
To generate a "classes.dot" and "packages.dot" file run::

    pyreverse <location of repository>

Finally, create the png file from "classes.dot", e.g. via::

    from subprocess import check_call
    check_call(['dot','-Tpng','classes.dot','-o','UML.png'])

.. |Codecov| image:: https://codecov.io/gh/theislab/moscot/branch/master/graph/badge.svg?token=Rgtm5Tsblo
    :target: https://codecov.io/gh/theislab/moscot
    :alt: Coverage
