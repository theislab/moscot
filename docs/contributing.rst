Contributing guide
~~~~~~~~~~~~~~~~~~

Table of Contents
=================
- `Contributing to moscot`_
- `Codebase structure`_
- `Code style guide`_
- `Testing`_
- `Writing documentation`_
- `Writing tutorials/examples`_
- `Submitting a PR`_

Contributing to moscot
-----------------------
Clone moscot from source as::

    git clone https://github.com/theislab/moscot
    cd moscot
    git checkout main

Install the test and development mode::

    pip install -e'.[dev,test]'

Optionally install pre-commit. This will ensure that the pushed code passes the linting steps::

    pre-commit install

Although the last step is not necessary, it is highly recommended, since it will help you to pass the linting step
(see `Code style guide`_). If you did install ``pre-commit`` but are unable to decipher some flags, you can
still commit using the ``--no-verify``.

Codebase structure
------------------
The moscot project:

- `moscot <../src/moscot>`_: the root of the package.

  - `moscot/backends <../src/moscot/backends>`_: the OTT backend module, which deals with OTT solvers and output functions.
  - `moscot/base <../src/moscot/base>`_: contains base moscot classes.
    - `moscot/base/problems <../src/moscot/base/problems>`_: the moscot problems module.
  - `moscot/costs <../src/moscot/costs>`_: contains different costs computation functions.
  - `moscot/plotting <../src/moscot/plotting>`_: the plotting module.
  - `moscot/problems <../src/moscot/problems>`_: the functionality of general problems classes, subdivided into problem types.
  - `moscot/utils <../src/moscot/utils>`_: contains various utility functions.
  - `moscot/datasets.py <../src/moscot/datasets.py>`_: contains loading and simulating functions for the datasets.

Tests structure:

- `tests <../tests>`_: the root of the package

  - `tests/backends <../tests/backends>`_: tests for the ott backend.
  - `tests/costs <../tests/costs>`_ tests for the solving costs.
  - `tests/data <../tests/data>`_: tests for the simulated data module.
  - `tests/datasets <../tests/datasets>`_: tests for the datasets.
  - `tests/plotting <../tests/plotting>`_ tests for the plotting module.
  - `tests/problems <../tests/problems>`_ tests for the problem classes, divided by problem type.
  - `tests/solvers <../tests/solvers>`_ tests for the solvers.
  - `tests/utils <../tests/utils>`_ tests for the utility functions.
  - `tests/conftest.py <../tests/conftest.py>`_: ``pytest`` fixtures and utility functions.

Code style guide
----------------
We rely on ``black`` and ``isort`` to do the most of the formatting - both of them are integrated as pre-commit hooks.
You can use ``tox`` to check the changes::

    tox -e lint-code

Furthermore, we also require that:

- functions are fully type-annotated.
- exception messages are capitalized and end with ``.``.
- warning messages are capitalized and do not end with ``.``.
- when referring to variable inside an error/warning message, enclose its name in \`.
- when referring to variable inside a docstrings, enclose its name in \``.

Testing
-------
We use ``tox`` to automate our testing, as well as linting and documentation creation. To run the tests, run::

    tox -e py{310,311}-{linux,macos}

depending on the Python version(s) in your ``PATH`` and your operating system. We use ``flake8`` and ``mypy`` to further
analyze the code. Use ``# noqa: <error1>,<error2>`` to ignore certain ``flake8`` errors and
``# type: ignore[error1,error2]`` to ignore specific ``mypy`` errors.

To run only a subset of tests, run::

    tox -e <environment> -- <name>

where ``<name>`` can be a path to a test file/directory or a name of a test function/class.
For example, to run only the tests in the ``plotting`` module, use::

    tox -e py310-linux -- tests/plotting/test_plotting.py

If needed, a specific ``tox`` environment can be recreated as::

    tox -e <environment> --recreate

Writing documentation
---------------------
We use ``numpy``-style docstrings for the documentation with the following additions and modifications:

- no type hints in the docstring (applies also for the return statement) are allowed,
  since all functions are required to have the type hints in their signatures.
- when referring to some argument within the same docstring, enclose that reference in \`\`.
- prefer putting references in the ``references.bib`` instead under the ``References`` sections of the docstring.
- use ``docrep`` for repeating documentation.

In order to build the documentation, run::

    tox -e build-docs

Since the tutorials are hosted on a separate repository (see `Writing tutorials/examples`_), we download the newest
tutorials/examples from there and build the documentation here.

To validate the links inside the documentation, run::

    tox -e lint-docs

If you need to clean the artifacts from previous documentation builds, run::

    tox -e clean-docs

Writing tutorials/examples
--------------------------
Tutorials and examples are hosted on a separate repository called `moscot_notebooks
<https://github.com/theislab/moscot_notebooks>`_.
Please refer to this `guide <https://github.com/theislab/moscot_notebooks/blob/main/CONTRIBUTING.rst>`_ for more information.

Submitting a PR
---------------
Before submitting a new pull request, please make sure you followed these instructions:

- make sure that you've branched off ``main`` and are merging into ``main``
- make sure that your code follows the above specified conventions
  (see `Code style guide`_ and `Writing documentation`_).
- if applicable, make sure you've added/modified at least 1 test to account for the changes you've made
- make sure that all tests pass locally (see `Testing`_).
- if there is no issue which this PR solves, create a new `one <https://github.com/theislab/moscot/issues/new>`_
  briefly explaining what the problem is.
- make sure that the section under ``## Description`` is properly formatted if automatically generating release notes.

