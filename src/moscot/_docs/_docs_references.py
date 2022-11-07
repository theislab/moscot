from docrep import DocstringProcessor

_ex_solve_quadratic = """\
See :ref:`sphx_glr_auto_examples_solvers_ex_quad_problems_basic.py` for a basic example
how to solve quadratic problems.
See :ref:`sphx_glr_auto_examples_solvers_ex_quad_problems_advanced.py` for an advanced
example how to solve quadratic problems.
"""
_ex_solve_linear = """\
See :ref:`sphx_glr_auto_examples_solvers_ex_linear_problems_basic.py` for a basic example
how to solve linear problems.
See :ref:`sphx_glr_auto_examples_solvers_ex_linear_problems_advanced.py` for an advanced
example how to solve linear problems.
"""

d_references = DocstringProcessor(
    ex_solve_quadratic=_ex_solve_quadratic,
    ex_solve_linear=_ex_solve_linear,
)
