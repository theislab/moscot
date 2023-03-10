from typing import Any, Union, Literal, TYPE_CHECKING

from . import utils

if TYPE_CHECKING:
    from . import ott


@utils.register_backend("ott")
def _(problem_kind: Literal["linear", "quadratic"], **kwargs: Any) -> Union["ott.SinkhornSolver", "ott.GWSolver"]:
    from . import ott

    if problem_kind == "linear":
        return ott.SinkhornSolver(**kwargs)
    if problem_kind == "quadratic":
        return ott.GWSolver(**kwargs)
    raise NotImplementedError(f"Unable to create solver for `{problem_kind!r}` problem.")
