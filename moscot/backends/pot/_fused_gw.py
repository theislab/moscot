from typing import Any, Dict, Tuple, Union, Optional

from ot import sinkhorn
from ot.utils import list_to_array
from ot.gromov import gwloss, gwggrad, init_matrix
from ot.backend import get_backend

import numpy as np
import numpy.typing as npt

__all__ = ("fused_gromov_wasserstein",)


def fused_gromov_wasserstein(
    M: npt.ArrayLike,
    C1: npt.ArrayLike,
    C2: npt.ArrayLike,
    p: npt.ArrayLike,
    q: npt.ArrayLike,
    epsilon: float,
    loss_fun: str = "square_loss",
    alpha: float = 0.5,
    G0: Optional[npt.ArrayLike] = None,
    max_iter: int = 1000,
    tol: float = 1e-9,
    log: bool = False,
    **kwargs: Any,
) -> Union[npt.ArrayLike, Tuple[npt.ArrayLike], Dict[str, Any]]:
    def update():
        ...

    p, q = list_to_array(p, q)
    p0, q0, C10, C20, M0 = p, q, C1, C2, M
    if G0 is None:
        nx = get_backend(p0, q0, C10, C20, M0)
    else:
        nx = get_backend(p0, q0, C10, C20, M0, G0)

    p = nx.to_numpy(p)
    q = nx.to_numpy(q)
    C1 = nx.to_numpy(C10)
    C2 = nx.to_numpy(C20)
    M = nx.to_numpy(M0)
    M /= M.max()

    if G0 is None:
        T = p[:, None] * q[None, :]
    else:
        T = nx.to_numpy(G0)
        np.testing.assert_allclose(T.sum(axis=1), p, atol=1e-08)
        np.testing.assert_allclose(T.sum(axis=0), q, atol=1e-08)

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)
    if log:
        log = {"err": []}

    err, converged = 1.0, True
    for i in range(max_iter):
        Tprev = T
        tens = gwggrad(constC, hC1, hC2, T)
        T = sinkhorn(p, q, alpha * tens + (1 - alpha) * M, epsilon, **kwargs)
        if i % 10 == 0:
            err = nx.norm(T - Tprev)
            if log:
                log["err"].append(err)
        if err <= tol:
            break
    else:
        converged = False

    if log:
        log["converged"] = converged
        log["fgw_dist"] = gwloss(constC, hC1, hC2, T)
        return T, log
    return T
