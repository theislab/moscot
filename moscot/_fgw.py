from types import MappingProxyType
from typing import Any, Tuple, Mapping, Callable, Optional
from functools import partial
from collections import namedtuple

from typing_extensions import Literal

from jax import numpy as jnp, random
from ott.tools.transport import Transport
from ott.geometry.geometry import Geometry
from ott.core.gromov_wasserstein import GWLoss, GWSqEuclLoss, _marginal_dependent_cost
import jax
import numpy as np

fgw_carry = namedtuple("fgw_carry", ["iter", "T", "prev_fval", "curr_fval"])
fgw_res = namedtuple("fgw_res", ["iter", "T"])


def _initialize(
    geom_a: Geometry,
    geom_b: Geometry,
    geom_ab: Geometry,
    a: jnp.ndarray,
    b: jnp.ndarray,
    *,
    alpha: float,
    epsilon: Optional[float],
    init_method: Literal["uniform", "random"],
    cost_fn: GWLoss,
    scale_fn: Optional[Callable[[jnp.ndarray], float]],
    seed: int,
) -> Tuple[jnp.ndarray, Geometry, jnp.ndarray]:
    C12 = _marginal_dep_term(geom_a, geom_b, a, b, cost_fn)
    T = _get_initial_coupling(a, b, method=init_method, epsilon=epsilon, seed=seed)

    scale_ab = 1.0 if scale_fn is None else scale_fn(geom_ab.cost_matrix)
    geom_ab = Geometry(cost_matrix=(1 - alpha) * geom_ab.cost_matrix / scale_ab)

    return T, geom_ab, C12


def _update_fgw(
    geom_a: Geometry,
    geom_b: Geometry,
    geom_ab: Geometry,
    alpha: float,
    epsilon: Optional[float],
    T: jnp.ndarray,
    C12: jnp.ndarray,
    cost_fn: GWLoss,
    scale_fn: Callable[[jnp.array], float],
) -> Geometry:
    h1 = cost_fn.left_x
    h2 = cost_fn.right_y
    C_ab = geom_ab.cost_matrix
    C_a, C_b = geom_a.cost_matrix, geom_b.cost_matrix
    # references:
    # https://github.com/PythonOT/POT/blob/master/ot/gromov.py#L205
    # https://github.com/PythonOT/POT/blob/master/ot/gromov.py#L137-L138
    C = 2 * (C12 - jnp.dot(h1(C_a), T).dot(h2(C_b).T))
    scale_c = 1.0 if scale_fn is None else scale_fn(C)
    # tmp = C_ab + self.alpha * 2 * (C12 - np.dot(h1(C_a), T).dot(h2(C_b).T))
    return Geometry(cost_matrix=C_ab + alpha * (C / scale_c), epsilon=epsilon)


def _get_initial_coupling(
    a: jnp.ndarray,
    b: jnp.ndarray,
    *,
    method: Literal["uniform", "random"],
    epsilon: Optional[float] = None,
    seed: int = 0,
) -> jnp.ndarray:
    key = random.PRNGKey(seed)

    if method == "uniform":
        return jnp.outer(a, b)
    if method == "random":
        # TODO(michalk8): RecursionError in `ott` if epsilon is `None`
        if epsilon is None:
            raise ValueError("Please specify `epsilon=...` when using `init_method='random'.`")
        geom = Geometry(kernel_matrix=random.uniform(key, shape=(len(a), len(b)), dtype=float), epsilon=epsilon)
        return Transport(geom, a=a, b=b).matrix

    raise NotImplementedError(method)


def _marginal_dep_term(geom_a: Geometry, geom_b: Geometry, a: jnp.ndarray, b: jnp.ndarray, cost_fn) -> jnp.ndarray:
    # TODO(michalk8): taken from ott, we could be more mem. efficient
    ab = a[:, None] * b[None, :]
    marginal_x = ab.sum(1)
    marginal_y = ab.sum(0)
    return _marginal_dependent_cost(marginal_x, marginal_y, geom_a, geom_b, cost_fn)


def _fgw(
    geom_a: Geometry,
    geom_b: Geometry,
    geom_ab: Geometry,
    a: jnp.ndarray,
    b: jnp.ndarray,
    *,
    alpha: float = 0.5,
    epsilon: Optional[float] = None,
    cost_fn: Optional[GWLoss] = None,
    scale_fn: Optional[Callable[[jnp.ndarray], float]],
    init_method: Literal["uniform", "random"] = "uniform",
    max_iterations: int = 20,
    rtol: Optional[float] = 1e-6,
    atol: Optional[float] = 1e-6,
    seed: int = 0,
    kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> fgw_res:
    def cond_fn(carry: fgw_carry) -> bool:
        abs_delta_fval = jnp.abs(carry.curr_fval - carry.prev_fval)
        relative_delta_fval = abs_delta_fval / jnp.abs(carry.curr_fval)
        return (carry.iter < max_iterations) & (relative_delta_fval > rtol) & (abs_delta_fval > atol)

    def while_body_fn(carry: fgw_carry) -> fgw_carry:
        geom = _update_fgw(geom_a, geom_b, geom_ab, alpha, epsilon, carry.T, C12, cost_fn, scale_fn)
        transport = Transport(geom, a=a, b=b, **kwargs)
        T_hat, f_val = transport.matrix, transport.reg_ot_cost
        del geom, transport

        return fgw_carry(carry.iter + 1, T_hat, carry.curr_fval, f_val)

    def for_body_fn(carry: jnp.ndarray, _: Any = None) -> Tuple[jnp.array, type(None)]:
        del _
        geom = _update_fgw(geom_a, geom_b, geom_ab, alpha, epsilon, carry, C12, cost_fn, scale_fn)
        return Transport(geom, a=a, b=b, **kwargs).matrix, None

    if cost_fn is None:
        cost_fn = GWSqEuclLoss()

    T, geom_ab, C12 = _initialize(
        geom_a,
        geom_b,
        geom_ab,
        a,
        b,
        alpha=alpha,
        epsilon=epsilon,
        init_method=init_method,
        cost_fn=cost_fn,
        scale_fn=scale_fn,
        seed=seed,
    )

    # TODO(michalk8): return more info (for warnings/etc.)
    if rtol is None or atol is None:
        iteration = max_iterations - 1
        T = jax.lax.scan(f=for_body_fn, init=T, xs=None, length=iteration)[0]
    else:
        res = jax.lax.while_loop(cond_fn, while_body_fn, fgw_carry(0.0, T, jnp.inf, 0.0))
        iteration, T = res.iter, res.T

    return fgw_res(iteration, T)


def fgw(
    geom_a: Geometry,
    geom_b: Geometry,
    geom_ab: Geometry,
    a: Optional[jnp.ndarray] = None,
    b: Optional[jnp.ndarray] = None,
    alpha: float = 0.5,
    epsilon: Optional[float] = None,
    init_method: Literal["uniform", "random"] = "uniform",
    cost_fn: Optional[GWLoss] = None,
    scale_fn: Optional[Callable[[jnp.ndarray], float]] = None,
    jit: bool = False,
    max_iterations: int = 20,
    rtol: Optional[float] = 1e-6,
    atol: Optional[float] = 1e-6,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> fgw_res:
    # fmt: off
    a = jnp.ones((geom_a.shape[0],)) / geom_a.shape[0] if a is None else a
    b = jnp.ones((geom_b.shape[0],)) / geom_b.shape[0] if b is None else b
    # fmt: on

    seed = 0 if init_method == "uniform" else np.random.RandomState(seed).randint(0, 2 ** 32 - 1)
    kwargs["jit"] = jit

    fgw_fn = partial(
        _fgw,
        alpha=alpha,
        epsilon=epsilon,
        init_method=init_method,
        cost_fn=cost_fn,
        scale_fn=scale_fn,
        max_iterations=max_iterations,
        rtol=rtol,
        atol=atol,
        seed=seed,
        kwargs=kwargs,
    )
    if jit:
        fgw_fn = jax.jit(fgw_fn)

    return fgw_fn(geom_a, geom_b, geom_ab, a, b)
