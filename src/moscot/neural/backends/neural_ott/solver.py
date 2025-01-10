import abc
import functools
import inspect
import math
import types
from typing import (
    Any,
    Hashable,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import optax

import jax
import jax.numpy as jnp
import numpy as np
from ott.geometry import costs, epsilon_scheduler, geodesic, geometry, pointcloud
from ott.neural.datasets import OTData, OTDataset
from ott.neural.methods.flows import dynamics, genot
from ott.neural.networks.layers import time_encoder
from ott.neural.networks.velocity_field import VelocityField
from ott.problems.linear import linear_problem
from ott.problems.quadratic import quadratic_problem
from ott.solvers.linear import sinkhorn, sinkhorn_lr
from ott.solvers.quadratic import gromov_wasserstein, gromov_wasserstein_lr
from ott.solvers.utils import uniform_sampler
from moscot.neural.data import PolicyDataLoader

from moscot._logging import logger
from moscot._types import (
    ArrayLike,
    LRInitializer_t,
    ProblemKind_t,
    QuadInitializer_t,
    SinkhornInitializer_t,
)
from moscot.backends.ott._utils import (
    InitializerResolver,
    Loader,
    MultiLoader,
    _instantiate_geodesic_cost,
    alpha_to_fused_penalty,
    check_shapes,
    convert_scipy_sparse,
    data_match_fn,
    densify,
    ensure_2d,
)
from moscot.utils.subset_policy import SubsetPolicy
from moscot.backends.ott.output import GraphOTTOutput, NeuralOutput, OTTOutput
from moscot.base.problems._utils import TimeScalesHeatKernel
from moscot.base.solver import BaseSolver
from moscot.neural.data import DistributionCollection
from typing import TypeVar

K = TypeVar("K", bound=Hashable)


__all__ = ["GENOTSolver"]


def _split_data(
    rng: jax.random.PRNGKey,
    distributions: DistributionCollection,
    train_size: float = 0.9,
) -> Any:
    train_collection: DistributionCollection = {}
    val_collection: DistributionCollection = {}
    for key, dist in distributions.items():
        n_train_x = math.ceil(train_size * dist.n_samples)
        idxs = jax.random.permutation(rng, jnp.arange(dist.n_samples))
        train_collection[key] = dist[idxs[:n_train_x]]
        val_collection[key] = dist[idxs[n_train_x:]]
    return train_collection, val_collection


class GENOTSolver(BaseSolver[NeuralOutput]):
    """Solver class for genot.GENOT linear :cite:`klein2023generative`."""

    def __init__(self, **kwargs: Any) -> None:
        """Initiate the class with any kwargs passed to the ott-jax class."""
        super().__init__()
        self._neural_kwargs = kwargs

    @property
    def problem_kind(self) -> ProblemKind_t:  # noqa: D102
        return "linear"

    def _prepare(  # type: ignore[override]
        self,
        distributions: DistributionCollection[K],
        policy: SubsetPolicy[K],
        train_size: float = 0.9,
        batch_size: int = 128,
        is_conditional: bool = False,
        seed: int = 0,
        device: Any = None,
    ) -> Tuple[PolicyDataLoader, PolicyDataLoader]:
        del device  # TODO: ignore for now, but we should handle this properly

        if train_size > 1.0 or train_size <= 0.0:
            raise ValueError("Invalid train_size. Must be: 0 < train_size <= 1")

        rng = jax.random.PRNGKey(seed)

        src_renames = tgt_renames = {
            "xy": "lin",
            "xx": "quad",
        }

        if train_size == 1.0:
            train_rng, valid_rng, rng = jax.random.split(rng, 3)
            train_loader = PolicyDataLoader(
                rng=train_rng,
                policy=policy,
                distributions=distributions,
                batch_size=batch_size,
                plan=policy.plan(),
                src_renames=src_renames,
                tgt_renames=tgt_renames,
            )
            validate_loader = PolicyDataLoader(
                rng=valid_rng,
                policy=policy,
                distributions=distributions,
                batch_size=batch_size,
                plan=policy.plan(),
                src_renames=src_renames,
                tgt_renames=tgt_renames,
            )

        else:
            train_rng, valid_rng, split_rng, rng = jax.random.split(rng, 4)
            train_dist, valid_dist = _split_data(split_rng, distributions, train_size=train_size)
            train_loader = PolicyDataLoader(
                rng=train_rng,
                policy=policy,
                distributions=train_dist,
                batch_size=batch_size,
                plan=policy.plan(),
                src_renames=src_renames,
                tgt_renames=tgt_renames,
            )
            validate_loader = PolicyDataLoader(
                rng=valid_rng,
                policy=policy,
                distributions=valid_dist,
                batch_size=batch_size,
                plan=policy.plan(),
                src_renames=src_renames,
                tgt_renames=tgt_renames,
            )
        self.train_loader = train_loader
        self.validate_loader = validate_loader
        source_dim = self._neural_kwargs.get("input_dim", 0)
        target_dim = source_dim
        condition_dim = self._neural_kwargs.get("cond_dim", 0)
        # TODO(ilan-gold): What are reasonable defaults here?
        neural_vf = VelocityField(
            output_dims=[*self._neural_kwargs.get("velocity_field_output_dims", []), target_dim],
            condition_dims=(
                self._neural_kwargs.get("velocity_field_condition_dims", [source_dim + condition_dim])
                if is_conditional
                else None
            ),
            hidden_dims=self._neural_kwargs.get("velocity_field_hidden_dims", [1024, 1024, 1024]),
            time_dims=self._neural_kwargs.get("velocity_field_time_dims", None),
            time_encoder=self._neural_kwargs.get(
                "velocity_field_time_encoder", functools.partial(time_encoder.cyclical_time_encoder, n_freqs=1024)
            ),
        )
        seed = self._neural_kwargs.get("seed", 0)
        rng = jax.random.PRNGKey(seed)
        data_match_fn_kwargs = self._neural_kwargs.get(
            "data_match_fn_kwargs",
            {} if "data_match_fn" in self._neural_kwargs else {"epsilon": 1e-1, "tau_a": 1.0, "tau_b": 1.0},
        )
        time_sampler = self._neural_kwargs.get("time_sampler", uniform_sampler)
        optimizer = self._neural_kwargs.get("optimizer", optax.adam(learning_rate=1e-4))
        self._solver = genot.GENOT(
            vf=neural_vf,
            flow=self._neural_kwargs.get(
                "flow",
                dynamics.ConstantNoiseFlow(0.1),
            ),
            data_match_fn=functools.partial(
                self._neural_kwargs.get("data_match_fn", data_match_fn), typ="lin", **data_match_fn_kwargs
            ),
            source_dim=source_dim,
            target_dim=target_dim,
            condition_dim=condition_dim if is_conditional else None,
            optimizer=optimizer,
            time_sampler=time_sampler,
            rng=rng,
            latent_noise_fn=self._neural_kwargs.get("latent_noise_fn", None),
            **self._neural_kwargs.get("velocity_field_train_state_kwargs", {}),
        )
        return train_loader, validate_loader


    @property
    def solver(self) -> genot.GENOT:
        """Underlying optimal transport solver."""
        return self._solver

    @classmethod
    def _call_kwargs(cls) -> Tuple[Set[str], Set[str]]:
        return {"batch_size", "train_size", "trainloader", "validloader", "seed"}, {}  # type: ignore[return-value]

    def _solve(self, data_samplers: Tuple[PolicyDataLoader, PolicyDataLoader]) -> NeuralOutput:  # type: ignore[override]
        seed = self._neural_kwargs.get("seed", 0)  # TODO(ilan-gold): unify rng hadnling like OTT tests
        rng = jax.random.PRNGKey(seed)
        logs = self.solver(
            data_samplers[0], n_iters=self._neural_kwargs.get("n_iters", 100), rng=rng
        )  # TODO(ilan-gold): validation and figure out defualts
        return NeuralOutput(self.solver, logs)
