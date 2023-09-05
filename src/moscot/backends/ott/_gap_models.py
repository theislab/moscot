import sys
from collections import defaultdict
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Literal, Mapping, Tuple, Union

import optax
from flax.core.scope import FrozenDict
from flax.training.early_stopping import EarlyStopping
from flax.training.train_state import TrainState
from tqdm import trange

import jax
import jax.nn as nn
import jax.numpy as jnp
import numpy as np
from ott.geometry import costs
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import acceleration, sinkhorn
from ott.solvers.nn.models import MLP, ModelBase
from ott.tools.sinkhorn_divergence import sinkhorn_divergence

from moscot._logging import logger
from moscot._types import ArrayLike
from moscot.backends.ott._jax_data import JaxSampler
from moscot.backends.ott._utils import (
    RunningAverageMeter,
    _compute_sinkhorn_divergence,
    _regularized_wasserstein,
    compute_ds_diff,
    mmd_rbf,
)

Train_t = Dict[str, Dict[str, List[float]]]


class MongeGap:
    r"""
    A class to define the Monge gap regularizer.

    From Monge Gap paper :cite:`uscidda2023monge`:
    For a cost function :`math:`c` and an empirical reference :math:`\rho_x`
    defined by samples :math:`x_i`, the (entropic) Monge gap of a vector field :math:`T`
    is defined as:
        :math:`\\mathcal{M}^c_{\rho_x, \\epsilon} = \frac{1}{n} \\sum_{i=1}^n c(x_i, T(x_i))
        - W_\\epsilon(\\mu_x, T \\# \rho_x)`.

    Args:
    geometry_kwargs: Holds the kwargs to instanciate the geometry to compute the ``reg_ot_cost``.
                     Default cost function (if no cost function is passed in ``geometry_kwargs``)
                     is the quadratic costs ``ott.geometry.costs.SqEuclidean()``.
    sinkhorn_kwargs: Holds the kwargs to instanciate the Sinkhorn solver to compute the ``reg_ot_cost``.
    """

    def __init__(
        self,
        geometry_kwargs: Mapping[str, Any] = MappingProxyType({}),
        sinkhorn_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ) -> None:
        self.geometry_kwargs = geometry_kwargs
        self.sinkhorn_kwargs = sinkhorn_kwargs

    def __call__(self, samples: jnp.ndarray, T: Callable[[jnp.ndarray], jnp.ndarray]) -> float:
        """TODO: add docstring."""
        T_samples = T(samples)
        geom = PointCloud(x=samples, y=T_samples, **self.geometry_kwargs)

        id_displacement = jnp.mean(jax.vmap(self.cost_fn)(samples, T_samples))

        opt_displacement = sinkhorn.Sinkhorn(**self.sinkhorn_kwargs)(linear_problem.LinearProblem(geom)).reg_ot_cost
        opt_displacement = jnp.add(opt_displacement, -2 * geom.epsilon * jnp.log(len(samples)))

        return id_displacement - opt_displacement

    @property
    def cost_fn(self) -> costs.CostFn:
        """TODO: add docstring."""
        if "cost_fn" in self.geometry_kwargs:
            return self.geometry_kwargs["cost_fn"]
        return costs.SqEuclidean()


class MongeGapSolver:
    def __init__(
        self,
        input_dim: int,
        batch_size: int = 256,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        epsilon: float = 0.1,
        seed: int = 0,
        best_model_metric: Literal["sinkhorn_forward", "sinkhorn"] = "sinkhorn_forward",
        max_iterations: int = 25000,
        valid_freq: int = 250,
        log_freq: int = 10,
        patience: int = 100,
        min_improvement: float = 1e-3,
        neural_net: ModelBase = MLP(dim_hidden=[128, 64, 64], is_potential=False, act_fn=nn.gelu),
        optimizer: optax.OptState = optax.adamw(
            learning_rate=1e-3,
            b1=0.5,
            b2=0.9,
            weight_decay=0.0,
        ),
        valid_sinkhorn_kwargs: Dict[str, Any] = MappingProxyType({}),
        compute_wasserstein_baseline: bool = True,
        lambda_monge_gap: float = 0.1,
        monge_gap_geom_kwargs: Dict[str, Any] = MappingProxyType({}),
        monge_gap_sinkhorn_kwargs: Dict[str, Any] = MappingProxyType({}),
        trial: Any = None,
        use_relative: bool = False,
    ) -> None:
        """ToDo: Add docstring."""
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.tau_a = 1.0 if tau_a is None else tau_a
        self.tau_b = 1.0 if tau_b is None else tau_b
        self.epsilon = epsilon if self.tau_a != 1.0 or self.tau_b != 1.0 else None
        self.best_model_metric = best_model_metric
        self.max_iterations = max_iterations
        self.valid_freq = valid_freq
        self.log_freq = log_freq
        self.patience = patience
        self.min_improvement = min_improvement
        self.lambda_monge_gap = lambda_monge_gap
        self._monge_gap_geom_kwargs = monge_gap_geom_kwargs
        self._monge_gap_sinkhorn_kwargs = monge_gap_sinkhorn_kwargs
        self._neural_net = neural_net
        self._optimizer = optimizer
        self.valid_sinkhorn_kwargs = dict(valid_sinkhorn_kwargs)
        self.valid_sinkhorn_kwargs.setdefault("tau_a", self.tau_a)
        self.valid_sinkhorn_kwargs.setdefault("tau_b", self.tau_b)
        self.valid_eps = self.valid_sinkhorn_kwargs.pop("epsilon", 1e-2)
        self.compute_wasserstein_baseline = compute_wasserstein_baseline
        self.key: ArrayLike = jax.random.PRNGKey(seed)
        self.trial = trial
        self.use_relative = use_relative if compute_wasserstein_baseline else False

        self._state_neural_net = None
        self._setup()

    def _setup(self):
        """Set up the Monge gap, neural network and optimizer to use. Get train and evaluation steps."""
        if self._monge_gap_geom_kwargs is None:
            self._monge_gap_geom_kwargs = {"epsilon": 1e-2, "relative_epsilon": True, "cost_fn": costs.Euclidean()}

        if self._monge_gap_sinkhorn_kwargs is None:
            self._monge_gap_sinkhorn_kwargs = {
                "momentum": acceleration.Momentum(value=1.0, start=25),
                "use_danskin": True,  # TODO: Can this be removed?
            }

        self._monge_gap = MongeGap(
            geometry_kwargs=self._monge_gap_geom_kwargs,
            sinkhorn_kwargs=self._monge_gap_sinkhorn_kwargs,
        )

        self._state_neural_net = self._neural_net.create_train_state(self.key, self._optimizer, self.input_dim)

        self._train_step = self.get_train_step()
        self._valid_step = self.get_eval_step()

    def __call__(
        self,
        trainloader: JaxSampler,
        validloader: JaxSampler,
    ) -> Tuple[TrainState, Train_t]:
        """
        Start the training pipeline of the :class:'moscot.backends.ott.MongeGapSolver.

        Parameters
        ----------
        :param trainloader: Data loader for the training data.
        :param validloader: Data loader for the validation data.

        :returns: trained model and training statistics.
        """
        # TODO: Check the return type. The object is actually descendent of TrainState, NeuralTrainState

        train_and_validation_logs = self.train_monge_gap(trainloader, validloader)

        # Using old syntax for backwards compatibility
        logs = {**train_and_validation_logs}

        return self._state_neural_net, logs

    def train_monge_gap(
        self,
        trainloader: JaxSampler,
        validloader: JaxSampler,
    ) -> Train_t:
        """
        Train the model.
        :param trainloader: Data loader for the training data.
        :param validloader: Data loader for the validation data.

        :returns:
        Training statistics.
        """
        # Set logging dictionaries
        train_logs: Dict[str, List[float]] = defaultdict(list)
        valid_logs: Dict[str, Union[List[float], float]] = defaultdict(list)
        average_meters: Dict[str, RunningAverageMeter] = defaultdict(RunningAverageMeter)
        valid_average_meters: Dict[str, RunningAverageMeter] = defaultdict(RunningAverageMeter)
        sink_dist: List[float] = []
        best_loss: float = jnp.inf
        best_params: jnp.ndarray = None

        # Define dict to contain source and target batch
        batch: Dict[str, jnp.ndarray] = {}
        valid_batch: Dict[Tuple[Any, Any], Dict[str, jnp.ndarray]] = {}
        for pair in trainloader.policy_pairs:
            valid_batch[pair] = {}
            valid_batch[pair]["source"], valid_batch[pair]["target"] = validloader(
                key=None, policy_pair=pair, full_dataset=True
            )
            if self.compute_wasserstein_baseline:
                if valid_batch[pair]["source"].shape[0] * valid_batch[pair]["source"].shape[1] > 25000000:
                    logger.warning(
                        "Validation Sinkhorn divergence is expensive to compute due to large size of the validation "
                        "set. Consider setting `valid_sinkhorn_divergence` to False."
                    )
                sink_dist.append(
                    _compute_sinkhorn_divergence(
                        point_cloud_1=valid_batch[pair]["source"],
                        point_cloud_2=valid_batch[pair]["target"],
                        **self.valid_sinkhorn_kwargs,
                    )
                )

        early_stop_callback = EarlyStopping(min_delta=self.min_improvement, patience=self.patience)

        # Training loop
        tbar = trange(self.max_iterations, leave=True, file=sys.stdout, colour="GREEN", ncols=100)
        for iteration in tbar:
            policy_key, target_key, self.key = jax.random.split(self.key, 3)
            policy_pair, batch["condition"] = trainloader.sample_policy_pair(policy_key)

            batch["target"] = trainloader(target_key, policy_pair, sample="target")

            if not self.is_balanced:
                source_key, self.key = jax.random.split(self.key, 2)
                curr_source = trainloader(source_key, policy_pair, sample="source")
                marginals_source, marginals_target = trainloader.compute_unbalanced_marginals(
                    curr_source, batch["target"]
                )

            source_key, self.key = jax.random.split(self.key, 2)

            if self.is_balanced:
                batch["source"] = trainloader(source_key, policy_pair, sample="source")
            else:
                batch["source"] = trainloader.unbalanced_resample(source_key, curr_source, marginals_source)

            resample_with_unbalanced_marginals = self.epsilon is not None
            if resample_with_unbalanced_marginals:
                target_key, self.key = jax.random.split(self.key, 2)
                batch["target"] = trainloader.unbalanced_resample(target_key, batch["target"], marginals_target)

            self._state_neural_net, train_metrics = self._train_step(self._state_neural_net, batch)
            for key, value in train_metrics.items():
                average_meters[key].update(value)

            is_logging_iteration = iteration % self.log_freq == 0
            if is_logging_iteration:
                for key, average_meter in average_meters.items():
                    train_logs[key].append(average_meter.avg)
                    average_meter.reset()

            is_validation_iteration = iteration % self.valid_freq == 0
            if is_validation_iteration:
                for index, pair in enumerate(trainloader.policy_pairs):
                    valid_metrics = self._valid_step(self._state_neural_net, valid_batch[pair])
                    for key, value in valid_metrics.items():
                        valid_logs[f"{pair[0]}_{pair[1]}_{key}"].append(value)
                        valid_average_meters[key].update(value)
                        if self.use_relative:
                            valid_logs[f"{pair[0]}_{pair[1]}_{key}" + "_relative"].append(value / sink_dist[0])

                if self.best_model_metric == "sinkhorn":
                    total_loss = jnp.abs(valid_average_meters["sinkhorn_loss_forward"].avg) + jnp.abs(
                        valid_average_meters["sink_loss_inverse"].avg
                    )
                elif self.best_model_metric == "sinkhorn_forward":
                    total_loss = jnp.abs(valid_average_meters["sinkhorn_loss_forward"].avg)
                else:
                    raise ValueError(f"Unknown metric: {self.best_model_metric}.")

                use_optuna = self.trial is not None
                if use_optuna:
                    self._report_to_optuna(iteration, total_loss)

                if total_loss < best_loss:
                    best_loss = total_loss
                    best_params = self._state_neural_net.params

                for key, average_meter in valid_average_meters.items():
                    valid_logs[f"mean_{key}"].append(average_meter.avg)
                    average_meter.reset()

                _, early_stop = early_stop_callback.update(total_loss)
                if early_stop.should_stop:
                    break

            if self.compute_wasserstein_baseline:
                self._state_neural_net = self._state_neural_net.replace(params=best_params)
                valid_logs["best_loss"] = float(best_loss)
                valid_logs["sinkhorn_dist"] = float(np.mean(sink_dist))

        return {
            "train_logs": train_logs,
            "valid_logs": valid_logs,
        }

    def get_train_step(
        self,
    ) -> Callable[[TrainState, TrainState, Dict[str, jnp.ndarray]], Tuple[TrainState, Dict[str, float]]]:
        """Get one training step."""

        def __loss_fn(
            params: FrozenDict,
            state_neural_net: TrainState,
            batch: Dict[str, jnp.ndarray],
        ) -> Tuple[float, Dict[str, float]]:
            """
            Loss function for training. Composed of:
            - Fitting loss: regularized wasserstein distance between source push-forward and target.
            - Monge gap loss: Monge gap as in :cite:`uscidda2023monge`.
            """
            pred_target = jax.vmap(lambda x: state_neural_net.apply_fn({"params": params}, x))(batch["source"])

            fitting_loss = _regularized_wasserstein(pred_target, batch["target"])

            monge_gap_loss = self._monge_gap(
                samples=batch["source"], T=lambda x: state_neural_net.apply_fn({"params": params}, x)
            )

            total_loss = fitting_loss + self.lambda_monge_gap * monge_gap_loss

            loss_logs = {
                "total_loss": total_loss,
                "fitting_loss": fitting_loss,
                "monge_gap_loss": monge_gap_loss,
            }

            return total_loss, loss_logs

        @jax.jit
        def step_fn(
            state_neural_net: TrainState,
            batch: Dict[str, jnp.ndarray],
        ) -> Tuple[TrainState, Dict[str, float]]:
            """Step function for training. Compute the value and gradient of the loss function and update accordingly."""
            grad_fn = jax.value_and_grad(__loss_fn, argnums=0, has_aux=True)

            (loss, raw_metrics), grads = grad_fn(state_neural_net.params, state_neural_net, batch)

            metrics = {
                "fitting_loss": raw_metrics["fitting_loss"],
                "monge_gap": raw_metrics["monge_gap_loss"],
                "loss": raw_metrics["total_loss"],
            }

            return state_neural_net.apply_gradients(grads=grads), metrics

        return step_fn

    def get_eval_step(
        self,
    ) -> Callable[[TrainState, TrainState, Dict[str, jnp.ndarray]], Dict[str, float]]:
        """Get one validation step."""

        def __loss_fn(
            params: FrozenDict,
            state_neural_net: TrainState,
            batch: Dict[str, jnp.ndarray],
        ) -> Tuple[float, Dict[str, float]]:
            """
            Loss function for validation. As per :cite:`bunne2022supervised`, we compute:
            - Sinkhorn divergence between predicted target and ground truth
            - Maximum Mean Discrepancy
            - Drug Signature difference, i.e. the euclidian distance between the vectors of means.
            """
            pred_target = jax.vmap(lambda x: state_neural_net.apply_fn({"params": params}, x))(batch["source"])

            sink_loss_forward = sinkhorn_divergence(
                PointCloud,
                x=pred_target,
                y=batch["target"],
                epsilon=self.valid_eps,
                sinkhorn_kwargs=self.valid_sinkhorn_kwargs,
            ).divergence

            mmd_dist = mmd_rbf(
                x=pred_target,
                y=batch["target"],
            )

            ds_diff = compute_ds_diff(
                control=batch["source"],
                treated=batch["target"],
                push_fwd=pred_target,
            )

            valid_loss_logs = {
                "ds_diff": ds_diff,
                "mmd_dist": mmd_dist,
            }

            return sink_loss_forward, valid_loss_logs

        @jax.jit
        def valid_step(
            state_neural_net: TrainState,
            batch: Dict[str, jnp.ndarray],
        ) -> Dict[str, float]:
            """Create a validation function."""
            # Compute gradient of loss function and loss on valid batch
            grad_fn = jax.value_and_grad(__loss_fn, argnums=0, has_aux=True)
            (sink_loss, raw_metrics), grads = grad_fn(state_neural_net.params, state_neural_net, batch)

            return {"sink_loss": sink_loss, **raw_metrics}

        return valid_step

    @property
    def is_balanced(self) -> bool:
        """Return whether the problem is balanced."""
        return self.tau_a == self.tau_b == 1.0
