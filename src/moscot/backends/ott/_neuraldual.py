from collections import abc, defaultdict
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import optax
from flax.core import freeze
from flax.core.scope import FrozenVariableDict
from flax.training.train_state import TrainState
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
import numpy as np
from ott.geometry import costs
from ott.problems.linear.potentials import DualPotentials

from moscot._logging import logger
from moscot._types import ArrayLike
from moscot.backends.ott._icnn import ICNN
from moscot.backends.ott._jax_data import JaxSampler
from moscot.backends.ott._utils import (
    ConditionalDualPotentials,
    RunningAverageMeter,
    _compute_metrics_sinkhorn,
    _get_icnn,
    _get_optimizer,
    sinkhorn_divergence,
)

Train_t = Dict[str, Dict[str, Union[float, List[float]]]]


class OTTNeuralDualSolver:
    """Solver of the ICNN-based Kantorovich dual.

    Optimal transport mapping via input convex neural networks,
    Makkuva-Taghvaei-Lee-Oh, ICML'20.
    http://proceedings.mlr.press/v119/makkuva20a/makkuva20a.pdf

    Parameters
    ----------
    input_dim
        Input dimension of data (without condition)
    conditional
        Whether to use partial input convex neural networks (:cite:`bunne2022supervised`).
    batch_size
        Batch size.
    tau_a
        Unbalancedness parameter in the source distribution in the inner sampling loop.
    tau_b
        Unbalancedness parameter in the target distribution in the inner sampling loop.
    epsilon
        Entropic regularisation parameter in the inner sampling loop.
    seed
        Seed for splitting the data.
    pos_weights
        If `True` enforces non-negativity of corresponding weights of ICNNs, else only penalizes negativity.
    dim_hidden
        The length of `dim_hidden` determines the depth of the ICNNs, while the entries of the list determine
        the layer widhts.
    beta
        If `pos_weights` is not `None`, this determines the multiplicative constant of L2-penalization of
        negative weights in ICNNs.
    best_model_metric
        Which metric to use to assess model training. The specified metric needs to be computed in the passed
        `callback_func`. By default `sinkhorn_loss_forward` only takes into account the error in the forward map,
        while `sinkhorn` computes the mean error between the forward and the inverse map.
    iterations
        Number of (outer) training steps (batches) of the training process.
    inner_iters
        Number of inner iterations for updating the convex conjugate.
    valid_freq
        Frequency at which the model is evaluated.
    log_freq
        Frequency at which training is logged.
    patience
        Number of iterations of no performance increase after which to apply early stopping.
    optimizer_f_kwargs
        Keyword arguments for the optimizer :class:`optax.adamw` for f.
    optimizer_g_kwargs
        Keyword arguments for the optimizer :class:`optax.adamw` for g.
    pretrain_iters
        Number of iterations (batches) for pretraining with the identity map.
    pretrain_scale
        Variance of Gaussian distribution used for pretraining.
    valid_sinkhorn_kwargs
        Keyword arguments for computing the discrete sinkhorn divergence for assessing model training.
        By default, the same `tau_a`, `tau_b` and `epsilon` are taken as for the inner sampling loop.
    compute_wasserstein_baseline
        Whether to compute the Sinkhorn divergence between the source and the target distribution as
        a baseline for the Wasserstein-2 distance computed with the neural solver.
    callback_func
        Callback function to compute metrics during training. The function takes as input the
        target and source batch and the predicted target and source batch and returns a dictionary of
        metrics.

    Warning
    -------
    If `compute_wasserstein_distance` is `True`, a discrete OT problem has to be solved on the validation
    dataset which scales linearly in the validation set size. If `train_size=1.0` the validation dataset size
    is the full dataset size, hence this is a source of prolonged run time or Out of Memory Error.
    """

    def __init__(
        self,
        input_dim: int,
        cond_dim: int = 0,
        batch_size: int = 1024,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        epsilon: float = 0.1,
        seed: int = 0,
        pos_weights: bool = False,
        f: Union[Dict[str, Any], ICNN] = MappingProxyType({}),
        g: Union[Dict[str, Any], ICNN] = MappingProxyType({}),
        beta: float = 1.0,
        best_model_metric: str = "sinkhorn_loss_forward",
        iterations: int = 25000,  # TODO(@MUCDK): rename to max_iterations
        inner_iters: int = 10,
        valid_freq: int = 250,
        log_freq: int = 10,
        patience: int = 100,
        optimizer_f: Union[Dict[str, Any], Type[optax.GradientTransformation]] = MappingProxyType({}),
        optimizer_g: Union[Dict[str, Any], Type[optax.GradientTransformation]] = MappingProxyType({}),
        pretrain_iters: int = 15001,
        pretrain_scale: float = 3.0,
        valid_sinkhorn_kwargs: Dict[str, Any] = MappingProxyType({}),
        compute_wasserstein_baseline: bool = False,
        callback_func: Optional[
            Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], Dict[str, float]]
        ] = None,
    ):
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.batch_size = batch_size
        self.tau_a = 1.0 if tau_a is None else tau_a
        self.tau_b = 1.0 if tau_b is None else tau_b
        self.epsilon = epsilon if self.tau_a != 1.0 or self.tau_b != 1.0 else None
        self.pos_weights = pos_weights
        self.beta = beta
        self.best_model_metric = best_model_metric
        self.iterations = iterations
        self.inner_iters = inner_iters
        self.valid_freq = valid_freq
        self.log_freq = log_freq
        self.patience = patience
        self.pretrain_iters = pretrain_iters
        self.pretrain_scale = pretrain_scale
        self.valid_sinkhorn_kwargs = dict(valid_sinkhorn_kwargs)
        self.valid_sinkhorn_kwargs.setdefault("tau_a", self.tau_a)
        self.valid_sinkhorn_kwargs.setdefault("tau_b", self.tau_b)
        self.valid_eps = self.valid_sinkhorn_kwargs.pop("epsilon", 1e-2)
        self.compute_wasserstein_baseline = compute_wasserstein_baseline
        self.key: jax.random.PRNGKeyArray = jax.random.PRNGKey(seed)

        self.optimizer_f = _get_optimizer(**optimizer_f) if isinstance(optimizer_f, abc.Mapping) else optimizer_f
        self.optimizer_g = _get_optimizer(**optimizer_g) if isinstance(optimizer_g, abc.Mapping) else optimizer_g
        self.neural_f = _get_icnn(input_dim=input_dim, cond_dim=cond_dim, **f) if isinstance(f, abc.Mapping) else f
        self.neural_g = _get_icnn(input_dim=input_dim, cond_dim=cond_dim, **g) if isinstance(g, abc.Mapping) else g
        self.callback_func = callback_func
        if self.callback_func is None:
            self.callback_func = lambda tgt, src, pred_tgt, pred_src: _compute_metrics_sinkhorn(
                tgt, src, pred_tgt, pred_src, self.valid_eps, self.valid_sinkhorn_kwargs
            )
        # set optimizer and networks
        self.setup(self.neural_f, self.neural_g, self.optimizer_f, self.optimizer_g)

    def setup(self, neural_f: ICNN, neural_g: ICNN, optimizer_f: optax.OptState, optimizer_g: optax.OptState):
        """Initialize all components required to train the :class:`moscot.backends.ott.NeuralDual`.

        Parameters
        ----------
        neural_f
            Network to parameterize the forward transport map.
        neural_g
            Network to parameterize the reverse transport map.
        optimizer_f
            Optimizer for `neural_f`.
        optimizer_g
            Optimizer for `neural_g`.
        """
        key_f, key_g, self.key = jax.random.split(self.key, 3)

        # check setting of network architectures
        if neural_g.pos_weights != self.pos_weights or neural_f.pos_weights != self.pos_weights:
            logger.warning(
                f"Setting of ICNN and the positive weights setting of the \
                      `NeuralDualSolver` are not consistent. Proceeding with \
                      the `NeuralDualSolver` setting, with positive weigths \
                      being {self.pos_weights}."
            )
            neural_g.pos_weights = self.pos_weights
            neural_f.pos_weights = self.pos_weights

        self.state_f = neural_f.create_train_state(key_f, optimizer_f, self.input_dim)
        self.state_g = neural_g.create_train_state(key_g, optimizer_g, self.input_dim)

        self.train_step_f = self.get_train_step(to_optimize="f")
        self.train_step_g = self.get_train_step(to_optimize="g")
        self.valid_step = self.get_eval_step()

    def __call__(
        self,
        trainloader: JaxSampler,
        validloader: JaxSampler,
    ) -> Tuple[DualPotentials, Train_t]:
        """Start the training pipeline of the :class:`moscot.backends.ott.NeuralDual`.

        Parameters
        ----------
        trainloader
            Data loader for the training data.
        validloader
            Data loader for the validation data.

        Returns
        -------
        The trained model and training statistics.
        """
        pretrain_logs = {}
        if self.pretrain_iters > 0:
            pretrain_logs = self.pretrain_identity(trainloader.conditions)

        train_logs = self.train_neuraldual(trainloader, validloader)
        res = self.to_dual_potentials()
        logs = pretrain_logs | train_logs

        return (res, logs)

    def pretrain_identity(
        self, conditions: Optional[jnp.ndarray]
    ) -> Train_t:  # TODO(@lucaeyr) conditions can be `None` right?
        """Pretrain the neural networks to parameterize the identity map.

        Parameters
        ----------
        conditions
            Conditions in the case of a conditional Neural OT model, otherwise `None`.

        Returns
        -------
        Pre-training statistics.
        """

        @jax.jit
        def pretrain_loss_fn(
            params: jnp.ndarray,
            data: jnp.ndarray,
            condition: jnp.ndarray,
            state: TrainState,
        ) -> float:
            """Loss function for the pretraining on identity."""
            grad_g_data = jax.vmap(jax.grad(lambda x: state.apply_fn({"params": params}, x, condition), argnums=0))(
                data
            )
            # loss is L2 reconstruction of the input
            return ((grad_g_data - data) ** 2).sum(axis=1).mean()  # TODO make nicer

        # @jax.jit
        def pretrain_update(state: TrainState, key: jax.random.KeyArray) -> Tuple[jnp.ndarray, TrainState]:
            """Update function for the pretraining on identity."""
            # sample gaussian data with given scale
            x = self.pretrain_scale * jax.random.normal(key, [self.batch_size, self.input_dim])
            condition = jax.random.choice(key, conditions) if self.cond_dim else None
            grad_fn = jax.value_and_grad(pretrain_loss_fn, argnums=0)
            loss, grads = grad_fn(state.params, x, condition, state)
            return loss, state.apply_gradients(grads=grads)

        pretrain_logs: Dict[str, List[float]] = {"loss": []}
        for iteration in range(self.pretrain_iters):
            key_pre, self.key = jax.random.split(self.key, 2)
            # train step for potential g directly updating the train state
            loss, self.state_g = pretrain_update(self.state_g, key_pre)
            # clip weights of g
            if not self.pos_weights:
                self.state_g = self.state_g.replace(params=self.clip_weights_icnn(self.state_g.params))
            if iteration % self.log_freq == 0:
                pretrain_logs["loss"].append(loss)
        # load params of g into state_f
        # this only works when f & g have the same architecture
        self.state_f = self.state_f.replace(params=self.state_g.params)
        return {"pretrain_logs": pretrain_logs}  # type:ignore[dict-item]

    def train_neuraldual(
        self,
        trainloader: JaxSampler,
        validloader: JaxSampler,
    ) -> Train_t:
        """Train the model.

        Parameters
        ----------
        trainloader
            Data loader for the training data.
        validloader
            Data loader for the validation data.

        Returns
        -------
        Training statistics.
        """
        # set logging dictionaries
        train_logs: Dict[str, List[float]] = defaultdict(list)
        valid_logs: Dict[str, Union[List[float], float]] = defaultdict(list)
        average_meters: Dict[str, RunningAverageMeter] = defaultdict(RunningAverageMeter)
        valid_average_meters: Dict[str, RunningAverageMeter] = defaultdict(RunningAverageMeter)
        sink_dist: List[float] = []
        curr_patience: int = 0
        best_loss: float = jnp.inf
        best_iter_distance: float = None
        best_params_f: jnp.ndarray = None
        best_params_g: jnp.ndarray = None

        # define dict to contain source and target batch
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
                    sinkhorn_divergence(
                        point_cloud_1=valid_batch[pair]["source"],
                        point_cloud_2=valid_batch[pair]["target"],
                        **self.valid_sinkhorn_kwargs,
                    )
                )

        for iteration in tqdm(range(self.iterations)):
            # sample policy and condition if given in trainloader
            policy_key, target_key, self.key = jax.random.split(self.key, 3)
            policy_pair, batch["condition"] = trainloader.sample_policy_pair(policy_key)
            # sample target batch
            batch["target"] = trainloader(target_key, policy_pair, sample="target")

            if not self.is_balanced:
                # sample source batch and compute unbalanced marginals
                source_key, self.key = jax.random.split(self.key, 2)
                curr_source = trainloader(source_key, policy_pair, sample="source")
                marginals_source, marginals_target = trainloader.compute_unbalanced_marginals(
                    curr_source, batch["target"]
                )

            for _ in range(self.inner_iters):
                source_key, self.key = jax.random.split(self.key, 2)

                if self.is_balanced:
                    # sample source batch
                    batch["source"] = trainloader(source_key, policy_pair, sample="source")
                else:
                    # resample source with unbalanced marginals
                    batch["source"] = trainloader.unbalanced_resample(source_key, curr_source, marginals_source)
                # train step for potential g directly updating the train state
                self.state_f, train_f_metrics = self.train_step_f(self.state_f, self.state_g, batch)
                for key, value in train_f_metrics.items():
                    average_meters[key].update(value)
            # resample target batch with unbalanced marginals
            if self.epsilon is not None:
                target_key, self.key = jax.random.split(self.key, 2)
                batch["target"] = trainloader.unbalanced_resample(target_key, batch["target"], marginals_target)
            # train step for potential f directly updating the train state
            self.state_g, train_g_metrics = self.train_step_g(self.state_f, self.state_g, batch)
            for key, value in train_g_metrics.items():
                average_meters[key].update(value)
            # clip weights of f
            if not self.pos_weights:
                self.state_g = self.state_g.replace(params=self.clip_weights_icnn(self.state_g.params))
            # log avg training values periodically
            if iteration % self.log_freq == 0:
                for key, average_meter in average_meters.items():
                    train_logs[key].append(average_meter.avg)
                    average_meter.reset()
            # evalute on validation set periodically
            if iteration % self.valid_freq == 0:
                for index, pair in enumerate(trainloader.policy_pairs):
                    condition = trainloader.conditions[index] if self.cond_dim else None
                    valid_metrics = self.valid_step(self.state_f, self.state_g, valid_batch[pair], condition)
                    for key, value in valid_metrics.items():
                        valid_logs[f"{pair[0]}_{pair[1]}_{key}"].append(value)  # type:ignore[union-attr]
                        valid_average_meters[key].update(value)
                # update best model and patience as necessary
                if self.best_model_metric == "sinkhorn":
                    total_loss = (
                        valid_average_meters["sinkhorn_loss_forward"].avg
                        + valid_average_meters["sinkhorn_loss_inverse"].avg
                    )
                else:
                    try:
                        total_loss = valid_average_meters[self.best_model_metric].avg
                    except ValueError:
                        f"Unknown metric: {self.best_model_metric}."
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_iter_distance = valid_average_meters["neural_dual_dist"].avg
                    best_params_f = self.state_f.params
                    best_params_g = self.state_g.params
                    curr_patience = 0
                else:
                    curr_patience += 1
                for key, average_meter in valid_average_meters.items():
                    valid_logs[f"mean_{key}"].append(average_meter.avg)  # type:ignore[union-attr]
                    average_meter.reset()
            if curr_patience >= self.patience:
                break
        self.state_f = self.state_f.replace(params=best_params_f)
        self.state_g = self.state_g.replace(params=best_params_g)
        valid_logs["best_loss"] = best_loss
        valid_logs["predicted_cost"] = None if best_iter_distance is None else float(best_iter_distance)
        if self.compute_wasserstein_baseline:
            valid_logs["sinkhorn_dist"] = np.mean(sink_dist)
        return {
            "train_logs": train_logs,  # type:ignore[dict-item]
            "valid_logs": valid_logs,
        }

    def get_train_step(
        self,
        to_optimize: Literal["f", "g"],
    ) -> Callable[[TrainState, TrainState, Dict[str, jnp.ndarray]], Tuple[TrainState, Dict[str, float]]]:
        """Get one training step."""

        def loss_f_fn(
            params_f: jnp.ndarray,
            params_g: jnp.ndarray,
            state_f: TrainState,
            state_g: TrainState,
            batch: Dict[str, jnp.ndarray],
        ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
            """Loss function for f."""
            # get loss terms of kantorovich dual
            grad_f_src = jax.vmap(
                jax.grad(lambda x: state_f.apply_fn({"params": params_f}, x, batch["condition"]), argnums=0)
            )(batch["source"])
            g_grad_f_src = jax.vmap(lambda x: state_g.apply_fn({"params": params_g}, x, batch["condition"]))(grad_f_src)
            src_dot_grad_f_src = jnp.sum(batch["source"] * grad_f_src, axis=1)
            # compute loss
            loss = jnp.mean(g_grad_f_src - src_dot_grad_f_src)
            if not self.pos_weights:
                penalty = self.beta * self.penalize_weights_icnn(params_f)
                loss += penalty
            else:
                penalty = 0
            return loss, [penalty]

        def loss_g_fn(
            params_f: jnp.ndarray,
            params_g: jnp.ndarray,
            state_f: TrainState,
            state_g: TrainState,
            batch: Dict[str, jnp.ndarray],
        ) -> Tuple[jnp.ndarray, List[float]]:
            """Loss function for g."""
            # get loss terms of kantorovich dual
            grad_f_src = jax.vmap(
                jax.grad(lambda x: state_f.apply_fn({"params": params_f}, x, batch["condition"]), argnums=0)
            )(batch["source"])
            g_grad_f_src = jax.vmap(lambda x: state_g.apply_fn({"params": params_g}, x, batch["condition"]))(grad_f_src)
            src_dot_grad_f_src = jnp.sum(batch["source"] * grad_f_src, axis=1)
            # compute loss
            f_tgt = jax.vmap(lambda x: state_g.apply_fn({"params": params_g}, x, batch["condition"]))(batch["target"])
            loss = jnp.mean(f_tgt - g_grad_f_src)
            total_loss = jnp.mean(g_grad_f_src - f_tgt - src_dot_grad_f_src)
            # compute wasserstein distance
            dist = 2 * total_loss + jnp.mean(
                jnp.sum(batch["target"] * batch["target"], axis=1)
                + 0.5 * jnp.sum(batch["source"] * batch["source"], axis=1)
            )
            return loss, [total_loss, dist]

        @jax.jit
        def step_fn(
            state_f: TrainState,
            state_g: TrainState,
            batch: Dict[str, jnp.ndarray],
        ) -> Tuple[TrainState, Dict[str, float]]:
            """Step function for training."""
            # get loss function for f or g
            if to_optimize == "f":
                grad_fn = jax.value_and_grad(loss_f_fn, argnums=1, has_aux=True)
                # compute loss, gradients and metrics
                (loss, raw_metrics), grads = grad_fn(state_f.params, state_g.params, state_f, state_g, batch)
                # return updated state and metrics dict
                metrics = {"loss_f": loss, "penalty": raw_metrics[0]}
                return state_f.apply_gradients(grads=grads), metrics
            if to_optimize == "g":
                grad_fn = jax.value_and_grad(loss_g_fn, argnums=0, has_aux=True)
                # compute loss, gradients and metrics
                (loss, raw_metrics), grads = grad_fn(state_f.params, state_g.params, state_f, state_g, batch)
                # return updated state and metrics dict
                metrics = {"loss_g": loss, "loss": raw_metrics[0], "w_dist": raw_metrics[1]}
                return state_g.apply_gradients(grads=grads), metrics
            raise NotImplementedError()

        return step_fn

    def get_eval_step(
        self,
    ) -> Callable[[TrainState, TrainState, Dict[str, jnp.ndarray], jnp.ndarray], Dict[str, float]]:
        """Get one validation step."""

        @jax.jit
        def valid_step(
            state_f: TrainState,
            state_g: TrainState,
            batch: Dict[str, jnp.ndarray],
            condition: jnp.ndarray,
        ) -> Dict[str, float]:
            """Create a validation function."""
            # get transported source and inverse transported target
            pred_target = jax.vmap(
                jax.grad(lambda x: state_f.apply_fn({"params": state_f.params}, x, condition), argnums=0)
            )(batch["source"])
            pred_source = jax.vmap(
                jax.grad(lambda x: state_g.apply_fn({"params": state_g.params}, x, condition), argnums=0)
            )(batch["target"])
            pred_target = pred_target
            pred_source = pred_source
            # get neural dual distance between true source and target
            g_tgt = jax.vmap(lambda x: state_g.apply_fn({"params": state_g.params}, x, condition))(batch["target"])
            g_grad_f_src = jax.vmap(lambda x: state_g.apply_fn({"params": state_g.params}, x, condition))(pred_target)
            src_dot_grad_f_src = jnp.sum(batch["source"] * pred_target, axis=-1)
            src_sq = jnp.mean(jnp.sum(batch["source"] ** 2, axis=-1))
            tgt_sq = jnp.mean(jnp.sum(batch["target"] ** 2, axis=-1))
            neural_dual_dist = tgt_sq + src_sq + 2.0 * (jnp.mean(g_grad_f_src - src_dot_grad_f_src) - jnp.mean(g_tgt))
            # calculate validation metrics
            metric_dict = self.callback_func(batch["target"], batch["source"], pred_target, pred_source)
            return metric_dict | {"neural_dual_dist": neural_dual_dist}

        return valid_step

    def clip_weights_icnn(self, params: FrozenVariableDict) -> FrozenVariableDict:
        """Clip weights of ICNN."""
        params = params.unfreeze()
        for key in params:
            if key.startswith("w_zs"):
                params[key]["kernel"] = jnp.clip(params[key]["kernel"], a_min=0)

        return freeze(params)

    def penalize_weights_icnn(self, params: FrozenVariableDict) -> float:
        """Penalize weights of ICNN."""
        penalty = 0
        for key in params:
            if key.startswith("w_z"):
                penalty += jnp.linalg.norm(jax.nn.relu(-params[key]["kernel"]))
        return penalty

    def to_dual_potentials(self, condition: Optional[ArrayLike] = None) -> DualPotentials:
        """Return the Kantorovich dual potentials from the trained potentials."""
        if self.cond_dim:
            return ConditionalDualPotentials(self.state_f, self.state_g)

        def f(x) -> float:
            return self.state_f.apply_fn({"params": self.state_f.params}, x)

        def g(x) -> float:
            return self.state_g.apply_fn({"params": self.state_g.params}, x)

        return DualPotentials(f, g, corr=True, cost_fn=costs.SqEuclidean())

    @property
    def is_balanced(self) -> bool:
        """Return whether the problem is balanced."""
        return self.tau_a == self.tau_b == 1.0
