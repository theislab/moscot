from collections import abc, defaultdict
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import optax
from flax.core.scope import FrozenVariableDict
from flax.training import train_state
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
import numpy as np
from ott.geometry import costs
from ott.problems.linear import potentials
from ott.problems.linear.potentials import DualPotentials
from ott.solvers.nn import models

from moscot._logging import logger
from moscot._types import ArrayLike
from moscot.backends.ott._jax_data import JaxSampler
from moscot.backends.ott._utils import (
    ConditionalDualPotentials,
    RunningAverageMeter,
    _compute_metrics_sinkhorn,
    _get_icnn,
    _get_optimizer,
    sinkhorn_divergence,
)
from moscot.backends.ott.nets._icnn import ICNN

Train_t = Dict[str, Dict[str, Union[float, List[float]]]]


class UnbalancedNeuralMixin:
    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        cond_dim: int,
        mlp_eta: Optional[models.ModelBase],
        mlp_xi: Optional[models.ModelBase],
        seed: Optional[int] = None,
        opt_eta: Optional[optax.GradientTransformation] = None,
        opt_xi: Optional[optax.GradientTransformation] = None,
        **_: Any,
    ) -> None:
        self.mlp_eta = mlp_eta
        self.mlp_xi = mlp_xi
        self.state_eta: Optional[train_state.TrainState] = None
        self.state_xi: Optional[train_state.TrainState] = None
        self.opt_eta = opt_eta
        self.opt_xi = opt_xi
        self._key: jax.random.PRNGKeyArray = jax.random.PRNGKey(seed) if seed is not None else jax.random.PRNGKey(0)

        self._setup(source_dim=source_dim, target_dim=target_dim, cond_dim=cond_dim)

    def _setup(self, source_dim: int, target_dim: int, cond_dim: int):
        self.unbalancedness_step_fn = self._get_step_fn()
        if self.mlp_eta is not None:
            self.opt_eta = (
                self.opt_eta if self.opt_eta is not None else optax.adamw(learning_rate=1e-4, weight_decay=1e-10)
            )
            self.state_eta = self.mlp_eta.create_train_state(self._key, self.opt_eta, source_dim + cond_dim)
        if self.mlp_xi is not None:
            self.opt_xi = (
                self.opt_xi if self.opt_xi is not None else optax.adamw(learning_rate=1e-4, weight_decay=1e-10)
            )
            self.state_xi = self.mlp_xi.create_train_state(self._key, self.opt_xi, target_dim + cond_dim)

    def _get_step_fn(self) -> Callable:  # type:ignore[type-arg]
        def loss_a_fn(
            params_eta: Optional[jnp.ndarray],
            apply_fn_eta: Callable[[Dict[str, jnp.ndarray], jnp.ndarray], jnp.ndarray],
            x: jnp.ndarray,
            a: jnp.ndarray,
            expectation_reweighting: float,
        ) -> Tuple[float, jnp.ndarray]:
            eta_predictions = apply_fn_eta({"params": params_eta}, x)
            return (
                optax.l2_loss(eta_predictions[:, 0], a).mean()
                + optax.l2_loss(jnp.mean(eta_predictions) - expectation_reweighting),
                eta_predictions,
            )

        def loss_b_fn(
            params_xi: Optional[jnp.ndarray],
            apply_fn_xi: Callable[[Dict[str, jnp.ndarray], jnp.ndarray], jnp.ndarray],
            x: jnp.ndarray,
            b: jnp.ndarray,
            expectation_reweighting: float,
        ) -> Tuple[float, jnp.ndarray]:
            xi_predictions = apply_fn_xi({"params": params_xi}, x)
            return (
                optax.l2_loss(xi_predictions[:, 0], b).mean()
                + optax.l2_loss(jnp.mean(xi_predictions) - expectation_reweighting),
                xi_predictions,
            )

        @jax.jit
        def step_fn(
            source: jnp.ndarray,
            target: jnp.ndarray,
            a: jnp.ndarray,
            b: jnp.ndarray,
            state_eta: Optional[train_state.TrainState] = None,
            state_xi: Optional[train_state.TrainState] = None,
            *,
            is_training: bool = True,
        ):
            if state_eta is not None:
                grad_a_fn = jax.value_and_grad(loss_a_fn, argnums=0, has_aux=True)
                (loss_a, eta_predictions), grads_eta = grad_a_fn(
                    state_eta.params,
                    state_eta.apply_fn,
                    source[:,],
                    a * len(a),
                    jnp.sum(b),
                )
                new_state_eta = state_eta.apply_gradients(grads=grads_eta) if is_training else None

            else:
                new_state_eta = eta_predictions = loss_a = None
            if state_xi is not None:
                grad_b_fn = jax.value_and_grad(loss_b_fn, argnums=0, has_aux=True)
                (loss_b, xi_predictions), grads_xi = grad_b_fn(
                    state_xi.params,
                    state_xi.apply_fn,
                    target,
                    b * len(b),
                    jnp.sum(a),
                )
                new_state_xi = state_xi.apply_gradients(grads=grads_xi) if is_training else None
            else:
                new_state_xi = xi_predictions = loss_b = None

            return new_state_eta, new_state_xi, eta_predictions, xi_predictions, loss_a, loss_b

        return step_fn

    @staticmethod
    def _update_unbalancedness_logs(
        logs: Dict[str, List[float]],
        loss_eta: Optional[jnp.ndarray],
        loss_xi: Optional[jnp.ndarray],
        *,
        is_train_set: bool = True,
    ) -> Dict[str, List[float]]:
        if is_train_set:
            if loss_eta is not None:
                logs["train_loss_eta"].append(float(loss_eta))
            if loss_xi is not None:
                logs["train_loss_xi"].append(float(loss_xi))
        else:
            if loss_eta is not None:
                logs["valid_loss_eta"].append(float(loss_eta))
            if loss_xi is not None:
                logs["valid_loss_xi"].append(float(loss_xi))
        return logs


class OTTNeuralDualSolver(UnbalancedNeuralMixin):
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
        mlp_eta: Callable[[jnp.ndarray], float] = None,
        mlp_xi: Callable[[jnp.ndarray], float] = None,
        unbalancedness_kwargs: Dict[str, Any] = MappingProxyType({}),
        epsilon: float = 0.1,
        seed: int = 0,
        pos_weights: bool = False,
        f: Union[Dict[str, Any], ICNN] = MappingProxyType({}),
        g: Union[Dict[str, Any], ICNN] = MappingProxyType({}),
        beta: float = 1.0,
        best_model_selection: bool = True,
        iterations: int = 25000,  # TODO(@MUCDK): rename to max_iterations
        inner_iters: int = 10,
        valid_freq: int = 250,
        log_freq: int = 10,
        patience: int = 100,
        patience_metric: Literal[
            "train_loss_f",
            "train_loss_g",
            "train_w_dist",
            "valid_loss_f",
            "valid_loss_g",
            "valid_w_dist",
        ] = "valid_w_dist",
        optimizer_f: Union[Dict[str, Any], Type[optax.GradientTransformation]] = MappingProxyType({}),
        optimizer_g: Union[Dict[str, Any], Type[optax.GradientTransformation]] = MappingProxyType({}),
        pretrain_iters: int = 0,
        pretrain_scale: float = 3.0,
        valid_sinkhorn_kwargs: Dict[str, Any] = MappingProxyType({}),
        compute_wasserstein_baseline: bool = False,
        callback_func: Optional[
            Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], Dict[str, float]]
        ] = None,
    ):
        super().__init__(
            source_dim=input_dim,
            target_dim=input_dim,
            cond_dim=cond_dim,
            mlp_eta=mlp_eta,
            mlp_xi=mlp_xi,
            seed=seed,
            **unbalancedness_kwargs,
        )
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.batch_size = batch_size
        self.tau_a = 1.0 if tau_a is None else tau_a
        self.tau_b = 1.0 if tau_b is None else tau_b
        self.epsilon = epsilon if self.tau_a != 1.0 or self.tau_b != 1.0 else None
        self.pos_weights = pos_weights
        self.beta = beta
        self.best_model_selection = best_model_selection
        self.iterations = iterations
        self.inner_iters = inner_iters
        self.valid_freq = valid_freq
        self.log_freq = log_freq
        self.patience = patience
        self.patience_metric = patience_metric
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
        self.f = _get_icnn(input_dim=input_dim, cond_dim=cond_dim, **f) if isinstance(f, abc.Mapping) else f
        self.g = _get_icnn(input_dim=input_dim, cond_dim=cond_dim, **g) if isinstance(g, abc.Mapping) else g
        self.callback_func = (
            lambda tgt, src, pred_tgt, pred_src: _compute_metrics_sinkhorn(
                tgt, src, pred_tgt, pred_src, self.valid_eps, self.valid_sinkhorn_kwargs
            )
            if callback_func is None
            else callback_func
        )
        # set optimizer and networks
        self.setup(self.f, self.g, self.optimizer_f, self.optimizer_g)

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

        self.train_step_f = self.get_step_fn(train=True, to_optimize="f")
        self.valid_step_f = self.get_step_fn(train=False, to_optimize="f")
        self.train_step_g = self.get_step_fn(train=True, to_optimize="g")
        self.valid_step_g = self.get_step_fn(train=False, to_optimize="g")

    def __call__(
        self,
        trainloader: JaxSampler,
        validloader: JaxSampler,
    ) -> Tuple[DualPotentials, "OTTNeuralDualSolver", Train_t]:
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

        train_logs, unbalancedness_logs = self.train_neuraldual(trainloader, validloader)
        res = self.to_dual_potentials()
        logs = {**pretrain_logs, **train_logs, **unbalancedness_logs}

        return (res, self, logs)

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
            state: train_state.TrainState,
        ) -> float:
            """Loss function for the pretraining on identity."""
            grad_g_data = jax.vmap(jax.grad(lambda x: state.apply_fn({"params": params}, x, condition), argnums=0))(
                data
            )
            # loss is L2 reconstruction of the input
            return ((grad_g_data - data) ** 2).sum(axis=1).mean()  # TODO make nicer

        # @jax.jit
        def pretrain_update(
            state: train_state.TrainState, key: jax.random.KeyArray
        ) -> Tuple[jnp.ndarray, train_state.TrainState]:
            """Update function for the pretraining on identity."""
            # sample gaussian data with given scale
            x = self.pretrain_scale * jax.random.normal(key, [self.batch_size, self.input_dim])
            condition = jax.random.choice(key, conditions, shape=(self.batch_size,)) if self.cond_dim else None
            grad_fn = jax.value_and_grad(pretrain_loss_fn, argnums=0)
            loss, grads = grad_fn(state.params, x, condition, state)
            return loss, state.apply_gradients(grads=grads)

        pretrain_logs: Dict[str, List[float]] = {"pretrain_loss": []}
        logger.info(f"Pretraining for {self.pretrain_iters} iterations.")
        for iteration in tqdm(range(self.pretrain_iters)):
            key_pre, self.key = jax.random.split(self.key, 2)
            # train step for potential g directly updating the train state
            loss, self.state_g = pretrain_update(self.state_g, key_pre)
            # clip weights of g
            if self.pos_weights:
                self.state_g = self.state_g.replace(params=self._clip_weights_icnn(self.state_g.params))
            if iteration % self.log_freq == 0:
                pretrain_logs["pretrain_loss"].append(loss)
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

        logs: Dict[str, List[float]] = defaultdict(list)
        unbalancedness_logs: Dict[str, List[float]] = defaultdict(list)

        average_meters: Dict[str, RunningAverageMeter] = defaultdict(RunningAverageMeter)
        valid_average_meters: Dict[str, RunningAverageMeter] = defaultdict(RunningAverageMeter)
        discrete_sinkhorn_div: List[float] = []
        curr_patience: int = 0
        best_loss: float = jnp.inf
        best_iter_distance: float = None
        best_params_f: Optional[jnp.ndarray] = None
        best_params_g: Optional[jnp.ndarray] = None

        # define dict to contain source and target batch
        batch: Dict[str, jnp.ndarray] = {}
        valid_batch: Dict[str, jnp.ndarray] = {}
        baseline_batch: Dict[Tuple[Any, Any], Dict[str, jnp.ndarray]] = {}
        for pair in trainloader.policy_pairs:
            baseline_batch[pair] = {}
            baseline_batch[pair]["source"], _, baseline_batch[pair]["target"] = validloader(
                key=None, policy_pair=pair, sample="both", full_dataset=True
            )
            if self.compute_wasserstein_baseline:
                if baseline_batch[pair]["source"].shape[0] * baseline_batch[pair]["source"].shape[1] > 25000000:
                    logger.warning(
                        "Validation Sinkhorn divergence is expensive to compute due to large size of the validation "
                        "set. Consider setting `valid_sinkhorn_divergence` to False."
                    )
                logger.info("Computing Sinkhorn divergence as a baseline.")
                discrete_sinkhorn_div.append(
                    sinkhorn_divergence(
                        point_cloud_1=baseline_batch[pair]["source"],
                        point_cloud_2=baseline_batch[pair]["target"],
                        **self.valid_sinkhorn_kwargs,
                    )
                )

        for iteration in tqdm(range(self.iterations)):
            # sample policy and condition if given in trainloader
            policy_key, target_key, self.key = jax.random.split(self.key, 3)
            policy_pair, batch["condition"] = trainloader.sample_policy_pair(policy_key)
            # sample target batch
            batch["target"] = trainloader(target_key, policy_pair, sample="target")

            if self.is_balanced:
                a = b = jnp.ones(self.batch_size) / self.batch_size
            else:
                # sample source batch and compute unbalanced marginals
                source_key, self.key = jax.random.split(self.key, 2)
                curr_source, _ = trainloader(source_key, policy_pair, sample="source")
                a, b = trainloader.compute_unbalanced_marginals(curr_source, batch["target"])
                (
                    self.state_eta,
                    self.state_xi,
                    _,
                    _,
                    loss_eta,
                    loss_xi,
                ) = self.unbalancedness_step_fn(curr_source, batch["target"], a, b, self.state_eta, self.state_xi)
                self._update_unbalancedness_logs(unbalancedness_logs, loss_eta, loss_xi, is_train_set=True)

            for _ in range(self.inner_iters):
                source_key, self.key = jax.random.split(self.key, 2)

                batch["source"], batch["condition"] = trainloader(source_key, policy_pair, sample="source")
                if not self.is_balanced:
                    # resample source with unbalanced marginals
                    batch["source"], batch["condition"] = trainloader.unbalanced_resample(
                        source_key, (batch["source"], batch["condition"]), a
                    )
                # train step for potential g directly updating the train state
                self.state_g, loss_g, _ = self.train_step_g(self.state_f, self.state_g, batch)
                average_meters["train_loss_g"].update(loss_g)
                logs = self._update_logs(logs, None, loss_g, None, is_train_set=True)
            # resample target batch with unbalanced marginals
            if not self.is_balanced:
                target_key, self.key = jax.random.split(self.key, 2)
                batch["target"] = trainloader.unbalanced_resample(target_key, (batch["target"],), b)
            # train step for potential f directly updating the train state
            self.state_f, loss_f, w_dist = self.train_step_f(self.state_f, self.state_g, batch)
            logs = self._update_logs(logs, loss_f, None, w_dist, is_train_set=True)
            # clip weights of f
            if self.pos_weights:
                self.state_f = self.state_f.replace(params=self._clip_weights_icnn(self.state_g.params))
            # log avg training values periodically
            if iteration % self.log_freq == 0:
                for key, average_meter in average_meters.items():
                    logs[key].append(average_meter.avg)
                    average_meter.reset()
            # evalute on validation set periodically
            if iteration % self.valid_freq == 0:
                for index, pair in enumerate(trainloader.policy_pairs):
                    # condition = validloader.conditions[index] if self.cond_dim else None
                    valid_batch["source"], valid_batch["condition"] = validloader(
                        source_key, policy_pair, sample="source"
                    )
                    valid_batch["target"] = validloader(source_key, policy_pair, sample="target")
                    valid_loss_f, _ = self.valid_step_f(self.state_f, self.state_g, valid_batch)
                    valid_loss_g, valid_w_dist = self.valid_step_g(self.state_f, self.state_g, valid_batch)
                    logs = self._update_logs(logs, valid_loss_f, valid_loss_g, valid_w_dist, is_train_set=False)
                    a, b = validloader.compute_unbalanced_marginals(valid_batch["source"], valid_batch["target"])
                    _, _, _, _, loss_eta, loss_xi = self.unbalancedness_step_fn(
                        valid_batch["source"], valid_batch["target"], a, b, self.state_eta, self.state_xi
                    )
                    unbalancedness_logs = self._update_unbalancedness_logs(
                        unbalancedness_logs, loss_eta, loss_xi, is_train_set=False
                    )

                # update best model and patience as necessary
                try:
                    total_loss = logs[self.patience_metric][-1]
                except ValueError:
                    f"Unknown metric: {self.patience_metric}."
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_iter_distance = valid_average_meters["valid_neural_dual_dist"].avg
                    best_params_f = self.state_f.params
                    best_params_g = self.state_g.params
                    curr_patience = 0
                else:
                    curr_patience += 1
            if curr_patience >= self.patience:
                break
        if self.best_model_selection:
            self.state_f = self.state_f.replace(params=best_params_f)
            self.state_g = self.state_g.replace(params=best_params_g)
        logs["best_loss"] = best_loss
        logs["predicted_cost"] = None if best_iter_distance is None else float(best_iter_distance)
        if self.compute_wasserstein_baseline:
            logs["sinkhorn_div"] = np.mean(discrete_sinkhorn_div)
        return logs, unbalancedness_logs

    def get_step_fn(self, train: bool, to_optimize: Literal["f", "g"]):
        """Create a parallel training and evaluation function."""

        def loss_fn(params_f, params_g, f_value, g_value, g_gradient, batch):
            """Loss function for both potentials."""
            # get two distributions
            source, target = batch["source"], batch["target"]

            init_source_hat = g_gradient(params_g)(target, batch["condition"])

            def g_value_partial(y: jnp.ndarray) -> jnp.ndarray:
                """Lazy way of evaluating g if f's computation needs it."""
                return g_value(params_g)(y, batch["condition"])

            f_value_partial = f_value(params_f, g_value_partial, batch["condition"])

            source_hat_detach = init_source_hat

            batch_dot = jax.vmap(jnp.dot)

            f_source = f_value_partial(source, batch["condition"])
            f_star_target = batch_dot(source_hat_detach, target) - f_value_partial(
                source_hat_detach, batch["condition"]
            )
            dual_source = f_source.mean()
            dual_target = f_star_target.mean()
            dual_loss = dual_source + dual_target

            f_value_parameters_detached = f_value(jax.lax.stop_gradient(params_f), g_value_partial, batch["condition"])
            amor_loss = (
                f_value_parameters_detached(init_source_hat, batch["condition"]) - batch_dot(init_source_hat, target)
            ).mean()
            if to_optimize == "f":
                loss = dual_loss
            elif to_optimize == "g":
                loss = amor_loss
            else:
                raise ValueError(f"Optimization target {to_optimize} has been misspecified.")

            if self.pos_weights:
                # Penalize the weights of both networks, even though one
                # of them will be exactly clipped.
                # Having both here is necessary in case this is being called with
                # the potentials reversed with the back_and_forth.
                loss += self.beta * self._penalize_weights_icnn(params_f) + self.beta * self._penalize_weights_icnn(
                    params_g
                )

            # compute Wasserstein-2 distance
            C = jnp.mean(jnp.sum(source**2, axis=-1)) + jnp.mean(jnp.sum(target**2, axis=-1))
            W2_dist = C - 2.0 * (f_source.mean() + f_star_target.mean())

            return loss, (dual_loss, amor_loss, W2_dist)

        @jax.jit
        def step_fn(state_f, state_g, batch):
            """Step function of either training or validation."""
            grad_fn = jax.value_and_grad(loss_fn, argnums=[0, 1], has_aux=True)
            if train:
                # compute loss and gradients
                (loss, (loss_f, loss_g, W2_dist)), (grads_f, grads_g) = grad_fn(
                    state_f.params,
                    state_g.params,
                    state_f.potential_value_fn,
                    state_g.potential_value_fn,
                    state_g.potential_gradient_fn,
                    batch,
                )

                if to_optimize == "f":
                    return state_f.apply_gradients(grads=grads_f), loss_f, W2_dist
                if to_optimize == "g":
                    return state_g.apply_gradients(grads=grads_g), loss_g, W2_dist
                raise ValueError("Optimization target has been misspecified.")

            # compute loss and gradients
            (loss, (loss_f, loss_g, W2_dist)), _ = grad_fn(
                state_f.params,
                state_g.params,
                state_f.potential_value_fn,
                state_g.potential_value_fn,
                state_g.potential_gradient_fn,
                batch,
            )

            # do not update state
            if to_optimize == "both":
                return loss_f, loss_g, W2_dist
            if to_optimize == "f":
                return loss_f, W2_dist
            if to_optimize == "g":
                return loss_g, W2_dist
            raise ValueError("Optimization target has been misspecified.")

        return step_fn

    def _clip_weights_icnn(self, params: FrozenVariableDict) -> FrozenVariableDict:
        """Clip weights of ICNN."""
        for key in params:
            if key.startswith("w_zs"):
                params[key]["kernel"] = jnp.clip(params[key]["kernel"], a_min=0)

        return params  # freeze(params)

    def _penalize_weights_icnn(self, params: FrozenVariableDict) -> float:
        """Penalize weights of ICNN."""
        penalty = 0
        for key in params:
            if key.startswith("w_z"):
                penalty += jnp.linalg.norm(jax.nn.relu(-params[key]["kernel"]))
        return penalty

    def to_dual_potentials_old(self, condition: Optional[ArrayLike] = None) -> DualPotentials:
        """Return the Kantorovich dual potentials from the trained potentials."""
        if self.cond_dim:
            return ConditionalDualPotentials(self.state_f, self.state_g)

        def f(x) -> float:
            return self.state_f.apply_fn({"params": self.state_f.params}, x)

        def g(x) -> float:
            return self.state_g.apply_fn({"params": self.state_g.params}, x)

        return DualPotentials(f, g, corr=True, cost_fn=costs.SqEuclidean())

    def to_dual_potentials(self, finetune_g: bool = False) -> potentials.DualPotentials:
        r"""Return the Kantorovich dual potentials from the trained potentials.

        Args:
        finetune_g: Run the conjugate solver to fine-tune the prediction.

        Returns
        -------
        A dual potential object
        """
        f_value = self.state_f.potential_value_fn(self.state_f.params)
        g_value_prediction = self.state_g.potential_value_fn(self.state_g.params, f_value)
        return potentials.DualPotentials(f=f_value, g=g_value_prediction, cost_fn=costs.SqEuclidean(), corr=True)

    @property
    def is_balanced(self) -> bool:
        """Return whether the problem is balanced."""
        return self.tau_a == self.tau_b == 1.0

    @staticmethod
    def _update_logs(
        logs: Dict[str, List[float]],
        loss_f: Optional[jnp.ndarray],
        loss_g: Optional[jnp.ndarray],
        w_dist: Optional[jnp.ndarray],
        *,
        is_train_set: bool,
    ) -> Dict[str, List[float]]:
        if is_train_set:
            if loss_f is not None:
                logs["train_loss_f"].append(float(loss_f))
            if loss_g is not None:
                logs["train_loss_g"].append(float(loss_g))
            if w_dist is not None:
                logs["train_w_dist"].append(float(w_dist))
        else:
            if loss_f is not None:
                logs["valid_loss_f"].append(float(loss_f))
            if loss_g is not None:
                logs["valid_loss_g"].append(float(loss_g))
            if w_dist is not None:
                logs["valid_w_dist"].append(float(w_dist))
        return logs
