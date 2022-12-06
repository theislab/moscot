from typing import Dict, List, Tuple, Literal, Callable, Optional
from collections import defaultdict

from flax.core import freeze
from tqdm.auto import tqdm
from flax.core.scope import FrozenVariableDict
from flax.training.train_state import TrainState
import optax

from ott.geometry.pointcloud import PointCloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence
from ott.problems.linear.potentials import DualPotentials
import jax
import jax.numpy as jnp

from moscot._logging import logger
from moscot.backends.ott._icnn import ICNN
from moscot.backends.ott._utils import RunningAverageMeter, _compute_sinkhorn_divergence
from moscot.backends.ott._jax_data import JaxSampler

Train_t = Dict[str, Dict[str, List[float]]]


class NeuralDualSolver:
    r"""Solver of the ICNN-based Kantorovich dual.

    Optimal transport mapping via input convex neural networks,
    Makkuva-Taghvaei-Lee-Oh, ICML'20.
    http://proceedings.mlr.press/v119/makkuva20a/makkuva20a.pdf
    """

    def __init__(
        self,
        input_dim: int,
        seed: int = 0,
        pos_weights: bool = False,
        dim_hidden: Optional[List[int]] = None,
        beta: float = 1.0,
        pretrain: bool = True,
        metric: str = "sinkhorn_forward",
        iterations: int = 25000,
        inner_iters: int = 10,
        valid_freq: int = 50,
        log_freq: int = 5,
        patience: int = 100,
        learning_rate: float = 1e-3,
        beta_one: float = 0.5,
        beta_two: float = 0.9,
        weight_decay: float = 0.0,
        pretrain_iters: int = 15001,
        pretrain_scale: float = 3.0,
        batch_size: int = 1024,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        split_indices: Optional[List[int]] = None,
    ):
        self.input_dim = input_dim
        self.pos_weights = pos_weights
        self.beta = beta
        self.pretrain = pretrain
        self.metric = metric
        self.iterations = iterations
        self.inner_iters = inner_iters
        self.valid_freq = valid_freq
        self.log_freq = log_freq
        self.patience = patience
        self.pretrain_iters = pretrain_iters
        self.pretrain_scale = pretrain_scale
        self.batch_size = batch_size
        self.split_indices = split_indices
        self.tau_a = 1.0 if tau_a is None else tau_a
        self.tau_b = 1.0 if tau_b is None else tau_b
        self.epsilon = 0.1 if self.tau_a != 1.0 or self.tau_b != 1.0 else None
        if dim_hidden is None:
            dim_hidden = [64, 64, 64, 64]

        self.key = jax.random.PRNGKey(seed)
        optimizer_f = optax.adamw(learning_rate=learning_rate, b1=beta_one, b2=beta_two, weight_decay=weight_decay)
        optimizer_g = optax.adamw(learning_rate=learning_rate, b1=beta_one, b2=beta_two, weight_decay=weight_decay)
        neural_f = ICNN(dim_hidden=dim_hidden, pos_weights=pos_weights, split_indices=split_indices)
        neural_g = ICNN(dim_hidden=dim_hidden, pos_weights=pos_weights, split_indices=split_indices)

        # set optimizer and networks
        self.setup(neural_f, neural_g, optimizer_f, optimizer_g)

    def setup(self, neural_f: ICNN, neural_g: ICNN, optimizer_f: optax.OptState, optimizer_g: optax.OptState):
        """Initialize all components required to train the `NeuralDual`."""
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
        """Call the training script, and return the trained neural dual."""
        pretrain_logs = {}
        if self.pretrain:
            pretrain_logs = self.pretrain_identity()

        train_logs = self.train_neuraldual(trainloader, validloader)
        res = self.to_dual_potentials()
        logs = pretrain_logs | train_logs

        return (res, logs)

    def pretrain_identity(self) -> Train_t:
        """Pretrain the neural networks to identity."""

        @jax.jit
        def pretrain_loss_fn(params: jnp.ndarray, data: jnp.ndarray, state: TrainState) -> float:
            """Loss function for the pretraining on identity."""
            grad_f_data = jax.vmap(jax.grad(lambda x: state.apply_fn({"params": params}, x), argnums=0))(data)
            # loss is L2 reconstruction of the input
            loss = ((grad_f_data - data) ** 2).sum(axis=1).mean()
            return loss

        @jax.jit
        def pretrain_update(state: TrainState, key: jax.random.KeyArray) -> Tuple[jnp.ndarray, TrainState]:
            """Update function for the pretraining on identity."""
            # sample gaussian data with given scale
            x = self.pretrain_scale * jax.random.normal(key, [self.batch_size, self.input_dim])
            grad_fn = jax.value_and_grad(pretrain_loss_fn, argnums=0)
            loss, grads = grad_fn(state.params, x, state)
            return loss, state.apply_gradients(grads=grads)

        pretrain_logs: Dict[str, List[float]] = {"pretrain_loss": []}
        for iteration in range(self.pretrain_iters):
            key_pre, self.key = jax.random.split(self.key, 2)
            # train step for potential f directly updating the train state
            loss, self.state_f = pretrain_update(self.state_f, key_pre)
            # clip weights of f
            if not self.pos_weights:
                self.state_f = self.state_f.replace(params=self.clip_weights_icnn(self.state_f.params))
            if iteration % self.log_freq == 0:
                pretrain_logs["pretrain_loss"].append(float(loss))
        # load params of f into state_g
        self.state_g = self.state_g.replace(params=self.state_f.params)
        return {"pretrain_logs": pretrain_logs}

    def train_neuraldual(
        self,
        trainloader: JaxSampler,
        validloader: JaxSampler,
    ) -> Train_t:
        """Train the neural dual and call evaluation script."""
        # define dict to contain source and target batch
        batch: Dict[str, jnp.ndarray] = {}
        valid_batch: Dict[str, jnp.ndarray] = {}
        valid_batch["source"] = validloader(key=None, full_dataset=True, source=True)
        valid_batch["target"] = validloader(key=None, full_dataset=True, source=False)
        if self.split_indices is None:
            sink_dist = _compute_sinkhorn_divergence(valid_batch["source"], valid_batch["target"])
        else:
            sink_dist = _compute_sinkhorn_divergence(valid_batch["source"][:,self.split_indices[0]], valid_batch["target"][:,self.split_indices[0]])
        # set logging dictionaries
        train_logs: Dict[str, List[float]] = defaultdict(list)
        valid_logs: Dict[str, List[float]] = defaultdict(list)
        average_meters: Dict[str, RunningAverageMeter] = defaultdict(RunningAverageMeter)
        curr_patience: int = 0
        best_loss: float = jnp.inf
        best_iter_distance: float = None
        best_params_f: jnp.ndarray = None
        best_params_g: jnp.ndarray = None

        for iteration in tqdm(range(self.iterations)):
            # sample target batch
            target_key, self.key = jax.random.split(self.key, 2)
            batch["target"] = trainloader(target_key, source=False)
            if self.epsilon is not None:
                # sample source batch and compute unbalanced marginals
                curr_source = trainloader(target_key, source=True)
                marginals_source, marginals_target = trainloader.compute_unbalanced_marginals(
                    curr_source, batch["target"]
                )

            for _ in range(self.inner_iters):
                source_key, self.key = jax.random.split(self.key, 2)
                if self.epsilon is None:
                    # sample source batch
                    batch["source"] = trainloader(source_key, source=True)
                else:
                    # resample source with unbalanced marginals
                    batch["source"] = trainloader.unbalanced_resample(source_key, curr_source, marginals_source)
                # train step for potential g directly updating the train state
                self.state_g, train_g_metrics = self.train_step_g(self.state_f, self.state_g, batch)
                for key, value in train_g_metrics.items():
                    average_meters[key].update(value)
            # resample target batch with unbalanced marginals
            if self.epsilon is not None:
                batch["target"] = trainloader.unbalanced_resample(target_key, batch["target"], marginals_target)
            # train step for potential f directly updating the train state
            self.state_f, train_f_metrics = self.train_step_f(self.state_f, self.state_g, batch)
            for key, value in train_f_metrics.items():
                average_meters[key].update(value)
            # clip weights of f
            if not self.pos_weights:
                self.state_f = self.state_f.replace(params=self.clip_weights_icnn(self.state_f.params))
            # log avg training values periodically
            if iteration % self.log_freq == 0:
                for key, average_meter in average_meters.items():
                    train_logs[key].append(average_meter.avg)
                    average_meter.reset()
            # evalute on validation set periodically
            if iteration % self.valid_freq == 0:
                sink_loss_forward, sink_loss_inverse, neural_dual_dist = self.valid_step(
                    self.state_f, self.state_g, batch
                )
                valid_logs["sinkhorn_loss_forward"].append(float(sink_loss_forward))
                valid_logs["sinkhorn_loss_inverse"].append(float(sink_loss_inverse))
                valid_logs["valid_w_dist"].append(float(neural_dual_dist))
                # update best model and patience as necessary
                if self.metric == "sinkhorn":
                    total_loss = jnp.abs(sink_loss_forward) + jnp.abs(sink_loss_inverse)
                elif self.metric == "sinkhorn_forward":
                    total_loss = jnp.abs(sink_loss_forward)
                else:
                    raise ValueError(f"Unknown metric: {self.metric}")
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_iter_distance = neural_dual_dist
                    best_params_f = self.state_f.params
                    best_params_g = self.state_g.params
                    curr_patience = 0
                else:
                    curr_patience += 1
            if curr_patience >= self.patience:
                break
        self.state_f = self.state_f.replace(params=best_params_f)
        self.state_g = self.state_g.replace(params=best_params_g)
        valid_logs["best_loss"] = [float(best_loss)]
        valid_logs["sink_dist"] = [float(sink_dist)]
        valid_logs["predicted_cost"] = [float(best_iter_distance)]
        return {
            "train_logs": train_logs,
            "valid_logs": valid_logs,
        }

    def get_train_step(
        self,
        to_optimize: Literal["f", "g"],
    ) -> Callable[[TrainState, TrainState, Dict[str, jnp.ndarray]], Tuple[TrainState, Dict[str, float]]]:
        """Get train step."""

        def loss_f_fn(
            params_f: jnp.ndarray,
            params_g: jnp.ndarray,
            state_f: TrainState,
            state_g: TrainState,
            batch: Dict[str, jnp.ndarray],
        ) -> Tuple[jnp.ndarray, List[float]]:
            """Loss function for f."""
            # get loss terms of kantorovich dual
            grad_g_src = jax.vmap(jax.grad(lambda x: state_g.apply_fn({"params": params_g}, x), argnums=0))(
                batch["source"]
            )
            if self.split_indices is not None:
                grad_g_src = grad_g_src[:,:self.split_indices[0]]
            f_grad_g_src = jax.vmap(lambda x: state_f.apply_fn({"params": params_f}, x))(grad_g_src)
            src_dot_grad_g_src = jnp.sum(batch["source"] * grad_g_src, axis=1)
            # compute loss
            f_tgt = jax.vmap(lambda x: state_f.apply_fn({"params": params_f}, x))(batch["target"])
            loss = jnp.mean(f_tgt - f_grad_g_src)
            total_loss = jnp.mean(f_grad_g_src - f_tgt - src_dot_grad_g_src)
            # compute wasserstein distance
            dist = 2 * (
                total_loss
                + jnp.mean(
                    0.5 * jnp.sum(batch["target"] * batch["target"], axis=1)
                    + 0.5 * jnp.sum(batch["source"] * batch["source"], axis=1)
                )
            )
            return loss, [total_loss, dist]

        def loss_g_fn(
            params_f: jnp.ndarray,
            params_g: jnp.ndarray,
            state_f: TrainState,
            state_g: TrainState,
            batch: Dict[str, jnp.ndarray],
        ) -> Tuple[jnp.ndarray, List[float]]:
            """Loss function for g."""
            # get loss terms of kantorovich dual
            grad_g_src = jax.vmap(jax.grad(lambda x: state_g.apply_fn({"params": params_g}, x), argnums=0))(
                batch["source"]
            )
            f_grad_g_src = jax.vmap(lambda x: state_f.apply_fn({"params": params_f}, x))(grad_g_src)
            src_dot_grad_g_src = jnp.sum(batch["source"] * grad_g_src, axis=1)
            # compute loss
            loss = jnp.mean(f_grad_g_src - src_dot_grad_g_src)
            if not self.pos_weights:
                penalty = self.beta * self.penalize_weights_icnn(params_g)
                loss += penalty
            else:
                penalty = 0
            return loss, [penalty]

        @jax.jit
        def step_fn(
            state_f: TrainState,
            state_g: TrainState,
            batch: Dict[str, jnp.ndarray],
        ) -> Tuple[TrainState, Dict[str, float]]:
            """Step function for training."""
            # get loss function for f or g
            if to_optimize == "f":
                grad_fn = jax.value_and_grad(loss_f_fn, argnums=0, has_aux=True)
                # compute loss, gradients and metrics
                (loss, raw_metrics), grads = grad_fn(state_f.params, state_g.params, state_f, state_g, batch)
                # return updated state and metrics dict
                metrics = {"loss_f": loss, "loss": raw_metrics[0], "w_dist": raw_metrics[1]}
                return state_f.apply_gradients(grads=grads), metrics
            elif to_optimize == "g":
                grad_fn = jax.value_and_grad(loss_g_fn, argnums=1, has_aux=True)
                # compute loss, gradients and metrics
                (loss, raw_metrics), grads = grad_fn(state_f.params, state_g.params, state_f, state_g, batch)
                # return updated state and metrics dict
                metrics = {"loss_g": loss, "penalty": raw_metrics[0]}
                return state_g.apply_gradients(grads=grads), metrics

        return step_fn

    def get_eval_step(
        self,
    ) -> Callable[[TrainState, TrainState, Dict[str, jnp.ndarray]], Tuple[float, float, float]]:
        """Get validation step."""

        @jax.jit
        def valid_step(
            state_f: TrainState,
            state_g: TrainState,
            batch: Dict[str, jnp.ndarray],
        ) -> Tuple[float, float, float]:
            """Create a validation function."""
            # get transported source and inverse transported target
            pred_target = jax.vmap(jax.grad(lambda x: state_g.apply_fn({"params": state_g.params}, x), argnums=0))(
                batch["source"]
            )
            pred_source = jax.vmap(jax.grad(lambda x: state_f.apply_fn({"params": state_f.params}, x), argnums=0))(
                batch["target"]
            )
            if self.split_indices is not None:
                pred_target = pred_target[:,:self.split_indices[0]]
                pred_source = pred_source[:,:self.split_indices[0]]
            # calculate sinkhorn loss between predicted and true samples
            # using sinkhorn_divergence because _compute_sinkhorn_divergence not jittable
            sink_loss_forward = sinkhorn_divergence(
                PointCloud,
                x=pred_target,
                y=batch["target"],
                epsilon=10,
                sinkhorn_kwargs={"tau_a": self.tau_a, "tau_b": self.tau_b},
            ).divergence
            sink_loss_inverse = sinkhorn_divergence(
                PointCloud,
                x=pred_source,
                y=batch["source"],
                epsilon=10,
                sinkhorn_kwargs={"tau_a": self.tau_a, "tau_b": self.tau_b},
            ).divergence
            # get neural dual distance between true source and target
            f_tgt = jax.vmap(lambda x: state_f.apply_fn({"params": state_f.params}, x))(batch["target"])
            f_grad_g_src = jax.vmap(lambda x: state_f.apply_fn({"params": state_f.params}, x))(pred_target)
            src_dot_grad_g_src = jnp.sum(batch["source"] * pred_target, axis=-1)
            src_sq = jnp.mean(jnp.sum(batch["source"] ** 2, axis=-1))
            tgt_sq = jnp.mean(jnp.sum(batch["target"] ** 2, axis=-1))
            neural_dual_dist = tgt_sq + src_sq + 2.0 * (jnp.mean(f_grad_g_src - src_dot_grad_g_src) - jnp.mean(f_tgt))
            return sink_loss_forward, sink_loss_inverse, neural_dual_dist

        return valid_step

    def clip_weights_icnn(self, params: FrozenVariableDict) -> FrozenVariableDict:
        """Clip weights of ICNN."""
        params = params.unfreeze()
        for key in params.keys():
            if key.startswith("w_zs"):
                params[key]["kernel"] = jnp.clip(params[key]["kernel"], a_min=0)

        return freeze(params)

    def penalize_weights_icnn(self, params: FrozenVariableDict) -> float:
        """Penalize weights of ICNN."""
        penalty = 0
        for key in params.keys():
            if key.startswith("w_z"):
                penalty += jnp.linalg.norm(jax.nn.relu(-params[key]["kernel"]))
        return penalty

    def to_dual_potentials(self) -> DualPotentials:
        """Return the Kantorovich dual potentials from the trained potentials."""
        return DualPotentials(self.state_f, self.state_g, cor=True)
