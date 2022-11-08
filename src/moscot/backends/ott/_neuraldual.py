from typing import Dict, List, Tuple, Callable, Optional
import warnings

from flax.core import freeze
from tqdm.auto import tqdm
from flax.training import train_state
import optax

from ott.core import potentials
from ott.geometry.pointcloud import PointCloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence
import jax
import jax.numpy as jnp

from moscot.backends.ott._icnn import ICNN
from moscot.backends.ott._utils import subtract_pytrees, _compute_sinkhorn_divergence
from moscot.backends.ott._jax_data import JaxSampler

Train_t = Dict[str, Dict[str, List[float]]]


class NeuralDualSolver:
    r"""Solver of the ICNN-based Kantorovich dual.

    Either apply the algorithm described in:
    Optimal transport mapping via input convex neural networks,
    Makkuva-Taghvaei-Lee-Oh, ICML'20.
    http://proceedings.mlr.press/v119/makkuva20a/makkuva20a.pdf
    or apply the algorithm described in:
    Wasserstein-2 Generative Networks
    https://arxiv.org/pdf/1909.13082.pdf
    """

    def __init__(
        self,
        input_dim: int,
        seed: int = 0,
        pos_weights: bool = False,
        beta: float = 1.0,
        pretrain: bool = True,
        metric: str = "sinkhorn_forward",
        iterations: int = 25000,
        inner_iters: int = 10,
        valid_freq: int = 500,
        log_freq: int = 50,
        patience: int = 500,
        dim_hidden: Optional[List[int]] = None,
        learning_rate: float = 1e-3,
        beta_one: float = 0.5,
        beta_two: float = 0.9,
        weight_decay: float = 0.0,
        pretrain_iters: int = 15001,
        pretrain_scale: float = 3.0,
        tau_a: Optional[float] = None,
        tau_b: Optional[float] = None,
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
        self.curr_patience = 0
        self.pretrain_iters = pretrain_iters
        self.pretrain_scale = pretrain_scale
        self.tau_a = 1.0 if tau_a is None else tau_a
        self.tau_b = 1.0 if tau_b is None else tau_b
        if dim_hidden is None:
            dim_hidden = [64, 64, 64, 64]

        self.key = jax.random.PRNGKey(seed)
        optimizer_f = optax.adamw(learning_rate=learning_rate, b1=beta_one, b2=beta_two, weight_decay=weight_decay)
        optimizer_g = optax.adamw(learning_rate=learning_rate, b1=beta_one, b2=beta_two, weight_decay=weight_decay)
        neural_f = ICNN(dim_hidden=dim_hidden, pos_weights=pos_weights)
        neural_g = ICNN(dim_hidden=dim_hidden, pos_weights=pos_weights)

        # set optimizer and networks
        self.setup(neural_f, neural_g, optimizer_f, optimizer_g)

    def setup(self, neural_f, neural_g, optimizer_f, optimizer_g):
        """Initialize all components required to train the `NeuralDual`."""
        key_f, key_g, self.key = jax.random.split(self.key, 3)

        # check setting of network architectures
        if neural_g.pos_weights != self.pos_weights or neural_f.pos_weights != self.pos_weights:
            warnings.warn(
                f"Setting of ICNN and the positive weights setting of the \
                      `NeuralDualSolver` are not consistent. Proceeding with \
                      the `NeuralDualSolver` setting, with positive weigths \
                      being {self.pos_weights}."
            )
            neural_g.pos_weights = self.pos_weights
            neural_f.pos_weights = self.pos_weights

        self.state_f = neural_f.create_train_state(key_f, optimizer_f, self.input_dim)
        self.state_g = neural_g.create_train_state(key_g, optimizer_g, self.input_dim)

        self.train_step = self.get_train_step()
        self.valid_step = self.get_eval_step()

    def __call__(
        self,
        trainloader: JaxSampler,
        validloader: JaxSampler,
    ) -> Tuple[potentials.DualPotentials, Train_t]:
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
        def pretrain_loss_fn(params: jnp.ndarray, data: jnp.ndarray, state: train_state.TrainState) -> float:
            grad_f_data = jax.vmap(jax.grad(lambda x: state.apply_fn({"params": params}, x), argnums=0))(data)
            loss = ((grad_f_data - data) ** 2).sum(axis=1).mean()
            return loss

        @jax.jit
        def pretrain_update(
            state: train_state.TrainState, key: jax.random.KeyArray
        ) -> Tuple[jnp.ndarray, train_state.TrainState]:
            x = self.pretrain_scale * jax.random.normal(key, [1024, self.input_dim])
            grad_fn = jax.value_and_grad(pretrain_loss_fn, argnums=0)
            loss, grads = grad_fn(state.params, x, state)
            return loss, state.apply_gradients(grads=grads)

        pretrain_logs: Dict[str, List[float]] = {"pretrain_loss": []}
        for iteration in range(self.pretrain_iters):
            key_pre, self.key = jax.random.split(self.key, 2)
            loss, self.state_f = pretrain_update(self.state_f, key_pre)
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
        self.sink_dist: float = None
        self.best_loss: float = jnp.inf
        self.best_iter_distance: float = None
        self.best_params_f: jnp.ndarray = None
        self.best_params_g: jnp.ndarray = None
        # define dict to contain source and target batch
        batch: Dict[str, jnp.ndarray] = {}
        valid_batch: Dict[str, jnp.ndarray] = {}
        valid_batch["source"], valid_batch["target"] = validloader(key=None, full_dataset=True)
        _compute_sinkhorn_divergence(valid_batch["source"], valid_batch["target"])
        # set logging dictionaries
        train_logs: Dict[str, List[float]] = {
            "train_loss": [],
            "train_loss_f": [],
            "train_loss_g": [],
            "train_w_dist": [],
            "weight_penalty": [],
        }
        valid_logs: Dict[str, List[float]] = {
            "sinkhorn_loss_forward": [],
            "sinkhorn_loss_inverse": [],
            "valid_w_dist": [],
        }

        for iteration in tqdm(range(self.iterations)):
            # set gradients of f to zero (is there a better way to do this?)
            grads_f_accumulated = jax.jit(jax.grad(lambda _: 0.0))(self.state_f.params)

            for inner_iter in range(self.inner_iters):
                batch_key, self.key = jax.random.split(self.key, 2)
                batch["source"], batch["target"] = trainloader(batch_key, inner_iter=inner_iter)
                # train step for potential g
                (
                    self.state_g,
                    grads_f,
                    loss,
                    w_dist,
                    penalty,
                    loss_f,
                    loss_g,
                ) = self.train_step(self.state_f, self.state_g, batch)
                # log training values periodically
                if (iteration * self.inner_iters + inner_iter) % self.log_freq == 0:
                    train_logs["train_loss"].append(float(loss))
                    train_logs["train_loss_f"].append(float(loss_f))
                    train_logs["train_loss_g"].append(float(loss_g))
                    train_logs["train_w_dist"].append(float(w_dist))
                    train_logs["weight_penalty"].append(float(penalty))
                # evalute on validation set periodically
                if (iteration * self.inner_iters + inner_iter) % self.valid_freq == 0:
                    sink_forward, sink_inverse, neural_dual_distance = self.valid_step(
                        self.state_f, self.state_g, valid_batch
                    )
                    valid_logs["sinkhorn_loss_forward"].append(float(sink_forward))
                    valid_logs["sinkhorn_loss_inverse"].append(float(sink_inverse))
                    valid_logs["valid_w_dist"].append(float(neural_dual_distance))
                # accumulate gradients
                grads_f_accumulated = subtract_pytrees(grads_f_accumulated, grads_f, self.inner_iters)

            # update potential f with accumulated gradients
            self.state_f = self.state_f.apply_gradients(grads=grads_f_accumulated)
            # clip weights of f
            if not self.pos_weights:
                self.state_f = self.state_f.replace(params=self.clip_weights_icnn(self.state_f.params))
            if self.curr_patience >= self.patience:
                break
        self.state_f = self.state_f.replace(params=self.best_params_f)
        self.state_g = self.state_g.replace(params=self.best_params_g)
        valid_logs["best_sinkhorn_loss_forward"] = [float(self.best_loss)]
        valid_logs["sink_dist"] = [float(self.sink_dist)]
        valid_logs["predicted_cost"] = [float(self.best_iter_distance)]
        return {
            "train_logs": train_logs,
            "valid_logs": valid_logs,
        }

    def get_train_step(
        self,
    ) -> Callable[
        [train_state.TrainState, train_state.TrainState, Dict[str, jnp.ndarray]],
        Tuple[train_state.TrainState, train_state.TrainState, float, float, float, float, float],
    ]:
        """Get train step."""

        @jax.jit
        def loss_fn(
            params_f: jnp.ndarray,
            params_g: jnp.ndarray,
            state_f: train_state.TrainState,
            state_g: train_state.TrainState,
            batch: Dict[str, jnp.ndarray],
        ):
            """Loss function."""
            # get loss terms of kantorovich dual
            f_tgt = jax.vmap(lambda x: state_f.apply_fn({"params": params_f}, x))(batch["target"])
            grad_g_src = jax.vmap(jax.grad(lambda x: state_g.apply_fn({"params": params_g}, x), argnums=0))(
                batch["source"]
            )
            f_grad_g_src = jax.vmap(lambda x: state_f.apply_fn({"params": params_f}, x))(grad_g_src)
            src_dot_grad_g_src = jnp.sum(batch["source"] * grad_g_src, axis=1)

            # compute loss
            loss_f = jnp.mean(f_tgt - f_grad_g_src)
            loss_g = jnp.mean(f_grad_g_src - src_dot_grad_g_src)
            loss = jnp.mean(f_grad_g_src - f_tgt - src_dot_grad_g_src)
            # compute wasserstein distance
            dist = 2 * (
                loss
                + jnp.mean(
                    0.5 * jnp.sum(batch["target"] * batch["target"], axis=1)
                    + 0.5 * jnp.sum(batch["source"] * batch["source"], axis=1)
                )
            )

            if not self.pos_weights:
                penalty = self.beta * self.penalize_weights_icnn(params_g)
                loss += penalty
            else:
                penalty = 0
            return loss, (dist, loss_f, loss_g, penalty)

        @jax.jit
        def step_fn(
            state_f: train_state.TrainState,
            state_g: train_state.TrainState,
            batch: Dict[str, jnp.ndarray],
        ):
            """Step function for training."""
            grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
            # compute loss and gradients
            (loss, (dist, loss_f, loss_g, penalty)), (grads_f, grads_g) = grad_fn(
                state_f.params, state_g.params, state_f, state_g, batch
            )
            # update state and return training stats
            return (
                state_g.apply_gradients(grads=grads_g),
                grads_f,
                loss,
                dist,
                penalty,
                loss_f,
                loss_g,
            )

        return step_fn

    def get_eval_step(
        self,
    ) -> Callable[[train_state.TrainState, train_state.TrainState, Dict[str, jnp.ndarray]], Tuple[float, float, float]]:
        """Get validation step."""

        @jax.jit
        def jit_valid_step(
            state_f: train_state.TrainState,
            state_g: train_state.TrainState,
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
            # calculate sinkhorn loss between predicted and true samples
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

        def valid_step(
            state_f: train_state.TrainState,
            state_g: train_state.TrainState,
            batch: Dict[str, jnp.ndarray],
        ) -> Tuple[float, float, float]:
            sink_loss_forward, sink_loss_inverse, neural_dual_dist = jit_valid_step(state_f, state_g, batch)
            # update best model and patience as necessary
            if self.metric == "sinkhorn":
                total_loss = jnp.abs(sink_loss_forward) + jnp.abs(sink_loss_inverse)
            elif self.metric == "sinkhorn_forward":
                total_loss = jnp.abs(sink_loss_forward)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
            if total_loss < self.best_loss:
                self.best_loss = total_loss
                self.best_iter_distance = neural_dual_dist
                self.best_params_f = state_f.params
                self.best_params_g = state_g.params
                self.curr_patience = 0
            else:
                self.curr_patience += 1
            return sink_loss_forward, sink_loss_inverse, neural_dual_dist

        return valid_step

    def clip_weights_icnn(self, params):
        """Clip weights of ICNN."""
        params = params.unfreeze()
        for k in params.keys():
            if k.startswith("w_z"):
                params[k]["kernel"] = jnp.clip(params[k]["kernel"], a_min=0)

        return freeze(params)

    def penalize_weights_icnn(self, params):
        """Penalize weights of ICNN."""
        penalty = 0
        for k in params.keys():
            if k.startswith("w_z"):
                penalty += jnp.linalg.norm(jax.nn.relu(-params[k]["kernel"]))
        return penalty

    def to_dual_potentials(self) -> potentials.DualPotentials:
        """Return the Kantorovich dual potentials from the trained potentials."""
        return potentials.DualPotentials(self.state_f, self.state_g, cor=True)
