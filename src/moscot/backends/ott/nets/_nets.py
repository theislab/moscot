from typing import Any, Callable

import flax.linen as nn
import optax
from flax.training import train_state

import jax
import jax.numpy as jnp
from ott.solvers.nn.models import ModelBase, NeuralTrainState


class Block(nn.Module):
    dim: int = 128
    num_layers: int = 3
    activation_fn: Any = nn.silu
    out_dim: int = 32

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_layers):
            x = nn.Dense(self.dim, name=f"fc{i}")(x)
            x = self.activation_fn(x)
        return nn.Dense(self.out_dim, name="fc_final")(x)


class MLP_marginal(ModelBase):
    hidden_dim: int
    is_potential: bool = False
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.selu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:  # noqa: D102
        x = Block(dim=self.hidden_dim, out_dim=self.hidden_dim, activation_fn=self.act_fn)(x)
        Wx = nn.Dense(1, use_bias=True, name="final_layer")
        z = Wx(x)
        return jnp.exp(z)

    def create_train_state(
        self,
        rng: jax.random.PRNGKeyArray,
        optimizer: optax.OptState,
        input_dim: int,
    ) -> NeuralTrainState:
        params = self.init(rng, jnp.ones((1, input_dim)))["params"]
        return NeuralTrainState.create(apply_fn=self.apply, params=params, tx=optimizer)
