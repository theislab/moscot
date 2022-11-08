from typing import Tuple, Union, Callable, Sequence

from flax import linen as nn
from flax.training import train_state
import optax

from ott.core.icnn import PositiveDense
import jax.numpy as jnp


class ICNN(nn.Module):
    """Input convex neural network (ICNN) architecture."""

    dim_hidden: Sequence[int]
    init_std: float = 0.1
    init_fn: Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]] = nn.initializers.normal
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu
    pos_weights: bool = False

    def setup(self):
        """Initialize ICNN architecture."""
        num_hidden = len(self.dim_hidden)

        if self.pos_weights:
            Dense = PositiveDense
        else:
            Dense = nn.Dense
        kernel_inits_wz = [self.init_fn(self.init_std) for _ in range(num_hidden + 1)]

        w_zs = []
        for i in range(1, num_hidden):
            w_zs.append(
                Dense(
                    self.dim_hidden[i],
                    kernel_init=kernel_inits_wz[i],
                    use_bias=False,
                )
            )
        w_zs.append(Dense(1, kernel_init=kernel_inits_wz[-1], use_bias=False))
        self.w_zs = w_zs

        w_xs = []
        for i in range(num_hidden):
            w_xs.append(
                nn.Dense(
                    self.dim_hidden[i],
                    kernel_init=self.init_fn(self.init_std),
                    bias_init=self.init_fn(self.init_std),
                    use_bias=True,
                )
            )
        w_xs.append(
            nn.Dense(
                1,
                kernel_init=self.init_fn(self.init_std),
                bias_init=self.init_fn(self.init_std),
                use_bias=True,
            )
        )
        self.w_xs = w_xs

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply ICNN module."""
        z = self.w_xs[0](x)
        z = jnp.multiply(z, z)
        for Wz, Wx in zip(self.w_zs[:-1], self.w_xs[1:-1]):
            z = self.act_fn(jnp.add(Wz(z), Wx(x)))
        y = jnp.add(self.w_zs[-1](z), self.w_xs[-1](x))
        return jnp.squeeze(y, axis=-1)

    def create_train_state(
        self,
        rng: jnp.ndarray,
        optimizer: optax.OptState,
        input_shape: Union[int, Tuple[int, ...]],
    ) -> train_state.TrainState:
        """Create initial `TrainState`."""
        params = self.init(rng, jnp.ones(input_shape))["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)
