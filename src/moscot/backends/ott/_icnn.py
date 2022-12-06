from typing import List, Tuple, Union, Callable, Optional, Sequence, TYPE_CHECKING

from flax import linen as nn
from flax.training import train_state
import optax

from ott.solvers.nn.layers import PositiveDense
import jax.numpy as jnp


class ICNN(nn.Module):
    """Input convex neural network (ICNN) architecture."""

    dim_hidden: Sequence[int]
    init_std: float = 0.1
    init_fn: Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]] = nn.initializers.normal
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu
    pos_weights: bool = False
    split_indices: Optional[List[int]] = None

    def setup(self):
        """Initialize ICNN architecture."""
        num_hidden = len(self.dim_hidden)

        if self.pos_weights:
            Dense = PositiveDense
        else:
            Dense = nn.Dense
        kernel_inits_wz = [self.init_fn(self.init_std) for _ in range(num_hidden + 1)]

        w_xs = []
        w_zs = []
        for i in range(1, num_hidden):
            w_xs.append(
                nn.Dense(
                    self.dim_hidden[i],
                    kernel_init=self.init_fn(self.init_std),
                    bias_init=self.init_fn(self.init_std),
                    use_bias=True,
                )
            )
            if i != 0:
                w_zs.append(
                    Dense(
                        self.dim_hidden[i],
                        kernel_init=kernel_inits_wz[i],
                        use_bias=False,
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
        w_zs.append(Dense(1, kernel_init=kernel_inits_wz[-1], use_bias=False))
        self.w_xs = w_xs
        self.w_zs = w_zs

        if self.split_indices is not None:
            if TYPE_CHECKING:
                assert isinstance(
                    self.split_indices,
                )
            w_zu = []
            w_xu = []
            w_u = []
            v = []

            if (
                self.combiner
            ):  # if combine different conditions into a single one, we assume conditions being concatenated
                cond_dim = self.combiner_output
                # add just one layer
                self.w_c = nn.Dense(  # combiner
                    self.combiner_output,
                    kernel_init=self.init_fn(self.init_std),
                    use_bias=True,
                    bias_init=self.init_fn(self.init_std),
                )

            else:
                cond_dim = len(self.split_indices)

            for i in range(1, num_hidden):
                w_zu.append(
                    nn.Dense(
                        self.dim_hidden[i],
                        kernel_init=self.init_fn(self.init_std),
                        use_bias=True,
                        bias_init=self.init_fn(self.init_std),
                    )
                )
                w_xu.append(  # this the matrix that multiply with x
                    nn.Dense(
                        self.dim_data,
                        kernel_init=self.init_fn(self.init_std),
                        use_bias=True,
                        bias_init=self.init_fn(self.init_std),
                    )
                )
                w_u.append(
                    nn.Dense(
                        self.dim_hidden[i],
                        kernel_init=self.init_fn(self.init_std),
                        use_bias=True,
                        bias_init=self.init_fn(self.init_std),
                    )
                )
                v.append(
                    nn.Dense(
                        cond_dim,
                        kernel_init=self.init_fn(self.init_std),
                        use_bias=True,
                        bias_init=self.init_fn(self.init_std),
                    )
                )
            w_u.append(
                nn.Dense(
                    1,
                    kernel_init=self.init_fn(self.init_std),
                    bias_init=self.init_fn(self.init_std),
                    use_bias=True,
                )
            )
            v.append(
                nn.Dense(
                    1,
                    kernel_init=self.init_fn(self.init_std),
                    bias_init=self.init_fn(self.init_std),
                    use_bias=True,
                )
            )

            self.w_zu = w_zu
            self.w_xu = w_xu
            self.w_u = w_u
            self.v = v

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply ICNN module."""

        if self.split_indices is None:
            z = self.w_xs[0](x)
            z = jnp.multiply(z, z)
            for Wz, Wx in zip(self.w_zs[:-1], self.w_xs[1:-1]):
                z = self.act_fn(jnp.add(Wz(z), Wx(x)))
            y = jnp.add(self.w_zs[-1](z), self.w_xs[-1](x))
        else:
            x = x[: self.split_indices[0]]
            c = x[self.split_indices[0] :]
            if self.combiner:
                c = self.w_c(c)
                c = self.act_fn(c)

            # Initialize
            mlp_condition_embedding = self.w_xu[0](c)
            x_hadamard_1 = jnp.multiply(x, mlp_condition_embedding)
            mlp_condition = self.w_u[0](c)
            z = jnp.add(mlp_condition, self.w_xs[0](x_hadamard_1))
            z = jnp.multiply(z, z)
            u = self.act_fn(self.v[0](c))

            for Wz, Wx, Wzu, Wxu, Wu, V in zip(
                self.w_zs[:-1], self.w_xs[1:-1], self.w_zu[:-1], self.w_xu[1:-1], self.w_u[1:-1], self.v[1:-1]
            ):
                mlp_convex = jnp.clip(Wzu(u), a_min=0)
                z_hadamard_1 = jnp.multiply(z, mlp_convex)
                mlp_condition_embedding = Wxu(u)
                x_hadamard_1 = jnp.multiply(x, mlp_condition_embedding)
                mlp_condition = Wu(u)
                z = self.act_fn(jnp.add(jnp.add(Wz(z_hadamard_1), Wx(x_hadamard_1)), mlp_condition))
                u = self.act_fn(V(u))

            mlp_convex = jnp.clip(self.w_zu[-1](u), a_min=0)  # bs x d
            z_hadamard_1 = jnp.multiply(z, mlp_convex)  # bs x d

            mlp_condition_embedding = self.w_xu[-1](u)  # bs x d
            x_hadamard_1 = jnp.multiply(x, mlp_condition_embedding)  # bs x d

            mlp_condition = self.w_u[-1](u)
            y = jnp.add(jnp.add(self.w_zs[-1](z_hadamard_1), self.w_xs[-1](x_hadamard_1)), mlp_condition)

        return jnp.squeeze(y, axis=-1)

    def create_train_state(
        self,
        rng: jnp.ndarray,
        optimizer: optax.OptState,
        input_shape: Union[int, Tuple[int, ...]],
    ) -> train_state.TrainState:
        """Create initial `TrainState`."""
        if self.split:
            params = self.init(rng, jnp.ones(self.split), jnp.ones(input_shape - self.split))["params"]
            return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)
        params = self.init(rng, jnp.ones(input_shape))["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)
