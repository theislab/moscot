from typing import Callable, Optional, Sequence, Tuple, Union

import optax
from flax import linen as nn
from flax.training import train_state

import jax.numpy as jnp
from ott.solvers.nn.layers import PositiveDense


class ICNN(nn.Module):
    """Input convex neural network (ICNN) architecture."""

    dim_hidden: Sequence[int]
    input_dim: int
    cond_dim: int
    init_std: float = 0.1
    init_fn: Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]] = nn.initializers.normal  # type: ignore[name-defined]  # noqa: E501
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu  # type: ignore[name-defined]
    pos_weights: bool = False

    def setup(self):
        """Initialize ICNN architecture."""
        num_hidden = len(self.dim_hidden)

        Dense = PositiveDense if self.pos_weights else nn.Dense
        kernel_inits_wz = [self.init_fn(self.init_std) for _ in range(num_hidden + 1)]

        w_xs = []
        w_zs = []
        for i in range(0, num_hidden):
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

        if self.cond_dim:
            w_zu = []
            w_xu = []
            w_u = []
            v = []

            for i in range(0, num_hidden):
                if i != 0:
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
                        self.input_dim,  # self.dim_hidden[i],
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
                        2,
                        kernel_init=self.init_fn(self.init_std),
                        use_bias=True,
                        bias_init=self.init_fn(self.init_std),
                    )
                )
            w_zu.append(
                nn.Dense(
                    self.dim_hidden[-1],
                    kernel_init=self.init_fn(self.init_std),
                    use_bias=True,
                    bias_init=self.init_fn(self.init_std),
                )
            )
            w_xu.append(  # this the matrix that multiply with x
                nn.Dense(
                    self.input_dim,
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
    def __call__(self, x: jnp.ndarray, c: Optional[jnp.ndarray] = None) -> jnp.ndarray:  # type: ignore[name-defined]
        """Apply ICNN module."""
        assert (c is not None) == (self.cond_dim > 0), "`conditional` flag and whether `c` is provided must match."

        if not self.cond_dim:
            z = self.w_xs[0](x)
            z = jnp.multiply(z, z)
            for Wz, Wx in zip(self.w_zs[:-1], self.w_xs[1:-1]):
                z = self.act_fn(jnp.add(Wz(z), Wx(x)))
            y = jnp.add(self.w_zs[-1](z), self.w_xs[-1](x))
        else:
            # Initialize
            mlp_condition_embedding = self.w_xu[0](c)
            x_hadamard_1 = jnp.multiply(x, mlp_condition_embedding)
            mlp_condition = self.w_u[0](c)
            z = jnp.add(mlp_condition, self.w_xs[0](x_hadamard_1))
            z = jnp.multiply(z, z)
            u = self.act_fn(self.v[0](c))

            for Wz, Wx, Wzu, Wxu, Wu, V in zip(
                self.w_zs[:-1], self.w_xs[:-1], self.w_zu[:-1], self.w_xu[1:-1], self.w_u[1:-1], self.v[1:-1]
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
        rng: jnp.ndarray,  # type: ignore[name-defined]
        optimizer: optax.OptState,
        input_shape: Union[int, Tuple[int, ...]],
    ) -> train_state.TrainState:
        """Create initial `TrainState`."""
        condition = (
            jnp.ones(
                shape=[
                    self.cond_dim,
                ]
            )
            if self.cond_dim
            else None
        )
        params = self.init(rng, x=jnp.ones(input_shape), c=condition)["params"]
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=optimizer)
