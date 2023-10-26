from typing import Any, Callable, Optional, Sequence, Tuple, Union

import optax
from flax import linen as nn
from flax.core import frozen_dict

import jax
import jax.numpy as jnp
from ott.solvers.nn.layers import PositiveDense
from ott.solvers.nn.models import ModelBase, NeuralTrainState

PotentialValueFn_t = Callable[[jnp.ndarray], jnp.ndarray]
PotentialGradientFn_t = Callable[[jnp.ndarray], jnp.ndarray]


class ICNN(ModelBase):
    """Input convex neural network (ICNN) architecture."""

    dim_hidden: Sequence[int]
    input_dim: int
    cond_dim: int
    init_std: float = 0.1
    init_fn: Callable[[jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]] = nn.initializers.normal  # noqa: E501
    act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu
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
    def __call__(self, x: jnp.ndarray, c: Optional[jnp.ndarray] = None) -> jnp.ndarray:
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
        rng: jnp.ndarray,
        optimizer: optax.OptState,
        input_shape: Union[int, Tuple[int, ...]],
        **kwargs: Any,
    ) -> NeuralTrainState:
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
        return NeuralTrainState.create(
            apply_fn=self.apply,
            params=params,
            tx=optimizer,
            potential_value_fn=self.potential_value_fn,
            potential_gradient_fn=self.potential_gradient_fn,
            **kwargs,
        )

    def is_potential(self) -> bool:
        """Indicate if the module implements a potential value or a vector field.

        Returns
        -------
        ``True`` if the module defines a potential, ``False`` if it defines a
        vector field.
        """
        return True

    def potential_value_fn(
        self,
        params: frozen_dict.FrozenDict[str, jnp.ndarray],
        other_potential_value_fn: Optional[PotentialValueFn_t] = None,
        c: Optional[jnp.ndarray] = None,
    ) -> PotentialValueFn_t:
        r"""Return a function giving the value of the potential.

        Applies the module if :attr:`is_potential` is ``True``, otherwise
        constructs the value of the potential from the gradient with

        .. math::

        g(y) = -f(\nabla_y g(y)) + y^T \nabla_y g(y)

        where :math:`\nabla_y g(y)` is detached for the envelope theorem
        :cite:`danskin:67,bertsekas:71`
        to give the appropriate first derivatives of this construction.

        Args:
        params: parameters of the module
        other_potential_value_fn: function giving the value of the other
            potential. Only needed when :attr:`is_potential` is ``False``.

        Returns
        -------
        A function that can be evaluated to obtain a potential value, or a linear
        interpolation of a potential.
        """
        if self.is_potential:
            return lambda x: self.apply({"params": params}, x, c=c)

        assert other_potential_value_fn is not None, (
            "The value of the gradient-based potential depends " "on the value of the other potential."
        )

        def value_fn(x: jnp.ndarray) -> jnp.ndarray:
            squeeze = x.ndim == 1
            if squeeze:
                x = jnp.expand_dims(x, 0)
            grad_g_x = jax.lax.stop_gradient(self.apply({"params": params}, x))
            value = -other_potential_value_fn(grad_g_x) + jax.vmap(jnp.dot)(grad_g_x, x)
            return value.squeeze(0) if squeeze else value

        return value_fn

    def potential_gradient_fn(
        self,
        params: frozen_dict.FrozenDict[str, jnp.ndarray],
        c: Optional[jnp.ndarray] = None,
    ) -> PotentialGradientFn_t:
        """Return a function returning a vector or the gradient of the potential.

        Args:
        params: parameters of the module

        Returns
        -------
        A function that can be evaluated to obtain the potential's gradient
        """
        if self.is_potential:
            return jax.vmap(jax.grad(self.potential_value_fn(params)))
        return lambda x: self.apply({"params": params}, x, c=c)
