from abc import ABC
from typing import Any, List, Tuple, Union, Optional
from functools import partial

from scipy.sparse.linalg import LinearOperator

import numpy as np
import numpy.typing as npt

from moscot.solvers._output import JointOperator, BaseSolverOutput


# TODO(michalk8): need to think about this a bit more
# TODO(MUCDK): remove ABC?
class AnalysisMixin(ABC):
    """Base Analysis Mixin."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _sample_from_tmap(
        self,
        start: Any,
        end: Any,
        n_samples: int,
        source_dim: int,
        target_dim: int,
        batch_size: int = 256,
        account_for_unbalancedness: bool = False,
        interpolation_parameter: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[npt.ArrayLike, List[npt.ArrayLike]]:

        rng = np.random.RandomState(seed)
        if account_for_unbalancedness and interpolation_parameter is None:

            raise ValueError(
                "TODO: if unbalancedness is to be accounted for `interpolation_parameter` must be provided."
            )
        if interpolation_parameter is not None and (0 > interpolation_parameter or interpolation_parameter > 1):
            raise ValueError(f"TODO: interpolation parameter must be between 0 and 1 but is {interpolation_parameter}.")
        mass = np.ones(target_dim)
        if account_for_unbalancedness:
            col_sums = self._apply(
                start=start,
                end=end,
                normalize=True,
                forward=True,
                scale_by_marginals=False,
                filter=[(start, end)],
            )[(start, end)]
            col_sums = np.asarray(col_sums).squeeze() + 1e-12
            mass = mass / np.power(col_sums, 1 - interpolation_parameter)

        row_probability = np.asarray(
            self._apply(
                start=start,
                end=end,
                data=mass,
                normalize=True,
                forward=False,
                scale_by_marginals=False,
                filter=[(start, end)],
            )[(start, end)]
        ).squeeze()

        rows_sampled = rng.choice(source_dim, p=row_probability / row_probability.sum(), size=n_samples)
        rows, counts = np.unique(rows_sampled, return_counts=True)
        all_cols_sampled = []
        for batch in range(0, len(rows), batch_size):
            rows_batch = rows[batch : batch + batch_size]
            counts_batch = counts[batch : batch + batch_size]
            data = np.zeros((source_dim, len(rows_batch)))
            data[rows_batch, range(len(rows_batch))] = 1

            col_p_given_row = np.asarray(
                self._apply(
                    start=start,
                    end=end,
                    data=data,
                    normalize=True,
                    forward=True,
                    scale_by_marginals=False,
                    filter=[(start, end)],
                )[(start, end)]
            ).squeeze()
            if account_for_unbalancedness:
                col_p_given_row = col_p_given_row / col_sums[:, None]

            cols_sampled = [
                rng.choice(a=target_dim, size=counts_batch[i], p=col_p_given_row[:, i] / col_p_given_row[:, i].sum())
                for i in range(len(rows_batch))
            ]
            all_cols_sampled.extend(cols_sampled)
        return rows, all_cols_sampled

    def _interpolate_transport(
        self, start: Any, end: Any, forward: bool = True, scale_by_marginals: bool = True
    ) -> Union[npt.ArrayLike, LinearOperator]:
        """Interpolate transport matrix."""
        # TODO(@MUCDK, @giovp, discuss what exactly this function should do, seems like it could be more generic)
        steps = self._policy.plan(start=start, end=end)[start, end]
        tmap = self._as_linear_operator(
            [self.problems[i].solution for i in steps], forward=forward, scale_by_marginals=scale_by_marginals
        )
        return tmap

    def _as_linear_operator(
        self, outputs: Tuple[BaseSolverOutput, ...], *, forward: bool, scale_by_marginals: bool
    ) -> LinearOperator:
        op = JointOperator(outputs)
        out = outputs[0]
        dtype = out.push(out._ones(out.shape[0])).dtype  # TODO(giovp): can't we set dtype as property of Output?
        push = partial(op.push, scale_by_marginals=scale_by_marginals)
        pull = partial(op.pull, scale_by_marginals=scale_by_marginals)

        return LinearOperator(
            shape=op.shape, dtype=dtype, matvec=push if forward else pull, rmatvec=pull if forward else push
        )


# TODO(michalk8): CompoundAnalysisMixin?
