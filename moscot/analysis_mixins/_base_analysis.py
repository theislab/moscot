from typing import Any, Dict, List, Tuple, Optional, Protocol, Sequence

from scipy.sparse.linalg import LinearOperator

import numpy as np

from moscot._types import ArrayLike, Numeric_t
from moscot.solvers._output import BaseSolverOutput
from moscot.problems._subset_policy import SubsetPolicy
from moscot.problems.base._compound_problem import B, K, Key, ApplyOutput_t


class AnalysisMixinProtocol(Protocol[K, B]):
    """Protocol class."""

    _policy: SubsetPolicy
    solutions: Dict[Key[K], BaseSolverOutput]
    problems: Optional[Dict[Key[K], B]]

    def _apply(
        self,
        start: Optional[K] = None,
        end: Optional[K] = None,
        normalize: bool = True,
        forward: bool = True,
        scale_by_marginals: bool = False,
        filter: Optional[Sequence[Key[K]]] = None,  # noqa: A002
        data: Optional[ArrayLike] = None,
    ) -> ApplyOutput_t[K]:
        ...


class AnalysisMixin:
    """Base Analysis Mixin."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _sample_from_tmap(
        self: AnalysisMixinProtocol[K],
        start: K,
        end: K,
        n_samples: int,
        source_dim: int,
        target_dim: int,
        batch_size: int = 256,
        account_for_unbalancedness: bool = False,
        interpolation_parameter: Optional[Numeric_t] = None,
        seed: Optional[int] = None,
    ) -> Tuple[ArrayLike, List[ArrayLike]]:

        rng = np.random.RandomState(seed)
        if account_for_unbalancedness and interpolation_parameter is None:

            raise ValueError(
                "TODO: if unbalancedness is to be accounted for `interpolation_parameter` must be provided."
            )
        if interpolation_parameter is not None and (0 > interpolation_parameter or interpolation_parameter > 1):
            raise ValueError(f"TODO: interpolation parameter must be between 0 and 1 but is {interpolation_parameter}.")
        mass = np.ones(target_dim)
        if account_for_unbalancedness and interpolation_parameter is not None:
            col_sums: ArrayLike = self._apply(  # type: ignore[assignment]
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
        self: AnalysisMixinProtocol[K], start: K, end: K, forward: bool = True, scale_by_marginals: bool = True
    ) -> LinearOperator:
        """Interpolate transport matrix."""
        # TODO(@MUCDK, @giovp, discuss what exactly this function should do, seems like it could be more generic)
        fst, *rest = self._policy.plan(start=start, end=end)[start, end]
        return self.solutions[fst].chain(
            [self.solutions[r] for r in rest], forward=forward, scale_by_marginals=scale_by_marginals
        )
