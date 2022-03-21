from abc import ABC
from typing import Number, Optional
import numpy.typing as npt
import numpy as np


# TODO(michalk8): need to think about this a bit more
class AnalysisMixin(ABC):
    def _sample_from_tmap(self, start: Number,
        end: Number,
        n_samples: int,
        source_dim: Optional[int] = None,
        target_dim: Optional[int] = None,
        batch_size: int = 256,
        account_for_unbalancedness: bool = False,
        interpolation_parameter: Optional[float] = None) -> npt.ArrayLike:

        if account_for_unbalancedness:
            if interpolation_parameter is None:
                raise ValueError("TODO: if unbalancedness is to be accounted for `interpolation_parameter` must be provided")

        mass = np.ones(target_dim)
        if account_for_unbalancedness:
            col_sums = (
                np.array(
                    np.squeeze(
                        self.push(
                            start=start,
                            end=end,
                            normalize=True,
                            scale_by_marginals=False,
                            plans={(start, end): [(start, end)]},
                        )
                    )
                )
                + 1e-12
            )
            mass = mass / np.power(col_sums, 1 - interpolation_parameter)
        row_probability = np.array(
            np.squeeze(
                self.pull(
                    start=start,
                    end=end,
                    data=mass,
                    normalize=True,
                    scale_by_marginals=False,
                    plans={(start, end): [(start, end)]},
                )
            )
        )
        rows_sampled = np.random.choice(source_dim, p=row_probability / row_probability.sum(), size=n_samples)
        rows, counts = np.unique(rows_sampled, return_counts=True)
        for batch in range(int(np.floor(len(rows) / batch_size))):
            rows_batch = rows[batch * batch_size : (batch + 1) * batch_size]
            counts_batch = counts[batch * batch_size : (batch + 1) * batch_size]
            data = np.zeros((source_dim, batch_size))
            data[rows_batch, range(batch_size)] = 1
            col_p_given_row = np.array(
                np.squeeze(
                    self.push(
                        start=start,
                        end=end,
                        data=data,
                        normalize=True,
                        scale_by_marginals=False,
                        plans={(start, end): [(start, end)]},
                    )
                )
            )
            if account_for_unbalancedness:
                col_p_given_row = col_p_given_row / col_sums[:, None]
            kwargs_list = [
                dict(size=counts_batch[i], p=col_p_given_row[:, i] / col_p_given_row[:, i].sum())
                for i in range(batch_size)
            ]
            cols_sampled = list(map(mappable_choice, [len(target_data)] * len(kwargs_list), kwargs_list))
            updated_index = current_index + np.sum(counts_batch)
            result[current_index:updated_index, :] = (
                source_data[np.repeat(rows_batch, counts_batch), :] * (1 - interpolation_parameter)
                + target_data[np.hstack(cols_sampled), :] * interpolation_parameter
            )
            
            current_index = updated_index



# TODO(michalk8): CompoundAnalysisMixin?
