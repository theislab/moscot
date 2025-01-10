from dataclasses import dataclass
from typing import Any, ClassVar, Hashable, Optional, TypeVar, Union

import jax
import jax.numpy as jnp

K = TypeVar("K", bound=Hashable)


@dataclass(frozen=True, repr=True)
class NeuralDistribution:
    """Data container representing a distribution to be used in OT-based flow models.

    Can be either a source or target distribution.
    Keep in mind that if a OT-based flow model is used,
    sizes such as `n_shared_features`, `n_flow` should be same
    in general for both source and target distributions.

    Parameters
    ----------
    shared_space (n_samples, n_shared_features)
        Distribution living in a shared space.
        Used for the linear term of matching step.
        I.e., given to the matching function of OT Based Flow Models
    incomparable_space (n_samples, n_incomparable_features)
        Distribution living in an incomparable space.
        Used for the quadratic term of matching step.
    condition (n_samples, n_conditions)
        Condition for the distributions.
    augment (n_samples, n_augment)
        Augmentation to be used in the flow model.
    flow (n_samples, n_flow)
        Often equal to `shared_space` but can be different.
        This will either be given to the flow model as primary input
        (not the case for GENOT as GENOT uses noise instead) or
        the output of the flow model.

    """

    shared_space: jax.Array
    incomparable_space: Optional[jax.Array] = None
    condition: Optional[jax.Array] = None
    augment: Optional[jax.Array] = None
    flow: Optional[jax.Array] = None

    FIELDS: ClassVar[tuple[str]] = ["shared_space", "incomparable_space", "condition", "augment", "flow"]

    def __post_init__(self) -> None:
        fields = ["shared_space", "incomparable_space", "condition", "augment", "flow"]
        if all(getattr(self, field) is None for field in fields):
            raise ValueError(f"At least one of the fields `{fields}` must be provided.")
        given_fields = [field for field in fields if getattr(self, field) is not None]
        # if all number of samples are not equal
        if len({getattr(self, field).shape[0] for field in given_fields}) > 1:
            raise ValueError("All fields must have the same number of samples.")

    @property
    def n_samples(self) -> int:
        """Number of samples in the distribution."""
        return self.shared_space.shape[0]

    def __getitem__(
        self, idx: Union[int, slice, jnp.ndarray, jax.Array, list[Any], tuple[Any]]
    ) -> "NeuralDistribution":
        """
        Return a new DistributionContainer where .xy, .xx, .conditions
        are sliced by `idx` (if they are not None).

        This allows usage like:
            new_container = distribution_container[train_ixs]
        """  # noqa: D205
        # TODO: Normally this is inefficient
        # But we first need to separate the slicing of training and validation data
        # Before creating this DistributionContainer!
        # Slice xy
        given_fields = [field for field in self.FIELDS if getattr(self, field) is not None]
        return NeuralDistribution(**{field: getattr(self, field)[idx] for field in given_fields})


@dataclass
class DistributionCollection(dict[K, NeuralDistribution]):
    """Collection of distributions."""

    def __post_init__(self) -> None:
        # check if all the shared spaces have the same shape[1]
        shared_spaces = {dist.shared_space.shape[1] for dist in self.values()}
        if len(shared_spaces) > 1:
            raise ValueError("All shared spaces must have the same number of features.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{list(self.keys())}"

    def __str__(self) -> str:
        return repr(self)
