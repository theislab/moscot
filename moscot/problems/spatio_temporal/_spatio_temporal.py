from types import MappingProxyType
from typing import Any, Tuple, Mapping, Optional

from typing_extensions import Literal

from moscot._docs import d
from moscot._types import Numeric_t
from moscot._constants._key import Key
from moscot._constants._constants import Policy, ScaleCost
from moscot.problems.time._mixins import TemporalMixin
from moscot.problems.space._mixins import SpatialAlignmentMixin
from moscot.problems.space._alignment import AlignmentProblem
from moscot.problems.base._birth_death import BirthDeathMixin, BirthDeathProblem


@d.dedent
class SpatioTemporalProblem(
    TemporalMixin[Numeric_t, BirthDeathProblem],
    BirthDeathMixin,
    AlignmentProblem[Numeric_t, BirthDeathProblem],
    SpatialAlignmentMixin[Numeric_t, BirthDeathProblem],
):
    """Spatio-Temporal problem."""

    @d.dedent
    def prepare(
        self,
        time_key: str,
        spatial_key: str = Key.obsm.spatial,
        joint_attr: Optional[Mapping[str, Any]] = MappingProxyType({"x_attr": "X", "y_attr": "X"}),
        policy: Literal[Policy.SEQUENTIAL, Policy.TRIL, Policy.TRIU, Policy.EXPLICIT] = Policy.SEQUENTIAL,
        marginal_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> "SpatioTemporalProblem":
        """
        Prepare the :class:`moscot.problems.spatio_temporal.SpatioTemporalProblem`.

        This method executes multiple steps to prepare the problem for the Optimal Transport solver to be ready
        to solve it

        Parameters
        ----------
        time_key
            Key in :attr:`anndata.AnnData.obs` which defines the time point each cell belongs to. It is supposed to be
            of numerical data type.
        spatial_key
            Specifies the way the lineage information is processed. TODO: Specify.
        joint_attr
            Parameter defining how to allocate the data needed to compute the transport maps. If None, the data is read
            from :attr:`anndata.AnnData.X` and for each time point the corresponding PCA space is computed.
            If `joint_attr` is a string the data is assumed to be found in :attr:`anndata.AnnData.obsm`.
            If `joint_attr` is a dictionary the dictionary is supposed to contain the attribute of
            :attr:`anndata.AnnData` as a key and the corresponding attribute as a value.

        %(policy)s
        %(marginal_kwargs)s
        %(a)s
        %(b)s
        subset
            subset of `anndata.AnnData.obs` [key] values of which the policy is to be applied to
        %(reference)s
        %(axis)s
        %(callback)s
        %(callback_kwargs)s
        kwargs
            Keyword arguments for :meth:`moscot.problems.BaseCompoundProblem._create_problems`

        Returns
        -------
        :class:`moscot.problems.spatio_temporal.SpatioTemporalProblem`

        Raises
        ------
        KeyError
            If `time_key` is not in :attr:`anndata.AnnData.obs`.
        KeyError
            If `spatial_key` is not in :attr:`anndata.AnnData.obs`.
        KeyError
            If `joint_attr` is a string and cannot be found in :attr:`anndata.AnnData.obsm`.
        ValueError
            If :attr:`adata.obsp` has no attribute `cost_matrices`.
        TypeError
            If `joint_attr` is not None, not a :class:`str` and not a :class:`dict`

        Notes
        -----
        If `a` and `b` are provided `marginal_kwargs` are ignored.
        """
        # spatial key set in AlignmentProblem
        self.temporal_key = time_key

        marginal_kwargs = dict(marginal_kwargs)
        if self.proliferation_key is not None:
            marginal_kwargs["proliferation_key"] = self.proliferation_key
            kwargs["a"] = True
        if self.apoptosis_key is not None:
            marginal_kwargs["proliferation_key"] = self.apoptosis_key
            kwargs["b"] = True

        return super().prepare(
            spatial_key=spatial_key,
            batch_key=time_key,
            joint_attr=joint_attr,
            policy=policy,
            reference=None,
            marginal_kwargs=marginal_kwargs,
            **kwargs,
        )

    @d.dedent
    def solve(
        self,
        alpha: Optional[float] = 0.5,
        epsilon: Optional[float] = 1e-3,
        scale_cost: str = ScaleCost.MEAN,
        **kwargs: Any,
    ) -> "SpatioTemporalProblem":
        """
        Solve optimal transport problems defined in :class:`moscot.problems.space.SpatioTemporalProblem`.

        Parameters
        ----------
        %(alpha)s
        %(epsilon)s
        %(scale_cost)s

        Returns
        -------
        :class:`moscot.problems.space.SpatioTemporalProblem`
        """
        scale_cost = ScaleCost(scale_cost)
        return super().solve(alpha=alpha, epsilon=epsilon, scale_cost=scale_cost, **kwargs)

    @property
    def _valid_policies(self) -> Tuple[Policy, ...]:
        return Policy.SEQUENTIAL, Policy.TRIL, Policy.TRIU, Policy.EXPLICIT
