from types import MappingProxyType
from typing import Any, Tuple, Mapping, Optional

from typing_extensions import Literal

from moscot._docs import d
from moscot.analysis_mixins import TemporalAnalysisMixin, SpatialAlignmentAnalysisMixin
from moscot.problems.mixins import BirthDeathMixin
from moscot.problems.space._alignment import AlignmentProblem


@d.dedent
class SpatioTemporalProblem(TemporalAnalysisMixin, BirthDeathMixin, AlignmentProblem, SpatialAlignmentAnalysisMixin):
    """Spatio-Temporal problem."""

    @d.dedent
    def prepare(
        self,
        time_key: str,
        spatial_key: str = "spatial",
        joint_attr: Optional[Mapping[str, Any]] = MappingProxyType({"x_attr": "X", "y_attr": "X"}),
        policy: Literal["sequential", "pairwise", "triu", "tril", "explicit"] = "sequential",
        marginal_kwargs: Mapping[str, Any] = MappingProxyType({}),
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> "AlignmentProblem":
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
            Keyword arguments for :meth:`moscot.problems.CompoundBaseProblem._create_problems`

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
        self.spatial_key = spatial_key
        self.temporal_key = time_key
        # TODO(michalk8): check for spatial key
        x = y = {"attr": "obsm", "key": self.spatial_key, "tag": "point_cloud"}

        if joint_attr is None:
            kwargs["callback"] = "local-pca"
            kwargs["callback_kwargs"] = {**kwargs.get("callback_kwargs", {}), **{"return_linear": True}}

        marginal_kwargs = dict(marginal_kwargs)
        marginal_kwargs["proliferation_key"] = self.proliferation_key
        marginal_kwargs["apoptosis_key"] = self.apoptosis_key
        if "a" not in kwargs:
            kwargs["a"] = self.proliferation_key is not None or self.apoptosis_key is not None
        if "b" not in kwargs:
            kwargs["b"] = self.proliferation_key is not None or self.apoptosis_key is not None

        return super().prepare(
            x=x, y=y, xy=joint_attr, policy=policy, batch_key=time_key, reference=reference, **kwargs
        )

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return "sequential", "pairwise", "triu", "tril", "explicit"
