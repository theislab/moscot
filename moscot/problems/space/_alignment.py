from typing import Any, Type, Tuple, Mapping, Optional

from typing_extensions import Literal

from moscot._docs import d
from moscot._constants._key import Key
from moscot._constants._constants import ScaleCost
from moscot.problems.space._mixins import SpatialAlignmentMixin
from moscot.problems.base._base_problem import OTProblem
from moscot.problems.base._compound_problem import B, K, CompoundProblem

__all__ = ["AlignmentProblem"]


# need generic type B for SpatioTemporal
@d.dedent
class AlignmentProblem(CompoundProblem[K, B], SpatialAlignmentMixin[K, B]):
    """
    Class for aligning spatial omics data, based on :cite:`zeira2022`.

    The `AlignmentProblem` allows to align spatial omics data via optimal transport.

    Parameters
    ----------
    %(adata)s

    Examples
    --------
    See notebook TODO(@giovp) LINK NOTEBOOK for how to use it
    """

    @d.dedent
    def _prepare(
        self,
        batch_key: str,
        spatial_key: str = Key.obsm.spatial,
        joint_attr: Optional[Mapping[str, Any]] = None,
        policy: Literal["sequential", "star"] = "sequential",
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Prepare the :class:`moscot.problems.space.AlignmentProblem`.

        This method prepares the data to be passed to the optimal transport solver.

        Parameters
        ----------
        %(spatial_key)s
        %(batch_key)s

        joint_attr
            Parameter defining how to allocate the data needed to compute the transport maps.
            If None, ``var_names`` is not an empty list, and ``adata_sc`` and ``adata_sp``
            share some genes in common, the corresponding PCA space is computed.
            If `joint_attr` is a dictionary the dictionary is supposed to contain the attribute of
            :attr:`anndata.AnnData` as a key and the corresponding attribute as a value.

        %(policy)s

        reference
            Only used if `policy="star"`, it's the value for reference stored
            in :attr:`adata.obs```["batch_key"]``.

        Returns
        -------
        :class:`moscot.problems.space.MappingProblem`
        """
        self.spatial_key = spatial_key

        x = y = {"attr": "obsm", "key": self.spatial_key, "tag": "point_cloud"}

        if joint_attr is None:
            kwargs["callback"] = "local-pca"
            kwargs["callback_kwargs"] = {**kwargs.get("callback_kwargs", {}), **{"return_linear": True}}

        return super()._prepare(x=x, y=y, xy=joint_attr, policy=policy, key=batch_key, reference=reference, **kwargs)

    @d.dedent
    def _solve(
        self,
        alpha: Optional[float] = 0.4,
        epsilon: Optional[float] = 1e-1,
        scale_cost: str = ScaleCost.MEAN,
        **kwargs: Any,
    ) -> None:
        """
        Solve optimal transport problems defined in :class:`moscot.problems.space.AlignmentProblem`.

        Parameters
        ----------
        %(alpha)s
        %(epsilon)s
        %(scale_cost)s

        Returns
        -------
        :class:`moscot.problems.space.AlignmentProblem`
        """
        return super()._solve(alpha=alpha, epsilon=epsilon, scale_cost=scale_cost, **kwargs)

    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem  # type: ignore[return-value]

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return "sequential", "star"
