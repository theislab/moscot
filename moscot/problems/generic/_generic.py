from types import MappingProxyType
from typing import Any, Type, Tuple, Union, Mapping, Optional

from typing_extensions import Literal

from anndata import AnnData

from moscot._docs import d
from moscot.problems.base import OTProblem, CompoundProblem  # type: ignore[attr-defined]
from moscot.problems.generic._mixins import GenericAnalysisMixin
from moscot.problems.base._compound_problem import B, K


@d.dedent
class SinkhornProblem(CompoundProblem[K, B], GenericAnalysisMixin[K, B]):
    """
    Class for solving linear OT problems.

    Parameters
    ----------
    %(adata)s

    Examples
    --------
    See notebook TODO(@MUCDK) LINK NOTEBOOK for how to use it
    """

    def __init__(self, adata: AnnData, **kwargs: Any):
        super().__init__(adata, **kwargs)

    @d.dedent
    def prepare(
        self,
        key: str,
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "pairwise", "explicit"] = "sequential",
        **kwargs: Any,
    ) -> "SinkhornProblem[K, B]":
        """
        Prepare the :class:`moscot.problems.generic.SinkhornProblem`.

        This method executes multiple steps to prepare the optimal transport problems.

        Parameters
        ----------
        %(key)s
        %(joint_attr)s
        %(policy)s
        %(marginal_kwargs)s
        %(a)s
        %(b)s
        %(subset)s
        %(reference)s
        %(axis)s
        %(callback)s
        %(callback_kwargs)s
        kwargs
            Keyword arguments for :meth:`moscot.problems.BaseCompoundProblem._create_problems`.

        Returns
        -------
        :class:`moscot.problems.generic.SinkhornProblem`

        Notes
        -----
        If `a` and `b` are provided `marginal_kwargs` are ignored.
        """
        self.batch_key = key
        if joint_attr is None:
            kwargs["callback"] = "local-pca"
            kwargs["callback_kwargs"] = {**kwargs.get("callback_kwargs", {}), **{"return_linear": True}}
        elif isinstance(joint_attr, str):
            kwargs["xy"] = {
                "x_attr": "obsm",
                "x_key": joint_attr,
                "y_attr": "obsm",
                "y_key": joint_attr,
            }
        elif isinstance(joint_attr, Mapping):
            kwargs["xy"] = joint_attr
        else:
            raise TypeError("TODO")

        return super().prepare(
            key=key,
            policy=policy,
            **kwargs,
        )

    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return "sequential", "pairwise", "explicit"


@d.get_sections(base="GWProblem", sections=["Parameters"])
@d.dedent
class GWProblem(CompoundProblem[K, B], GenericAnalysisMixin[K, B]):
    """
    Class for solving Gromov-Wasserstein problems.

    Parameters
    ----------
    %(adata)s

    Examples
    --------
    See notebook TODO(@MUCDK) LINK NOTEBOOK for how to use it
    """

    def __init__(self, adata: AnnData, **kwargs: Any):
        super().__init__(adata, **kwargs)

    @d.dedent
    def prepare(
        self,
        key: str,
        GW_attr: Mapping[str, Any] = MappingProxyType({}),
        policy: Literal["sequential", "pairwise", "explicit"] = "sequential",
        **kwargs: Any,
    ) -> "GWProblem[K, B]":
        """
        GWProblem.
        """
        self.batch_key = key
        # TODO(michalk8): use and
        if not len(GW_attr):
            if "cost_matrices" not in self.adata.obsp:
                raise ValueError(
                    "TODO: default location for quadratic loss is `adata.obsp[`cost_matrices`]` \
                        but adata has no key `cost_matrices` in `obsp`."
                )
        # TODO(michalk8): refactor me
        GW_attr = dict(GW_attr)
        GW_attr.setdefault("attr", "obsp")
        GW_attr.setdefault("key", "cost_matrices")
        GW_attr.setdefault("loss", "Euclidean")
        GW_attr.setdefault("tag", "cost")
        GW_attr.setdefault("loss_kwargs", {})
        x = y = GW_attr

        return super().prepare(
            key,
            x=x,
            y=y,
            policy=policy,
            **kwargs,
        )

    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return "sequential", "pairwise", "explicit"


@d.dedent
class FGWProblem(GWProblem[K, B]):
    """
    Class for solving Fused Gromov-Wasserstein problems.

    Parameters
    ----------
    %(adata)s

    Examples
    --------
    See notebook TODO(@MUCDK) LINK NOTEBOOK for how to use it
    """

    @d.dedent
    def prepare(
        self,
        key: str,
        joint_attr: Mapping[str, Any] = MappingProxyType({}),
        GW_attr: Mapping[str, Any] = MappingProxyType({}),
        policy: Literal["sequential", "pairwise", "explicit"] = "sequential",
        **kwargs: Any,
    ) -> "FGWProblem[K, B]":
        """
        Prepare the :class:`moscot.problems.generic.GWProblem`.

        Parameters
        ----------
        %(GWProblem.parameters)s
        %(joint_attr)s

        """
        return super().prepare(key=key, GW_attr=GW_attr, joint_attr=joint_attr, policy=policy, **kwargs)
