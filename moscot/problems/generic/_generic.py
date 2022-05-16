from types import MappingProxyType
from typing import Any, Literal, Mapping, Optional, Tuple, Type, Union

from anndata import AnnData

from moscot._docs import d
from moscot.problems import OTProblem, SingleCompoundProblem
from moscot.analysis_mixins import AnalysisMixin
from moscot.problems._compound_problem import B


@d.dedent
class SinkhornProblem(SingleCompoundProblem, AnalysisMixin):
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
    ) -> "SinkhornProblem":
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
            Keyword arguments for :meth:`moscot.problems.CompoundBaseProblem._create_problems`.

        Returns
        -------
        :class:`moscot.problems.generic.SinkhornProblem`

        Notes
        -----
        If `a` and `b` are provided `marginal_kwargs` are ignored.
        """
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
class GWProblem(SingleCompoundProblem, AnalysisMixin):
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
    ) -> "GWProblem":
        """
        Prepare the :class:`moscot.problems.generic.GWProblem`.

        This method executes multiple steps to prepare the problem for the Optimal Transport solver to be ready
        to solve it

        Parameters
        ----------
        %(key)s
        GW_attr
            Specifies the way the GW information is processed. TODO: Specify.
        %(joint_attr)
        %(policy)s
        %(marginal_kwargs)s
        %(a)s
        %(b)s
        %(subset)s
        %(reference)s
        %(callback)s
        %(callback_kwargs)s
        kwargs
            Keyword arguments for :meth:`moscot.problems.CompoundBaseProblem._create_problems`

        Returns
        -------
        :class:`moscot.problems.generic.GWProblem`

        Notes
        -----
        If `a` and `b` are provided `marginal_kwargs` are ignored.
        """
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
        GW_attr.setdefault("loss", None)
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
class FGWProblem(GWProblem):
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
        *args,
        joint_attr: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> "FGWProblem":
        """
        Prepare the :class:`moscot.problems.generic.GWProblem`.

        Parameters
        ----------
        %(GWProblem.parameters)s
        %(joint_attr)s

        """
        kwargs["joint_attr"] = joint_attr
        return super().prepare(*args, joint_attr=joint_attr, **kwargs)
