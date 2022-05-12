from typing import Any, Dict, Tuple, Type, Optional, Union, Mapping, Literal
from types import MappingProxyType

from anndata import AnnData

from moscot._docs import d
from moscot.problems import SingleCompoundProblem, OTProblem
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
        Prepare the :class:`moscot.problems.time.TemporalProblem`.

        This method executes multiple steps to prepare the optimal transport problems.

        Parameters
        ----------
        %(key)s
        %(joint_attr)s
        %(policy)s
        %(marginal_kwargs)s
        %(a)s
        %(b)s
        subset
            Subset of `anndata.AnnData.obs` [key] values of which the policy is to be applied to.
        %(reference)s
        %(axis)s
        %(callback)s
        %(callback_kwargs)s
        kwargs
            Keyword arguments for :meth:`moscot.problems.CompoundBaseProblem._create_problems`.

        Returns
        -------
        :class:`moscot.problems.generic.SinkhornProblem`

        Raises
        ------
        KeyError
            If `key` is not in :attr:`anndata.AnnData.obs`.
        KeyError
            If `joint_attr` is a string and cannot be found in :attr:`anndata.AnnData.obsm`.

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
        Prepare the :class:`moscot.problems.time.LineageProblem`.

        This method executes multiple steps to prepare the problem for the Optimal Transport solver to be ready
        to solve it

        Parameters
        ----------
        %(time_key)s
        lineage_attr
            Specifies the way the lineage information is processed. TODO: Specify.
        joint_attr
            Parameter defining how to allocate the data needed to compute the transport maps. If None, the data is read
            from :attr:`anndata.AnnData.X` and for each time point the corresponding PCA space is computed.
            If `joint_attr` is a string the data is assumed to be found in :attr:`anndata.AnnData.obsm`.
            If `joint_attr` is a dictionary the dictionary is supposed to contain the attribute of
            :attr:`anndata.AnnData` as a key and the corresponding attribute as a value.
        policy
            defines which transport maps to compute given different cell distributions
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
        :class:`moscot.problems.time.LineageProblem`

        Raises
        ------
        KeyError
            If `time_key` is not in :attr:`anndata.AnnData.obs`.
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
        # TODO(michalk8): use and
        if not len(lineage_attr):
            if "cost_matrices" not in self.adata.obsp:
                raise ValueError(
                    "TODO: default location for quadratic loss is `adata.obsp[`cost_matrices`]` \
                        but adata has no key `cost_matrices` in `obsp`."
                )
        # TODO(michalk8): refactor me
        lineage_attr = dict(lineage_attr)
        lineage_attr.setdefault("attr", "obsp")
        lineage_attr.setdefault("key", "cost_matrices")
        lineage_attr.setdefault("loss", None)
        lineage_attr.setdefault("tag", "cost")
        lineage_attr.setdefault("loss_kwargs", {})
        x = y = lineage_attr

        return super().prepare(
            time_key,
            joint_attr=joint_attr,
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
class FGWProblem(SingleCompoundProblem, AnalysisMixin):
    """
    Class for solving Fused Gromov-Wasserstein problems.
    
    Parameters
    ----------
    %(adata)s

    Examples
    --------
    See notebook TODO(@MUCDK) LINK NOTEBOOK for how to use it
    """
    def __init__(self, adata: AnnData, **kwargs: Any):
        super().__init__(adata, **kwargs)

    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem

    @property
    def _valid_policies(self) -> Tuple[str, ...]:
        return "sequential", "pairwise", "explicit"