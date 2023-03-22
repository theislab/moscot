from types import MappingProxyType
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, Type, Union

from anndata import AnnData
import numpy as np
import scipy.spatial
import scanpy as sc
import anndata as ad
from sklearn.preprocessing import normalize


from moscot import _constants
from moscot._docs._docs import d
from moscot._types import (
    ArrayLike,
    Policy_t,
    ProblemStage_t,
    QuadInitializer_t,
    ScaleCost_t,
    Str_Dict_t,
)
from moscot.base.problems.compound_problem import B, CompoundProblem, K
from moscot.base.problems.problem import OTProblem
from moscot.problems._utils import handle_cost, handle_joint_attr
from moscot.problems.cross_modality._mixins import CrossModalityIntegrationMixin
from moscot.utils.subset_policy import DummyPolicy
from moscot.base.output import BaseSolverOutput

__all__ = ["IntegrationProblem"]


@d.dedent
class IntegrationProblem(CompoundProblem[K, OTProblem], CrossModalityIntegrationMixin[K, OTProblem]):
    """
    Class for integrating single cell multiomics data.

    Parameters
    ----------
    adata_src
        Instance of :class:`anndata.AnnData` containing the source data.
    adata_tgt
        Instance of :class:`anndata.AnnData` containing the target data.
    """

    def __init__(self, adata_src: AnnData, adata_tgt: AnnData, **kwargs: Any):
        super().__init__(adata_src, **kwargs)
        self._adata_tgt = adata_tgt
        self.filtered_vars: Optional[Sequence[str]] = None

    def _create_problem(
        self,
        src: K,
        tgt: K,
        src_mask: ArrayLike,
        tgt_mask: ArrayLike,
        **kwargs: Any,
    ) -> OTProblem:
        return self._base_problem_type(
            adata=self.adata_src,
            adata_tgt=self.adata_tgt,
            src_obs_mask=src_mask,
            tgt_obs_mask=None,
            src_key=src,
            tgt_key=tgt,
            **kwargs,
        )

    @d.dedent
    def prepare(
        self,
        src_attr: Str_Dict_t,
        tgt_attr: Str_Dict_t,
        var_names: Optional[Sequence[Any]] = None,
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["dummy"] = "dummy",
        cost: Union[
            Literal["sq_euclidean", "cosine", "bures", "unbalanced_bures"],
            Mapping[str, Literal["sq_euclidean", "cosine", "bures", "unbalanced_bures"]],
        ] = "sq_euclidean",
        a: Optional[str] = None,
        b: Optional[str] = None,
        **kwargs: Any,
    ) -> "IntegrationProblem[K]":
        """
        Prepare the :class:`moscot.problems.cross_modality.IntegrationProblem`.

        This method prepares the data to be passed to the optimal transport solver.

        Parameters
        ----------
        

        Returns
        -------
        :class:`moscot.problems.cross_modality.IntegrationProblem`.

        Examples
        --------
        """
        self._src_attr = src_attr['key']
        self._tgt_attr = tgt_attr['key']

        x = {"attr": "obsm", "key": src_attr} if isinstance(src_attr, str) else src_attr
        y = {"attr": "obsm", "key": tgt_attr} if isinstance(tgt_attr, str) else tgt_attr
        

        xy, kwargs = handle_joint_attr(joint_attr, kwargs)
        xy, _, _ = handle_cost(xy=xy, cost=cost)
        return super().prepare(x=x, y=y, xy=xy, policy="dummy", key=None, cost=cost, a=a, b=b, **kwargs)

    @d.dedent
    def solve(
        self,
        alpha: Optional[float] = 0.5,
        epsilon: Optional[float] = 1e-2,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        rank: int = -1,
        scale_cost: ScaleCost_t = "mean",
        batch_size: Optional[int] = None,
        stage: Union[ProblemStage_t, Tuple[ProblemStage_t, ...]] = ("prepared", "solved"),
        initializer: QuadInitializer_t = None,
        initializer_kwargs: Mapping[str, Any] = MappingProxyType({}),
        jit: bool = True,
        min_iterations: int = 5,
        max_iterations: int = 50,
        threshold: float = 1e-3,
        gamma: float = 10.0,
        gamma_rescale: bool = True,
        ranks: Union[int, Tuple[int, ...]] = -1,
        tolerances: Union[float, Tuple[float, ...]] = 1e-2,
        linear_solver_kwargs: Mapping[str, Any] = MappingProxyType({}),
        device: Optional[Literal["cpu", "gpu", "tpu"]] = None,
        **kwargs: Any,
    ) -> "IntegrationProblem[K]":
        """
        Solve optimal transport problems defined in :class:`moscot.problems.cross_modality.IntegrationProblem`.

        Parameters
        ----------
        %(alpha)s
        %(epsilon)s
        %(tau_a)s
        %(tau_b)s
        %(rank)s
        %(scale_cost)s
        %(pointcloud_kwargs)s
        %(stage)s
        %(initializer_quad)s
        %(initializer_kwargs)s
        %(gw_kwargs)s
        %(sinkhorn_lr_kwargs)s
        %(gw_lr_kwargs)s
        %(linear_solver_kwargs)s
        %(device_solve)s
        %(kwargs_quad_fused)s

        Returns
        -------
        :class:`moscot.problems.space.MappingProblem`.

        Examples
        --------
        %(ex_solve_quadratic)s
        """
        return super().solve(
            alpha=alpha,
            epsilon=epsilon,
            tau_a=tau_a,
            tau_b=tau_b,
            rank=rank,
            scale_cost=scale_cost,
            batch_size=batch_size,
            stage=stage,
            initializer=initializer,
            initializer_kwargs=initializer_kwargs,
            jit=jit,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            threshold=threshold,
            gamma=gamma,
            gamma_rescale=gamma_rescale,
            ranks=ranks,
            tolerances=tolerances,
            linear_solver_kwargs=linear_solver_kwargs,
            device=device,
            **kwargs,
        )  # type: ignore[return-value]

    def normalize(
            self,
            norm="l2", 
            bySample=True
    ):
        assert (norm in ["l1","l2","max"]), "Norm argument has to be either one of 'max', 'l1', or 'l2'."
        if (bySample==True or bySample==None):
            axis=1
        else:
            axis=0
        self.adata_src.obsm[self._src_attr] = normalize(self.adata_src.obsm[self._src_attr], norm=norm, axis=axis)
        self.adata_tgt.obsm[self._tgt_attr] = normalize(self.adata_tgt.obsm[self._tgt_attr], norm=norm, axis=axis)

    def barycentric_projection(
            self,
            SRContoTGT=True
    ):
        if SRContoTGT:
            # Projecting the source domain onto the target domain
            self._tgt_aligned = self.adata_tgt.obsm[self._tgt_attr]
            self.coupling = self[('src', 'tgt')].solution.transport_matrix
            weights = np.sum(self.coupling, axis = 1)
            self._src_aligned = np.matmul(self.coupling, self._tgt_aligned) / weights[:, None]
        else:
            # Projecting the target domain onto the source domain
            self._src_aligned = self.adata_src.obsm[self._src_attr]
            self.coupling = self[('src', 'tgt')].solution.transport_matrix
            weights = np.sum(self.coupling, axis = 1)
            self._tgt_aligned = np.matmul(np.transpose(self.coupling), self._src_aligned) / weights[:, None]
        
        self.adata_src.obsm["X_aligned"] = self._src_aligned
        self.adata_tgt.obsm["X_aligned"] = self._tgt_aligned
        return self._src_aligned, self._tgt_aligned
    
    def integrate(
            self,
            normalize = True,
            norm = "l2",
            SRContoTGT=True,
            **kwargs:Any,
    ) -> ArrayLike:
        """
        Integrate source and target objects
        """
        if normalize:
            self.normalize(norm=norm) # überschreibt so die adata objecte, evlt. lieber neues feld in obsm hinzufügen?
        
        src_aligned, tgt_aligned = self.barycentric_projection(SRContoTGT=SRContoTGT)

        self.src_aligned, self.tgt_aligned = src_aligned, tgt_aligned
        return (self.src_aligned, self.tgt_aligned)

    def plotting(
            self,
            color : Union[str, Sequence[str], None] = None, # add cell type here
            **kwargs:Any, 
    ):
        adata_comb = ad.concat([self.adata_src, self.adata_tgt], join = 'outer', label='batch', index_unique = '-')
        sc.pp.neighbors(adata_comb, use_rep="X_aligned")
        sc.tl.umap(adata_comb)
        if isinstance(color, str):
            col = ["batch", color]
        elif isinstance(color, list):
            col = ['batch']+ color
        else:
            raise ValueError("Input color must be a string or a list of strings.")

        sc.pl.umap(adata_comb, color=col)
        self.adata_comb = adata_comb
        
    @property
    def solution(self) -> Optional[BaseSolverOutput]:
        """Solution of the optimal transport problem."""
        return self._solution
    
    @property
    def adata_tgt(self) -> AnnData:
        """Target data."""
        return self._adata_tgt

    @property
    def adata_src(self) -> AnnData:
        """Source data."""
        return self.adata

    @property
    def _base_problem_type(self) -> Type[B]:
        return OTProblem  # type: ignore[return-value]

    @property
    def _valid_policies(self) -> Tuple[Policy_t, ...]:
        return _constants.DUMMY  # type: ignore[return-value]

    @property
    def _secondary_adata(self) -> Optional[AnnData]:
        return self._adata_tgt
    