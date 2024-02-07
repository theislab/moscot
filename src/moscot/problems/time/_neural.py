from types import MappingProxyType
from typing import Any, Dict, Literal, Mapping, Optional, Tuple, Type, Union

import optax

from moscot import _constants
from moscot._types import Numeric_t, Policy_t
from moscot.base.problems._mixins import NeuralAnalysisMixin
from moscot.base.problems.birth_death import BirthDeathMixin, BirthDeathNeuralProblem
from moscot.base.problems.compound_problem import CompoundProblem
from moscot.problems._utils import handle_joint_attr


class TemporalNeuralProblem(  # type: ignore[misc]
    NeuralAnalysisMixin[Numeric_t, BirthDeathNeuralProblem],
    BirthDeathMixin,
    CompoundProblem[Numeric_t, BirthDeathNeuralProblem],
):
    """Class for analyzing time series single cell data with MongeVelor based on :cite:`eyring2022modeling`.

    The `TemporalNeuralProblem` allows to model and analyze time series single cell data by matching
    cells across time points with Neural Optimal Transport.
    This yields velocity vectors in the underlying space, allowing to study the dynamics of time-resolved
    single-cell data.

    Parameters
    ----------
    adata
        Annotated data object of :class:`anndata.AnnData`.
    """

    def prepare(
        self,
        time_key: str,
        joint_attr: Optional[Union[str, Mapping[str, Any]]] = None,
        policy: Literal["sequential", "tril", "triu", "explicit"] = "sequential",
        a: Optional[str] = None,
        b: Optional[str] = None,
        marginal_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> "TemporalNeuralProblem":
        r"""Prepare the :class:`moscot.problems.time.TemporalNeuralProblem`.

        Parameters
        ----------
        time_key
            Time point key in :attr:`anndata.AnnData.obs`.
        joint_attr
            - If `None`, PCA on :attr:`anndata.AnnData.X` is computed.
            - If `str`, it must refer to a key in :attr:`anndata.AnnData.obsm`.
            - If `dict`, the dictionary stores `attr` (attribute of :class:`anndata.AnnData`) and `key`
              (key of :class:`anndata.AnnData` ``['{attr}']``).
        policy
            Defines the rule according to which pairs of distributions are selected to compute the transport map
            between.
        a
            Specifies the left marginals. If

                - ``a`` is :class:`str` - the left marginals are taken from :attr:`anndata.AnnData.obs`,
                - if :meth:`~moscot.problems.base._birth_death.BirthDeathMixin.score_genes_for_marginals` was run and
                  if ``a`` is `None`, marginals are computed based on a birth-death process as suggested in
                  :cite:`schiebinger:19`,
                - if :meth:`~moscot.problems.base._birth_death.BirthDeathMixin.score_genes_for_marginals` was run and
                  if ``a`` is `None`, and additionally ``'scaling'`` is provided in ``marginal_kwargs``,
                  the marginals are computed as
                  :math:`\\exp(\frac{(\textit{proliferation} -
                  \textit{apoptosis}) \\cdot (t_2 - t_1)}{\textit{scaling}})`
                  rather than using a birth-death process,
                - otherwise or if ``a`` is :obj:`False`, uniform marginals are used.

        b
            Specifies the right marginals. If

                - ``b`` is :class:`str` - the left marginals are taken from :attr:`anndata.AnnData.obs`,
                - if :meth:`~moscot.problems.base._birth_death.BirthDeathMixin.score_genes_for_marginals` was run
                  uniform (mean of left marginals) right marginals are used,
                - otherwise or if ``b`` is :obj:`False`, uniform marginals are used.

        marginal_kwargs
            Keyword arguments for :meth:`~moscot.problems.BirthDeathProblem._estimate_marginals`. If ``'scaling'``
            is in ``marginal_kwargs``, the left marginals are computed as
            :math:`\\exp(\frac{(\textit{proliferation} - \textit{apoptosis}) \\cdot (t_2 - t_1)}{\textit{scaling}})`.
            Otherwise, the left marginals are computed using a birth-death process. The keyword arguments
            are either used for :func:`~moscot.problems.time._utils.beta`, i.e. one of:

                - beta_max: float
                - beta_min: float
                - beta_center: float
                - beta_width: float

            or for :func:`~moscot.problems.time._utils.delta`, i.e. one of:

                - delta_max: float
                - delta_min: float
                - delta_center: float
                - delta_width: float
        kwargs
            Keyword arguments, see notebooks TODO.

        Returns
        -------
        :class:`moscot.problems.time.TemporalNeuralProblem`.

        Notes
        -----
        If `a` and `b` are provided `marginal_kwargs` are ignored.

        Examples
        --------
        See :ref:`sphx_glr_auto_examples_problems_ex_different_policies.py` for an example how to
        use different policies. See :ref:`sphx_glr_auto_examples_problems_ex_passing_marginals.py`
        for an example how to pass marginals.
        """
        self.temporal_key = time_key
        xy, kwargs = handle_joint_attr(joint_attr, kwargs)

        marginal_kwargs = dict(marginal_kwargs)
        estimate_marginals = self.proliferation_key is not None or self.apoptosis_key is not None
        a = estimate_marginals if a is None else a
        b = estimate_marginals if b is None else b

        return super().prepare(  # type:ignore[return-value]
            key=time_key,
            xy=xy,
            policy=policy,
            a=a,
            b=b,
            marginal_kwargs=marginal_kwargs,
            **kwargs,
        )

    def solve(  # type:ignore[override]
        self,
        batch_size: int = 1024,
        tau_a: float = 1.0,
        tau_b: float = 1.0,
        epsilon: float = 0.1,
        seed: int = 0,
        pos_weights: bool = False,
        beta: float = 1.0,
        best_model_selection: bool = True,
        iterations: int = 25000,  # TODO(@MUCDK): rename to max_iterations
        inner_iters: int = 10,
        valid_freq: int = 50,
        log_freq: int = 5,
        patience: int = 100,
        patience_metric: Literal[
            "train_loss_f",
            "train_loss_g",
            "train_w_dist",
            "valid_loss_f",
            "valid_loss_g",
            "valid_w_dist",
        ] = "valid_w_dist",
        f: Union[Dict[str, Any], Any] = MappingProxyType({}), # TODO(ilan-gold): replace with corect type
        g: Union[Dict[str, Any], Any] = MappingProxyType({}),
        optimizer_f: Union[Dict[str, Any], Type[optax.GradientTransformation]] = MappingProxyType({}),
        optimizer_g: Union[Dict[str, Any], Type[optax.GradientTransformation]] = MappingProxyType({}),
        pretrain_iters: int = 0,
        pretrain_scale: float = 3.0,
        valid_sinkhorn_kwargs: Dict[str, Any] = MappingProxyType({}),
        compute_wasserstein_baseline: bool = True,
        train_size: float = 1.0,
        solver_name: Literal["LinearConditionalNeuralSolver"] = "LinearConditionalNeuralSolver",
        **kwargs: Any,
    ) -> "TemporalNeuralProblem":
        """Solve optimal transport problems defined in :class:`moscot.problems.time.TemporalNeuralProblem`.

        Parameters
        ----------
        batch_size
            Batch size.
        tau_a
            Unbalancedness parameter for left marginal between 0 and 1. `tau_a=1` means no unbalancedness
            in the source distribution. The limit of `tau_a` going to 0 ignores the left marginals.
            Unbalancedness is implemented as described in :cite:`eyring2022modeling`.
        tau_b
            Unbalancedness parameter for right marginal between 0 and 1. `tau_b=1` means no unbalancedness
            in the target distribution. The limit of `tau_b` going to 0 ignores the right marginals.
            Unbalancedness is implemented as described in :cite:`eyring2022modeling`.
        epsilon
            Entropic regularisation parameter in batch-wise inner loop. Only relevant if the problem is
            unbalanced.
        seed
            Seed for splitting the data.
        pos_weights
            If `True` enforces non-negativity of corresponding weights of ICNNs, else only penalizes negativity.
        beta
            If `pos_weights` is not `None`, this determines the multiplicative constant of L2-penalization of
            negative weights in ICNNs.
        best_model_selection
            TODO
        iterations
            Number of (outer) training steps (batches) of the training process.
        inner_iters
            Number of inner iterations for updating the convex conjugate.
        valid_freq
            Frequency at which the model is evaluated.
        log_freq
            Frequency at which training is logged.
        patience
            Number of iterations of no performance increase after which to apply early stopping.
        optimizer_f_kwargs
            Keyword arguments for the optimizer :class:`optax.adamw` for f.
        optimizer_g_kwargs
            Keyword arguments for the optimizer :class:`optax.adamw` for g.
        pretrain_iters
            Number of iterations (batches) for pretraining with the identity map.
        pretrain_scale
            Variance of Gaussian distribution used for pretraining.
        combiner_kwargs
            Keyword arguments for the combiner module in the PICNN (:cite:`bunne2022supervised`).
        valid_sinkhorn_kwargs
            Keyword arguments for computing the discrete sinkhorn divergence for assessing model training.
            By default, the same `tau_a`, `tau_b` and `epsilon` are taken as for the inner sampling loop.
        train_size
            Fraction of dataset to use as a training set. The remaining data is used for validation.
        compute_wasserstein_baseline
            Whether to compute the Sinkhorn divergence between the source and the target distribution as
            a baseline for the Wasserstein-2 distance computed with the neural solver.
        kwargs
            Keyword arguments.

        Warning
        -------
        If `compute_wasserstein_distance` is `True`, a discrete OT problem has to be solved on the validation
        dataset which scales linearly in the validation set size. If `train_size=1.0` the validation dataset size
        is the full dataset size, hence this is a source of prolonged run time or Out of Memory Error.
        """
        if solver_name not in self._valid_solver_names:
            raise ValueError(f"Solver name {solver_name} not in {self._valid_solver_names}.")
        return super().solve(
            batch_size=batch_size,
            tau_a=tau_a,
            tau_b=tau_b,
            epsilon=epsilon,
            seed=seed,
            pos_weights=pos_weights,
            beta=beta,
            best_model_selection=best_model_selection,
            iterations=iterations,
            inner_iters=inner_iters,
            valid_freq=valid_freq,
            log_freq=log_freq,
            patience=patience,
            f=f,
            g=g,
            optimizer_f=optimizer_f,
            optimizer_g=optimizer_g,
            pretrain_iters=pretrain_iters,
            pretrain_scale=pretrain_scale,
            valid_sinkhorn_kwargs=valid_sinkhorn_kwargs,
            compute_wasserstein_baseline=compute_wasserstein_baseline,
            train_size=train_size,
            solver_name="LinearConditionalNeuralSolver",
            **kwargs,
        )  # type:ignore[return-value]

    @property
    def _base_problem_type(self) -> Type[BirthDeathNeuralProblem]:
        return BirthDeathNeuralProblem

    @property
    def _valid_policies(self) -> Tuple[Policy_t, ...]:
        return _constants.SEQUENTIAL, _constants.TRIU, _constants.EXPLICIT  # type: ignore[return-value]

    @property
    def _valid_solver_names(self) -> Tuple[str, ...]:
        return ("LinearConditionalNeuralSolver",)
