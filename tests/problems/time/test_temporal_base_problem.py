from typing import Any, List, Mapping, Optional

import pytest

import numpy as np

from anndata import AnnData

from moscot.base.problems import BirthDeathProblem


# TODO(@MUCDK) put file in different folder according to moscot.problems structure
class TestBirthDeathProblem:
    @pytest.mark.fast
    def test_initialization_pipeline(self, adata_time_marginal_estimations: AnnData):
        t1, t2 = 0, 1
        adata_x = adata_time_marginal_estimations[adata_time_marginal_estimations.obs["time"] == t1]
        adata_y = adata_time_marginal_estimations[adata_time_marginal_estimations.obs["time"] == t2]

        prob = BirthDeathProblem(adata_x, adata_y, src_key=t1, tgt_key=t2)
        prob = prob.prepare(
            xy={},
            x={"attr": "X"},
            y={"attr": "X"},
            a=True,
            b=True,
            marginal_kwargs={
                "proliferation_key": "proliferation",
                "apoptosis_key": "apoptosis",
            },
        )

        assert prob.delta == (t2 - t1)
        assert isinstance(prob.a, np.ndarray)
        assert isinstance(prob.b, np.ndarray)

    # TODO(MUCDK): break this test
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "adata_obs_keys",
        [
            ["proliferation", "apoptosis"],
            ["error_proliferation", "error_apoptosis"],
            ["proliferation", None],
            [None, None],
        ],
    )
    @pytest.mark.parametrize("source", [True, False])
    def test_estimate_marginals_pipeline(
        self, adata_time_marginal_estimations: AnnData, adata_obs_keys: List[Optional[str]], source: bool
    ):
        proliferation_key, apoptosis_key = adata_obs_keys
        t1, t2 = 0, 1

        adata_x = adata_time_marginal_estimations[adata_time_marginal_estimations.obs["time"] == t1]
        adata_y = adata_time_marginal_estimations[adata_time_marginal_estimations.obs["time"] == t2]
        prob = BirthDeathProblem(adata_x, adata_y, src_key=t1, tgt_key=t2)
        adata = adata_x if source else adata_y

        if proliferation_key is not None and "error" in proliferation_key:
            with pytest.raises(KeyError, match=r"Unable to find proliferation"):
                _ = prob.estimate_marginals(
                    adata, source=source, proliferation_key=proliferation_key, apoptosis_key=apoptosis_key
                )
        elif proliferation_key is None and apoptosis_key is None:
            with pytest.raises(
                ValueError, match=r"At least one of `proliferation_key` or `apoptosis_key` must be specified"
            ):
                _ = prob.estimate_marginals(
                    adata, source=source, proliferation_key=proliferation_key, apoptosis_key=apoptosis_key
                )
        else:
            a_estimated = prob.estimate_marginals(
                adata, source=source, proliferation_key=proliferation_key, apoptosis_key=apoptosis_key
            )
            assert isinstance(a_estimated, np.ndarray)
            if not source:
                assert len(np.unique(a_estimated)) == 1

    @pytest.mark.fast
    def test_prior_growth_rates(self, adata_time_marginal_estimations: AnnData):
        t1, t2 = 0, 1
        adata_x = adata_time_marginal_estimations[adata_time_marginal_estimations.obs["time"] == t1]
        adata_y = adata_time_marginal_estimations[adata_time_marginal_estimations.obs["time"] == t2]
        prob = BirthDeathProblem(adata_x, adata_y, src_key=t1, tgt_key=t2)
        prob = prob.prepare(
            xy={},
            x={"attr": "X"},
            y={"attr": "X"},
            a=True,
            b=True,
            marginal_kwargs={
                "proliferation_key": "proliferation",
            },
        )
        assert prob.delta == (t2 - t1)

        gr = prob.prior_growth_rates
        assert isinstance(gr, np.ndarray)

    def test_posterior_growth_rates(self, adata_time_marginal_estimations: AnnData):
        t1, t2 = 0, 1
        adata_x = adata_time_marginal_estimations[adata_time_marginal_estimations.obs["time"] == t1]
        adata_y = adata_time_marginal_estimations[adata_time_marginal_estimations.obs["time"] == t2]
        prob = BirthDeathProblem(adata_x, adata_y, src_key=t1, tgt_key=t2)
        prob = prob.prepare(
            xy={},
            x={"attr": "X"},
            y={"attr": "X"},
            a=True,
            b=True,
            marginal_kwargs={"proliferation_key": "proliferation"},
        )
        prob = prob.solve(max_iterations=10, alpha=1.0)
        assert prob.delta == (t2 - t1)

        gr = prob.posterior_growth_rates
        assert isinstance(gr, np.ndarray)

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "marginal_kwargs", [{}, {"delta_width": 0.9}, {"delta_center": 0.9}, {"beta_width": 0.9}, {"beta_center": 0.9}]
    )
    def test_marginal_kwargs(self, adata_time_marginal_estimations: AnnData, marginal_kwargs: Mapping[str, Any]):
        t1, t2 = 0, 1
        adata_x = adata_time_marginal_estimations[adata_time_marginal_estimations.obs["time"] == t1]
        adata_y = adata_time_marginal_estimations[adata_time_marginal_estimations.obs["time"] == t2]

        prob = BirthDeathProblem(adata_x, adata_y, src_key=t1, tgt_key=t2)
        prob = prob.prepare(
            xy={},
            x={"attr": "X"},
            y={"attr": "X"},
            a=True,
            b=True,
            marginal_kwargs={
                "proliferation_key": "proliferation",
                "apoptosis_key": "apoptosis",
            },
        )

        gr1 = prob.prior_growth_rates
        prob = prob.prepare(
            xy={},
            x={"attr": "X"},
            y={"attr": "X"},
            a=True,
            b=True,
            marginal_kwargs={
                "proliferation_key": "proliferation",
                "apoptosis_key": "apoptosis",
                **marginal_kwargs,
            },
        )
        gr2 = prob.prior_growth_rates
        if len(marginal_kwargs) > 0:
            assert not np.allclose(gr1, gr2)
        else:
            assert np.allclose(gr1, gr2)
