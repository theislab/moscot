from typing import List, Optional

from _utils import MockBaseSolverOutput
import pytest

import numpy as np

from anndata import AnnData

from moscot.problems.time._lineage import BirthDeathBaseProblem


class TestBirthDeathBaseProblem:
    def test_initialisation_pipeline(self, adata_time_marginal_estimations: AnnData):
        adata_x = adata_time_marginal_estimations[adata_time_marginal_estimations.obs["time"] == 0]
        adata_y = adata_time_marginal_estimations[adata_time_marginal_estimations.obs["time"] == 1]
        prob = BirthDeathBaseProblem(adata_x, adata_y, source=0, target=1)
        prob = prob.prepare(x={"attr": "X"}, y={"attr": "X"}, marginal_kwargs={"proliferation_key": "proliferation"})

        assert isinstance(prob._a, list)
        assert isinstance(prob._b, list)
        assert isinstance(prob.a, np.ndarray)
        assert isinstance(prob.b, np.ndarray)

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
        adata_x = adata_time_marginal_estimations[adata_time_marginal_estimations.obs["time"] == 0]
        adata_y = adata_time_marginal_estimations[adata_time_marginal_estimations.obs["time"] == 1]
        prob = BirthDeathBaseProblem(adata_x, adata_y, source=0, target=1)
        prob = prob.prepare(x={"attr": "X"}, y={"attr": "X"}, marginal_kwargs={"proliferation_key": "proliferation"})

        adata = adata_x if source else adata_y

        if proliferation_key is not None and "error" in proliferation_key:
            with np.testing.assert_raises(KeyError):
                prob._estimate_marginals(
                    adata, source=source, proliferation_key=proliferation_key, apoptosis_key=apoptosis_key
                )
        elif proliferation_key is None and apoptosis_key is None:
            with np.testing.assert_raises(ValueError):
                prob._estimate_marginals(
                    adata, source=source, proliferation_key=proliferation_key, apoptosis_key=apoptosis_key
                )
        else:
            a_estimated = prob._estimate_marginals(
                adata, source=source, proliferation_key=proliferation_key, apoptosis_key=apoptosis_key
            )
            assert isinstance(a_estimated, np.ndarray)
            if not source:
                assert len(np.unique(a_estimated)) == 1

    def test_add_marginals_pipeline(self, adata_time_marginal_estimations: AnnData):
        adata_x = adata_time_marginal_estimations[adata_time_marginal_estimations.obs["time"] == 0]
        adata_y = adata_time_marginal_estimations[adata_time_marginal_estimations.obs["time"] == 1]
        sol = MockBaseSolverOutput(len(adata_x), len(adata_y))
        prob = BaseProblem(adata_x, adata_y, source=0, target=1)
        prob = prob.prepare(x={"attr": "X"}, y={"attr": "X"}, marginal_kwargs={"proliferation_key": "proliferation"})

        assert isinstance(prob._a, list)
        assert isinstance(prob._b, list)
        assert isinstance(prob.a, np.ndarray)
        assert isinstance(prob.b, np.ndarray)
        assert len(prob._a) == 1
        assert len(prob._b) == 1

        prob._add_marginals(sol)

        assert isinstance(prob._a, list)
        assert isinstance(prob._b, list)
        assert isinstance(prob.a, np.ndarray)
        assert isinstance(prob.b, np.ndarray)
        assert len(prob._a) == 2
        assert len(prob._b) == 2

        assert len(np.unique(prob._b[-1])) == 1

    def test_growth_rates(self, adata_time_marginal_estimations: AnnData):
        adata_x = adata_time_marginal_estimations[adata_time_marginal_estimations.obs["time"] == 0]
        adata_y = adata_time_marginal_estimations[adata_time_marginal_estimations.obs["time"] == 1]
        prob = BirthDeathBaseProblem(adata_x, adata_y, source=0, target=1)
        prob = prob.prepare(x={"attr": "X"}, y={"attr": "X"}, marginal_kwargs={"proliferation_key": "proliferation"})

        gr = prob.growth_rates
        assert isinstance(gr, np.ndarray)
