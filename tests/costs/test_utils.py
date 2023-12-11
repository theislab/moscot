import pytest

from moscot.costs._utils import (
    _get_available_backends_n_costs,
    get_available_costs,
    get_cost,
)


class TestCostUtils:
    ALL_BACKENDS_N_COSTS = {
        "moscot": ("barcode_distance", "leaf_distance"),
        "ott": (
            "euclidean",
            "sq_euclidean",
            "cosine",
            "pnorm_p",
            "sq_pnorm",
            "elastic_l1",
            "elastic_l2",
            "elastic_stvs",
            "elastic_sqk_overlap",
        ),
    }

    @staticmethod
    def test_get_available_backends_n_costs():
        assert dict(_get_available_backends_n_costs()) == {
            k: list(v) for k, v in _get_available_backends_n_costs().items()
        }

    @staticmethod
    def test_get_available_costs():
        assert get_available_costs() == TestCostUtils.ALL_BACKENDS_N_COSTS
        assert get_available_costs("moscot") == {"moscot": (TestCostUtils.ALL_BACKENDS_N_COSTS["moscot"])}
        assert get_available_costs("ott") == {"ott": TestCostUtils.ALL_BACKENDS_N_COSTS["ott"]}
        with pytest.raises(KeyError):
            get_available_costs("foo")

    @staticmethod
    def test_get_cost_fails():
        invalid_cost = "foo"
        invalid_backend = "bar"
        with pytest.raises(ValueError):
            get_cost(invalid_cost, backend=invalid_backend)
        for backend in TestCostUtils.ALL_BACKENDS_N_COSTS.keys():
            with pytest.raises(ValueError):
                get_cost(invalid_cost, backend=backend)
