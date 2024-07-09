from typing import Any, Mapping

import pytest

import scanpy as sc

from moscot.utils.tagged_array import Tag, TaggedArray


class TestTaggedArray:
    @pytest.mark.parametrize(
        ("cost", "cost_kwargs"),
        [
            ("euclidean", {}),
            ("sq_euclidean", {}),
            ("cosine", {}),
            ("pnorm_p", {"p": 3}),
            ("sq_pnorm", {"p": 2}),
        ],
    )
    def test_from_adata_ott_cost_from_pointcloud(self, adata_time, cost: str, cost_kwargs: Mapping[str, Any]):
        tagged_array = TaggedArray.from_adata(
            adata_time, dist_key="time", attr="obsm", key="X_pca", tag=Tag.POINT_CLOUD, cost=cost, **cost_kwargs
        )
        assert isinstance(tagged_array, TaggedArray)
        assert tagged_array.tag == Tag.POINT_CLOUD

    def test_from_adata_geodesic_cost(self, adata_time):
        adata_time = adata_time[adata_time.obs["time"].isin((0, 1))]
        sc.pp.neighbors(adata_time, key_added="0_1")
        tagged_array = TaggedArray.from_adata(
            adata_time, dist_key=(0, 1), attr="obsp", key="connectivities", tag=Tag.GRAPH, cost="geodesic"
        )
        assert isinstance(tagged_array, TaggedArray)
        assert tagged_array.tag == Tag.GRAPH
