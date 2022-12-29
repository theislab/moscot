import scanpy as sc

from moscot.datasets import simulate_data
from moscot.problems.generic import ConditionalNeuralProblem  # type: ignore[attr-defined]


class TestConditionalNeuralProblem:
    def test_pipeline(self):
        adata = simulate_data()
        sc.pp.pca(adata)
        cnp = ConditionalNeuralProblem(adata)
        cnp = cnp.prepare(key="batch", joint_attr="X_pca", cond_dim=1)
        cnp = cnp.solve(batch_size=8, iterations=10)
