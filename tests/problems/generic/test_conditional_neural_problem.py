import scanpy as sc

from moscot.datasets import simulate_data
from moscot.problems.generic import ConditionalNeuralProblem  # type: ignore[attr-defined]


class TestConditionalNeuralProblem:
    def test_pipeline(self):
        adata = simulate_data()
        sc.pp.pca(adata)
        cnp = ConditionalNeuralProblem(adata)
<<<<<<< HEAD
        cnp = cnp.prepare(key="batch", joint_attr="X_pca")
=======
        cnp = cnp.prepare(key="batch", joint_attr="X_pca", cond_dim=1)
>>>>>>> origin/conditional_not_precommit
        cnp = cnp.solve(batch_size=8, iterations=10)
