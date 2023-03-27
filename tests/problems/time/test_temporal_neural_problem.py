from moscot.datasets import simulate_data
from moscot.problems.time import TemporalNeuralProblem


class TestNeuralProblem:
    def test_pipeline():
        adata = simulate_data(key="time")
        tnp = TemporalNeuralProblem(adata)
        tnp = tnp.prepare("batch")
        tnp = tnp.solve(batch_size=8, iterations=10, valid_sinkhorn_kwargs={"tau_a": 1.0, "tau_b": 1.0})
