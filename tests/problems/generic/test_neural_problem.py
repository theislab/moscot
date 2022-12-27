from typing import Any, Tuple, Literal, Mapping

import pandas as pd
import pytest
from moscot.datasets import simulate_data
from moscot.problems.generic import NeuralProblem, ConditionalNeuralProblem

class TestNeuralProblem:
     def test_pipeline():
        adata = simulate_data()
        np = NeuralProblem(adata)
        np = np.prepare("batch")
        np = np.solve(batch_size=8, iterations=10, valid_sinkhorn_kwargs={"tau_a": 1.0, "tau_b": 1.0})

        