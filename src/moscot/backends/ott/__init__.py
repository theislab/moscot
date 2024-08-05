from ott.geometry import costs

from moscot.backends.ott._utils import sinkhorn_divergence
from moscot.backends.ott.output import GraphOTTOutput, OTTNeuralOutput, OTTOutput
from moscot.backends.ott.solver import GENOTLinSolver, GWSolver, SinkhornSolver
from moscot.costs import register_cost

__all__ = ["OTTOutput", "GWSolver", "SinkhornSolver", "OTTNeuralOutput", "sinkhorn_divergence", "GENOTLinSolver"]


register_cost("euclidean", backend="ott")(costs.Euclidean)
register_cost("sq_euclidean", backend="ott")(costs.SqEuclidean)
register_cost("cosine", backend="ott")(costs.Cosine)
register_cost("pnorm_p", backend="ott")(costs.PNormP)
register_cost("sq_pnorm", backend="ott")(costs.SqPNorm)
