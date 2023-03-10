from ott.geometry import costs

from moscot.costs import register_cost

register_cost("euclidean", backend="ott")(costs.Euclidean)
register_cost("sq_euclidean", backend="ott")(costs.SqEuclidean)
register_cost("cosine", backend="ott")(costs.Cosine)
register_cost("pnorm_p", backend="ott")(costs.PNormP)
register_cost("sq_pnorm", backend="ott")(costs.SqPNorm)
register_cost("elastic_l1", backend="ott")(costs.ElasticL1)
register_cost("elastic_stvs", backend="ott")(costs.ElasticSTVS)
