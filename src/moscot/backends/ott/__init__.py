# Compatibility shim for ott-jax<=0.6.0 with jax>=0.8.
# Released ott-jax (<=0.6.0) does `from jax.interpreters import batching` and then
# calls `batching.is_vmappable`, but jax removed that name from the public
# `jax.interpreters.batching` namespace after 0.7.2. The symbol still lives in the
# private module, which is exactly what ott switched to upstream (ott-jax#701).
# Re-expose it here so ott's `batched_vmap` works until that fix is released.
# TODO(moscot): drop this once a released ott-jax imports batching from jax._src.
import jax.interpreters.batching as _public_batching

if not hasattr(_public_batching, "is_vmappable"):
    import jax._src.interpreters.batching as _src_batching

    _public_batching.is_vmappable = _src_batching.is_vmappable

from ott.geometry import costs

from moscot.backends.ott._utils import sinkhorn_divergence
from moscot.backends.ott.output import GraphOTTOutput, OTTOutput
from moscot.backends.ott.solver import GWSolver, SinkhornSolver
from moscot.costs import register_cost

__all__ = [
    "OTTOutput",
    "GWSolver",
    "SinkhornSolver",
    "sinkhorn_divergence",
    "GENOTLinSolver",
    "GraphOTTOutput",
]


register_cost("euclidean", backend="ott")(costs.Euclidean)
register_cost("sq_euclidean", backend="ott")(costs.SqEuclidean)
register_cost("cosine", backend="ott")(costs.Cosine)
register_cost("pnorm_p", backend="ott")(costs.PNormP)
register_cost("sq_pnorm", backend="ott")(costs.SqPNorm)
