"""Strict peak-memory tests for :meth:`~moscot.base.output.BaseDiscreteSolverOutput.sparsify`.

Deselected by default (``addopts = -m 'not memory'``); run them with ``pytest -m memory``.
They are excluded from CI because peak RSS depends on the allocator and the runner, and because
they are deliberately large enough that the pre-fix implementation exhausts memory.

Each case runs in a fresh subprocess: peak RSS is a whole-process high-water mark, so it cannot be
measured reliably inside a shared (and possibly `xdist`-parallel) test session.
"""

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.memory

N = 2048
_CHILD = textwrap.dedent(
    """
    import resource, sys
    import numpy as np, jax.numpy as jnp
    from ott.geometry.pointcloud import PointCloud
    from ott.problems.linear.linear_problem import LinearProblem
    from ott.solvers.linear.sinkhorn import Sinkhorn
    from ott.solvers.linear.sinkhorn_lr import LRSinkhorn
    from moscot.backends.ott.output import OTTOutput

    n, batch_size, low_rank = int(sys.argv[1]), int(sys.argv[2]), bool(int(sys.argv[3]))
    rng = np.random.RandomState(0)
    x, y = jnp.asarray(rng.randn(n, 5)), jnp.asarray(rng.randn(n, 5))
    prob = LinearProblem(PointCloud(x, y, epsilon=0.1, scale_cost="mean"))
    out = OTTOutput(LRSinkhorn(rank=5)(prob) if low_rank else Sinkhorn(max_iterations=20)(prob))
    out.push(np.ones((n, 1)))  # warm up jit/solver buffers so they land in the baseline

    peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    before = peak()
    # `threshold` with a huge value keeps nothing, so only the transient blocks are measured
    out.sparsify(mode="threshold", value=1e9, batch_size=batch_size)
    print(peak() - before)
    """
)


def _peak_delta_bytes(batch_size: int, *, low_rank: bool = False, n: int = N) -> int:
    scale = 1 if sys.platform == "darwin" else 1024  # `ru_maxrss` is bytes on macOS, KiB on Linux
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _CHILD, str(n), str(batch_size), str(int(low_rank))],
        capture_output=True,
        text=True,
        check=True,
        env={"JAX_PLATFORMS": "cpu", "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PATH": "/usr/bin:/bin"},
    )
    return int(res.stdout.strip().splitlines()[-1]) * scale


@pytest.mark.parametrize("batch_size", [64, 1024])
def test_peak_memory_follows_batch_not_shape(batch_size: int) -> None:
    # a `[batch_size, n]` block of float32, with generous slack for the sparse result and allocator noise
    block = batch_size * N * 4
    assert _peak_delta_bytes(batch_size) < 32 * block + 64 * 2**20


def test_peak_memory_does_not_scale_with_batch_size() -> None:
    # the old implementation materialized `[n, m, batch_size]`, so this ratio was ~16x
    small, large = _peak_delta_bytes(64), _peak_delta_bytes(1024)
    assert large < max(2 * small, 64 * 2**20)


def test_low_rank_never_materializes_the_cost() -> None:
    # a low-rank output must never touch a geometry: `[n, n]` would already exceed this bound
    assert _peak_delta_bytes(1024, low_rank=True) < N * N * 4
