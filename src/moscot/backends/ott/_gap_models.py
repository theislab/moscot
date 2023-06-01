from collections import defaultdict
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union

import optax
from flax.core import freeze
from flax.core.scope import FrozenVariableDict
from flax.training.train_state import TrainState
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
import numpy as np
from ott.geometry import costs
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.potentials import DualPotentials
from ott.tools.sinkhorn_divergence import sinkhorn_divergence

from moscot.backends.ott._icnn import ICNN
from moscot.backends.ott._jax_data import JaxSampler
from moscot.backends.ott._utils import (
    RunningAverageMeter,
    _compute_sinkhorn_divergence,
)

Train_t = Dict[str, Dict[str, Union[float, List[float]]]]


class MongeGapSolver:
    pass