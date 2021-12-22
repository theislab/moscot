import networkx as nx
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Union, Literal, Optional
from numbers import Number
import jax.numpy as jnp


def lca_cost(trees: Dict[int, Any]) -> List[jnp.ndarray]: #TODO: specify Any, i.e. which data type trees have
    """
    uses nx.algorithms.lowest_common_ancestors
    """
    for key, tree in trees.items():
        n_nodes = len(tree.nodes)
        cost = jnp.zeros((n_nodes, n_nodes))

