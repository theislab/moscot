
from jax.config import config
config.update("jax_enable_x64", True)

from ott.geometry.geometry import Geometry
from ott.geometry.pointcloud import PointCloud
from jax import numpy as jnp
import seaborn as sns

from time import perf_counter
from moscot import FusedGW, Regularized, GW
import pickle
import os
import networkx as nx

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ot

import lineageot.simulation as sim
import lineageot.evaluation as sim_eval
import lineageot.inference as sim_inf

from typing import Optional, Sequence, Dict
from typing_extensions import Literal
from collections import namedtuple, defaultdict
from copy import deepcopy

from pathlib import Path

ROOT = Path(__file__).parent.resolve().parent.resolve()

DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

flow_types = ["bifurcation", "convergent", "partial_convergent", "mismatched_clusters"]
flow_types_str = ["bifurcation (B)", "convergent (C)", "partial \nconvergent (PC)", "mismatched \nclusters (MC)"]
palette = sns.color_palette("Set1")
data_path = DATA_DIR / 'simulations/'
figs_path = FIG_DIR / 'simulations/'

bnt = namedtuple("bnt", "tmat early_cost late_cost norm_diff converged time")
stn = namedtuple("sim",
                 "ancestor_info "
                 "rna_arrays "
                 "true_coupling "
                 "true_distances "
                 "barcode_arrays "
                 "ec_scale lc_scale "
                 "early_time_rna_cost "
                 "late_time_rna_cost "
                 "dimensions_to_plot")


def plot2D_samples_mat(xs, xt, G, thr=1e-8, alpha_scale=1, ax=None, **kwargs):
    """ !! Adapted from LineageOT !!
    Plot matrix M  in 2D with  lines using alpha values
    Plot lines between source and target 2D samples with a color
    proportional to the value of the matrix G between samples.
    Copied function from PythonOT and added alpha_scale parameter
    Parameters
    ----------
    xs : ndarray, shape (ns,2)
        Source samples positions
    b : ndarray, shape (nt,2)
        Target samples positions
    G : ndarray, shape (na,nb)
        OT matrix
    thr : float, optional
        threshold above which the line is drawn
    **kwargs : dict
        parameters given to the plot functions (default color is black if
        nothing given)
    """
    size = 4
    if ('color' not in kwargs) and ('c' not in kwargs):
        kwargs['color'] = 'k'
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(size, size))

    mx = G.max()
    for i in range(xs.shape[0]):
        for j in range(xt.shape[0]):
            if G[i, j] / mx > thr:
                ax.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]],
                        alpha=alpha_scale * G[i, j] / mx, zorder=0, **kwargs)


def create_geometry(cost_matrix: np.ndarray, scale='max') -> Geometry:
    cost_matrix = jnp.array(cost_matrix)
    if scale is None:
        pass
    elif scale == 'max':
        cost_matrix /= cost_matrix.max()
        assert cost_matrix.max() == 1.0
    elif scale == 'mean':
        cost_matrix /= np.mean(cost_matrix)
    elif scale == 'median':
        cost_matrix /= np.median(cost_matrix)
    else:
        raise NotImplementedError(scale)

    assert (cost_matrix >= 0).all()
    return Geometry(cost_matrix=cost_matrix)


def init_sim(flow_type: Literal['bifurcation', 'convergent', 'partial_convergent', 'mistmatched_clusters'],
             seed: int = 257,  **kwargs):
    """
    Create a simulated trajectory
    Parameters
    ----------
    flow_type
        the type of flow to simulate
    seed
        seed for reproducibility
    kwargs

    Returns
    -------
    stn
        simulation object
    """
    print(f'simulating `{flow_type}` flow with {seed} seed.')
    fpath = f"{flow_type}_{seed}_sim.pickle"

    start = perf_counter()
    np.random.seed(seed)
    if flow_type == 'bifurcation':
        timescale = 1
    else:
        timescale = 100

    x0_speed = 1 / timescale
    sim_params = sim.SimulationParameters(division_time_std=0.01 * timescale,
                                          flow_type=flow_type,
                                          x0_speed=x0_speed,
                                          mutation_rate=1 / timescale,
                                          mean_division_time=1.1 * timescale,
                                          timestep=0.001 * timescale,
                                          **kwargs)

    # These parameters can be adjusted freely.
    # As is, they replicate the plots in the paper for the fully convergent simulation.
    mean_x0_early = 2
    time_early = 7.4 * timescale  # Time when early cells are sampled
    time_late = time_early + 4 * timescale  # Time when late cells are sampled
    x0_initial = mean_x0_early - time_early * x0_speed
    initial_cell = sim.Cell(np.array([x0_initial, 0, 0]), np.zeros(sim_params.barcode_length))
    sample_times = {'early': time_early, 'late': time_late}

    # Choosing which of the three dimensions to show in later plots
    if flow_type == 'mismatched_clusters':
        dimensions_to_plot = [1, 2]
    else:
        dimensions_to_plot = [0, 1]

    ## Running the simulation
    sample = sim.sample_descendants(initial_cell.deepcopy(), time_late, sim_params)

    # Extracting trees and barcode matrices
    true_trees = {'late': sim_inf.list_tree_to_digraph(sample)}
    true_trees['late'].nodes['root']['cell'] = initial_cell
    true_trees['early'] = sim_inf.truncate_tree(true_trees['late'], sample_times['early'], sim_params)

    # Computing the ground-truth coupling
    true_coupling = sim_inf.get_true_coupling(true_trees['early'], true_trees['late'])

    data_arrays = {'late': sim_inf.extract_data_arrays(true_trees['late']),
                   'early': sim_inf.extract_data_arrays(true_trees['early'])}
    rna_arrays = {'late': data_arrays['late'][0]}
    barcode_arrays = {'early': data_arrays['early'][1], 'late': data_arrays['late'][1]}

    rna_arrays['early'] = sim_inf.extract_data_arrays(true_trees['early'])[0]
    num_cells = {'early': rna_arrays['early'].shape[0], 'late': rna_arrays['late'].shape[0]}

    print("Times:", sample_times)
    print("Number of cells:", num_cells)

    # Creating a copy of the true tree for use in LineageOT
    true_trees['late, annotated'] = copy.deepcopy(true_trees['late'])
    sim_inf.add_node_times_from_division_times(true_trees['late, annotated'])

    sim_inf.add_nodes_at_time(true_trees['late, annotated'], sample_times['early'])


    # Infer ancestor locations for the late cells based on the true lineage tree
    observed_nodes = [n for n in sim_inf.get_leaves(true_trees['late, annotated'], include_root=False)]
    sim_inf.add_conditional_means_and_variances(true_trees['late, annotated'], observed_nodes)

    ancestor_info = {'true tree': sim_inf.get_ancestor_data(true_trees['late, annotated'], sample_times['early'])}

    # True distances
    true_distances = {key: sim_inf.compute_tree_distances(true_trees[key]) for key in true_trees}

    rate_estimate = sim_inf.rate_estimator(barcode_arrays['late'], sample_times['late'])

    print("Fraction unmutated barcodes: ", {key: np.sum(barcode_arrays[key] == 0) / barcode_arrays[key].size
                                            for key in barcode_arrays})
    print("Rate estimate: ", rate_estimate)
    print("True rate: ", sim_params.mutation_rate / sim_params.barcode_length)
    print("Rate accuracy: ", rate_estimate * sim_params.barcode_length / sim_params.mutation_rate)

    # Compute Hamming distance matrices for neighbor joining

    early_time_rna_cost = ot.utils.dist(rna_arrays['early'],
                                        sim_inf.extract_ancestor_data_arrays(true_trees['late'], sample_times['early'],
                                                                             sim_params)[0])
    late_time_rna_cost = ot.utils.dist(rna_arrays['late'], rna_arrays['late'])



    indep = np.ones(true_coupling.shape) / true_coupling.size
    ind_ancestor_error = sim_inf.OT_cost(indep, early_time_rna_cost)
    ind_descendant_error = sim_inf.OT_cost(sim_eval.expand_coupling(indep,
                                                                    true_coupling,
                                                                    late_time_rna_cost),
                                           late_time_rna_cost)

    res = stn(ancestor_info, rna_arrays, true_coupling, true_distances,
              barcode_arrays, ind_ancestor_error, ind_descendant_error,
              early_time_rna_cost, late_time_rna_cost, dimensions_to_plot)

    with open(fpath, "wb") as fout:
        pickle.dump(tuple(res), fout)

    return res

