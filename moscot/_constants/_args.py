from typing import List

pointcloud_attr = {
    "cost": "cost_fn",
    "power": "power",
    "batch_size": "_batch_size",
    "scale_cost": "_scale_cost",
}

geometry_attr = {"epsilon": "_epsilon_init", "scale_cost": "_scale_cost"}

linear_solver_attr = {
    "lse_mode": "lse_mode",
    "threshold": "threshold",
    "norm_error": "norm_error",
    "inner_iterations": "inner_iterations",
    "min_iterations": "min_iterations",
    "max_iterations": "max_iterations",
    "initializer": "initializer",
    "initializer_kwargs": "kwargs_init",
    "jit": "jit",
    "rank": "rank",
    "gamma": "gamma",
    "gamma_rescale": "gamma_rescale",
}

lin_prob_attr = {"epsilon": "_epsilon_init", "scale_cost": "_scale_cost"}

quad_solver_attr = {
    "epsilon": "epsilon",
    "rank": "rank",
    "lse_mode": "lse_mode",
    "threshold": "threshold",
    "norm_error": "norm_error",
    "inner_iterations": "inner_iterations",
    "min_iterations": "min_iterations",
    "max_iterations": "max_iterations",
    "initializer": "quad_initializer",
    "initializer_kwargs": "kwargs_init",
    "jit": "jit",
    "warm_start": "_warm_start",
    "quad_initializer": "quad_initializer",
}
quad_prob_attr = {
    "tau_a": "_tau_a",
    "tau_b": "_tau_b",
    "gw_unbalanced_correction": "gw_unbalanced_correction",
    "ranks": "ranks",
    "tolerances": "tolerances",
}


linear_init_kwargs_list: List[str] = list(linear_solver_attr.keys())
linear_prepare_kwargs_list: List[str] = list(set(lin_prob_attr.keys()).union(set(pointcloud_attr.keys())))
linear_solve_kwargs_list: List[str] = []

quad_init_kwargs_list: List[str] = list(quad_solver_attr.keys())
quad_prepare_kwargs_list: List[str] = list(set(lin_prob_attr.keys()).union(set(pointcloud_attr.keys())))
quad_solve_kwargs_list: List[str] = []

quad_f_init_kwargs_list: List[str] = ["alpha"] + quad_init_kwargs_list
quad_f_prepare_kwargs_list: List[str] = quad_prepare_kwargs_list
quad_f_solve_kwargs_list: List[str] = []
