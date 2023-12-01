import math
import random
import numpy as np
import jax.numpy as jnp

from datetime import datetime
from numpy.random import default_rng
from torch.utils.tensorboard import SummaryWriter

from MyTyping import *

_FINFO = np.finfo(float)
# Numerical error.
_EPS = _FINFO.eps

non_zero_eps = 1e-9


# Set seed and return a random generator of numpy.
def set_seed_and_get_rng(seed: int) -> np.random._generator.Generator:
    np.random.seed(seed)
    random.seed(seed)
    return default_rng(seed)


def get_format_time():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def check_gossip(U: Array, M: int):
    assert U.shape == (M, M)
    assert jnp.allclose(U, U.T)
    assert jnp.allclose(jnp.sum(U, axis=0), 1.)
    assert jnp.allclose(jnp.sum(U, axis=1), 1.)
    return


def count_float_nonzero(arr: Array) -> int:
    return int(jnp.count_nonzero(jnp.abs(arr) > non_zero_eps))


def count_overlap_float_nonzero(arr1: Array, arr2: Array) -> int:
    return int(jnp.count_nonzero((jnp.abs(arr1) > non_zero_eps) & (jnp.abs(arr2) > non_zero_eps)))


# Non-batch Lp norm for jax.
def Lp_norm(w: Array, p: float) -> float:
    w_abs = jnp.abs(w)
    return (jnp.sum(w_abs ** p) ** (1. / p))


def replace_nan_with_zero(arr: Array) -> Array:
    return jnp.nan_to_num(arr, nan=0.0)


def add_scalar_dict(input_dict: Dict, writer: SummaryWriter, global_step: int):
    for key, val in input_dict.items():
        if isinstance(val, float) or isinstance(val, int):
            writer.add_scalar(tag=f'{key}', scalar_value=val, global_step=global_step)
    return


def add_suffix_to_keys(dictionary: dict, suffix: str):
    new_dict = {}
    for key, value in dictionary.items():
        new_key = key + f'_{suffix}'
        new_dict[new_key] = value
    return new_dict


def get_value_by_key_containing_substr(dictionary: dict, substr: str):
    for key, value in dictionary.items():
        if substr in key:
            return value


def dict_add(dict_1: dict, dict_2: dict):
    if dict_2 is None:
        return dict_1
    sum_dict = {}
    for key in dict_1.keys():
        sum_dict[key] = dict_1[key] + dict_2[key]
    return sum_dict


def dict_max(dict_1: dict, dict_2: dict):
    if dict_2 is None:
        return dict_1
    max_dict = {}
    for key in dict_1.keys():
        max_dict[key] = max(dict_1[key], dict_2[key])
    return max_dict


def dict_min(dict_1: dict, dict_2: dict):
    if dict_2 is None:
        return dict_1
    min_dict = {}
    for key in dict_1.keys():
        min_dict[key] = min(dict_1[key], dict_2[key])
    return min_dict


def dict_divide(dict_1: dict, divide: float):
    result_dict = {}
    for key in dict_1.keys():
        result_dict[key] = dict_1[key] / divide
    return result_dict


def dict_log(dict_1: dict):
    tgt_keys_lst = ('obj_gap', 'grad_norm', 'norm_gap')
    dict_1_copy = dict_1.copy()
    for key in dict_1_copy.keys():
        for tgt_key in tgt_keys_lst:
            if tgt_key in key:
                dict_1[f'log_{key}'] = max(math.log(1. + dict_1[key]), 0.)
                break
    return dict_1


def sparsify(w: Array, sparsity: int) -> Array:
    top_s_indices = jnp.argsort(jnp.abs(w))[-sparsity:]
    return jnp.zeros_like(w).at[top_s_indices].set(w[top_s_indices])


def vec_sparsify(batch_w: Array, sparsity: int) -> Array:
    top_s_indices = jnp.argsort(jnp.abs(batch_w), axis=1)[:, -sparsity:]
    idx_array = jnp.repeat(jnp.arange(batch_w.shape[0])[:, None], repeats=sparsity, axis=1)
    return jnp.zeros_like(batch_w).at[idx_array, top_s_indices].set(batch_w[idx_array, top_s_indices])


def project_onto_l1_ball(batch_x: Array, eps: float) -> Array:
    """
    Compute Euclidean projection onto the L1 ball for a batch.

      min_u ||x - u||_2 s.t. ||u||_1 <= eps

    Inspired by the corresponding numpy version by Adrien Gaidon.

    Parameters
    ----------
    x: (batch_size, dim_x) JAX array
      batch of arbitrary-size tensors to project, possibly on GPU

    eps: float
      radius of l-1 ball to project onto

    Returns
    -------
    u: (batch_size, dim_x) JAX array
      batch of projected tensors, reshaped to match the original
    """
    x = batch_x
    mask = (jnp.linalg.norm(x, ord=1, axis=1) < eps).astype(jnp.float32).reshape(-1, 1)
    mu = jnp.abs(x)
    mu_sorted = jnp.sort(mu, axis=1)[:, ::-1]
    cumsum = jnp.cumsum(mu_sorted, axis=1)
    arange = jnp.arange(1, x.shape[1] + 1)
    rho = jnp.max((mu_sorted * arange > (cumsum - eps)) * arange, axis=1)
    theta = (cumsum[jnp.arange(x.shape[0]), rho - 1] - eps) / rho
    proj = jnp.clip(jnp.abs(x) - theta.reshape(-1, 1), a_min=0)
    return mask * x + (1 - mask) * proj * jnp.sign(x)


# min_u ||x - u||_2 s.t. ||u - w||_1 <= eps
# batch_x.shape = batch_w.shape = (batch_size, dim_x)
def project_onto_l1_ball_center_w(batch_x: Array, batch_w: Array, eps: float) -> Array:
    return project_onto_l1_ball(batch_x-batch_w, eps)

