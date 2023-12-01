import numpy as np
import jax.numpy as jnp

from MyTyping import *


def full_connected_gossip(M: int) -> Array:
    return jnp.ones((M, M), dtype=float) / M


def chain_gossip(M: int) -> Array:
    assert M >= 3
    adjacency_matrix = np.eye(M) * (1. / 3.)
    adjacency_matrix[0, 0] = (2. / 3.)
    adjacency_matrix[-1, -1] = (2. / 3.)
    idx_0_Mm2 = np.arange(M-1)
    idx_1_Mm1 = idx_0_Mm2 + 1
    adjacency_matrix[idx_0_Mm2, idx_1_Mm1] = (1. / 3.)
    adjacency_matrix[idx_1_Mm1, idx_0_Mm2] = (1. / 3.)
    return jnp.array(adjacency_matrix)


def circle_gossip(M: int) -> Array:
    assert M >= 3
    adjacency_matrix = np.eye(M) * (1. / 3.)
    adjacency_matrix[0, -1] = (1. / 3.)
    adjacency_matrix[-1, 0] = (1. / 3.)
    idx_0_Mm2 = np.arange(M - 1)
    idx_1_Mm1 = idx_0_Mm2 + 1
    adjacency_matrix[idx_0_Mm2, idx_1_Mm1] = (1. / 3.)
    adjacency_matrix[idx_1_Mm1, idx_0_Mm2] = (1. / 3.)
    return jnp.array(adjacency_matrix)


gossip_register = {
    'full_connected': full_connected_gossip,
    'chain': chain_gossip,
    'circle': circle_gossip,
}
