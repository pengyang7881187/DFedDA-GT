from jax import Array
from numpy import ndarray
from jax.typing import ArrayLike
from jax._src.prng import PRNGKeyArray

from typing import Callable, Tuple, List, Union, Dict
KeyArray = Union[Array, PRNGKeyArray]
