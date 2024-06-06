from typing import Union
import jax.numpy as jnp

Number = Union[float, jnp.ndarray]
CoordLike = Union[
    list[tuple[Number, Number, Number]], list[tuple[Number, Number]], jnp.ndarray
]
ArrayLike = Union[list[float], jnp.ndarray]
