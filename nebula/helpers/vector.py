import jax.numpy as jnp

class VectorHelper:
    @staticmethod
    def normalize(arr: jnp.ndarray) -> jnp.ndarray:
        arr_min = jnp.min(arr)
        arr_max = jnp.max(arr)
        normalized = (arr - arr_min) / (arr_max - arr_min)
        return normalized
