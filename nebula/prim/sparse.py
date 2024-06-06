from typing import Optional, Protocol, Union
import jax_dataclasses as jdc
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt


class SparseIndexable(Protocol):
    index: jdc.Static[npt.NDArray[np.int32]]

    def mask(self, mask: Union[jnp.ndarray, np.ndarray]) -> "SparseIndexable": ...

    @property
    def last_index(self):
        return np.max(self.index).astype(np.int32) if len(self.index) else 0

    @property
    def count(self):
        return SparseIndexable.get_count(self.index)

    def add_indices(self, indices: np.ndarray, reorder_index: bool = False):
        if reorder_index:
            indices = SparseIndexable.reorder_index(indices)

        new_indices = indices + self.count
        return np.concatenate([self.index, new_indices])

    @staticmethod
    def get_count(index: np.ndarray):
        return np.max(index).astype(np.int32) + 1 if len(index) else 0

    @staticmethod
    def reorder_index(index: np.ndarray):
        return np.unique(index, return_inverse=True)[1]

    @staticmethod
    def expanded_index(index: np.ndarray, repeats: int):
        count = SparseIndexable.get_count(index)
        index_increment = (np.arange(0, repeats) * count).repeat(len(index))
        return np.repeat(index[None, :], repeats, axis=0).flatten() + index_increment


@jdc.pytree_dataclass
class SparseArray(SparseIndexable):
    val: jnp.ndarray
    index: jdc.Static[npt.NDArray[np.int32]]

    @staticmethod
    def empty(shape=(0,)):
        return SparseArray(jnp.empty(shape), np.empty((0,), dtype=np.int32))

    @staticmethod
    def from_array(array: jnp.ndarray):
        return SparseArray(array, np.full(len(array), 0, dtype=np.int32))

    def add(self, item: "SparseArray"):
        val = jnp.concatenate([self.val, item.val])
        index = self.add_indices(item.index)
        return SparseArray(val, index)

    def flip_winding(self, mask: Optional[jnp.ndarray] = None):
        if mask is None:
            return SparseArray(self.val[::-1], self.index[::-1])

        new_index = self.index.copy()
        new_index[mask] = np.flip(new_index[mask], axis=0)

        return SparseArray(self.val, new_index)

    def mask(self, mask: Union[jnp.ndarray, np.ndarray]) -> "SparseArray":
        return SparseArray(
            val=self.val[mask],
            index=SparseIndexable.reorder_index(self.index[mask]),
        )

    # TODO: make this faster
    def reorder(self, index: np.ndarray):
        """
        Reorder the array according to the given index.

        :param index: The new order of the array.
        :type index: np.ndarray
        :return: The reordered array.
        :rtype: SparseArray
        """

        new_val = jnp.empty((0, *self.val.shape[1:]))
        new_index = np.empty((0,), dtype=np.int32)
        for i in index:
            index_mask = self.index == i
            new_val = jnp.concatenate([new_val, self.val[index_mask]])
            new_index = np.concatenate([new_index, self.index[index_mask]])

        return SparseArray(new_val, new_index)

    def __repr__(self) -> str:
        return f"SparseArray(count={self.count})"
