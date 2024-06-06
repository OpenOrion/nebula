from typing import Union
import jax_dataclasses as jdc
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from nebula.topology.topology import Topology
from nebula.topology.faces import Faces


@jdc.pytree_dataclass
class Shells(Topology):
    faces: Faces
    index: jdc.Static[npt.NDArray[np.int32]]

    def __repr__(self) -> str:
        return f"Shells(count={self.count}, faces={self.faces})"

    def add(self, shells: "Shells", reorder_index: bool = False):
        faces = self.faces.add(shells.faces)
        index = self.add_indices(shells.index, reorder_index)
        return Shells(faces, index)

    @staticmethod
    def empty():
        return Shells(faces=Faces.empty(), index=np.empty((0,), dtype=np.int32))

    @staticmethod
    def from_faces(faces: Faces):
        return Shells(faces=faces, index=np.full(len(faces.index), 0, dtype=np.int32))

    def mask(self, mask: Union[jnp.ndarray, np.ndarray]):
        """Get shells from mask

        :param mask: Mask for shells
        :type mask: Union[jnp.ndarray, np.ndarray]
        :return: Shells
        :rtype: Shells
        """
        return Shells(
            self.faces.mask(mask),
            Topology.reorder_index(self.index[mask]),
        )
