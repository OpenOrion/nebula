from typing import Union
import jax_dataclasses as jdc

import numpy as np
import numpy.typing as npt
from nebula.topology.topology import Topology
from nebula.topology.shells import Shells
import jax.numpy as jnp


@jdc.pytree_dataclass
class Solids(Topology):
    shells: Shells
    index: jdc.Static[npt.NDArray[np.int32]]

    @staticmethod
    def empty():
        return Solids(shells=Shells.empty(), index=np.empty((0,), dtype=np.int32))

    @staticmethod
    def from_shells(shells: Shells):
        return Solids(shells=shells, index=np.full(len(shells.index), 0, dtype=np.int32))

    def add(self, solids: "Solids", reorder_index: bool = False):
        shells = self.shells.add(solids.shells)
        index = self.add_indices(solids.index, reorder_index)
        return Solids(shells, index)

    def __repr__(self) -> str:
        return f"Solids(count={self.count}, shells={self.shells})"

    def mask(self, mask: Union[jnp.ndarray, np.ndarray]):
        """Get solids from mask

        :param mask: Mask for solids
        :type mask: Union[jnp.ndarray, np.ndarray]
        :return: Masked solids
        :rtype: Shells
        """
        return Solids(
            self.shells.mask(mask),
            Topology.reorder_index(self.index[mask]),
        )
