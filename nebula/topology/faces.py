from dataclasses import dataclass
from typing import Union
import jax_dataclasses as jdc
import numpy as np
import numpy.typing as npt
import jax.numpy as jnp
from nebula.prim.bspline_curves import BSplineCurves
from nebula.prim.bspline_surfaces import BSplineSurfaces
from nebula.prim.axes import Axes
from nebula.topology.topology import Topology
from nebula.topology.wires import WireLike, Wires


@jdc.pytree_dataclass
class Faces(WireLike):
    wires: Wires
    index: jdc.Static[npt.NDArray[np.int32]]

    @property
    def bspline_surfaces(self):
        return BSplineSurfaces(all_wires=self.wires)

    @property
    def edges(self):
        return self.wires.edges

    def add(self, faces: "Faces", reorder_index: bool = False):
        wires = self.wires.add(faces.wires)
        index = self.add_indices(faces.index, reorder_index)
        return Faces(wires, index)

    @staticmethod
    def empty():
        return Faces(wires=Wires.empty(), index=np.empty((0,), dtype=np.int32))

    @staticmethod
    def from_wires(wires: Wires, is_exterior: bool = False):
        """
        Translates a Wires into Faces. 

        :param wires: Wires to convert to faces
        :type wires: Wires
        :param is_exterior: transfers the index of the wires to the faces otherwise treat the wires all as one face
        :type is_exterior: bool, optional
        """
        if is_exterior:
            return Faces(wires=wires, index=wires.index)
        return Faces(wires=wires, index=np.full(len(wires.index), 0, dtype=np.int32))

    def translate(self, translation: jnp.ndarray):
        return Faces(wires=self.wires.translate(translation), index=self.index)

    def flip_winding(self):
        return Faces(wires=self.wires.flip_winding(), index=self.index[::-1])

    def project(self, new_axes: Axes):
        return Faces(wires=self.wires.project(new_axes), index=self.index)

    def __repr__(self) -> str:
        return f"Faces(count={self.count}, wires={self.wires})"

    def mask(self, mask: Union[jnp.ndarray, np.ndarray]):
        """Get wires from mask

        :param mask: Mask for faces
        :type mask: Union[jnp.ndarray, np.ndarray]
        :return: Masked faces
        :rtype: Faces
        """
        return Faces(
            self.wires.mask(mask),
            Topology.reorder_index(self.index[mask]),
        )

