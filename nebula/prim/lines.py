import jax_dataclasses as jdc
import numpy as np
import jax.numpy as jnp
from typing import Optional, Union
from nebula.prim.axes import Axes
import numpy.typing as npt

@jdc.pytree_dataclass
class Lines:
    vertices: jnp.ndarray
    index: jdc.Static[npt.NDArray[np.int32]]

    def add(
        self,
        lines: "Lines",
    ):
        vertices = lines.vertices
        # Allocate new indices for the new edges ahead of the current indices
        new_line_indices = lines.index + len(self.vertices)

        vertices = jnp.concatenate([self.vertices, vertices], axis=0)
        index = np.concatenate([self.index, new_line_indices], axis=0)
        return Lines(vertices, index)

    @staticmethod
    def empty():
        return Lines(vertices=jnp.empty((0, 3)), index=np.empty((0, 2), dtype=np.int32))

    @staticmethod
    def from_segments(segments: jnp.ndarray):
        vertices = segments.reshape(-1, 3)
        index = np.arange(vertices.shape[0], dtype=np.int32).reshape(-1, 2)
        return Lines(vertices=vertices, index=index)

    def get_segments(self):
        return self.vertices[self.index]

    def clone(
        self,
        vertices: Optional[jnp.ndarray] = None,
        index: Optional[np.ndarray] = None,
    ):
        return Lines(
            vertices=vertices if vertices is not None else self.vertices,
            index=index if index is not None else self.index,
        )

    def evaluate_at(self, u: jnp.ndarray, index: jnp.ndarray):
        vertices = self.vertices[self.index[index]]
        return (1 - u) * vertices[0] + u * vertices[1]

    def project(self, curr_axes: Axes, new_axes: Axes):
        local_coords = curr_axes.to_local_coords(self.vertices)
        
        new_vertices = new_axes.to_world_coords(local_coords).reshape(-1, 3)

        max_index = np.max(self.index) if len(self.index) else 0
        index_increment = (np.arange(0, new_axes.count) * max_index).repeat(len(self.index))
        new_index = np.repeat(self.index[None, :], new_axes.count, axis=0).reshape(
            -1, 2
        ) + np.expand_dims(index_increment, axis=1)

        return Lines(vertices=new_vertices, index=new_index)


    def translate(self, translation: jnp.ndarray):
        if len(translation.shape) == 1:
            new_vertices = self.vertices + translation
        else:
            new_vertices = self.vertices.at[self.index].add(
                jnp.expand_dims(translation, axis=1)
            )
        
        return self.clone(vertices=new_vertices)
        

    def mask(self, mask: Union[jnp.ndarray, np.ndarray]):
        return Lines.from_segments(self.vertices[self.index[mask]])

    def flip_winding(self, mask: Optional[jnp.ndarray] = None):
        if mask is None:
            return Lines(vertices=self.vertices, index=np.flip(self.index, (0, 1)))
        new_index = self.index.copy()
        new_index[mask] = np.flip(new_index[mask], (0, 1))
        return Lines(vertices=self.vertices, index=new_index)

    def reorder(self, index: np.ndarray):
        new_index = self.index[index]
        return self.clone(index=new_index)

    def __repr__(self) -> str:
        return f"Lines(count={len(self.index)})"

    @property
    def count(self) -> int:
        return len(self.index)
