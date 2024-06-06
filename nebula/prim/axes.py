from dataclasses import field
from functools import cached_property
from typing import Literal
import jax_dataclasses as jdc
import jax
import jax.numpy as jnp


@jax.jit
def get_rotation_matrix(a: jnp.ndarray, b: jnp.ndarray):
    """Returns a rotation matrix that rotates vector a to vector b.

    :param a: vector a
    :type a: jnp.ndarray
    :param b: vector b
    :type b: jnp.ndarray
    :return: rotation matrix
    :rtype: jnp.ndarray
    """
    # Tolerance for floating point errors
    eps = 1.0e-10

    # Normalize the vectors
    a = a / jnp.linalg.norm(a)
    b = b / jnp.linalg.norm(b)

    # dimension of the space and identity
    I = jnp.identity(a.size)

    # Get the cross product of the two vectors
    v = jnp.cross(a, b)

    # Get the dot product of the two vectors
    c = jnp.dot(a, b=b)

    # the cross product matrix of a vector to rotate around
    K = jnp.outer(b, a) - jnp.outer(a, b)

    return jnp.where(
        # same direction
        jnp.abs(c - 1.0) < eps,
        I,
        jnp.where(
            # opposite direction
            jnp.abs(c + 1.0) < eps,
            -I,
            # Rodrigues' formula
            I + K + (K @ K) / (1 + c),
        ),
    )


AxesString = Literal["XY", "YX", "XZ", "ZX", "YZ", "ZY"]




@jdc.pytree_dataclass
class Axes:
    normals: jnp.ndarray
    origins: jnp.ndarray = field(default_factory=lambda: jnp.array([[0.0, 0.0, 0.0]]))

    @cached_property
    def local_rotation_matrix(self):
        return jax.vmap(get_rotation_matrix, (0, None))(self.normals, XY.normals[0])

    @cached_property
    def world_rotation_matrix(self):
        return jax.vmap(get_rotation_matrix, (None, 0))(XY.normals[0], self.normals)

    @property
    def count(self):
        return self.normals.shape[0]

    @property
    def local_origins(self):
        return self.to_local_coords(self.origins)

    @property
    def shape(self):
        return self.origins.shape

    def __add__(self, translation: jnp.ndarray):
        return Axes(origins=self.origins + translation, normals=self.normals)

    def to_local_coords(self, world_coords: jnp.ndarray):
        """Returns a rotation matrix that rotates the plane to the xy-plane.

        :param origin: origin of the plane
        :type origin: djnp.ndarray
        :param normal: normal of the plane
        :type normal: jnp.ndarray
        :param world_coords: world coordinates to rotate
        :type world_coords: jnp.ndarray
        :return: rotated coordinates
        :rtype: jnp.ndarray

        """
        # Translation vector
        translation_vector = jnp.expand_dims(XY.origins[0] - self.origins, axis=1)

        # Rotate the points
        return jax.jit(jax.vmap(jnp.matmul, (None, 0)))(
            world_coords + translation_vector, self.local_rotation_matrix
        )

    def to_world_coords(self, local_coords: jnp.ndarray):
        """Returns a rotation matrix that rotates the plane to the xy-plane.

        :param origin: origin of the plane
        :type origin: jnp.ndarray
        :param normal: normal of the plane
        :type normal: jnp.ndarray
        :param local_coords: local coordinates to rotate
        :type local_coords: jnp.ndarray
        :return: rotated coordinates
        :rtype: jnp.ndarray
        """
        if local_coords.shape[-1] == 2:
            local_coords = jnp.concatenate(
                [local_coords, jnp.zeros((local_coords.shape[0], 1))], axis=-1
            )

        # Translation vector
        translation_vector = jnp.expand_dims(self.origins - XY.origins[0], axis=1)

        # Rotate the points
        return (
            jax.jit(jax.vmap(jnp.matmul, (None, 0)))(
                local_coords, self.world_rotation_matrix
            )
            + translation_vector
        )

    def __repr__(self) -> str:
        return f"Axes(origin={self.origins}, normal={self.normals})"

    @staticmethod
    def from_str(str: AxesString):
        if str == "XY":
            return Axes(
                origins=jnp.array([[0.0, 0.0, 0.0]]), normals=jnp.array([[0.0, 0.0, 1.0]])
            )
        elif str == "YX":
            return Axes(
                origins=jnp.array([[0.0, 0.0, 0.0]]),
                normals=jnp.array([[0.0, 0.0, -1.0]]),
            )
        elif str == "XZ":
            return Axes(
                origins=jnp.array([[0.0, 0.0, 0.0]]), normals=jnp.array([[0.0, 1.0, 0.0]])
            )
        elif str == "ZX":
            return Axes(
                origins=jnp.array([[0.0, 0.0, 0.0]]),
                normals=jnp.array([[0.0, -1.0, 0.0]]),
            )
        elif str == "YZ":
            return Axes(
                origins=jnp.array([[0.0, 0.0, 0.0]]), normals=jnp.array([[1.0, 0.0, 0.0]])
            )
        elif str == "ZY":
            return Axes(
                origins=jnp.array([[0.0, 0.0, 0.0]]),
                normals=jnp.array([[-1.0, 0.0, 0.0]]),
            )


XY = Axes(
    origins=jnp.array([[0.0, 0.0, 0.0]]), normals=jnp.array(object=[[0.0, 0.0, 1.0]])
)
