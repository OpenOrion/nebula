import jax
import jax.numpy as jnp

from nebula.helpers.types import CoordLike
from nebula.prim.axes import Axes


@jax.jit
def is_inside(polygon_segment: jnp.ndarray, point: jnp.ndarray):
    x1, y1, x2, y2 = polygon_segment[:, :2].flatten()
    xp, yp = point[:2]

    return ((yp < y1) != (yp < y2)) & (xp < (x1 + ((yp - y1) / (y2 - y1)) * (x2 - x1)))


class WireHelper:
    @staticmethod
    def contains_vertex(
        polygon_segments: jnp.ndarray,
        vertex: jnp.ndarray,
    ):
        """
        Checks if polygon contains vertex.

        :param polygon_segments: polygon segments (num_segments, 2, 3)
        :type polygon_segments: jnp.ndarray
        :param vertex: vertex to check (3,)
        :type vertex: jnp.ndarray
        :return: True if polygon contains vertex
        :rtype: bool

        """
        ray_intersection = jax.vmap(is_inside, in_axes=(0, None))(
            polygon_segments, vertex
        ).astype(jnp.int32)

        # if odd number of intersections, then segment is inside polygon
        total_intersections = jnp.sum(ray_intersection)
        return jnp.mod(total_intersections, 2) == 1

    @staticmethod
    def contains_segment(
        polygon_segments: jnp.ndarray,
        segment: jnp.ndarray,
    ):
        """
        Checks if segment is inside polygon.

        :param polygon_segments: polygon segments (num_segments, 2, 3)
        :type polygon_segments: jnp.ndarray
        :param segment: segment to check (2, 3)
        :type segment: jnp.ndarray
        :return: True if segment is inside polygon
        :rtype: bool

        """
        mean_vertex = jnp.mean(segment, axis=0)
        return WireHelper.contains_vertex(polygon_segments, mean_vertex)

    @staticmethod
    def fully_contains_segment(
        polygon_segments: jnp.ndarray,
        segment: jnp.ndarray,
    ):
        """
        Checks if segment is inside polygon.

        :param polygon_segments: polygon segments (num_segments, 2, 3)
        :type polygon_segments: jnp.ndarray
        :param segment: segment to check (2, 3)
        :type segment: jnp.ndarray
        :return: True if segment is inside polygon
        :rtype: bool

        """
        return jax.vmap(WireHelper.contains_vertex, in_axes=(None, 0))(
            polygon_segments, segment
        ).all()

    @staticmethod
    def to_3d_vertices(vertices: CoordLike):
        vertices = jnp.asarray(vertices)
        if vertices.shape[-1] == 2:
            return jnp.concatenate([vertices, jnp.zeros((len(vertices), 1))], axis=-1)
        return vertices

    @staticmethod
    @jax.jit
    def is_clockwise(first_segment: jnp.ndarray, last_segment: jnp.ndarray):
        """
        Finds if the segments are in clockwise order. IMPORTANT: This assumes that segments are on the XY plane
        
        """
        normal = WireHelper.get_normal(first_segment, last_segment)
        return normal.sum() < 0

    @staticmethod
    def get_normal(first_segment: jnp.ndarray, last_segment: jnp.ndarray):
        start_vecs = first_segment[1] - first_segment[0]
        end_vecs = last_segment[1] - last_segment[0]
        normals = jnp.cross(end_vecs, start_vecs)
        return normals / jnp.linalg.norm(normals, axis=-1, keepdims=True)


