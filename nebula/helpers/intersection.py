import jax_dataclasses as jdc
import jax
import jax.numpy as jnp


@jdc.pytree_dataclass
class IntersectionResult:
    intersected_vertices: jnp.ndarray
    is_intersection: jnp.ndarray
    num_intersections: jnp.ndarray


class Intersection:
    @staticmethod
    @jax.jit
    def on_line(point: jnp.ndarray, line: jnp.ndarray):
        """
        Determines if a point is on a line.

        :param point: point to check (3,)
        :type point: jnp.ndarray
        :param line: start and endpoint of line (2, 3)
        :type line: jnp.ndarray
        :return: True if point is on line, False otherwise
        :rtype: bool
        """
        line_x1, line_y1, line_x2, line_y2 = line[:, :2].flatten()

        return jnp.all(
            jnp.array(
                [
                    jnp.minimum(line_x1, line_x2) <= point[0],
                    jnp.minimum(line_y1, line_y2) <= point[1],
                    jnp.maximum(line_x1, line_x2) >= point[0],
                    jnp.maximum(line_y1, line_y2) >= point[1],
                ]
            )
        )

    @staticmethod
    @jax.jit
    def line_segment_intersection(
        line1: jnp.ndarray, line2: jnp.ndarray, decimal_accuracy=5
    ):
        """
        Finds the intersection of two 3D line segment given endpoints of each line.
        https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection

        :param line1: start and endpoint of line 1 (2, 3)
        :type line1: jnp.ndarray
        :param line2: start and endpoint of line 2 (2, 3)
        :type line2: jnp.ndarray
        :return: intersection point, nan if lines are parallel
        :rtype: jnp.ndarray
        """
        x1, y1, x2, y2 = line1[:, :2].flatten()
        x3, y3, x4, y4 = line2[:, :2].flatten()

        nx = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        ny = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        unconstrained_intersection_vertices = jnp.array([nx / denom, ny / denom, 0])
        rounded_unconstrained_intersection_vertices = jnp.round(
            unconstrained_intersection_vertices, decimal_accuracy
        )
        
        intersection_on_line1 = Intersection.on_line(
            rounded_unconstrained_intersection_vertices,
            jnp.round(line1, decimal_accuracy),
        )
        intersection_on_line2 = Intersection.on_line(
            rounded_unconstrained_intersection_vertices,
            jnp.round(line2, decimal_accuracy),
        )
        is_intersection = (denom != 0) & intersection_on_line1 & intersection_on_line2

        nan_array = jnp.full((3,), jnp.nan)

        intersection_vertices = jnp.where(
            is_intersection, unconstrained_intersection_vertices, nan_array
        )
        return IntersectionResult(
            intersected_vertices=intersection_vertices,
            is_intersection=is_intersection,
            num_intersections=jnp.where(is_intersection, 1, 0),
        )
