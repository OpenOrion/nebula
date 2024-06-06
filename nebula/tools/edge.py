import jax
from typing import Optional
import jax.numpy as jnp
from nebula.evaluators.bspline import BSplineEvaluator
from nebula.prim.bspline_curves import BSplineCurves
from nebula.helpers.clipper import Clipper
from nebula.helpers.types import ArrayLike, Number
from nebula.helpers.wire import WireHelper
from nebula.prim.sparse import SparseArray
from nebula.topology.edges import Edges
from nebula.topology.wires import Wires


class EdgeTool:
    @staticmethod
    @jax.jit
    def make_rect(
        x_len: Number, y_len: Number, origins: Optional[jnp.ndarray] = None, centered: bool = True
    ):
        """Creates a rectangle.

        :param x_len: length of the rectangle along x-axis
        :type x_len: float
        :param y_len: length of the rectangle along y-axis
        :type y_len: float
        :param axes: axes to create the rectangle on, defaults to XY
        :type axes: Axes, optional
        :param center: flag to create the rectangle at the center
        :type center: bool
        :return: rectangle array of size (num_axis, 4, 2, 3)
        :rtype: jnp.ndarray
        """
        if origins is None:
            origins = jnp.zeros((1,3))


        vertices = jnp.array(
            [
                (0.0, 0.0, 0.0),
                (x_len, 0.0, 0.0),
                (x_len, y_len, 0.0),
                (0.0, y_len, 0.0),
                (0.0, 0.0, 0.0),
            ]
        )

        # center the rectangle if requested
        vertices = jnp.where(
            centered, vertices - jnp.array([x_len / 2, y_len / 2, 0.0]), vertices
        )
        return EdgeTool.make_polyline(vertices + origins)

    @staticmethod
    @jax.jit
    def make_polyline(vertices: jnp.ndarray):
        segments = jnp.stack([vertices[:-1], vertices[1:]], axis=1)
        return Edges.from_line_segments(segments)

    @staticmethod
    @jax.jit
    def make_line(start_vertex: jnp.ndarray, end_vertex: jnp.ndarray):
        line_segments = jnp.stack([start_vertex, end_vertex])
        return Edges.from_line_segments(jnp.expand_dims(line_segments, axis=0))

    @staticmethod
    @jax.jit
    def make_polar_line(start_vertex: jnp.ndarray, distance: Number, angle: Number):
        end_vertex = jnp.array(
            [
                jnp.cos(jnp.radians(angle)) * distance,
                jnp.sin(jnp.radians(angle)) * distance,
            ]
        )

        return EdgeTool.make_line(start_vertex, end_vertex)

    @staticmethod
    def make_bspline_curve(
        ctrl_pnts: jnp.ndarray,
        degree: jnp.ndarray,
        knots: Optional[ArrayLike] = None,
    ):
        if knots is None:
            knots = BSplineEvaluator.generate_clamped_knots(degree, len(ctrl_pnts))
        knots = jnp.asarray(knots)

        return BSplineCurves(
            degree=jnp.array([degree], dtype=jnp.int32),
            ctrl_pnts=SparseArray.from_array(ctrl_pnts),
            knots=SparseArray.from_array(knots),
        )

    @staticmethod
    def consolidate_wires(wires: Wires, pending_edges: Edges, validate: bool):
        """
        Consolidates the pending edges with the existing wires.

        :param wires: The existing wires
        :type wires: Wires
        :param pending_edges: The pending edges
        :type pending_edges: Edges
        :param validate: Flag to validate the pending edges for sort order before consolidation. No validation is more performant.
        :type validate: bool
        
        """
        if validate:
            pending_edge_result = pending_edges.get_sorted()
            pending_edges = pending_edge_result.edges
            is_clockwise = WireHelper.is_clockwise(pending_edge_result.segments[0], pending_edge_result.segments[-1])
            if is_clockwise:
                pending_edges = pending_edges.flip_winding()

        clipper_wire = Wires.from_edges(pending_edges)
        if wires.count > 0:
            return Clipper.cut_polygons(wires, clipper_wire)
        else:
            return clipper_wire
