from typing import Optional, Protocol, Sequence, Union
import jax
import jax_dataclasses as jdc
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from nebula.helpers.types import Number
from nebula.helpers.wire import WireHelper
from nebula.prim.axes import Axes
from nebula.topology.topology import SparseIndexable, Topology
from nebula.topology.edges import DEFAULT_CURVE_SAMPLES, BSplineCurves, Edges


class BoundingBox:
    def __init__(self, min: jnp.ndarray, max: jnp.ndarray) -> None:
        self.min = min
        self.max = max

    @staticmethod
    def from_points(points: jnp.ndarray):
        flattened_points = points.reshape(-1, 3)
        return BoundingBox(
            jnp.min(flattened_points, axis=0), jnp.max(flattened_points, axis=0)
        )

    def contains(self, point: jnp.ndarray):
        return jnp.all(point >= self.min, axis=-1) & jnp.all(point <= self.max, axis=-1)

    def __repr__(self) -> str:
        return f"BoundingBox(min={self.min}, max={self.max})"


class WireLike(Topology, Protocol):
    edges: Edges

    @classmethod
    def combine(cls, topologies: Sequence["Topology"], reorder_index: bool = True):
        """Combine topologies into single

        :param topologies: Topology to combine
        :type topologies: Topology
        :return: Combined topologies
        :rtype: Topology
        """
        new_topology = cls.empty()  # type: ignore
        for topology in topologies:
            new_topology = new_topology.add(topology, reorder_index)

        return new_topology

    def get_segment_index(self, curve_samples: Optional[int] = None):
        if curve_samples is None:
            curve_samples = 0
        # Get the wire segment index including repeats for curve segments
        return jnp.concatenate(
            [
                self.index[~self.edges.is_curve],
                self.index[self.edges.is_curve].repeat(curve_samples, axis=0),
            ]
        )

    def get_axis_normals(self) -> jnp.ndarray:
        """Compute the normals of the wire axes. Only works for planar wires."""
        line_segments = self.edges.get_segments()
        # Get the first and last edges
        first_edges = (
            jnp.zeros(shape=(self.count, 2, 3))
            .at[self.index[::-1]]
            .set(line_segments[::-1])
        )
        last_edges = jnp.zeros(shape=(self.count, 2, 3)).at[self.index].set(line_segments)

        return jax.vmap(WireHelper.get_normal, in_axes=(0, 0))(first_edges, last_edges)

    def get_axes(self):
        """Compute the axis of the faces."""
        return Axes(normals=self.get_axis_normals())
    
    def get_lengths(self, curve_samples=DEFAULT_CURVE_SAMPLES):
        segments = self.edges.evaluate(curve_samples)
        segment_index = self.get_segment_index(curve_samples)

        start_vertices = segments[:, 0]
        end_vertices = segments[:, 1]

        lengths = jnp.linalg.norm(end_vertices - start_vertices)
        return jnp.zeros(self.count).at[segment_index].add(lengths)

    def get_edge_index(self, u: Number):
        if isinstance(u, (int, float)):
            u = jnp.array([u])
        wire_lengths = self.get_lengths()
        length_ratios = wire_lengths / jnp.sum(wire_lengths)

        wire_ratios = jnp.cumsum(length_ratios)
        # TODO: check this for correctness
        # return jnp.arange(len(wire_ratios))[(wire_ratios <= u)][-1]
        return jnp.array(0)
    
    def dir_vec_at(self, u: Number):
        if isinstance(u, (int, float)):
            u = jnp.array([u])

        edge_index = self.get_edge_index(u)
        # TODO: add this later
        # edge_u = (u - length_ratios[edge_index]) / length_ratios[edge_index]
        return self.edges.dir_vec_at(u, edge_index)
    
    def evaluate_at(self, u: Number):
        if isinstance(u, (int, float)):
            u = jnp.array([u])

        edge_index = self.get_edge_index(u)

        # TODO: add this later
        # edge_u = (u - length_ratios[edge_index]) / length_ratios[edge_index]
        return self.edges.evaluate_at(u, edge_index)

    def get_centroids(self, curve_samples=DEFAULT_CURVE_SAMPLES):
        """Compute the centers of the faces."""
        segments = self.edges.evaluate(curve_samples)
        segment_index = self.get_segment_index(curve_samples)
        num_segments = jnp.expand_dims(
            (jnp.zeros(self.count).at[segment_index].add(values=jnp.ones(len(segments)))),
            axis=1,
        )

        start_vertices = segments[:, 0]

        # Sum vertices and divide by number of vertices
        vertex_sum = (
            jnp.zeros(shape=(self.count, 3)).at[segment_index].add(start_vertices)
        )  # (num_wires, 3)
        centers = vertex_sum / num_segments
        return centers

    def get_num_edges(self):
        """Compute the number of edges in each wire."""
        return (
            jnp.zeros(shape=self.count)
            .at[self.index]
            .add(values=jnp.ones(shape=self.edges.count))
        )

    def get_num_curves(self):
        """Compute the number of edges in each wire."""
        return (
            jnp.zeros(shape=self.count)
            .at[self.index]
            .add(values=self.edges.is_curve.astype(jnp.int32))
        )

    def get_bounding_box(self):
        start_vertices = self.edges.evaluate()[:, 0]

        max_coords = (
            jnp.zeros(shape=(self.count, 3)).at[self.index].max(start_vertices)
        )  # (num_wires, 3)

        min_coords = (
            jnp.zeros(shape=(self.count, 3)).at[self.index].min(start_vertices)
        )  # (num_wires, 3)

        return BoundingBox(min_coords, max_coords)


@jdc.pytree_dataclass
class Wires(WireLike):
    edges: Edges
    index: jdc.Static[npt.NDArray[np.int32]]
    is_interior: jdc.Static[npt.NDArray[np.bool_]]
    is_planar: jdc.Static[npt.NDArray[np.bool_]]

    def get_segment_is_interior(self, curve_samples: int):
        # Get the wire segment index including repeats for curve segments
        return jnp.concatenate(
            [
                self.is_interior[~self.edges.is_curve],
                self.is_interior[self.edges.is_curve].repeat(curve_samples, axis=0),
            ]
        )

    @property
    def is_single(self):
        return self.count == 1

    # TODO: make this more efficient
    @staticmethod
    def convert_to_bspline(wires: "Wires"):
        """
        :param wires: Wires to convert to bspline if they are planar and quads
        :type wires: Wires
        :return: BSpline curves
        :rtype: BSplineCurves
        """
        eval_wires = Wires.empty()
        for i in range(wires.count):
            if wires.is_planar[i]:
                curr_wire = wires.mask(wires.index == i)
                segments = curr_wire.edges.get_segments()
                assert segments.shape[0] == 4, "Only quads are supported"
                top_curve = BSplineCurves.from_line_segment(segments[0])
                bottom_curve = BSplineCurves.from_line_segment(segments[2][::-1])
                bspline_plane_wire = Wires.skin([top_curve, bottom_curve])
            else:
                bspline_plane_wire = wires.mask(wires.index == i)
            eval_wires = eval_wires.add(bspline_plane_wire)
        return eval_wires

    def mask(self, mask: Union[jnp.ndarray, np.ndarray]):
        """Get wires from mask

        :param mask: Mask for wires
        :type mask: Union[jnp.ndarray, np.ndarray]
        :return: Masked wires
        :rtype: Wires
        """
        wires = Wires(
            edges=self.edges.mask(mask),
            index=Topology.reorder_index(self.index[mask]),
            is_interior=self.is_interior[mask],
            is_planar=self.is_planar[mask],
        )
        assert len(wires.index) == wires.edges.count, "index and node count mismatch"
        return wires

    @staticmethod
    def empty():
        return Wires(
            edges=Edges.empty(),
            index=np.empty((0,), dtype=np.int32),
            is_interior=np.empty((0,), dtype=bool),
            is_planar=np.empty((0,), dtype=bool),
        )

    @staticmethod
    def skin(sections: Sequence[BSplineCurves]):
        """Create Wires from bspline surfaces

        :param bspline_surfaces: bspline surfaces
        :type bspline_surfaces: BSplineSurfaces
        :return: Wires
        :rtype: Wires
        """
        new_curves = BSplineCurves.empty()
        index = np.empty((0,), dtype=np.int32)
        num_section_pnts = sections[0].ctrl_pnts.count
        for section in sections:
            assert section.ctrl_pnts.count == num_section_pnts, "section size mismatch"
            index = np.concatenate([index, np.arange(section.count, dtype=np.int32)])
            new_curves = new_curves.add(section)

        edges = Edges.from_bspline_curves(new_curves)
        num_edges = edges.count

        return Wires(
            edges=edges,
            index=index,
            is_interior=np.full(num_edges, False),
            is_planar=np.full(num_edges, False),
        )

    @staticmethod
    def from_edges(edges: Edges):
        """Create Wires from edges

        :param edges: edges
        :type edges: Edges
        :return: Wires
        :rtype: Wires
        """
        num_edges = edges.count
        return Wires(
            edges=edges,
            index=np.full(num_edges, 0, dtype=np.int32),
            is_interior=np.full(num_edges, False),
            is_planar=np.full(num_edges, True),
        )

    @staticmethod
    def from_line_segments(
        segments: jnp.ndarray,
        is_interior: bool = False,
    ):
        """Create Wires from edge segments for each wire (num_wires, num_edges, 2, 3)

        :param edge_segments: edge segments for each wire (num_wires, num_edges, 2, 3)
        :type edge_segments: jnp.ndarray
        :return: Wires
        :rtype: Wires
        """
        if len(segments.shape) == 3:
            segments = jnp.expand_dims(segments, axis=0)

        edges = Edges.from_line_segments(segments)

        index = (
            # create new indices for each wire
            np.arange(0, len(segments))
            # repeat for each edge
            .repeat(segments.shape[1], axis=0)
        )

        return Wires(
            edges=edges,
            index=index,
            is_interior=np.full(edges.count, is_interior),
            is_planar=np.full(edges.count, True),
        )

    def add(self, wires: "Wires", reorder_index: bool = False):
        is_interior = np.concatenate([self.is_interior, wires.is_interior])
        is_planar = np.concatenate([self.is_planar, wires.is_planar])
        edges = self.edges.add(wires.edges)
        index = self.add_indices(wires.index, reorder_index)
        assert len(index) == edges.count, "index and edges count mismatch"

        return Wires(edges, index, is_interior, is_planar)

    def clone(
        self,
        edges: Optional[Edges] = None,
        index: Optional[np.ndarray] = None,
        is_interior: Optional[np.ndarray] = None,
        is_planar: Optional[np.ndarray] = None,
    ):
        edges = edges if edges is not None else self.edges
        return Wires(
            edges=edges,
            index=index if index is not None else self.index,
            is_interior=np.full(
                edges.count, is_interior if is_interior is not None else False
            ),
            is_planar=np.full(edges.count, is_planar if is_planar is not None else True),
        )

    def project(self, new_axes: Axes):
        """
        Project the wires onto the given axes from XY plane. Assumes all wires are on XY plane.
        :param axes: Axes to project on
        :type axes: Axes
        :return: Projected wires
        :rtype: Wires
        """
        curr_axes = self.get_axes()
        new_edges = self.edges.project(curr_axes, new_axes)
        index = SparseIndexable.expanded_index(self.index, new_axes.count)
        is_interior = np.repeat(
            self.is_interior[None, :], new_axes.count, axis=0
        ).flatten()
        is_planar = np.repeat(self.is_planar[None, :], new_axes.count, axis=0).flatten()

        return Wires(new_edges, index, is_interior, is_planar)

    def translate(self, translation: jnp.ndarray):
        """Translate the wires by a given translation.

        :param translation: The translation to apply.
        :type translation: jnp.ndarray
        """
        if len(translation.shape) == 1:
            new_edges = self.edges.translate(translation)
        elif len(translation.shape) == 2:
            new_edges = self.edges.translate(translation[self.index])
        else:
            raise ValueError(
                f"Translation must be of shape (3,) or (num_wires, 3), got {translation.shape}"
            )
        return self.clone(new_edges)

    def wind(self, is_clockwise: bool = False):
        normals = self.get_axis_normals()
        # checks if winding order is clockwise, if so it needs to be flipped to counter clockwise
        flip_mask = jnp.sum(normals, axis=1) < 0
        if is_clockwise:
            # if we want it to be clockwise, it is the inverse of the flip mask
            flip_mask = ~flip_mask
        new_flipped = self.flip_winding(flip_mask)
        return new_flipped

    def flip_winding(self, mask: Optional[jnp.ndarray] = None):
        """Returns the wires with fliped winding order.

        :param edges: Wires to flip
        :type wires: Wires
        :return: Wire with flipped winding order
        :rtype: jnp.ndarray
        """
        # assign to new Wires
        edge_flip_mask = mask[self.index] if mask is not None else None
        new_edges = self.edges.flip_winding(edge_flip_mask)
        new_wires = self.clone(
            new_edges, self.index[::-1], is_interior=self.is_interior[::-1]
        )

        return new_wires

    @property
    def num_interior(self):
        return jnp.sum(jnp.any(self.is_interior))

    def __repr__(self) -> str:
        return f"Wires(count={self.count}, edges={self.edges}, num_interior={self.num_interior})"

    def plot(self, is_3d=True):
        """Plotly plot the wires."""

        import plotly.graph_objects as go

        x = []
        y = []
        z = []
        segments = self.edges.evaluate()
        for segment in segments:
            x += [*segment[:, 0], None]
            y += [*segment[:, 1], None]
            z += [*segment[:, 2], None]
        if is_3d:
            fig = go.Figure(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    line=dict(color="blue", width=2),
                )
            )
            fig.update_scenes(aspectmode="data")

        else:
            fig = go.Figure(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color="blue", width=2),
                )
            )
            # equal scale
            fig.update_xaxes(scaleanchor="y", scaleratio=1)
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.show()
