from typing import Any, Optional, Union
import jax_dataclasses as jdc
import jax.numpy as jnp
import numpy as np
from nebula.evaluators.bspline import get_sampling
from nebula.prim import BSplineCurves
from nebula.prim import Lines
import numpy.typing as npt
from nebula.helpers.wire import WireHelper
from nebula.prim.axes import Axes
from nebula.topology.topology import Topology

DEFAULT_CURVE_SAMPLES = 20


@jdc.pytree_dataclass
class SortedEdgeResult:
    edges: "Edges"
    index: jdc.Static[npt.NDArray[np.int32]]
    segments: jnp.ndarray


@jdc.pytree_dataclass
class Edges(Topology):
    lines: Lines
    curves: BSplineCurves
    is_curve: jdc.Static[npt.NDArray[np.bool_]]

    def add(
        self,
        edges: Union["Edges", Lines, BSplineCurves, jnp.ndarray],
        reorder_index: bool = False,
    ):
        """
        Add a new edge to the topology.

        :param properties: Either new vertices of the edges or new Edges to add.
        :type a: jnp.ndarray

        :returns: The indices of the new edges.
        :rtype: jnp.ndarray
        """
        # Add new unqiue vertices to the list of vertices
        if isinstance(edges, Edges):
            lines = self.lines.add(edges.lines)
            curves = self.curves.add(edges.curves)
            is_curve = np.concatenate([self.is_curve, edges.is_curve])

        else:
            if isinstance(edges, BSplineCurves):
                curves = self.curves.add(edges)
                lines = self.lines
            else:
                if isinstance(edges, jnp.ndarray):
                    edges = Lines.from_segments(edges)
                lines = self.lines.add(edges)
                curves = self.curves

            new_indices = np.arange(edges.count, dtype=np.int32)
            is_curve = np.concatenate(
                [self.is_curve, np.full(len(new_indices), isinstance(edges, BSplineCurves))]
            )

        return Edges(
            lines,
            curves,
            is_curve,
        )

    @staticmethod
    def empty():
        return Edges(
            lines=Lines.empty(),
            curves=BSplineCurves.empty(),
            is_curve=np.empty((0,), dtype=bool),
        )

    @staticmethod
    def from_line_segments(segments: jnp.ndarray):
        """Create edges from segments

        :param segments: segments to create edges from
        :type segments: jnp.ndarray
        :return: Edges
        :rtype: Edges
        """
        lines = Lines.from_segments(segments)
        return Edges(
            lines=lines,
            curves=BSplineCurves.empty(),
            is_curve=np.zeros(lines.count, dtype=bool),
        )

    @staticmethod
    def from_bspline_curves(
        bspline_curves: BSplineCurves,
    ):
        """Create edges from bspline curves

        :param bspline_curves: bspline curves
        :type bspline_curves: BSplineCurves
        :return: Edges
        :rtype: Edges
        """
        new_edges = Edges.empty()
        return new_edges.add(bspline_curves)

    def clone(
        self,
        lines: Optional[Lines] = None,
        curves: Optional[BSplineCurves] = None,
        is_curve: Optional[np.ndarray] = None,
    ):
        return Edges(
            lines=lines if lines is not None else self.lines,
            curves=curves if curves is not None else self.curves,
            is_curve=is_curve if is_curve is not None else self.is_curve,
        )

    @property
    def vertices(self):
        return jnp.concatenate(
            [self.lines.get_segments()[:, 0], self.curves.ctrl_pnts.val]
        )

    # TODO: make this simpler and more efficient
    def get_sorted(self):
        segments = self.get_segments()
        sorted_segments = jnp.zeros((len(segments), 2, 3))
        sorting_order = np.zeros(self.index.shape[0], dtype=int)

        order_index = np.arange(self.count)
        for i in range(0, len(segments)):
            if i == 0:
                sorted_segments = sorted_segments.at[i].set(segments[0])
                sorting_order[i] = self.index[0]
            else:
                previous_segment_end = sorted_segments[i - 1][1]
                # check which start segments match the previous segment's end
                sort_mask = jnp.all(
                    jnp.abs(segments[:, 0] - previous_segment_end) <= 1e-5, axis=1
                )
                sorted_segments = sorted_segments.at[i].set(segments[sort_mask][0])
                sorting_order[i] = order_index[sort_mask][0]

        sorted_is_curve = self.is_curve[sorting_order]
        sorted_index = self.index[sorting_order]

        new_edges = self.clone(
            lines=self.lines.reorder(sorted_index[~sorted_is_curve]),
            curves=self.curves.reorder(sorted_index[sorted_is_curve]),
            is_curve=sorted_is_curve,
        )

        return SortedEdgeResult(new_edges, sorted_index, sorted_segments)
        # if return_index:
        #     return self.clone(), self.index
        # return self.clone()

    # def get_segments(self):
    #     line_segments = self.lines.get_segments()
    #     curve_segments = self.curves.get_segments()

    #     total_size = line_segments.shape[0] + curve_segments.shape[0]
    #     padded_curve_segments = jnp.pad(
    #         curve_segments, ((0, total_size - curve_segments.shape[0]), (0, 0), (0, 0))
    #     )
    #     padded_line_segments = jnp.pad(
    #         line_segments, ((0, total_size - line_segments.shape[0]), (0, 0), (0, 0))
    #     )

    #     return jnp.where(
    #         self.is_curve[:, None, None], padded_curve_segments, padded_line_segments
    #     )

    def get_segments(self):
        line_segments = self.lines.get_segments()
        curve_segments = self.curves.get_segments()

        return (
            jnp.empty((self.count, 2, 3))
            .at[~self.is_curve]
            .set(line_segments)
            .at[self.is_curve]
            .set(curve_segments)
        )

    @property
    def index(self):
        """
        Generates edge index for each component, use mask self.is_curve to mask off each index
        """
        index = np.empty(self.count, dtype=np.int32)
        index[~self.is_curve] = np.arange(self.lines.count)
        index[self.is_curve] = np.arange(self.curves.count)
        return index

    @property
    def count(self):
        return len(self.is_curve)

    def evaluate(
        self,
        curve_samples: Optional[int] = DEFAULT_CURVE_SAMPLES,
        is_cosine_sampling: bool = False,
    ):
        """
        Evaluate edges, not in order

        :param curve_samples: Number of samples to use for curves,
        if None then uses ctrl pnt start and end as segments, defaults to DEFAULT_CURVE_SAMPLES
        :type curve_samples: Optional[int], optional
        :return: Segments
        :rtype: jnp.ndarray
        """
        u = (
            get_sampling(0.0, 1.0, curve_samples + 1, is_cosine_sampling)
            if curve_samples
            else None
        )
        curve_segments = self.curves.evaluate(u)
        line_segments = self.lines.get_segments()

        return jnp.concatenate([line_segments, curve_segments], axis=0)

    def dir_vec_at(self, u: jnp.ndarray, index: jnp.ndarray, eps=1e-5):
        u_next = jnp.where(u >= 1.0, u, u + eps)
        u = jnp.where(u >= 1.0, u_next - eps, u)
        
        start_vertices = self.evaluate_at(u, index)
        end_vertices = self.evaluate_at(u_next, index)

        return end_vertices - start_vertices

    def evaluate_at(self, u: jnp.ndarray, index: jnp.ndarray):
        """
        Evaluate the edge at the given u values.

        :param u: The u values to evaluate the edge at.
        :type u: jnp.ndarray
        :param index: The index of the edge to evaluate.
        :type index: jnp.ndarray
        :return: The vertices of the edge.
        :rtype: jnp.ndarray
        """
        if self.is_curve[index]:
            return self.curves.evaluate_at(u, self.index[index])
        return self.lines.evaluate_at(u, self.index[index])

    def mask(self, mask: Union[jnp.ndarray, np.ndarray]):
        """Get edges from mask

        :param mask: Mask for edges
        :type mask: Union[jnp.ndarray, np.ndarray]
        :return: Edges
        :rtype: Edges
        """

        line_mask = mask[~self.is_curve]
        curve_mask = mask[self.is_curve]

        return self.clone(
            self.lines.mask(line_mask),
            self.curves.mask(curve_mask),
            self.is_curve[mask],
        )

    def project(self, curr_axes: Axes, new_axes: Axes):
        new_lines = self.lines.project(curr_axes, new_axes)
        new_curves = self.curves.project(curr_axes, new_axes)
        is_curve = np.repeat(self.is_curve[None, :], new_axes.count, axis=0).flatten()
        return Edges(lines=new_lines, curves=new_curves, is_curve=is_curve)

    def translate(self, translation: jnp.ndarray):
        line_translation = (
            translation if len(translation.shape) == 1 else translation[~self.is_curve]
        )

        curve_translation = (
            translation if len(translation.shape) == 1 else translation[self.is_curve]
        )

        new_lines = self.lines.translate(line_translation)
        new_curves = self.curves.translate(curve_translation)
        return self.clone(new_lines, new_curves)

    def flip_winding(self, mask: Optional[jnp.ndarray] = None):
        line_mask = mask[~self.is_curve] if mask is not None else None
        curve_mask = mask[self.is_curve] if mask is not None else None
        flipped_edges = self.clone(
            self.lines.flip_winding(line_mask),
            self.curves.flip_winding(curve_mask),
            self.is_curve[::-1],
        )

        return flipped_edges

    def __repr__(self) -> str:
        return f"Edges(count={self.count})"
