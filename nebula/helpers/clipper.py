from typing import Union
import jax_dataclasses as jdc
import jax
import jax.numpy as jnp
import numpy as np
from nebula.helpers.intersection import Intersection
from nebula.helpers.wire import WireHelper
from nebula.topology.edges import Edges
from nebula.topology.wires import Wires


@jdc.pytree_dataclass
class PolygonContainsResult:
    subject_in_clipper: jnp.ndarray
    clipper_in_subject: jnp.ndarray


@jdc.pytree_dataclass
class PolygonSplitResult:
    split_subject_segments: jnp.ndarray
    split_clipper_segments: jnp.ndarray
    subject_intersected: jnp.ndarray
    clipper_intersected: jnp.ndarray


@jdc.pytree_dataclass
class WireSplitResult:
    subject_segments: jnp.ndarray
    clipper_segments: jnp.ndarray
    unsplit_wires: Wires


@jdc.pytree_dataclass
class PolygonClipResult:
    clip_edges: Edges
    unclipped_wires: Wires


class Clipper:
    @staticmethod
    def cut_polygons(
        subject_wires: Wires, clipper: Union[Wires, Edges], sort_edges: bool = True
    ):
        cutter_union_result = Clipper.union(
            subject_wires.mask(subject_wires.is_interior),
            clipper,
        )

        intersection_result = Clipper.intersect(
            subject_wires.mask(~subject_wires.is_interior),
            cutter_union_result.clip_edges,
        )
        clip_edges = intersection_result.clip_edges
        if sort_edges:
            clip_edges = clip_edges.get_sorted().edges

        intersection_wires = Wires.from_edges(clip_edges)
        return Wires.combine(
            [
                intersection_wires,
                cutter_union_result.unclipped_wires,
                intersection_result.unclipped_wires,
            ]
        )

    @staticmethod
    def intersect(subject_wires: Wires, clipper: Union[Wires, Edges]):
        # Split subject and clipper edges at intersection points
        split_result = Clipper.split_wires(subject_wires, clipper)

        if subject_wires.edges.curves.count:
            raise NotImplementedError("Intersecting curves is not supported yet.")

        new_subject_segments = (
            split_result.subject_segments
            if len(split_result.subject_segments)
            else subject_wires.edges.lines.get_segments()
        )

        # Get which subject and clipper edges are in the other polygon
        polygon_contains = Clipper.contains_polygons(
            new_subject_segments, split_result.clipper_segments
        )
        # TODO: check for orientation of clipper
        filtered_subject_segments, filtered_clipper_segments = (
            new_subject_segments[~polygon_contains.subject_in_clipper],
            split_result.clipper_segments[polygon_contains.clipper_in_subject],
        )
        

        if len(split_result.subject_segments):
            filtered_clipper_segments = jnp.flip(filtered_clipper_segments, (0,1))
            clip_edges = Edges.from_line_segments(
                jnp.concatenate(
                    [
                        filtered_subject_segments,
                        filtered_clipper_segments,
                    ]
                ),
            )
            return PolygonClipResult(
                clip_edges=clip_edges,
                unclipped_wires=split_result.unsplit_wires,
            )

        return PolygonClipResult(
            clip_edges=Edges.from_line_segments(filtered_subject_segments),
            unclipped_wires=Wires.from_line_segments(
                filtered_clipper_segments, is_interior=True
            ),
        )

    @staticmethod
    def union(subject_wires: Wires, clipper: Union[Wires, Edges]):
        # Split subject and clipper edges at intersection points
        split_result = Clipper.split_wires(subject_wires, clipper)

        # Get which subject and clipper edges are in the other polygon
        polygon_contains = Clipper.contains_polygons(
            split_result.subject_segments, split_result.clipper_segments
        )

        clip_edges = Edges.from_line_segments(
            jnp.concatenate(
                [
                    split_result.subject_segments[~polygon_contains.subject_in_clipper],
                    split_result.clipper_segments[~polygon_contains.clipper_in_subject],
                ]
            ),
        )
        return PolygonClipResult(
            clip_edges=clip_edges,
            unclipped_wires=split_result.unsplit_wires,
        )

    # TODO: make segments have same count for jit
    @staticmethod
    @jax.jit
    def contains_polygons(
        subject_segments: jnp.ndarray,
        clipper_segments: jnp.ndarray,
    ):
        subject_in_clipper = jax.vmap(WireHelper.contains_segment, in_axes=(None, 0))(
            clipper_segments, subject_segments
        )

        clipper_in_subject = jax.vmap(WireHelper.contains_segment, in_axes=(None, 0))(
            subject_segments, clipper_segments
        )

        return PolygonContainsResult(
            subject_in_clipper=subject_in_clipper, clipper_in_subject=clipper_in_subject
        )

    @staticmethod
    def split_polygon(subject_segments: jnp.ndarray, clipper_segments: jnp.ndarray):
        intersections = jax.vmap(
            jax.vmap(Intersection.line_segment_intersection, in_axes=(None, 0)),
            in_axes=(0, None),
        )(subject_segments, clipper_segments)
        # find which subject and clipper edges intersect
        subject_intersected = jnp.any(intersections.is_intersection, axis=(1,))
        clipper_intersected = jnp.any(intersections.is_intersection, axis=(0,))

        # split subject edges at intersection points
        subject_split_segments = Clipper.split_segments(
            subject_segments[subject_intersected],
            intersections.intersected_vertices[intersections.is_intersection],
        )

        # Split clipper edges at intersection points
        clipper_split_segments = Clipper.split_segments(
            clipper_segments[clipper_intersected],
            jnp.swapaxes(intersections.intersected_vertices, 0, 1)[
                jnp.swapaxes(intersections.is_intersection, 0, 1)
            ],
        )

        return PolygonSplitResult(
            split_subject_segments=subject_split_segments,
            split_clipper_segments=clipper_split_segments,
            subject_intersected=subject_intersected,
            clipper_intersected=clipper_intersected,
        )

    @staticmethod
    def split_wires(subject_wires: Wires, clipper: Union[Wires, Edges]):
        clipper_edges = clipper.edges if isinstance(clipper, Wires) else clipper
        if subject_wires.edges.curves.count or clipper_edges.curves.count:
            raise NotImplementedError("Intersecting curves is not supported yet.")
        subject_segments = subject_wires.edges.lines.get_segments()
        clipper_segments = clipper_edges.lines.get_segments()
        split_result = Clipper.split_polygon(subject_segments, clipper_segments)

        # Exclude all subject wires that are not affected from final subject segements
        is_split_wire = (
            jnp.zeros(subject_wires.count)
            .at[subject_wires.index]
            .add(split_result.subject_intersected.astype(jnp.int32))[subject_wires.index]
            .astype(bool)
        )
        unsplit_wires = subject_wires.mask(~is_split_wire)
        unsplit_affected_wire_segments = subject_segments[
            ~split_result.subject_intersected & is_split_wire
        ]

        new_subject_segments = jnp.concatenate(
            [unsplit_affected_wire_segments, split_result.split_subject_segments]
        )

        new_clipper_segments = jnp.concatenate(
            [
                clipper_segments[~split_result.clipper_intersected],
                split_result.split_clipper_segments,
            ]
        )

        return WireSplitResult(
            subject_segments=new_subject_segments,
            clipper_segments=new_clipper_segments,
            unsplit_wires=unsplit_wires,
        )

    # TODO: make segments have same count for jit
    @staticmethod
    @jax.jit
    def split_segments(segments: jnp.ndarray, intersection_vertices: jnp.ndarray):
        fun_split_edges = lambda edge_vertices, intersection_point: jnp.array(
            [
                [edge_vertices[0], intersection_point],
                [intersection_point, edge_vertices[1]],
            ]
        )  # (2, 3)

        # Split subject edges at intersection points
        segment_groups = jax.vmap(fun_split_edges, in_axes=(0, 0))(
            segments,
            intersection_vertices,
        )
        return (
            jnp.concatenate(segment_groups)
            if len(segment_groups) > 0
            else jnp.empty((0, 2, 3))
        )
