import jax
import jax.numpy as jnp
import numpy as np
from nebula.helpers.types import Number
from nebula.prim.axes import Axes
from nebula.topology.faces import Faces
from nebula.topology.shells import Shells
from nebula.topology.solids import Solids
from nebula.topology.wires import Wires


class SolidTool:
    @staticmethod
    # @jax.jit
    def extrude_line_segments(segments: jnp.ndarray, translation: Number):
        """
        Returns the segments for the extruded edge points

        :param segment: edge segments to extrude
        :type segment: jnp.ndarray
        :param translation: translation to extrude by
        :type translation: Number
        :return: extruded segments (num_edges, 2, 3)


        """
        bottom_edge = segments
        top_edge = bottom_edge[::-1] + translation
        right_edge = jnp.stack([bottom_edge[-1], top_edge[0]])
        left_edge = jnp.stack([top_edge[-1], bottom_edge[0]])

        return jnp.stack([bottom_edge, right_edge, top_edge, left_edge])

    @staticmethod
    @jax.jit
    def extruded_planar_faces(
        faces: Faces, translation: Number, is_interior: jnp.ndarray
    ):
        # Handle planar extrusion faces
        line_segments = faces.wires.edges.lines.get_segments()
        plane_segments = jnp.where(
            is_interior[:, None, None],
            jnp.flip(line_segments, (0, 1)),
            line_segments,
        )

        extruded_plane_segments = jax.vmap(
            SolidTool.extrude_line_segments, in_axes=(0, None)
        )(
            plane_segments, translation
        )  # (num_edges, 4, 2, 3)

        return Faces.from_wires(
            Wires.from_line_segments(extruded_plane_segments),
            is_exterior=True,
        )

    # TODO: this is a work in progress.
    # @staticmethod
    # def sweep(wires: Wires, path: Wires):
    #     assert path.is_single, "Only one path is supported for sweep"
    #     faces = Faces.from_wires(wires)
    #     start_face = faces.flip_winding()


    #     sample = jnp.linspace(0.1, 1.0, 10)

    #     new_axes = Axes(
    #         normals=path.dir_vec_at(sample),
    #         origins=path.evaluate_at(sample),
    #     )

    #     end_face = faces.project(new_axes)
    #     end_face.wires.plot()


    # @staticmethod
    # def revolve(wires: Wires, angle: Number, axis: Axes):
    #     return SolidTool.sweep(wires, Wires.make_circle(1.0, angle, axis))

    @staticmethod
    def extrude(wires: Wires, distance: Number):
        """Extrudes a wire by a distance on axis.

              E-------F
             /|      /|
            / |     / |
           A--|----B  |
           |  H----|--G
           | /     ^ /
           |/      |/
           C--->---D

        :param wire: Wires to extrude
        :type wire: Wire
        :param distance: distance to extrude
        :type distance: float
        :return: new extruded Solids
        :rtype: Solids
        """

        faces = Faces.from_wires(wires)

        faces_normals = faces.get_axis_normals()
        translation = faces_normals * distance

        start_face = faces.flip_winding()
        end_face = faces.translate(translation)

        # Handle planar extrusion faces
        line_segments = faces.wires.edges.lines.get_segments()
        planar_is_interior = faces.wires.is_interior[~faces.wires.edges.is_curve]
        plane_segments = line_segments.at[planar_is_interior].set(
            jnp.flip(line_segments[planar_is_interior], (0, 1))
        )
        extruded_plane_segments = jax.jit(
            jax.vmap(SolidTool.extrude_line_segments, in_axes=(0, None))
        )(plane_segments, translation)

        exterior_extruded_plane_faces = Faces.from_wires(
            Wires.from_line_segments(extruded_plane_segments[~planar_is_interior]),
            is_exterior=True,
        )
        interior_extruded_plane_faces = Faces.from_wires(
            Wires.from_line_segments(extruded_plane_segments[planar_is_interior]),
            is_exterior=True,
        )

        # Handle non-planar extrusion faces
        start_curves = faces.wires.edges.curves
        end_curves = end_face.wires.edges.curves
        bspline_surface_faces = Faces.from_wires(Wires.skin((start_curves, end_curves)))

        exterior_faces = Faces.combine(
            (start_face, end_face, exterior_extruded_plane_faces, bspline_surface_faces)
        )
        start_face_exterior_index = start_face.index[~wires.is_interior][0]
        exterior_shell_index = (
            np.ones(len(exterior_faces.index), dtype=np.int32) * start_face_exterior_index
        )
        interior_shell_index = np.repeat(wires.index[wires.is_interior], 4)
        shell_index = np.concatenate([exterior_shell_index, interior_shell_index])

        new_faces = Faces.combine((exterior_faces, interior_extruded_plane_faces))
        shells = Shells(faces=new_faces, index=shell_index)

        return Solids.from_shells(shells)
