from dataclasses import dataclass, field
from typing import Sequence, Union
import jax
import numpy as np
from scipy.spatial import Delaunay
import jax.numpy as jnp
from nebula.evaluators.bspline import BSplineEvaluator, get_sampling
from nebula.prim.bspline_curves import BSplineCurves
from nebula.helpers.wire import WireHelper
from nebula.topology.edges import DEFAULT_CURVE_SAMPLES
from nebula.topology.faces import Faces
from nebula.topology.solids import Solids
from nebula.topology.wires import Wires
from nebula.workplane import Workplane


@dataclass
class Mesh:
    vertices: jnp.ndarray
    simplices: jnp.ndarray
    normals: jnp.ndarray
    edges: jnp.ndarray

    @staticmethod
    def empty():
        return Mesh(
            vertices=jnp.empty((0, 3)),
            simplices=jnp.empty((0, 3), dtype=jnp.int32),
            normals=jnp.empty((0, 3)),
            edges=jnp.empty((0, 2, 3)),
        )

    @staticmethod
    def combine(meshes: Sequence["Mesh"]):

        vertices = jnp.concatenate([mesh.vertices for mesh in meshes])
        simplices = jnp.concatenate(
            [
                mesh.simplices + (len(meshes[i - 1].vertices) if i > 0 else 0)
                for i, mesh in enumerate(meshes)
            ]
        )
        normals = jnp.concatenate([mesh.normals for mesh in meshes])
        edges = jnp.concatenate([mesh.edges for mesh in meshes])
        return Mesh(vertices, simplices, normals, edges)


class Tesselator:
    @staticmethod
    def get_mesh(solids: Solids):
        faces = solids.shells.faces
        planar_mesh = Tesselator.get_planar_mesh(faces.mask(faces.wires.is_planar))
        non_planar_mesh = Tesselator.get_non_planar_mesh(
            faces.mask(~faces.wires.is_planar)
        )
        return Mesh.combine([planar_mesh, non_planar_mesh])

    @staticmethod
    def get_differentiable_mesh(target: Union[Workplane, Solids]):
        solids = target.base_solids if isinstance(target, Workplane) else target
        faces = solids.shells.faces
        planar_mesh = Tesselator.get_differentiable_planar_mesh(
            faces.mask(faces.wires.is_planar)
        )
        non_planar_mesh = Tesselator.get_non_planar_mesh(
            faces.mask(~faces.wires.is_planar)
        )
        return Mesh.combine([planar_mesh, non_planar_mesh])

    @staticmethod
    def get_trimmed_simplices(
        vertices: jnp.ndarray,
        simplices: jnp.ndarray,
        exterior_segments: jnp.ndarray,
        interior_segments: jnp.ndarray,
    ):
        triangles = vertices[simplices]
        triangle_centers = jnp.mean(triangles, axis=1)
        is_exterior_triangle = jax.vmap(WireHelper.contains_vertex, in_axes=(None, 0))(
            exterior_segments, triangle_centers
        )
        is_interior_triangle = jax.vmap(WireHelper.contains_vertex, in_axes=(None, 0))(
            interior_segments, triangle_centers
        )
        return simplices[is_exterior_triangle & ~is_interior_triangle]

    @staticmethod
    def get_tri_normal(tri_vertices: Union[jnp.ndarray, np.ndarray]) -> jnp.ndarray:
        # Calculate two vectors along the edges of the triangle
        normal = jnp.cross(
            tri_vertices[1] - tri_vertices[0], tri_vertices[2] - tri_vertices[0]
        )
        unit_normal = normal / jnp.linalg.norm(normal, axis=0, keepdims=True)
        return unit_normal

    @staticmethod
    def get_differentiable_planar_mesh(faces: Faces, curve_samples: int = 50):
        is_quad_face = ((faces.get_num_edges() - faces.get_num_curves()) == 4).repeat(4)
        quad_faces = faces.mask(is_quad_face)
        bspline_faces = Faces.from_wires(
            Wires.convert_to_bspline(quad_faces.wires), is_exterior=True
        )
        bspline_face_mesh = Tesselator.get_non_planar_mesh(bspline_faces, curve_samples)
        return bspline_face_mesh

    @staticmethod
    def get_planar_mesh(faces: Faces, curve_samples: int = 50):
        vertices = jnp.empty((0, 3))
        simplices = jnp.empty((0, 3), dtype=jnp.int32)
        normals = jnp.empty((0, 3))

        # TODO: replace this with jax implementation of Delaunay triangulation
        for i in range(faces.count):
            face = faces.mask(faces.index == i)
            assert face.wires.is_planar.all() == True, "Only planar faces supported"

            axis = face.get_axes()
            axis_normal = axis.normals[0]

            # triangulate face
            segments = face.wires.edges.evaluate(curve_samples, is_cosine_sampling=True)
            local_segments = axis.to_local_coords(segments)[0]

            delaunay = Delaunay(local_segments[:, 0, 0:2])

            tri_local_vertices, tri_simplices = (
                jnp.array(delaunay.points),
                jnp.array(delaunay.simplices),
            )
            tri_vertices = axis.to_world_coords(tri_local_vertices)[0]

            # Flip triangle normals to correct direction as plane
            tri_normal = Tesselator.get_tri_normal(
                tri_vertices[tri_simplices[0]],
            )
            if not (tri_normal == axis_normal).all():
                tri_simplices = jnp.flip(tri_simplices, axis=1)

            # Trim triangles that are not interior
            is_interior = face.wires.get_segment_is_interior(curve_samples)
            trimmed_simplices = Tesselator.get_trimmed_simplices(
                tri_local_vertices,
                tri_simplices,
                local_segments[~is_interior],
                local_segments[is_interior],
            )
            new_normals = axis.normals.repeat(len(vertices), axis=0)

            simplices = jnp.concatenate([simplices, trimmed_simplices + len(vertices)])
            vertices = jnp.concatenate([vertices, tri_vertices])

            normals = jnp.concatenate([normals, new_normals])

        return Mesh(
            vertices,
            simplices,
            normals,
            faces.edges.evaluate(curve_samples),
        )

    @staticmethod
    @jax.jit
    def get_bspline_tri_index(i: jnp.ndarray, j: jnp.ndarray, num_v: jnp.ndarray):
        index = jnp.array(
            [
                j + (i * num_v),
                j + ((i + 1) * num_v),
                j + 1 + ((i + 1) * num_v),
                j + 1 + (i * num_v),
            ],
            dtype=jnp.int32,
        )
        # + curr_index

        return jax.vmap(
            lambda tri_index, i: jnp.array(
                # [tri_index[i + 1], tri_index[i], tri_index[0]],
                [tri_index[0], tri_index[i], tri_index[i + 1]],
                dtype=jnp.int32,
            ),
            in_axes=(None, 0),
        )(index, jnp.arange(1, 3))

    @staticmethod
    def get_non_planar_mesh(
        faces: Faces,
        num_u: int = 20,
        num_v: int = 20,
        curve_samples: int = DEFAULT_CURVE_SAMPLES,
    ):
        mesh = Mesh.empty()
        if len(faces.wires.is_planar) > 0:
            # TODO: get rid of for loop here
            for i in range(faces.count):
                face = faces.mask(faces.index == i)
                assert (
                    faces.wires.is_planar.all() == False
                ), "Only non-planar faces supported"
                i, j = jnp.arange(num_u - 1), jnp.arange(num_v - 1)

                u = get_sampling(0.0, 1.0, num_u)
                v = get_sampling(0.0, 1.0, num_v)
                vertices = face.bspline_surfaces.evaluate(u, v)

                simplicies = jax.vmap(
                    jax.vmap(
                        Tesselator.get_bspline_tri_index,
                        in_axes=(None, 0, None),
                    ),
                    in_axes=(0, None, None),
                )(i, j, jnp.array(num_v)).reshape(-1, 3)

                u_curve = get_sampling(0.0, 1.0, curve_samples)
                line_segments = faces.wires.edges.curves.evaluate(u_curve)
                # faces.wires.plot()
                # TODO: fix these normals frp, Bspline derivatives later
                normals = jax.vmap(Tesselator.get_tri_normal, in_axes=(0,))(
                    vertices[simplicies]
                )

                new_mesh = Mesh(vertices, simplicies, normals, line_segments)
                mesh = Mesh.combine([mesh, new_mesh])
        return mesh
