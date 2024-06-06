import jax.numpy as jnp
from typing import Optional, Union
from nebula.helpers.types import ArrayLike, CoordLike, Number
from nebula.helpers.wire import WireHelper
from nebula.tools.solid import SolidTool
from nebula.tools.edge import EdgeTool
from nebula.prim.axes import Axes, AxesString
from nebula.topology.edges import Edges
from nebula.topology.solids import Solids
from nebula.topology.wires import Wires
import jax_dataclasses as jdc


@jdc.pytree_dataclass
class Workplane:
    axes: Axes
    "The axes of the workplane"

    base_solids: Solids = jdc.field(default_factory=Solids.empty)
    "The base solids of the workplane"

    pending_edges: Edges = jdc.field(default_factory=Edges.empty)
    "The pending edges of the workplane"

    pending_wires: Wires = jdc.field(default_factory=Wires.empty)
    "The pending wires of the workplane"

    trace_vertex: jnp.ndarray = jdc.field(
        default_factory=lambda: jnp.array([0, 0, 0], dtype=jnp.float32)
    )
    "The trace vertex of the workplane"

    @staticmethod
    def init(
        axes: Union[AxesString, Axes] = "XY",
    ):
        "Initialize the workplane"
        axes = axes if isinstance(axes, Axes) else Axes.from_str(axes)
        workplane = Workplane(axes)
        return workplane

    def clone(
        self,
        axes: Optional[Axes] = None,
        base_solids: Optional[Solids] = None,
        pending_edges: Optional[Edges] = None,
        pending_wires: Optional[Wires] = None,
        trace_vertex: Optional[jnp.ndarray] = None,
    ):
        "Clone the workplane"
        return Workplane(
            axes if axes is not None else self.axes,
            base_solids if base_solids is not None else self.base_solids,
            pending_edges if pending_edges is not None else self.pending_edges,
            pending_wires if pending_wires is not None else self.pending_wires,
            trace_vertex if trace_vertex is not None else self.trace_vertex,
        )

    def plot(self):
        self.base_solids.shells.faces.wires.plot()

    @property
    def axis_locked(self):
        return self.pending_wires.count > 0

    def consolidateWires(self, validate: bool = False):
        """Consolidate pending wires into a single wire

        :param validate: Validate the edges for order

        """
        pending_wires = EdgeTool.consolidate_wires(
            self.pending_wires, self.pending_edges, validate
        )
        # pending_wires = Wires.empty()
        pending_edges = Edges.empty()
        trace_vertex = jnp.array([0, 0, 0])
        return self.clone(
            pending_edges=pending_edges,
            pending_wires=pending_wires,
            trace_vertex=trace_vertex,
        )

    def moveTo(self, x: Number = 0.0, y: Number = 0.0):
        "Move to the specified point, without drawing"
        trace_vertex = jnp.array([x, y, 0])
        return self.clone(trace_vertex=trace_vertex)

    def move(self, xDist: Number = 0.0, yDist: Number = 0.0):
        "Move the specified distance from the current point, without drawing"
        trace_vertex = self.trace_vertex + jnp.array([xDist, yDist, 0])
        return self.clone(trace_vertex=trace_vertex)

    def polarLineTo(self, distance: Number, angle: Number, forConstruction: bool = False):
        """
        Make a line from the current point to the given polar coordinates

        Useful if it is more convenient to specify the end location rather than
        the distance and angle from the current point

        :param distance: distance of the end of the line from the origin
        :param angle: angle of the vector to the end of the line with the x-axis
        :return: the Workplane object with the current point at the end of the new line
        """
        edge = EdgeTool.make_polar_line(self.trace_vertex, distance, angle)

        new_workplane = self.moveTo(
            edge.lines.vertices[-1][0], edge.lines.vertices[-1][1]
        )
        if not forConstruction:
            new_pending_edges = self.pending_edges.add(edge)
            new_workplane = new_workplane.clone(pending_edges=new_pending_edges)
        return new_workplane

    def box(self, length: Number, width: Number, height: Number, centered=True):
        "Make a box for each item on the stack"
        return self.rect(length, width).extrude(height)

    def lineTo(self, x: Number, y: Number, forConstruction=False):
        "Make a line from the current point to the provided point"
        edge = EdgeTool.make_line(self.trace_vertex, jnp.array([x, y, 0]))

        new_workplane = self.moveTo(x, y)
        if not forConstruction:
            new_pending_edges = self.pending_edges.add(edge)
            new_workplane = new_workplane.clone(pending_edges=new_pending_edges)
        return new_workplane

    def rect(self, xLen: Number, yLen: Number, centered=True, forConstruction=False):
        "Make a rectangle for each item on the stack"
        edges = EdgeTool.make_rect(xLen, yLen, self.trace_vertex, centered=centered)
        if not forConstruction:
            new_pending_edges = self.pending_edges.add(edges)
            new_workplane = self.clone(pending_edges=new_pending_edges).consolidateWires(
                validate=False
            )

        return new_workplane

    # def show(self):

    def polyline(
        self,
        vertices: CoordLike,
        forConstruction: bool = False,
        includeCurrent: bool = False,
    ):
        "Create a polyline from a list of points"

        vertices = WireHelper.to_3d_vertices(vertices)

        if includeCurrent:
            vertices = jnp.concatenate(
                [jnp.expand_dims(self.trace_vertex, axis=0), vertices]
            )

        edges = EdgeTool.make_polyline(vertices)
        new_workplane = self.moveTo(vertices[-1][0], vertices[-1][1])
        if not forConstruction:
            new_pending_edges = self.pending_edges.add(edges)
            new_workplane = new_workplane.clone(pending_edges=new_pending_edges)
        return new_workplane

    def bspline(
        self,
        ctrl_pnts: CoordLike,
        degree: Number = 3,
        knots: Optional[ArrayLike] = None,
        includeCurrent: bool = False,
        forConstruction: bool = False,
    ):
        ctrl_pnts = WireHelper.to_3d_vertices(ctrl_pnts)
        if includeCurrent:
            ctrl_pnts = jnp.concatenate(
                [jnp.expand_dims(self.trace_vertex, axis=0), ctrl_pnts]
            )

        curve = EdgeTool.make_bspline_curve(ctrl_pnts, jnp.array(degree), knots)
        edges = Edges.from_bspline_curves(curve)
        new_workplane = self.moveTo(ctrl_pnts[-1][0], ctrl_pnts[-1][1])
        if not forConstruction:
            new_pending_edges = self.pending_edges.add(edges)
            new_workplane = new_workplane.clone(pending_edges=new_pending_edges)
        return new_workplane

    def close(self, validate: bool = True):
        """End construction, and attempt to build a closed wire

        If the wire is already closed, nothing happens

        :param validate: Validate the edges for order
        """
        assert self.pending_edges.count > 0, "No segments to close"

        start_point = self.pending_edges.vertices[0]
        is_closed = jnp.allclose(self.trace_vertex, start_point, atol=1e-3)
        assert ~is_closed, "Wire already closed"

        return self.lineTo(start_point[0], start_point[1]).consolidateWires(validate)

    def extrude(
        self,
        until: Number,
    ):
        "Use all un-extruded wires in the parent chain to create a prismatic solid"
        projected_wires = self.pending_wires.project(self.axes)
        extruded_solids = SolidTool.extrude(projected_wires, until)
        base_solids = self.base_solids.add(extruded_solids)
        return Workplane(self.axes, base_solids)

    # TODO: implement this later
    # def sweep(self, path: Union[Wires, Edges, "Workplane"]):
    #     "Use all un-extruded wires in the parent chain to create a swept solid"
    #     if isinstance(path, Workplane):
    #         path = path.pending_wires
    #     elif isinstance(path, Edges):
    #         path = Wires.from_edges(path)

    #     projected_wires = self.pending_wires.project(self.axes)
    #     swept_solids = SolidTool.sweep(projected_wires, path)
    #     self.base_solids = self.base_solids.add(swept_solids)
    #     self.reset()

    #     return self
