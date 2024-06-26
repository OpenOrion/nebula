from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal, Optional
import plotly.graph_objects as go
from nebula.evaluators.bspline import BSplineEvaluator, get_sampling
import jax.numpy as jnp
from nebula.tools.edge import EdgeTool


def get_thickness_dist_ctrl_pnts(
    camber: jnp.ndarray,
    camber_normal: jnp.ndarray,
    thickness_dist: jnp.ndarray,
    thickness_sampling: jnp.ndarray,
    degree: jnp.ndarray,
):
    "get thickness distribution control points"
    camber_normal_thickness = BSplineEvaluator.eval_curve(
        degree, thickness_dist, thickness_sampling
    )

    return jnp.concatenate(
        [
            jnp.array([camber[0]]),
            camber + camber_normal * camber_normal_thickness,
            jnp.array([camber[-1]]),
        ]
    )


@dataclass
class CamberThicknessAirfoil:
    "parametric airfoil using B-splines"

    inlet_angle: jnp.ndarray
    "inlet angle (rad)"

    outlet_angle: jnp.ndarray
    "outlet angle (rad)"

    upper_thick_prop: jnp.ndarray
    "upper thickness proportion to chord length (length)"

    lower_thick_prop: jnp.ndarray
    "lower thickness proportion to chord length (length)"

    leading_prop: jnp.ndarray
    "leading edge tangent line proportion [0.0-1.0] (dimensionless)"

    trailing_prop: jnp.ndarray
    "trailing edge tangent line proportion [0.0-1.0] (dimensionless)"

    chord_length: jnp.ndarray
    "chord length (length)"

    stagger_angle: Optional[jnp.ndarray] = None
    "stagger angle (rad)"

    num_samples: int = 50
    "number of samples"

    is_cosine_sampling: bool = True
    "use cosine sampling"

    leading_ctrl_pnt: jnp.ndarray = field(
        default_factory=lambda: jnp.array([0.0, 0.0, 0.0])
    )
    "leading control point (length)"

    angle_units: Literal["rad", "deg"] = "rad"
    "angle units"

    def __post_init__(self):
        if self.angle_units == "deg":
            self.inlet_angle = jnp.radians(self.inlet_angle)
            self.outlet_angle = jnp.radians(self.outlet_angle)

        if self.upper_thick_prop is not None:
            self.upper_thick_dist = [
                self.chord_length * prop for prop in self.upper_thick_prop
            ]

        if self.lower_thick_prop is not None:
            self.lower_thick_dist = [
                self.chord_length * prop for prop in self.lower_thick_prop
            ]

        if self.stagger_angle is None:
            self.stagger_angle = (self.inlet_angle + self.outlet_angle) / 2

        self.degree = jnp.array(3)
        self.num_thickness_dist_pnts = len(self.upper_thick_dist) + 4
        self.camber_knots = BSplineEvaluator.generate_clamped_knots(self.degree, 4)

        self.thickness_dist_sampling = jnp.linspace(
            0, 1, self.num_thickness_dist_pnts, endpoint=True
        )
        self.sampling = get_sampling(0.0, 1.0, self.num_samples, self.is_cosine_sampling)
        self.axial_chord_length = self.chord_length * jnp.cos(self.stagger_angle)
        self.height = self.chord_length * jnp.sin(self.stagger_angle)

    @cached_property
    def camber_ctrl_pnts(self):
        assert self.stagger_angle is not None, "stagger angle is not defined"
        p_le = jnp.array(self.leading_ctrl_pnt)

        p_te = p_le + jnp.array(
            [
                self.chord_length * jnp.cos(self.stagger_angle),
                self.chord_length * jnp.sin(self.stagger_angle),
                0.0,
            ]
        )

        # leading edge tangent control point
        p1 = p_le + self.leading_prop * jnp.array(
            [
                self.chord_length * jnp.cos(self.inlet_angle),
                self.chord_length * jnp.sin(self.inlet_angle),
                0.0,
            ]
        )

        # trailing edge tangent control point
        p2 = p_te - self.trailing_prop * jnp.array(
            [
                self.chord_length * jnp.cos(self.outlet_angle),
                self.chord_length * jnp.sin(self.outlet_angle),
                0.0,
            ]
        )

        return jnp.vstack((p_le, p1, p2, p_te))

    @cached_property
    def top_ctrl_pnts(self):
        "upper side bspline"
        assert (
            self.upper_thick_dist is not None
        ), "upper thickness distribution is not defined"
        thickness_dist = jnp.vstack(self.upper_thick_dist)
        return get_thickness_dist_ctrl_pnts(
            self.camber_coords,
            self.camber_normal_coords,
            thickness_dist,
            self.thickness_dist_sampling,
            self.degree,
        )

    @cached_property
    def bottom_ctrl_pnts(self):
        "lower side bspline"
        assert (
            self.lower_thick_dist is not None
        ), "lower thickness distribution is not defined"
        thickness_dist = -jnp.vstack(self.lower_thick_dist)
        return get_thickness_dist_ctrl_pnts(
            self.camber_coords,
            self.camber_normal_coords,
            thickness_dist,
            self.thickness_dist_sampling,
            self.degree,
        )

    @cached_property
    def camber_coords(self):
        "camber line coordinates"
        return BSplineEvaluator.eval_curve(
            self.degree, self.camber_ctrl_pnts, self.thickness_dist_sampling
        )

    @cached_property
    def camber_normal_coords(self):
        "camber normal line coordinates"
        dy = jnp.gradient(self.camber_coords[:, 1])
        dx = jnp.gradient(self.camber_coords[:, 0])
        normal = jnp.vstack([-dy, dx, jnp.zeros(len(self.camber_coords))]).T
        return normal / jnp.linalg.norm(normal, axis=1)[:, None]

    def get_coords(self):
        "airfoil coordinates"
        top_coords = BSplineEvaluator.eval_curve(
            self.degree, self.top_ctrl_pnts, self.sampling
        )
        bottom_coords = BSplineEvaluator.eval_curve(
            self.degree, self.bottom_ctrl_pnts, self.sampling
        )

        return jnp.concatenate([top_coords[1:-1], bottom_coords[::-1]])

    def get_edges(self):
        airfoil_top_edge = EdgeTool.make_bspline_curve(self.top_ctrl_pnts, self.degree)
        airfoil_bottom_edge = EdgeTool.make_bspline_curve(
            self.bottom_ctrl_pnts, self.degree
        )
        return airfoil_top_edge, airfoil_bottom_edge

    def visualize(
        self,
        include_camber=True,
        include_camber_ctrl_pnts=False,
        filename: Optional[str] = None,
    ):
        fig = go.Figure(layout=go.Layout(title=go.layout.Title(text="Airfoil")))
        if include_camber_ctrl_pnts:
            fig.add_trace(
                go.Scatter(
                    x=self.camber_ctrl_pnts[:, 0],
                    y=self.camber_ctrl_pnts[:, 1],
                    name=f"Camber Control Points",
                )
            )

        if include_camber:
            camber_coords = self.camber_coords
            fig.add_trace(
                go.Scatter(x=camber_coords[:, 0], y=camber_coords[:, 1], name=f"Camber")
            )

        coords = self.get_coords()
        fig.add_trace(go.Scatter(x=coords[:, 0], y=coords[:, 1], name=f"Airfoil"))

        fig.layout.yaxis.scaleanchor = "x"  # type: ignore
        if filename:
            fig.write_image(filename, width=500, height=500)
        else:
            fig.show()


