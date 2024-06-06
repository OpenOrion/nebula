import jax_dataclasses as jdc
import jax.numpy as jnp
from nebula.evaluators.bspline import BSplineEvaluator
from nebula.helpers.types import Number
from nebula.topology.wires import Wires

@jdc.pytree_dataclass
class BSplineSurfaces:
    all_wires: Wires

    def evaluate(self, u: jnp.ndarray, v: jnp.ndarray, degree: Number = 1):
        bspline_wires = self.wires
        bspline_curves = bspline_wires.edges.curves
        num_edges = bspline_wires.get_num_edges()

        vertices = jnp.empty((0, 3))
        for i in range(bspline_wires.count):
            curve = bspline_curves.mask(bspline_wires.index == i)
            # TODO: this is only if all curves in group are the same length
            ctrl_pnts = curve.ctrl_pnts.val.reshape(
                curve.count,-1, 3
            )
            v_degree = curve.degree[0]
            v_knots = curve.knots.val[curve.knots.index == 0]
            u_degree = degree
            u_knots = BSplineEvaluator.generate_clamped_knots(u_degree, num_edges[i])
            surface_pnts = BSplineEvaluator.eval_surface(
                u_degree, v_degree, ctrl_pnts, u_knots, v_knots, u, v
            )
            new_vertices = surface_pnts.reshape(-1, 3)
            vertices = jnp.concatenate([vertices, new_vertices])
        return vertices

    def __repr__(self) -> str:
        return f"BSplineSurfaces(count={self.count})"

    @property
    def wires(self):
        return self.all_wires.mask(~self.all_wires.is_planar)

    @property
    def count(self) -> int:
        return self.wires.count


    # TODO: switch to this later
    # @property
    # def count(self) -> int:
    #     index = SparseIndexable.reorder_index(self.all_wires.index[~self.all_wires.is_planar])
    #     return SparseIndexable.get_count(index)
