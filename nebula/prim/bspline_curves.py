import jax
import jax_dataclasses as jdc
from nebula.evaluators.bspline import BSplineEvaluator
from nebula.prim.axes import Axes
from nebula.prim import SparseArray, SparseIndexable
from typing import Optional, Sequence, Union
import jax.numpy as jnp
import numpy as np


@jdc.pytree_dataclass
class BSplineCurves:
    degree: jnp.ndarray
    ctrl_pnts: SparseArray
    knots: SparseArray

    @staticmethod
    def empty():
        return BSplineCurves(
            degree=jnp.empty((0,), dtype=np.int32),
            ctrl_pnts=SparseArray.empty((0, 3)),
            knots=SparseArray.empty(),
        )


    # TODO: make this more efficient
    def evaluate(self, u: Optional[jnp.ndarray] = None):
        """
        Evaluate the curves at the given u values.

        :param u: The u values to evaluate the curves at. If None then gives sements of start and end ctrl pnts
        :type u: jnp.ndarray
        :returns: The vertices of the curves.
        :rtype: jnp.ndarray

        """
        if not self.ctrl_pnts.count:
            return jnp.empty((0, 2, 3))
        curve_segments = []
        for i in range(self.ctrl_pnts.count):
            curve = self.mask(i)
            if u is not None:
                vertices = BSplineEvaluator.eval_curve(
                    curve.degree, curve.ctrl_pnts.val, u, curve.knots.val
                )

                segments = jnp.stack([vertices[:-1], vertices[1:]], axis=1)
            else:
                segments = jnp.array([[curve.ctrl_pnts.val[0], curve.ctrl_pnts.val[-1]]])
            curve_segments.append(segments)
        return jnp.concatenate(curve_segments, axis=0)

    def evaluate_at(self, u: jnp.ndarray, index: jnp.ndarray):
        curve = self.mask(index)
        return BSplineEvaluator.eval_curve(
            curve.degree, curve.ctrl_pnts.val, u, curve.knots.val
        )

    # TODO: Add and test this later when converting planes to bspline surfaces
    @staticmethod
    @jax.jit
    def from_line_segment(segment: jnp.ndarray):
        # other segments are the connectors between top and bottom segments in quad
        ctrl_pnts = segment.reshape(-1, 3)
        line_knots = BSplineEvaluator.generate_line_knots()
        return BSplineCurves(
            jnp.array([1]),
            SparseArray.from_array(ctrl_pnts),
            SparseArray.from_array(line_knots),
        )

    def get_segments(self):
        last_vertices = (
            jnp.empty((self.count, 3)).at[self.ctrl_pnts.index].set(self.ctrl_pnts.val)
        )

        first_vertices = (
            jnp.empty((self.count, 3))
            .at[self.ctrl_pnts.index[::-1]]
            .set(self.ctrl_pnts.val[::-1])
        )

        return jnp.stack([first_vertices, last_vertices], axis=1)


    def add(
        self,
        curves: "BSplineCurves",
    ):
        return BSplineCurves(
            degree=jnp.concatenate([self.degree, curves.degree]),
            ctrl_pnts=self.ctrl_pnts.add(curves.ctrl_pnts),
            knots=self.knots.add(curves.knots),
        )


    @classmethod
    def combine(cls, all_curves: Sequence["BSplineCurves"]):
        """Combine curves into single

        :param curves: Topology to combine
        :type curves: Topology
        :return: Combined curves
        :rtype: Topology
        """
        new_curves = cls(None)  # type: ignore
        for curves in all_curves:
            new_curves.add(curves)
        return new_curves

    def clone(
        self,
        degree: Optional[jnp.ndarray] = None,
        ctrl_pnts: Optional[SparseArray] = None,
        knots: Optional[SparseArray] = None,
    ):
        return BSplineCurves(
            degree=degree if degree is not None else self.degree,
            ctrl_pnts=(
                SparseArray(ctrl_pnts.val, ctrl_pnts.index)
                if ctrl_pnts is not None
                else self.ctrl_pnts
            ),
            knots=(
                SparseArray(knots.val, knots.index) if knots is not None else self.knots
            ),
        )

    # TODO: this is assuming it is already on the XY plane
    def project(self, curr_axes: Axes, new_axes: Axes):
        local_coords = curr_axes.to_local_coords(self.ctrl_pnts.val)
        return BSplineCurves(
            degree=jnp.repeat(self.degree[None, :], new_axes.count, axis=0).flatten(),
            ctrl_pnts=SparseArray(
                val=new_axes.to_world_coords(local_coords).reshape(-1, 3),
                index=SparseIndexable.expanded_index(self.ctrl_pnts.index, new_axes.count),
            ),
            knots=SparseArray(
                val=jnp.repeat(self.knots.val[None, :], new_axes.count, axis=0).flatten(),
                index=SparseIndexable.expanded_index(self.knots.index, new_axes.count),
            ),
        )

    def translate(self, translation: jnp.ndarray):
        return self.clone(
            ctrl_pnts=SparseArray(self.ctrl_pnts.val + translation, index=self.ctrl_pnts.index),
        )

    def mask(self, mask: Union[jnp.ndarray, np.ndarray, int]):
        index = np.arange(self.count)[mask]
        ctrl_pnts_mask = jnp.isin(self.ctrl_pnts.index, index)
        knots_mask = jnp.isin(self.knots.index, index)
        return BSplineCurves(
            degree=self.degree[mask],
            ctrl_pnts=self.ctrl_pnts.mask(ctrl_pnts_mask),
            knots=self.knots.mask(knots_mask),
        )

    def reorder(self, index: np.ndarray):
        return BSplineCurves(
            degree=self.degree[index],
            ctrl_pnts=self.ctrl_pnts.reorder(index),
            knots=self.knots.reorder(index),
        )

    def flip_winding(self, mask: Optional[jnp.ndarray] = None):
        return BSplineCurves(
            degree=self.degree[::-1],
            ctrl_pnts=self.ctrl_pnts.flip_winding(mask),
            knots=self.knots.flip_winding(mask),
        )

    def __repr__(self) -> str:
        return f"BSplineCurves(count={self.count})"

    @property
    def count(self):
        return self.ctrl_pnts.count
