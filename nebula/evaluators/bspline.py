
from nebula.helpers.types import Number
import jax
import jax.numpy as jnp
from typing import Callable, Optional


SpanFunction = Callable[[int, jnp.ndarray, int, jnp.ndarray], jnp.ndarray]


def get_degree(degree: Number):
    degree = jnp.asarray(degree)
    return degree[0] if len(degree.shape) > 0 else degree

def get_sampling(start: Number, end: Number, num_points: int, is_cosine_sampling: bool = False):
    if is_cosine_sampling:
        beta = jnp.linspace(0.0,jnp.pi, num_points, endpoint=True)
        return 0.5*(1.0-jnp.cos(beta))
    return jnp.linspace(start, end, num_points, endpoint=True)


class BSplineEvaluator:
    @staticmethod
    def find_spans(
        degree: jnp.ndarray,
        knot_vector: jnp.ndarray,
        num_ctrlpts: int,
        knot_samples: jnp.ndarray,
    ):
        """Finds the span of a single knot over the knot vector using linear search.

        Alternative implementation for the Algorithm A2.1 from The NURBS Book by Piegl & Tiller.

        :param degree: degree, :math:`p`
        :type degree: jnp.ndarray, (1,)
        :param knot_vector: knot vector, :math:`U`
        :type knot_vector: torch.Tensor
        :param num_ctrlpts: number of control points, :math:`n + 1`
        :type num_ctrlpts: int
        :param knot: knot or parameter, :math:`u`
        :type knot: float
        :return: knot span
        :rtype: int
        """
        span_start = degree + 1
        span_offset = jnp.sum(
            jnp.expand_dims(knot_samples, axis=-1) > knot_vector[span_start:], axis=-1
        )
        span = jnp.clip(span_start + span_offset, a_max=num_ctrlpts)
        return span - 1

    @staticmethod
    def basis_functions(
        degree: Number,
        knot_vector: jnp.ndarray,
        span: jnp.ndarray,
        knot_samples: jnp.ndarray,
    ):
        """Computes the non-vanishing basis functions for a single parameter.

        Implementation of Algorithm A2.2 pg 70 from The NURBS Book by Piegl & Tiller.
        Uses recurrence to compute the basis functions, also known as Cox - de
        Boor recursion formula.

        :param degree: degree, :math:`p`
        :type degree: jnp.ndarray (1,)
        :param knot_vector: knot vector, :math:`U`
        :type knot_vector: list, tuple
        :param span: knot span, :math:`i`
        :type span: int
        :param knot: knot or parameter, :math:`u`
        :type knot: float
        :return: basis functions
        :rtype: list
        """
        N = jnp.ones((degree + 1, len(knot_samples)))
        left = jnp.expand_dims(knot_samples, axis=0) - knot_vector[span + 1 - jnp.arange(degree + 1)[:, None]]
        right = knot_vector[span + jnp.arange(degree + 1)[:, None]] - jnp.expand_dims(knot_samples, axis=0)

        def inner_body_fun(r, init_value):
            j, saved, new_N = init_value
            temp = new_N[r] / (right[r + 1] + left[j - r])
            next_N = new_N.at[r].set(saved + right[r + 1] * temp)
            saved = left[j - r] * temp
            return j, saved, next_N

        def outer_body_fun(j, N: jnp.ndarray):
            saved = jnp.zeros(len(knot_samples))
            _, saved, N = jax.lax.fori_loop(0, j, inner_body_fun, (j, saved, N))
            return N.at[j].set(saved)

        return jax.lax.fori_loop(1, degree+1, outer_body_fun, N)


    @staticmethod
    def generate_line_knots():
        return BSplineEvaluator.generate_clamped_knots(1, 2)

    @staticmethod
    def generate_clamped_knots(degree: Number, num_ctrlpts: Number):
        """Generates a clamped knot vector.

        :param degree: non-zero degree of the curve
        :type degree: int
        :param num_ctrlpts: non-zero number of control points
        :type num_ctrlpts: int
        :return: clamped knot vector
        :rtype: Array
        """
        # Number of repetitions at the start and end of the array
        num_repeat = degree
        # Number of knots in the middle
        num_segments = int(num_ctrlpts - (degree + 1))

        return jnp.concatenate(
            (
                jnp.zeros(num_repeat),
                jnp.linspace(0.0, 1.0, num_segments + 2),
                jnp.ones(num_repeat),
            )
        )

    @staticmethod
    def generate_unclamped_knots(degree: int, num_ctrlpts: int):
        """Generates a unclamped knot vector.

        :param degree: non-zero degree of the curve
        :type degree: int
        :param num_ctrlpts: non-zero number of control points
        :type num_ctrlpts: int
        :return: clamped knot vector
        :rtype: Array
        """
        # Should conform the rule: m = n + p + 1
        return jnp.linspace(0.0, 1.0, degree + num_ctrlpts + 1)


    @staticmethod
    def eval_curve_pnt(
        degree: jnp.ndarray, ctrl_pnts: jnp.ndarray, basis: jnp.ndarray, span: jnp.ndarray
    ):
        dim = ctrl_pnts.shape[-1]
        if len(ctrl_pnts) < degree + 1:
            raise ValueError("Invalid size of control points for the given degree.")

        ctrl_pnt_slice = jax.lax.dynamic_slice(
            ctrl_pnts, (span - degree, 0), (1 + degree, dim)
        )
        return jnp.sum(ctrl_pnt_slice * jnp.expand_dims(basis, axis=1), axis=0)

    @staticmethod
    def eval_curve(
        degree: jnp.ndarray, ctrl_pnts: jnp.ndarray, u: jnp.ndarray, knots: Optional[jnp.ndarray] = None
    ):
        degree = get_degree(degree)
        knots = (
            BSplineEvaluator.generate_clamped_knots(degree, len(ctrl_pnts))
            if knots is None
            else knots
        )

        span = BSplineEvaluator.find_spans(degree, knots, len(ctrl_pnts), u)
        basis = BSplineEvaluator.basis_functions(degree, knots, span, u)

        return jax.vmap(BSplineEvaluator.eval_curve_pnt, in_axes=(None, None, 1, 0))(
            degree, ctrl_pnts, basis, span
        )

    @staticmethod
    def eval_surface_pnt(
        u_degree: jnp.ndarray,
        v_degree: jnp.ndarray,
        ctrl_pnts: jnp.ndarray,
        basis_u: jnp.ndarray,
        basis_v: jnp.ndarray,
        span_u: jnp.ndarray,
        span_v: jnp.ndarray,
    ):
        ctrl_pnt_slice = jax.lax.dynamic_slice(
            ctrl_pnts,
            (span_u - u_degree, span_v - v_degree, 0),
            (1 + u_degree, 1 + v_degree, 3),
        )

        eval_prev_pnt = jnp.sum(ctrl_pnt_slice * jnp.expand_dims(basis_v, axis=1), axis=1)
        return jnp.sum(eval_prev_pnt * jnp.expand_dims(basis_u, axis=1), axis=0)

    @staticmethod
    def eval_surface(
        u_degree: Number,
        v_degree: Number,
        ctrl_pnts: jnp.ndarray,
        u_knots: jnp.ndarray,
        v_knots: jnp.ndarray,
        u: jnp.ndarray,
        v: jnp.ndarray,
    ):
        u_degree = get_degree(u_degree)
        v_degree = get_degree(v_degree)

        assert (
            ctrl_pnts.shape[0] >= u_degree + 1
        ), f"Number of curves should be at least {u_degree + 1}"
        assert (
            ctrl_pnts.shape[1] >= v_degree + 1
        ), f"Number of control points should be at least {v_degree + 1}"

        span_u = BSplineEvaluator.find_spans(
            u_degree, u_knots, ctrl_pnts.shape[0], u
        )
        basis_u = BSplineEvaluator.basis_functions(u_degree, u_knots, span_u, u)

        span_v = BSplineEvaluator.find_spans(
            v_degree, v_knots, ctrl_pnts.shape[1], v
        )
        basis_v = BSplineEvaluator.basis_functions(v_degree, v_knots, span_v, v)

        return jax.vmap(
            jax.vmap(
                BSplineEvaluator.eval_surface_pnt,
                in_axes=(None, None, None, None, 1, None, 0),
            ),
                in_axes=(None, None, None, 1, None, 0, None),
        )(u_degree, v_degree, ctrl_pnts, basis_u, basis_v, span_u, span_v)

