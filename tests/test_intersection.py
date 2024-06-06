import jax.numpy as jnp
from pytest import mark
from nebula.helpers.intersection import Intersection


@mark.parametrize(
    "line1, line2, expected",
    [
        (
            jnp.array([[50, 50], [163, 215]]),
            jnp.array([[50, 250], [170, 170]]),
            jnp.array([144.036, 187.309, 0]),
        ),
    ],
)
def test_intersection(line1: jnp.ndarray, line2: jnp.ndarray, expected: jnp.ndarray):
    intersection = Intersection.line_segment_intersection(line1, line2)
    assert jnp.allclose(intersection.intersected_vertices, expected, atol=1e-3)


@mark.parametrize(
    "line1, line2",
    [
        (
            jnp.array([[50, 50], [205, 172]]),
            jnp.array([[50, 250], [170, 170]]),
        ),
    ],
)
def test_no_intersect(line1: jnp.ndarray, line2: jnp.ndarray):
    intersection = Intersection.line_segment_intersection(line1, line2)
    assert jnp.isnan(intersection.intersected_vertices).all()
