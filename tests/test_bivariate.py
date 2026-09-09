"""Tests for `nalib.bivariate`.

The easy claim, that a grid makes two dimensional interpolation into two one dimensional
problems, is tested by building the surface three ways. The hard claim, that scattered data in
two dimensions can have no solution at all, is tested by constructing a singular case, which is
the thing that cannot happen in one dimension for any distinct nodes whatsoever.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import bivariate as bv


GRIDS = [(3, 3), (4, 6), (6, 4), (5, 5), (7, 8)]
SURFACES = [
    ("separable", lambda a, b: np.exp(a) * np.sin(2.0 * b)),
    ("polynomial", lambda a, b: 1.0 + a - 2.0 * b + a * b - 0.5 * a ** 2 * b),
    ("mixed", lambda a, b: 1.0 / (1.0 + a + b)),
]


def build(f, nx, ny, lo=0.0, hi=1.0):
    x = np.linspace(lo, hi, nx)
    y = np.linspace(lo, hi, ny)
    Z = np.asarray([[f(a, b) for b in y] for a in x], dtype=float)
    return x, y, Z


# ---------------------------------------------------------------- the tensor product


@pytest.mark.parametrize("nx,ny", GRIDS)
@pytest.mark.parametrize("name,f", SURFACES)
def test_the_order_of_the_two_directions_does_not_matter(nx, ny, name, f):
    """**What "tensor product" means**, and the reason two dimensions is no harder than two
    one dimensional problems on a grid."""
    x, y, Z = build(f, nx, ny)
    out = bv.order_does_not_matter(x, y, Z, np.linspace(0.0, 1.0, 9), np.linspace(0.0, 1.0, 9))
    assert out["order_gap"] < 1e-10
    assert out["lagrange_gap"] < 1e-10


@pytest.mark.parametrize("nx,ny", GRIDS)
@pytest.mark.parametrize("name,f", SURFACES)
def test_the_surface_reproduces_the_data_at_every_grid_point(nx, ny, name, f):
    x, y, Z = build(f, nx, ny)
    got = bv.newton_grid(x, y, Z, x, y)
    assert float(np.max(np.abs(got - Z))) < 1e-8 * max(float(np.max(np.abs(Z))), 1.0)


@pytest.mark.parametrize("nx,ny", GRIDS)
def test_bilinear_reproduces_the_data_and_is_exact_on_a_bilinear_surface(nx, ny):
    """The degree one case, which is used more than every other case combined."""
    f = lambda a, b: 2.0 + 3.0 * a - b + 4.0 * a * b
    x, y, Z = build(f, nx, ny)
    assert float(np.max(np.abs(bv.bilinear(x, y, Z, x, y) - Z))) < 1e-12
    ps = np.linspace(0.0, 1.0, 11)
    truth = np.asarray([[f(a, b) for b in ps] for a in ps])
    assert float(np.max(np.abs(bv.bilinear(x, y, Z, ps, ps) - truth))) < 1e-12


@pytest.mark.parametrize("nx,ny", [(4, 4), (5, 6)])
def test_a_polynomial_surface_is_reproduced_exactly(nx, ny):
    f = SURFACES[1][1]
    x, y, Z = build(f, nx, ny)
    ps = np.linspace(0.0, 1.0, 13)
    truth = np.asarray([[f(a, b) for b in ps] for a in ps])
    got = bv.newton_grid(x, y, Z, ps, ps)
    assert float(np.max(np.abs(got - truth))) < 1e-10


@pytest.mark.parametrize("nx,ny", [(4, 5), (6, 6), (8, 7)])
def test_the_tensor_interpolant_beats_bilinear_on_a_smooth_surface(nx, ny):
    out = bv.tensor_error(SURFACES[0][1], nx, ny)
    assert out["tensor_error"] < out["bilinear_error"]
    assert out["n_values_used"] == nx * ny


def test_a_mismatched_grid_is_rejected():
    with pytest.raises(ValueError):
        bv.lagrange_grid([0.0, 1.0], [0.0, 1.0, 2.0], np.zeros((2, 2)), 0.5, 0.5)


def test_repeated_nodes_in_either_direction_are_rejected():
    with pytest.raises(ValueError):
        bv.lagrange_grid([0.0, 0.0], [0.0, 1.0], np.zeros((2, 2)), 0.5, 0.5)


# ---------------------------------------------------------------- the curse


@pytest.mark.parametrize("d", [1, 2, 4, 8])
def test_the_sample_count_is_the_degree_plus_one_to_the_dimension(d):
    out = bv.curse_of_dimensionality(d, dimensions=[1, 2, 3, 5])
    for k, c in zip(out["dimensions"], out["values_needed"]):
        assert abs(c - (d + 1.0) ** int(k)) < 1e-6 * max(c, 1.0)


def test_the_curse_reaches_impossible_numbers_by_ten_dimensions():
    """**The whole argument for every gridless method later in the course.** At degree 4:
    5 values in one dimension, 25 in two, 9765625 in ten, and 9.5e13 in twenty."""
    out = bv.curse_of_dimensionality(4, dimensions=[1, 2, 3, 5, 10, 20])
    v = dict(zip([int(k) for k in out["dimensions"]], out["values_needed"]))
    assert v[1] == 5.0
    assert v[2] == 25.0
    assert v[10] > 9.7e6
    assert v[20] > 9.0e13


@pytest.mark.parametrize("k", [1, 2, 3, 6])
@pytest.mark.parametrize("p", [1, 2, 4])
def test_the_sample_count_exponent_is_the_dimension_over_the_order(k, p):
    """``target^(-k/p)``. Raising the dimension raises the exponent, and only raising the order
    can offset it, which is the curse stated exactly."""
    got = bv.samples_for_accuracy(1e-3, p, k)
    assert abs(np.log(got) / np.log(1e-3) + k / p) < 1e-9


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 2.0])
def test_a_target_outside_zero_to_one_is_rejected(bad):
    with pytest.raises(ValueError):
        bv.samples_for_accuracy(bad, 2, 3)


def test_a_degenerate_order_is_rejected():
    with pytest.raises(ValueError):
        bv.samples_for_accuracy(1e-3, 0, 3)


# ---------------------------------------------------------------- Mairhuber


def test_two_dimensional_interpolation_can_be_exactly_singular():
    """**Mairhuber's theorem, constructed.** In one dimension the Vandermonde matrix on distinct
    nodes is nonsingular for every node set, always. In two dimensions no basis has that
    property: moving two points continuously around each other until they swap exchanges two rows
    of the interpolation matrix, so the determinant changes sign, so somewhere on the path it was
    zero.

    Measured on a fixed quadratic basis with six points, walking that path: the determinant
    changes sign 3 times. Bisecting on the first sign change gives a configuration whose
    determinant is **9.9e-19** with condition number **2.0e16**, which is singular to working
    precision.

    This is why scattered multivariate interpolation is done with radial basis functions or
    triangulation rather than polynomials. It is a difference in kind, not in degree.
    """
    out = bv.mairhuber_example(n_probe=400, rng=np.random.default_rng(1))
    assert out["swaps_sign"]
    assert out["sign_changes"] >= 1
    assert abs(out["determinant_at_the_root"]) < 1e-15
    assert out["condition_at_the_root"] > 1e12


@pytest.mark.parametrize("n_probe", [100, 200, 400, 800])
def test_the_bisected_root_is_the_same_however_the_path_was_sampled(n_probe):
    """The grid only has to bracket the root. Where the grid points happen to land does not
    matter, and it is not monotone: a 100 point grid came within 3.2e-6 of zero and a 400 point
    grid within 6.9e-6. The bisected root is the same to working precision from all of them.
    """
    out = bv.mairhuber_example(n_probe=n_probe, rng=np.random.default_rng(1))
    reference = bv.mairhuber_example(n_probe=100, rng=np.random.default_rng(1))
    assert abs(out["determinant_at_the_root"]) < 1e-15
    assert abs(out["bisected_angle"] - reference["bisected_angle"]) < 1e-9


@pytest.mark.parametrize("n", [3, 5, 6, 9])
def test_one_dimensional_interpolation_is_never_singular(n):
    """The contrast. No amount of searching finds a singular node set, because there is none."""
    out = bv.one_dimension_never_fails(n_trials=300, n_nodes=n, rng=np.random.default_rng(2))
    assert not out["any_singular"]
    assert out["smallest_determinant_found"] > 0.0