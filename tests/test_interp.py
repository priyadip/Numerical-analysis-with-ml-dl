"""Tests for `nalib.interp`.

The central claim of lesson 44 is that the five forms are **the same polynomial**, so almost
every test here builds an interpolant several ways and checks they agree. Where they do not
agree, the disagreement is the conditioning, and that is tested too rather than hidden by a
loose tolerance.

Sizes, intervals and functions are all parametrized. Nothing is checked at one size only,
because the whole subject is about what happens as the degree grows.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import interp as ip


SIZES = [1, 2, 3, 5, 8, 13, 21]
INTERVALS = [(-1.0, 1.0), (0.0, 1.0), (-5.0, 3.0), (99.0, 101.0), (-0.001, 0.001)]
FUNCS = [
    ("exp", np.exp),
    ("sin", np.sin),
    ("cubic", lambda t: 1.0 + 2.0 * t - 0.5 * t ** 2 + 0.25 * t ** 3),
    ("gauss", lambda t: np.exp(-(t ** 2))),
]


def nodes_on(lo, hi, n):
    return np.linspace(lo, hi, n) if n > 1 else np.array([0.5 * (lo + hi)])


# ---------------------------------------------------------------- existence and uniqueness


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("lo,hi", INTERVALS)
def test_the_two_lagrange_forms_reproduce_the_data_at_the_nodes_exactly(n, lo, hi, rng):
    """**Exactly**, not to a tolerance, and at every degree and interval.

    Both forms contain a factor that is exactly zero at a node, so the arithmetic returns the
    data value with no rounding at all. Measured at 0.00e+00 for every size from 5 to 31 while
    the coefficient-based forms below degrade by twelve orders of magnitude over the same range.
    """
    x = nodes_on(lo, hi, n)
    y = rng.standard_normal(n)
    assert np.array_equal(ip.evaluate_barycentric(x, y, x), y)
    assert np.max(np.abs(ip.evaluate_lagrange(x, y, x) - y)) == 0.0


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("lo,hi", INTERVALS)
def test_the_coefficient_forms_reproduce_the_nodes_to_eps_times_the_conditioning(n, lo, hi, rng):
    """The Newton and power forms solve for coefficients, so they inherit the conditioning of
    that step. The bound they must meet is ``eps * kappa``, not a fixed tolerance: at 31 equally
    spaced nodes the measured error is 6e-4, which is correct behaviour for ``kappa = 5.6e13``
    and would be a bug for a well conditioned problem.

    The ``kappa`` used is that of the **normalised** system, since that is the intrinsic
    difficulty of the degree. Using the raw one would make the bound vacuous.
    """
    x = nodes_on(lo, hi, n)
    y = rng.standard_normal(n)
    scale = max(float(np.max(np.abs(y))), 1.0)
    c = 0.5 * (float(np.max(x)) + float(np.min(x)))
    h = 0.5 * (float(np.max(x)) - float(np.min(x))) or 1.0
    kappa = ip.vandermonde_condition((x - c) / h)
    bound = max(np.finfo(float).eps * kappa, 1e-14)

    newton = np.max(np.abs(ip.evaluate_newton(ip.newton_coefficients(x, y), x, x) - y)) / scale
    a, cc, hh = ip.normalised_power_form(x, y)
    power = np.max(np.abs(ip.evaluate_normalised(a, x, cc, hh) - y)) / scale
    assert newton <= bound, f"newton {newton:.2e} exceeds eps*kappa {bound:.2e}"
    assert power <= bound, f"normalised power {power:.2e} exceeds eps*kappa {bound:.2e}"


@pytest.mark.parametrize("n", [2, 3, 5, 8, 13])
def test_the_five_forms_are_the_same_polynomial(n, rng):
    """**The uniqueness theorem, checked rather than assumed.** They must agree away from the
    nodes as well as at them, since a polynomial is determined by its values everywhere."""
    x = np.linspace(-1.0, 1.0, n)
    y = np.cos(3.0 * x)
    out = ip.uniqueness_report(x, y, n_trials=9, rng=rng)
    for name, gap in out["max_disagreement"].items():
        assert gap < 1e-10, f"{name} disagrees by {gap:.2e}"


@pytest.mark.parametrize("n", SIZES)
def test_a_polynomial_of_low_degree_is_reproduced_exactly(n):
    """Interpolating a cubic at four or more nodes must return the cubic itself."""
    cubic = FUNCS[2][1]
    x = np.linspace(-2.0, 2.0, n)
    y = cubic(x)
    probe = np.linspace(-2.0, 2.0, 51)
    got = ip.evaluate_barycentric(x, y, probe)
    if n >= 4:
        assert np.max(np.abs(got - cubic(probe))) < 1e-11
    else:
        assert np.max(np.abs(ip.evaluate_barycentric(x, y, x) - y)) < 1e-12


# ---------------------------------------------------------------- the node hypothesis


@pytest.mark.parametrize("bad", [[1.0, 1.0], [0.0, 1.0, 0.0], [2.0, 3.0, 4.0, 3.0]])
def test_repeated_nodes_are_rejected(bad):
    """Interpolation is not defined on repeated abscissas, and every entry point says so."""
    y = np.arange(len(bad), dtype=float)
    for fn in (ip.newton_coefficients, ip.power_form, ip.shifted_power_form):
        with pytest.raises(ValueError):
            fn(bad, y)
    with pytest.raises(ValueError):
        ip.barycentric_weights(bad)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        ip.newton_coefficients([0.0, 1.0, 2.0], [1.0, 2.0])


def test_an_empty_node_set_is_rejected():
    with pytest.raises(ValueError):
        ip.barycentric_weights([])


# ---------------------------------------------------------------- the Newton form's property


@pytest.mark.parametrize("n", [1, 2, 5, 9, 14])
@pytest.mark.parametrize("lo,hi", [(-1.0, 1.0), (0.0, 3.0)])
def test_adding_a_point_leaves_every_existing_coefficient_unchanged(n, lo, hi):
    """**The property no other form has.** It is what makes the Newton form the one an adaptive
    method carries, and it is exact rather than approximate: the old coefficients are the same
    floating point numbers, not merely close ones."""
    f = np.exp
    x = np.linspace(lo, hi, n)
    c = ip.newton_coefficients(x, f(x))
    x_new = hi + 0.37 * (hi - lo) + 0.11
    c2, x2 = ip.newton_add_point(c, x, x_new, float(f(x_new)))
    assert np.array_equal(c, c2[:c.size])
    assert c2.size == c.size + 1
    assert abs(float(ip.evaluate_newton(c2, x2, x_new)) - float(f(x_new))) < 1e-9


def test_adding_an_existing_node_is_rejected():
    x = np.linspace(0.0, 1.0, 4)
    c = ip.newton_coefficients(x, np.exp(x))
    with pytest.raises(ValueError):
        ip.newton_add_point(c, x, float(x[2]), 1.0)


@pytest.mark.parametrize("n", [2, 4, 7, 11])
def test_reordering_the_nodes_gives_a_different_form_and_the_same_polynomial(n, rng):
    """The Newton coefficients depend on the node ORDER; the polynomial does not."""
    x = np.linspace(-1.0, 1.0, n)
    y = np.sin(2.0 * x)
    c = ip.newton_coefficients(x, y)
    order = rng.permutation(n)
    z = x[order]
    c2 = ip.change_of_centre(c, x, z)
    probe = np.linspace(-1.0, 1.0, 37)
    a = ip.evaluate_newton(c, x, probe)
    b = ip.evaluate_newton(c2, z, probe)
    assert np.max(np.abs(a - b)) < 1e-10


# ---------------------------------------------------------------- barycentric


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("lo,hi", INTERVALS)
def test_barycentric_weights_match_their_definition(n, lo, hi):
    """``w_i = 1 / prod_{j != i} (x_i - x_j)``, up to the common scaling the formula allows.

    The barycentric formula is invariant to a common factor in the weights, so the test compares
    the ratios rather than the values, which is what the definition actually pins down.
    """
    x = nodes_on(lo, hi, n)
    w = ip.barycentric_weights(x)
    if n == 1:
        assert w.shape == (1,)
        return
    direct = np.array([1.0 / np.prod([x[i] - x[j] for j in range(n) if j != i])
                       for i in range(n)])
    r = w / direct
    assert np.max(np.abs(r - r[0])) < 1e-8 * max(abs(r[0]), 1.0)


@pytest.mark.parametrize("n", [2, 5, 12, 20])
def test_barycentric_handles_evaluation_exactly_at_a_node(n, rng):
    """The formula divides by ``t - x_i``, so the node itself is a special case that must be
    handled rather than allowed to produce a nan."""
    x = np.linspace(-1.0, 1.0, n)
    y = rng.standard_normal(n)
    got = ip.evaluate_barycentric(x, y, x)
    assert np.all(np.isfinite(got))
    assert np.max(np.abs(got - y)) < 1e-13


def test_barycentric_accepts_a_scalar_and_returns_a_scalar():
    x = np.linspace(0.0, 1.0, 5)
    out = ip.evaluate_barycentric(x, np.exp(x), 0.42)
    assert np.isscalar(out) or np.ndim(out) == 0


# ---------------------------------------------------------------- conditioning


@pytest.mark.parametrize("n", [5, 10, 15, 20])
def test_the_vandermonde_condition_number_grows_with_the_degree(n):
    """It is exponential in ``n``, which is the reason the power form is not used."""
    x = np.linspace(-1.0, 1.0, n)
    bigger = np.linspace(-1.0, 1.0, n + 5)
    assert ip.vandermonde_condition(bigger) > 10.0 * ip.vandermonde_condition(x)


@pytest.mark.parametrize("centre", [0.0, 100.0, 1000.0, 10000.0])
def test_centring_the_nodes_makes_the_conditioning_independent_of_where_they_sit(centre):
    """**The whole point of the shifted power form.** The condition number of the centred system
    is the same wherever the interval sits; the uncentred one grows without bound."""
    n = 10
    x = np.linspace(-1.0, 1.0, n) + centre
    centred = ip.vandermonde_condition(x - np.mean(x))
    reference = ip.vandermonde_condition(np.linspace(-1.0, 1.0, n))
    assert abs(centred - reference) < 1e-6 * reference
    if centre >= 100.0:
        assert ip.vandermonde_condition(x) > 1e6 * centred


def test_the_singular_vandermonde_says_why_it_failed():
    """A bare "singular matrix" tells the caller nothing. This one names the conditioning and
    the two forms that do not have the problem."""
    x = np.linspace(999.0, 1001.0, 12)
    with pytest.raises(np.linalg.LinAlgError, match="shifted_power_form"):
        ip.power_form(x, np.sin(x))


# ---------------------------------------------------------------- accuracy against a function


@pytest.mark.parametrize("name,f", FUNCS)
@pytest.mark.parametrize("n", [4, 8, 12])
def test_the_forms_agree_with_each_other_on_a_smooth_function(name, f, n):
    """The five forms are one polynomial, so their errors against ``f`` must be nearly equal.

    Note this asserts **agreement**, not accuracy. How close the interpolant gets to ``f`` is a
    property of ``f`` and the degree, not of the form: a cubic through four points is 5.8e-2
    away from a Gaussian no matter how it is written, and lesson 46 is where that error is
    analysed. Asserting a fixed accuracy here would be testing the wrong thing.
    """
    out = ip.compare_forms(f, n, lo=-1.0, hi=1.0)
    errs = np.array(list(out.errors.values()))
    spread = float(np.max(errs) - np.min(errs))
    assert spread < 1e-6 + 1e-6 * float(np.max(errs)), \
        f"the forms disagree on {name} at n={n}: {out.errors}"
    for form, res in out.node_residuals.items():
        assert res < 1e-8, f"{form} on {name} at n={n} missed its own nodes by {res:.2e}"


@pytest.mark.parametrize("n", [2, 4, 6, 8, 10, 12])
def test_more_nodes_improve_a_smooth_function_while_the_error_is_above_rounding(n):
    """Below machine precision the approximation error dominates and more nodes always help."""
    coarse = ip.compare_forms(np.exp, n, lo=-1.0, hi=1.0).errors["barycentric"]
    fine = ip.compare_forms(np.exp, n + 2, lo=-1.0, hi=1.0).errors["barycentric"]
    assert fine < coarse


def test_equally_spaced_interpolation_gets_worse_past_its_minimum():
    """**On `exp`, an entire function**, so this is not approximation error at all.

    Measured maximum error of the barycentric interpolant on ``[-1, 1]``:

    ====  ================  ============
    n     equally spaced    Chebyshev
    ====  ================  ============
    14    1.29e-14          9.80e-16
    16    3.61e-14          3.27e-16
    24    3.88e-12          4.90e-16
    40    1.91e-07          4.90e-16
    60    1.60e-01          4.90e-16
    ====  ================  ============

    The equally spaced error bottoms out near ``n = 14`` and then climbs by **thirteen orders of
    magnitude**, ending at 16 percent. The approximation error of a polynomial to ``exp`` goes to
    zero, so every bit of that rise is rounding, amplified by the Lebesgue constant, which grows
    exponentially for equally spaced nodes. Lessons 46 and 47 develop both halves.
    """
    errs = {n: ip.compare_forms(np.exp, n, lo=-1.0, hi=1.0).errors["barycentric"]
            for n in (14, 24, 40, 60)}
    assert errs[24] > errs[14]
    assert errs[40] > errs[24]
    assert errs[60] > 1e-3


@pytest.mark.parametrize("n", [16, 24, 40, 60])
def test_chebyshev_nodes_stay_at_machine_precision_however_high_the_degree(n):
    """The same measurement with the nodes moved. The error stops at rounding and stays there,
    which is what makes high degree interpolation usable at all."""
    err = ip.compare_forms(np.exp, n, lo=-1.0, hi=1.0, chebyshev=True).errors["barycentric"]
    assert err < 1e-14


@pytest.mark.parametrize("n", [10, 20, 30])
def test_chebyshev_nodes_beat_equally_spaced_on_runge(n):
    """Lesson 46's phenomenon and lesson 47's fix, asserted here as a regression guard."""
    runge = lambda t: 1.0 / (1.0 + 25.0 * t * t)
    equal = ip.compare_forms(runge, n, chebyshev=False)
    cheb = ip.compare_forms(runge, n, chebyshev=True)
    assert cheb.errors["barycentric"] < equal.errors["barycentric"]


def test_compare_forms_rejects_a_degenerate_size():
    with pytest.raises(ValueError):
        ip.compare_forms(np.exp, 0)


# ---------------------------------------------------------------- cost


@pytest.mark.parametrize("n", [4, 16, 64, 256])
def test_the_cost_model_ranks_the_forms_the_way_the_lesson_claims(n):
    """Barycentric is linear per point, naive Lagrange is quadratic, and the gap is the reason
    the rearrangement exists."""
    c = ip.interpolation_cost(n)
    assert c["barycentric"] < c["lagrange_naive"]
    assert c["lagrange_naive"] / c["barycentric"] > 0.3 * n
    assert c["setup_barycentric"] < c["setup_power"] or n < 8