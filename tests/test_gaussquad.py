"""Tests for nalib.gaussquad.

Five groups. The small rules are checked against their closed forms, which are the only exactly
known answers. Every family is checked against the closed form moments of its own weight
function, so a wrong recurrence coefficient cannot pass. The orthogonality argument is checked
step by step rather than restated. The Kronrod extension is checked against the published
QUADPACK table and against the three properties it must have. The comparisons against
Newton-Cotes and Simpson are made at equal node counts and equal evaluation counts.

Every test sweeps node counts, families and integrands.
"""
import math

import numpy as np
import pytest

from nalib import gaussquad as gq
from nalib import newtoncotes as nc

COUNTS = [1, 2, 3, 4, 5, 8, 12, 16]
FAMILIES = sorted(gq.FAMILIES)
FINITE = ["legendre", "chebyshev_t", "chebyshev_u"]
INTERVALS = [(0.0, 1.0), (-1.0, 2.0), (0.5, 3.0), (-2.5, -0.25)]

#: The Gauss-Legendre rules everyone knows, to full double precision.
KNOWN = {
    1: ([0.0], [2.0]),
    2: ([-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)], [1.0, 1.0]),
    3: ([-math.sqrt(0.6), 0.0, math.sqrt(0.6)], [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]),
}

#: The QUADPACK 15 point Gauss-Kronrod table, positive half, outermost first.
QUADPACK_15 = (
    [0.991455371120813, 0.949107912342759, 0.864864423359769, 0.741531185599394,
     0.586087235467691, 0.405845151377397, 0.207784955007898, 0.000000000000000],
    [0.022935322010529, 0.063092092629979, 0.104790010322250, 0.140653259715525,
     0.169004726639267, 0.190350578064785, 0.204432940075298, 0.209482141084728],
)

SMOOTH = {
    "sin": (np.sin, lambda a, b: math.cos(a) - math.cos(b)),
    "exp": (np.exp, lambda a, b: math.exp(b) - math.exp(a)),
}


# --------------------------------------------------------------------------- nodes and weights


@pytest.mark.parametrize("n", sorted(KNOWN))
def test_the_small_rules_are_the_ones_in_every_textbook(n):
    x, w = gq.nodes_and_weights(n)
    want_x, want_w = KNOWN[n]
    assert np.max(np.abs(x - np.asarray(want_x))) < 1e-14
    assert np.max(np.abs(w - np.asarray(want_w))) < 1e-14


@pytest.mark.parametrize("n", COUNTS)
@pytest.mark.parametrize("family", FAMILIES)
def test_the_weights_sum_to_the_mass_of_the_weight_function(n, family):
    _, _, mass = gq.recurrence(n, family)
    _, w = gq.nodes_and_weights(n, family)
    assert float(np.sum(w)) == pytest.approx(mass, rel=1e-11)


@pytest.mark.parametrize("n", COUNTS)
@pytest.mark.parametrize("family", FAMILIES)
def test_the_nodes_are_distinct_and_ordered(n, family):
    x, _ = gq.nodes_and_weights(n, family)
    assert x.size == n
    if n > 1:
        assert np.all(np.diff(x) > 0.0)


@pytest.mark.parametrize("n", COUNTS)
@pytest.mark.parametrize("family", FINITE)
def test_the_nodes_lie_strictly_inside_the_interval(n, family):
    lo, hi = gq.FAMILIES[family][0]
    x, _ = gq.nodes_and_weights(n, family)
    assert float(np.min(x)) > lo
    assert float(np.max(x)) < hi


@pytest.mark.parametrize("n", COUNTS)
@pytest.mark.parametrize("family", ["legendre", "chebyshev_t", "chebyshev_u", "hermite"])
def test_the_symmetric_families_have_symmetric_nodes(n, family):
    x, w = gq.nodes_and_weights(n, family)
    assert np.max(np.abs(x + x[::-1])) < 1e-12
    assert np.max(np.abs(w - w[::-1])) < 1e-12


def test_an_unknown_family_is_rejected():
    with pytest.raises(ValueError):
        gq.nodes_and_weights(4, "not a family")


def test_zero_nodes_is_rejected():
    with pytest.raises(ValueError):
        gq.nodes_and_weights(0)


@pytest.mark.parametrize("family", ["laguerre", "hermite"])
def test_an_infinite_domain_family_cannot_be_mapped_to_an_interval(family):
    with pytest.raises(ValueError):
        gq.rule_on(4, 0.0, 1.0, family)


def test_a_reversed_interval_is_rejected():
    with pytest.raises(ValueError):
        gq.rule_on(4, 1.0, 0.0)


# --------------------------------------------------------------------------- exactness


@pytest.mark.parametrize("n", COUNTS)
@pytest.mark.parametrize("family", FAMILIES)
def test_every_moment_up_to_degree_two_n_minus_one_is_exact(n, family):
    out = gq.moments_are_exact(n, family)
    assert out["all_exact"], f"{family} n={n} worst gap {out['worst_gap']:.2e}"


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 12, 16])
def test_the_degree_of_precision_is_two_n_minus_one_where_it_can_be_measured(n):
    assert gq.degree_of_precision(n) == 2 * n - 1


def test_the_degree_measurement_stops_working_and_says_so():
    """Past about 16 nodes the boundary between exact and not exact is below roundoff.

    The rule's relative error on ``x^(2n)`` falls to 2.8e-12 by n = 20, so a tolerance based
    walk steps straight over the true degree and reports 41 instead of 39. The report says which
    node counts can still be trusted rather than leaving the reader to find out.
    """
    out = gq.degree_boundary_report()
    nodes = np.asarray(out["nodes"])
    measurable = np.asarray(out["measurable"], dtype=bool)
    measured = np.asarray(out["measured_degree"])
    true = np.asarray(out["true_degree"])
    assert np.all(measured[measurable] == true[measurable])
    assert np.any(~measurable)
    assert np.all(measured[~measurable] > true[~measurable])
    # the margin falls monotonically, which is why the limit exists at all
    assert np.all(np.diff(np.asarray(out["margin"], dtype=float)) < 0.0)
    assert out["last_measurable"] is not None


@pytest.mark.parametrize("interval", INTERVALS)
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_a_mapped_rule_is_exact_on_polynomials_of_degree_two_n_minus_one(interval, n):
    a, b = interval
    rng = np.random.default_rng(42)
    c = rng.standard_normal(2 * n)

    def f(x):
        return sum(c[k] * np.asarray(x, dtype=float) ** k for k in range(c.size))

    exact = sum(c[k] * (b ** (k + 1) - a ** (k + 1)) / (k + 1) for k in range(c.size))
    got = gq.integrate(f, a, b, n)
    assert got == pytest.approx(exact, rel=1e-9, abs=1e-9 * max(1.0, abs(exact)))


@pytest.mark.parametrize("interval", INTERVALS)
@pytest.mark.parametrize("panels", [1, 2, 5, 11])
def test_composite_agrees_with_a_single_rule_on_a_polynomial(interval, panels):
    a, b = interval
    n = 4
    rng = np.random.default_rng(7)
    c = rng.standard_normal(2 * n)

    def f(x):
        return sum(c[k] * np.asarray(x, dtype=float) ** k for k in range(c.size))

    exact = sum(c[k] * (b ** (k + 1) - a ** (k + 1)) / (k + 1) for k in range(c.size))
    got = gq.composite(f, a, b, panels, n)
    assert got == pytest.approx(exact, rel=1e-9, abs=1e-9 * max(1.0, abs(exact)))


def test_zero_panels_is_rejected():
    with pytest.raises(ValueError):
        gq.composite(np.sin, 0.0, 1.0, 0)


# --------------------------------------------------------------------------- why it works


@pytest.mark.parametrize("n", [2, 3, 5, 8, 12])
@pytest.mark.parametrize("family", FINITE)
def test_the_nodes_are_the_roots_of_the_orthogonal_polynomial(n, family):
    out = gq.why_the_roots(n, family)
    assert out["relative_residual_at_the_nodes"] < 1e-11


@pytest.mark.parametrize("n", [2, 3, 5, 8, 12])
@pytest.mark.parametrize("family", FINITE)
def test_the_polynomial_is_orthogonal_to_every_lower_degree(n, family):
    out = gq.why_the_roots(n, family)
    assert float(np.max(out["orthogonality_residuals"])) < 1e-10


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13, 21, 34, 55, 89])
@pytest.mark.parametrize("family", FAMILIES)
def test_no_gauss_weight_is_ever_negative(n, family):
    """The property Newton-Cotes loses at 9 nodes and Gauss never loses.

    Stated as non-negative rather than strictly positive on purpose: in exact arithmetic every
    weight is strictly positive, but the outermost Hermite and Laguerre weights underflow to
    zero in double precision at large node counts. Negative is the thing that would break the
    rule, and that never happens.
    """
    _, w = gq.nodes_and_weights(n, family)
    assert bool(np.all(w >= 0.0))


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13, 21, 34, 55, 89])
@pytest.mark.parametrize("family", FINITE)
def test_the_finite_interval_families_keep_every_weight_strictly_positive(n, family):
    _, w = gq.nodes_and_weights(n, family)
    assert float(np.min(w)) > 0.0


@pytest.mark.parametrize("family", ["hermite", "laguerre"])
def test_the_underflowed_weights_cost_nothing(family):
    """Zero weights at nodes where the weight function is already negligible.

    The Laguerre rule at 120 nodes has five weights that underflowed and a largest node at
    x = 453, where exp(-x) is far below anything representable. The total mass is still right
    to 1e-15, so the rule is unaffected.
    """
    out = gq.weight_underflow(family=family)
    assert not out["any_negative"]
    assert float(np.max(out["total_mass_error"])) < 1e-13
    # the smallest weight does keep shrinking, which is why the zeros appear at all
    smallest = np.asarray(out["smallest_positive_weight"], dtype=float)
    assert np.all(np.diff(smallest) < 0.0)


def test_gauss_weights_stay_positive_where_newton_cotes_weights_do_not():
    g = gq.weights_stay_positive()
    assert g["always_positive"]
    assert g["never_negative"]
    # and the sum of absolute weights never grows, because there is nothing to cancel
    sums = np.asarray(g["sum_of_absolute_weights"], dtype=float)
    assert np.max(np.abs(sums - 2.0)) < 1e-12
    # while Newton-Cotes at the same counts does grow
    assert not bool(np.all(nc.weights(21, True) >= 0.0))
    assert float(np.sum(np.abs(nc.weights(21, True)))) > 100.0


def test_gauss_reaches_about_twice_the_degree_per_node():
    out = gq.degree_against_newton_cotes()
    ratio = np.asarray(out["ratio"], dtype=float)
    assert np.all(ratio > 1.6)
    assert np.all(np.asarray(out["gauss_weights_positive"], dtype=bool))
    counts = np.asarray(out["nodes"])
    positive = np.asarray(out["newton_cotes_weights_positive"], dtype=bool)
    assert np.all(positive[counts <= 8])
    assert not np.any(positive[counts >= 9])


def test_golub_welsch_is_far_better_than_root_finding_in_the_power_basis():
    """Why the Jacobi matrix and not a root finder.

    Expanding the orthogonal polynomial in the power basis spreads its coefficients over 11
    orders of magnitude by n = 32, and the roots are exponentially sensitive to them. The
    eigenvalue route never forms those coefficients and stays at rounding level.
    """
    out = gq.golub_welsch_beats_root_finding([4, 8, 16, 24, 32])
    stable = np.asarray(out["golub_welsch_residual"], dtype=float)
    naive = np.asarray(out["power_basis_residual"], dtype=float)
    spread = np.asarray(out["coefficient_range"], dtype=float)
    assert np.all(stable < 1e-12)
    assert naive[-1] / stable[-1] > 1e6
    assert np.all(np.diff(spread) > 0.0)


# --------------------------------------------------------------------------- convergence


@pytest.mark.parametrize("kind", sorted(SMOOTH))
@pytest.mark.parametrize("interval", INTERVALS)
def test_an_analytic_integrand_converges_geometrically(interval, kind):
    a, b = interval
    f, antiderivative = SMOOTH[kind]
    out = gq.convergence(f, antiderivative(a, b), a, b)
    assert out["geometric_wins"]
    assert out["fitted_rate"] > 0.0


def test_an_endpoint_branch_point_converges_only_as_a_power():
    """Smoothness sets the rate, and Gauss cannot manufacture what is not there."""
    f = lambda x: np.sqrt(np.asarray(x, dtype=float) + 1.0)
    exact = 2.0 / 3.0 * 2.0 ** 1.5
    out = gq.convergence(f, exact, -1.0, 1.0)
    assert not out["geometric_wins"]
    assert out["fitted_power"] == pytest.approx(3.0, abs=0.5)


def test_a_pole_near_the_interval_still_converges_geometrically_but_slowly():
    f = lambda x: 1.0 / (1.0 + 25.0 * np.asarray(x, dtype=float) ** 2)
    exact = 2.0 * math.atan(5.0) / 5.0
    out = gq.convergence(f, exact, -1.0, 1.0)
    assert out["geometric_wins"]
    assert 0.0 < out["fitted_rate"] < 2.0


def test_gauss_beats_simpson_once_the_budget_is_large_enough():
    """And loses at the smallest budgets, which is the honest form of the claim."""
    f = lambda x: 1.0 / (1.0 + 25.0 * np.asarray(x, dtype=float) ** 2)
    exact = 2.0 * math.atan(5.0) / 5.0
    out = gq.against_newton_cotes(f, exact, -1.0, 1.0, [5, 9, 17, 33, 65])
    wins = np.asarray(out["gauss_wins"], dtype=bool)
    assert not wins[0]
    assert bool(np.all(wins[2:]))
    assert float(out["gauss_error"][-1]) < float(out["simpson_error"][-1]) / 1e3


# --------------------------------------------------------------------------- Kronrod


def test_the_fifteen_point_rule_matches_the_quadpack_table():
    """The one place a published table exists, so it is used."""
    want_x, want_w = QUADPACK_15
    x, w = gq.kronrod(7)
    assert x.size == 15
    for i, (qx, qw) in enumerate(zip(want_x, want_w)):
        assert float(x[14 - i]) == pytest.approx(qx, abs=1e-14)
        assert float(w[14 - i]) == pytest.approx(qw, abs=1e-14)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 10, 15, 20])
def test_the_kronrod_rule_has_two_n_plus_one_points_and_contains_the_gauss_nodes(n):
    out = gq.kronrod_report(n)
    assert out["point_count"] == 2 * n + 1
    assert out["gauss_nodes_are_contained"]
    assert out["worst_containment_gap"] < 1e-12


@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 10, 15, 20])
def test_the_kronrod_weights_are_positive_and_sum_to_two(n):
    out = gq.kronrod_report(n)
    assert out["weights_positive"]
    assert out["weight_sum_error"] < 1e-13


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7])
def test_the_kronrod_degree_is_three_n_plus_one_plus_the_parity(n):
    """The half of the statement the usual "at least 3n+1" leaves out.

    An odd Gauss count buys one more degree, which is why the standard pairs are 7 in 15,
    10 in 21 and 15 in 31 rather than the even counts.
    """
    out = gq.kronrod_report(n)
    assert out["predicted_degree"] == 3 * n + 1 + (n % 2)
    assert out["measured_degree"] == out["predicted_degree"]


def test_the_kronrod_degree_measurement_also_runs_out():
    out = gq.kronrod_report(12)
    assert not out["measurable"]
    assert out["measured_degree"] > out["predicted_degree"]


def test_a_zero_point_kronrod_rule_is_rejected():
    with pytest.raises(ValueError):
        gq.kronrod(0)


@pytest.mark.parametrize("interval", INTERVALS)
@pytest.mark.parametrize("n", [3, 7, 10])
def test_the_pair_costs_only_the_kronrod_points(interval, n):
    a, b = interval
    counter = [0]
    out = gq.gauss_kronrod_estimate(np.sin, a, b, n, counter)
    assert counter[0] == 2 * n + 1
    assert out["evaluations"] == 2 * n + 1


@pytest.mark.parametrize("interval", INTERVALS)
@pytest.mark.parametrize("kind", sorted(SMOOTH))
def test_the_kronrod_value_is_the_better_of_the_two(interval, kind):
    a, b = interval
    f, antiderivative = SMOOTH[kind]
    want = antiderivative(a, b)
    out = gq.gauss_kronrod_estimate(f, a, b, 7)
    # both are at machine precision on these integrands, so the comparison is only meaningful
    # above the roundoff of the answer itself
    floor = 8.0 * np.finfo(float).eps * max(abs(want), 1.0)
    assert abs(out["value"] - want) <= abs(out["gauss_value"] - want) + floor


def test_the_difference_estimates_the_gauss_error_not_the_returned_error():
    """The thing a Gauss-Kronrod pair is quietly doing.

    The difference between the two rules is an excellent estimate of the **Gauss** error, and
    the value returned is the **Kronrod** one, which is far better. At 3 Gauss points on sin
    over [0, 1] the difference matches the Gauss error to three digits while the returned
    answer is already at machine precision, so the reported error is an overestimate by eight
    orders of magnitude. Safe, and not what it looks like.
    """
    out = gq.kronrod_estimate_quality(np.sin, 1.0 - math.cos(1.0), 0.0, 1.0, [3, 5])
    over_gauss = np.asarray(out["difference_over_gauss_error"], dtype=float)
    over_kronrod = np.asarray(out["difference_over_kronrod_error"], dtype=float)
    assert np.all(np.abs(over_gauss - 1.0) < 0.2)
    assert float(np.max(over_kronrod)) > 1e5


def test_the_ratio_is_floored_at_roundoff_rather_than_at_the_smallest_float():
    out = gq.kronrod_estimate_quality(np.sin, 1.0 - math.cos(1.0), 0.0, 1.0, [3, 5, 7, 10, 15])
    ratios = np.asarray(out["difference_over_kronrod_error"], dtype=float)
    assert np.all(np.isfinite(ratios))
    # dividing by 1e-300 instead would produce numbers around 1e287
    assert float(np.max(ratios)) < 1e20
    assert out["roundoff_floor"] > 0.0
