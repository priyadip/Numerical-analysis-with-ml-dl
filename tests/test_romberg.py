"""Tests for nalib.romberg.

Four groups. The trapezoid ladder is checked against the composite rule it is supposed to
reproduce, so the evaluation reuse cannot be hiding an error. The table is checked against the
Newton-Cotes rules its columns secretly are, and its column orders are measured. The
Euler-Maclaurin series is checked against a measured trapezoid error, which is the only way to
know the even power structure is real. The failure modes are checked by measuring how much the
extrapolation gains, which is the number that collapses.

Every test sweeps integrands, intervals and level counts.
"""
import math
from fractions import Fraction

import numpy as np
import pytest

from nalib import newtoncotes as nc
from nalib import romberg as rb

INTERVALS = [(0.0, 1.0), (-1.0, 2.0), (0.5, 3.0), (-2.5, -0.25)]
LEVELS = [1, 2, 4, 6, 8]

#: Integrands with an exact antiderivative and an exact ``k``th derivative, both callable.
SMOOTH = {
    "sin": (np.sin,
            lambda a, b: math.cos(a) - math.cos(b),
            lambda x, k: math.sin(x + k * math.pi / 2.0)),
    "exp": (np.exp,
            lambda a, b: math.exp(b) - math.exp(a),
            lambda x, k: math.exp(x)),
    "quartic": (lambda x: np.asarray(x, dtype=float) ** 4 + 2.0 * np.asarray(x, dtype=float),
                lambda a, b: (b ** 5 - a ** 5) / 5.0 + (b ** 2 - a ** 2),
                lambda x, k: (0.0 if k > 4 else
                              [x ** 4 + 2.0 * x, 4.0 * x ** 3 + 2.0,
                               12.0 * x ** 2, 24.0 * x, 24.0][k])),
}

#: Known Bernoulli numbers, so a recurrence error cannot pass.
BERNOULLI = {0: "1", 1: "-1/2", 2: "1/6", 3: "0", 4: "-1/30", 5: "0",
             6: "1/42", 7: "0", 8: "-1/30", 9: "0", 10: "5/66", 12: "-691/2730"}


def sqrt_case():
    return (lambda x: np.sqrt(np.asarray(x, dtype=float))), 2.0 / 3.0, 0.0, 1.0


# --------------------------------------------------------------------------- the ladder


@pytest.mark.parametrize("interval", INTERVALS)
@pytest.mark.parametrize("kind", sorted(SMOOTH))
def test_the_ladder_matches_the_composite_trapezoid_rule(interval, kind):
    a, b = interval
    f = SMOOTH[kind][0]
    ladder = rb.trapezoid_sequence(f, a, b, 6)
    for level in range(ladder.size):
        direct = nc.composite_shared(f, a, b, 2 ** level, 2)
        assert float(ladder[level]) == pytest.approx(direct, rel=1e-13, abs=1e-14)


@pytest.mark.parametrize("levels", [0, 1, 5, 10])
def test_the_reuse_costs_exactly_two_to_the_levels_plus_one(levels):
    out = rb.evaluations_saved(levels)
    assert out["with_reuse"] == 2 ** levels + 1
    assert out["without_reuse"] == sum(2 ** j + 1 for j in range(levels + 1))


def test_the_saving_tends_to_a_factor_of_two():
    ratios = [rb.evaluations_saved(k)["ratio"] for k in (4, 8, 12, 16)]
    assert np.all(np.asarray(ratios) < 2.2)
    assert ratios[-1] == pytest.approx(2.0, abs=0.01)
    # and it is monotone towards 2 from above, never a bigger win at larger sizes
    assert ratios[-1] < ratios[0]


def test_the_ladder_rejects_a_reversed_interval():
    with pytest.raises(ValueError):
        rb.trapezoid_sequence(np.sin, 1.0, 0.0, 4)


# --------------------------------------------------------------------------- the table


@pytest.mark.parametrize("interval", INTERVALS)
@pytest.mark.parametrize("levels", LEVELS)
def test_the_table_is_lower_triangular_with_the_ladder_in_column_zero(interval, levels):
    a, b = interval
    R = rb.table(np.sin, a, b, levels)
    assert R.shape == (levels + 1, levels + 1)
    ladder = rb.trapezoid_sequence(np.sin, a, b, levels)
    assert np.max(np.abs(R[:, 0] - ladder)) == 0.0
    for i in range(levels + 1):
        for j in range(i + 1, levels + 1):
            assert R[i, j] == 0.0


@pytest.mark.parametrize("interval", INTERVALS)
@pytest.mark.parametrize("kind", sorted(SMOOTH))
def test_column_one_is_composite_simpson_and_column_two_is_composite_boole(interval, kind):
    a, b = interval
    f = SMOOTH[kind][0]
    out = rb.column_is_a_newton_cotes_rule(f, a, b, 6)
    assert out["column_1_is_simpson"]
    assert out["column_2_is_boole"]
    assert float(np.max(out["simpson_relative_gap"])) < 1e-12
    assert float(np.max(out["boole_relative_gap"])) < 1e-12


@pytest.mark.parametrize("kind", sorted(SMOOTH))
def test_each_column_has_order_two_j_plus_two(kind):
    f, antiderivative, _ = SMOOTH[kind]
    a, b = 0.0, 1.0
    out = rb.column_orders(f, antiderivative(a, b), a, b, 12)
    fitted = np.asarray(out["fitted_order"], dtype=float)
    predicted = np.asarray(out["predicted_order"], dtype=float)
    used = np.asarray(out["rows_used"])
    checked = 0
    for j in range(fitted.size):
        if math.isnan(fitted[j]):
            assert used[j] < 3
            continue
        assert fitted[j] == pytest.approx(predicted[j], abs=0.3)
        checked += 1
    # the guard must not let every column opt out
    assert checked >= 2


def test_a_polynomial_is_integrated_exactly_by_the_first_column_it_can_reach():
    """The quartic is exact in column 2, which is composite Boole, at every level.

    Nothing beyond that column can improve on an exact answer, so the rest of the row stays
    exact too. This is the check that the extrapolation formula does not damage a value that
    is already right.
    """
    f, antiderivative, _ = SMOOTH["quartic"]
    a, b = 0.0, 1.0
    want = antiderivative(a, b)
    R = rb.table(f, a, b, 6)
    for i in range(2, 7):
        for j in range(2, i + 1):
            assert float(R[i, j]) == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize("interval", INTERVALS)
@pytest.mark.parametrize("kind", sorted(SMOOTH))
def test_integrate_returns_the_corner_and_is_accurate(interval, kind):
    a, b = interval
    f, antiderivative, _ = SMOOTH[kind]
    want = antiderivative(a, b)
    got = rb.integrate(f, a, b, 8)
    assert got == pytest.approx(float(rb.table(f, a, b, 8)[-1, -1]))
    assert got == pytest.approx(want, rel=1e-11, abs=1e-11)


# --------------------------------------------------------------------------- Euler-Maclaurin


@pytest.mark.parametrize("n", sorted(BERNOULLI))
def test_the_bernoulli_numbers_are_the_known_ones(n):
    assert str(rb.bernoulli(n)) == BERNOULLI[n]


@pytest.mark.parametrize("n", [3, 5, 7, 9, 11, 13, 15])
def test_every_odd_bernoulli_number_past_one_is_zero(n):
    assert rb.bernoulli(n) == Fraction(0)


def test_a_negative_index_is_rejected():
    with pytest.raises(ValueError):
        rb.bernoulli(-1)


@pytest.mark.parametrize("kind", ["sin", "exp"])
@pytest.mark.parametrize("interval", INTERVALS)
def test_the_series_predicts_the_measured_trapezoid_error(interval, kind):
    """The check that makes the even power claim more than a quotation.

    Euler-Maclaurin is an **asymptotic** series, so a fixed number of terms is not accurate at
    a fixed panel count: on [-2.5, -0.25] the four term sum is off by 6e-7 relative at 2 panels,
    where the step is still 1.125. What has to happen is that the gap falls as the panels are
    refined, and it does, by four orders of magnitude or more before roundoff in the measured
    error puts a floor under it. Asserting a flat tolerance instead is a test of the interval,
    not of the series.
    """
    a, b = interval
    f, antiderivative, derivatives = SMOOTH[kind]
    out = rb.euler_maclaurin_predicts_the_error(f, derivatives, antiderivative(a, b), a, b)
    gaps = np.asarray(out["relative_gap"], dtype=float)
    assert float(np.min(gaps)) < 1e-11
    assert gaps[0] / float(np.min(gaps)) > 1e4
    # and the fall is monotone until the floor
    assert gaps[1] < gaps[0] and gaps[2] < gaps[1]


@pytest.mark.parametrize("kind", ["sin", "exp"])
def test_the_full_series_beats_the_leading_term(kind):
    f, antiderivative, derivatives = SMOOTH[kind]
    a, b = 0.0, 1.0
    out = rb.euler_maclaurin_predicts_the_error(f, derivatives, antiderivative(a, b), a, b,
                                                panel_counts=[2, 4])
    measured = np.asarray(out["measured_error"], dtype=float)
    leading = np.asarray(out["leading_term"], dtype=float)
    full = np.asarray(out["series_sum"], dtype=float)
    assert np.all(np.abs(full - measured) < np.abs(leading - measured))


@pytest.mark.parametrize("panels", [1, 2, 8, 32])
def test_the_series_uses_only_even_powers(panels):
    out = rb.euler_maclaurin_terms(lambda x, k: math.sin(x + k * math.pi / 2.0),
                                   0.0, 1.0, panels, terms=5)
    assert list(np.asarray(out["power"])) == [2, 4, 6, 8, 10]
    # and the coefficients are B_2k / (2k)!
    for k, c in zip(out["power"], out["coefficient"]):
        assert c == pytest.approx(float(rb.bernoulli(int(k))) / math.factorial(int(k)))


def test_zero_panels_is_rejected():
    with pytest.raises(ValueError):
        rb.euler_maclaurin_terms(lambda x, k: 1.0, 0.0, 1.0, 0)


# --------------------------------------------------------------------------- periodic


def periodic_case():
    """Integral of exp(cos x) over a full period, which is 2 pi I_0(1)."""
    series = sum(1.0 / (4.0 ** k * math.factorial(k) ** 2) for k in range(60))
    return (lambda x: np.exp(np.cos(np.asarray(x, dtype=float)))), 2.0 * math.pi * series


@pytest.mark.parametrize("n", [8, 12, 16, 24, 32])
def test_the_periodic_trapezoid_reaches_machine_precision(n):
    f, want = periodic_case()
    out = rb.periodic_trapezoid(f, want, 0.0, 2.0 * math.pi, [n])
    assert float(out["errors"][0]) < 1e-5 if n < 12 else float(out["errors"][0]) < 1e-10


def test_the_periodic_error_falls_geometrically_not_polynomially():
    f, want = periodic_case()
    out = rb.periodic_convergence_is_geometric(f, want, 0.0, 2.0 * math.pi,
                                               [2, 4, 6, 8, 10, 12, 16, 20])
    assert out["geometric_wins"]
    assert out["geometric_residual"] < out["power_residual"]
    assert out["fitted_rate"] > 0.0


def test_a_geometric_fit_is_refused_when_there_is_nothing_to_fit():
    f, want = periodic_case()
    # every count here is already at the roundoff floor
    out = rb.periodic_convergence_is_geometric(f, want, 0.0, 2.0 * math.pi, [32, 64, 128])
    assert not out["geometric_wins"]
    assert math.isnan(out["geometric_residual"])
    assert out["note"]


def test_extrapolation_makes_the_periodic_case_much_worse():
    """The result that is easy to get backwards.

    On a periodic integrand the Euler-Maclaurin series is identically zero, so the Romberg
    table is extrapolating a series that is not there. It combines coarse rows that are still
    badly wrong, and at equal cost its diagonal is far behind the plain trapezoid rule.
    """
    f, want = periodic_case()
    out = rb.periodic_against_romberg(f, want, 0.0, 2.0 * math.pi, 6)
    plain = np.asarray(out["periodic_trapezoid_error"], dtype=float)
    diagonal = np.asarray(out["romberg_diagonal_error"], dtype=float)
    # at the finest cost the gap is many orders of magnitude, in the trapezoid rule's favour
    assert plain[-1] < diagonal[-1]
    assert diagonal[-1] / max(plain[-1], 1e-300) > 1e4


def test_the_first_column_is_the_plain_trapezoid_rule_over_a_period():
    # the two endpoints carry the same value, so the composite rule collapses to an average
    f, want = periodic_case()
    out = rb.periodic_against_romberg(f, want, 0.0, 2.0 * math.pi, 6)
    a = np.asarray(out["periodic_trapezoid_error"], dtype=float)
    b = np.asarray(out["romberg_first_column_error"], dtype=float)
    assert np.max(np.abs(a - b)) < 1e-12


# --------------------------------------------------------------------------- the limits


def test_extrapolation_gains_orders_on_a_smooth_integrand():
    out = rb.on_a_singularity(np.sin, 1.0 - math.cos(1.0), 0.0, 1.0, 8)
    assert out["first_column_order"] == pytest.approx(2.0, abs=0.3)
    assert out["extrapolation_gained"] > 3.0


def test_and_gains_almost_nothing_without_endpoint_derivatives():
    f, want, a, b = sqrt_case()
    out = rb.on_a_singularity(f, want, a, b, 12)
    assert out["first_column_order"] == pytest.approx(1.5, abs=0.15)
    assert out["extrapolation_gained"] < 0.5


def test_the_stopping_test_never_fires_on_the_singular_case():
    """A Romberg routine running to its level limit is reporting a failed assumption.

    It still returns a usable number here, which is exactly what makes it dangerous: the
    caller who ignores `converged` and `levels_used` sees a good answer bought with a million
    function evaluations, and has no warning when the next integrand is worse.
    """
    f, want, a, b = sqrt_case()
    counter = [0]
    out = rb.to_tolerance(f, a, b, 1e-10, 14, counter)
    assert not out["converged"]
    assert out["levels_used"] == 14
    assert counter[0] == 2 ** 14 + 1


@pytest.mark.parametrize("kind", sorted(SMOOTH))
@pytest.mark.parametrize("interval", INTERVALS)
def test_the_stopping_test_fires_quickly_on_a_smooth_integrand(kind, interval):
    a, b = interval
    f, antiderivative, _ = SMOOTH[kind]
    want = antiderivative(a, b)
    counter = [0]
    out = rb.to_tolerance(f, a, b, 1e-10, 20, counter)
    assert out["converged"]
    assert out["levels_used"] <= 10
    assert out["value"] == pytest.approx(want, rel=1e-8, abs=1e-9)
    assert counter[0] == 2 ** out["levels_used"] + 1


@pytest.mark.parametrize("power", [0.5, 1.0, 2.0])
def test_the_modified_table_reduces_to_the_ordinary_one_at_power_two(power):
    f, want, a, b = sqrt_case()
    M = rb.modified(f, a, b, 6, power)
    if power == 2.0:
        assert np.max(np.abs(M - rb.table(f, a, b, 6))) < 1e-14
    else:
        assert np.max(np.abs(M - rb.table(f, a, b, 6))) > 1e-8


def test_the_right_power_recovers_the_singular_integral():
    f, want, a, b = sqrt_case()
    out = rb.modified_helps_only_with_the_right_power(f, want, a, b, [0.5, 1.0, 1.5, 2.0, 3.0], 10)
    assert out["best_power"] == 0.5
    errors = np.asarray(out["corner_error"], dtype=float)
    best = float(np.min(errors))
    classical = float(errors[list(out["assumed_power"]).index(2.0)])
    # the right power is worth several orders of magnitude over the classical table
    assert best < 1e-11
    assert classical / best > 1e4


def test_the_classical_table_still_beats_doing_nothing_on_the_singular_case():
    # a wrong power is a lost opportunity here, not a catastrophe, and saying so is the point
    f, want, a, b = sqrt_case()
    levels = 10
    R = rb.table(f, a, b, levels)
    plain = abs(float(R[levels, 0]) - want)
    corner = abs(float(R[levels, levels]) - want)
    assert corner < plain
