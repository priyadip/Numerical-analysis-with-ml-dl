"""Tests for `nalib.equalinterval`.

The claim these formulas exist to support is that they are all the same polynomial. That is
tested where each can be formed at full order, and the place where it stops being true is tested
separately, because that is the actual reason both the diagonal and the central families exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from nalib import equalinterval as ei, interp as ip


SIZES = [5, 7, 9, 11, 13]
SPANS = [(0.0, 1.0), (-1.0, 1.0), (2.0, 8.0), (-0.5, -0.1)]
FUNCS = [("exp", lambda t: np.exp(2.0 * t)), ("sin", np.sin),
         ("quartic", lambda t: 1.0 - 2.0 * t + t ** 2 - 0.5 * t ** 3 + 0.25 * t ** 4)]


def grid(lo, hi, n):
    return np.linspace(lo, hi, n)


# ---------------------------------------------------------------- the grid hypothesis


def test_an_unequally_spaced_grid_is_rejected_with_a_pointer_to_the_general_code():
    """These formulas are the equally spaced special case and say so rather than returning a
    plausible wrong answer."""
    x = np.array([0.0, 0.1, 0.4, 1.0])
    with pytest.raises(ValueError, match="EQUALLY"):
        ei.gregory_newton_forward(x, np.exp(x), 0.5)


def test_a_single_node_is_rejected():
    with pytest.raises(ValueError):
        ei.gregory_newton_forward([1.0], [2.0], 1.0)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        ei.gregory_newton_forward([0.0, 1.0, 2.0], [1.0, 2.0], 0.5)


# ---------------------------------------------------------------- the real binomial


@pytest.mark.parametrize("k", [0, 1, 2, 3, 5])
def test_the_generalised_binomial_matches_the_integer_one(k):
    """``C(s, k)`` for real ``s`` must agree with the integer coefficient at integer ``s``."""
    import math

    for s in range(k, k + 6):
        assert abs(float(ei.binomial_s(float(s), k)) - math.comb(s, k)) < 1e-9


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_the_generalised_binomial_vanishes_at_the_right_integers(k):
    """``C(s, k)`` is zero at ``s = 0, 1, ..., k-1``, which is what makes each formula reproduce
    the data at the nodes."""
    for s in range(k):
        assert abs(float(ei.binomial_s(float(s), k))) < 1e-12


# ---------------------------------------------------------------- reproducing the data


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("lo,hi", SPANS)
def test_the_diagonal_formulas_reproduce_the_data_at_every_node(n, lo, hi):
    """Gregory-Newton forward and backward always reach full degree, so they interpolate."""
    x = grid(lo, hi, n)
    y = np.exp(x)
    scale = float(np.max(np.abs(y)))
    for fn in (ei.gregory_newton_forward, ei.gregory_newton_backward):
        got = np.asarray([float(np.atleast_1d(fn(x, y, t))[0]) for t in x])
        assert float(np.max(np.abs(got - y))) < 1e-8 * scale


@pytest.mark.parametrize("name,f", FUNCS)
@pytest.mark.parametrize("n", SIZES)
def test_the_two_diagonal_formulas_agree_everywhere(name, f, n):
    """**They always reach full degree**, so they are the same polynomial at every point, at
    every size, measured at 2e-15 or better."""
    x = grid(0.0, 1.0, n)
    y = np.asarray([f(v) for v in x])
    for t in np.linspace(0.02, 0.98, 13):
        out = ei.all_formulas_agree(x, y, t)
        assert out["diagonal_spread"] < 1e-10 * max(float(np.max(np.abs(y))), 1.0)


@pytest.mark.parametrize("n", [5, 9, 13])
@pytest.mark.parametrize("name,f", FUNCS)
def test_every_formula_agrees_where_the_central_ones_reach_full_degree(n, name, f):
    """At the middle of the table all eight are the same polynomial. Measured at 0.00e+00 for
    ``exp(2x)`` on nine nodes at ``t = 0.25``, ``0.5`` and ``0.75``."""
    x = grid(0.0, 1.0, n)
    y = np.asarray([f(v) for v in x])
    out = ei.all_formulas_agree(x, y, 0.5)
    assert out["central_formulas_reach_full_degree"]
    assert out["relative_spread"] < 1e-12


def test_the_central_formulas_cannot_be_formed_near_the_ends():
    """**The reason both families exist.** A central expansion needs differences on both sides of
    its node, so at the first and last node it reaches order 0. Measured on nine nodes over
    ``[0, 1]`` with ``f = exp(2x)``:

    =======  ============  ============  ============  ============================
    ``t``    centre node   room below    room above    highest central order usable
    =======  ============  ============  ============  ============================
    0.06     0             0             8             0
    0.25     2             2             6             4
    0.50     4             4             4             8
    0.94     8             8             0             0
    =======  ============  ============  ============  ============================

    At ``t = 0.94`` the central formulas err by 8.4e-1 while Gregory-Newton backward, which
    reads a diagonal instead of a centred band, errs by 1.3e-7, and that 1.3e-7 is the
    interpolation error of the degree 8 polynomial rather than any defect of the formula.
    """
    x = grid(0.0, 1.0, 9)
    y = np.exp(2.0 * x)
    assert ei.usable_central_order(x, 0.06) == 0
    assert ei.usable_central_order(x, 0.5) == 8
    assert ei.usable_central_order(x, 0.94) == 0

    near_end = ei.all_formulas_agree(x, y, 0.94)
    assert not near_end["central_formulas_reach_full_degree"]
    assert near_end["diagonal_spread"] < 1e-12
    assert near_end["spread"] > 1e-3

    truth = np.exp(2.0 * 0.94)
    diag_err = abs(near_end["values"]["gregory-newton backward"] - truth)
    cent_err = abs(near_end["values"]["gauss forward"] - truth)
    assert diag_err < 1e-5
    assert cent_err > diag_err


@pytest.mark.parametrize("n", [7, 9, 11, 15])
def test_the_usable_central_order_is_twice_the_distance_to_the_nearer_end(n):
    x = grid(0.0, 1.0, n)
    for i, t in enumerate(x):
        assert ei.usable_central_order(x, t) == 2 * min(i, n - 1 - i)


# ---------------------------------------------------------------- against the general code


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("lo,hi", SPANS)
def test_the_diagonal_formulas_match_the_general_interpolant(n, lo, hi):
    """They are the Newton form on an equally spaced grid, so they must agree with lesson 44's
    barycentric evaluation of the same data."""
    x = grid(lo, hi, n)
    y = np.sin(3.0 * x)
    probe = np.linspace(lo, hi, 17)
    ref = ip.evaluate_barycentric(x, y, probe)
    got = np.asarray([float(np.atleast_1d(ei.gregory_newton_forward(x, y, t))[0])
                      for t in probe])
    assert float(np.max(np.abs(got - ref))) < 1e-8 * max(float(np.max(np.abs(ref))), 1.0)


# ---------------------------------------------------------------- Everett and Steffensen


@pytest.mark.parametrize("n", [7, 9, 11, 13])
def test_everett_and_steffensen_agree_exactly_at_the_centre_of_the_table(n):
    """They are duals, each implemented from its own formula rather than one calling the other,
    so agreement is evidence and not a tautology. At the centre both reach full degree and the
    measured difference is exactly zero."""
    x = grid(0.0, 1.0, n)
    y = np.exp(2.0 * x)
    a = float(np.atleast_1d(ei.everett(x, y, 0.5))[0])
    b = float(np.atleast_1d(ei.steffensen(x, y, 0.5))[0])
    ref = float(ip.evaluate_barycentric(x, y, 0.5))
    assert a == b
    assert abs(a - ref) < 1e-12


@pytest.mark.parametrize("n", [7, 9, 11, 13])
def test_everett_favours_the_left_half_and_steffensen_the_right(n):
    """**They truncate differently, and the asymmetry is systematic.** Away from the centre
    neither reaches full degree, and Steffensen's extra odd term is anchored at the right hand
    node. Errors against the true interpolant of ``exp(2x)``:

    ====  ========  ==============  =================
    n     t         Everett         Steffensen
    ====  ========  ==============  =================
    9     0.32      9.42e-7         2.22e-6
    9     0.50      0.00e+0         0.00e+0
    9     0.68      2.55e-6         5.07e-7
    13    0.32      2.39e-10        5.72e-10
    13    0.68      6.51e-10        9.31e-11
    ====  ========  ==============  =================

    Both are correct formulas. Which is more accurate depends on which side of the centre the
    point sits, and that is a genuine reason to keep both rather than a defect in either.
    """
    x = grid(0.0, 1.0, n)
    y = np.exp(2.0 * x)
    err = lambda fn, t: abs(float(np.atleast_1d(fn(x, y, t))[0])
                            - float(ip.evaluate_barycentric(x, y, t)))
    assert err(ei.everett, 0.32) < err(ei.steffensen, 0.32)
    assert err(ei.steffensen, 0.68) < err(ei.everett, 0.68)


@pytest.mark.parametrize("n", [9, 11, 13])
def test_everett_uses_only_even_differences_and_still_interpolates(n):
    """Half the table is never read, which is what the formula was worth a name for."""
    x = grid(0.0, 1.0, n)
    y = np.exp(2.0 * x)
    mid = n // 2
    for t in (x[mid], x[mid] + 0.5 * (x[1] - x[0])):
        got = float(np.atleast_1d(ei.everett(x, y, t))[0])
        ref = float(ip.evaluate_barycentric(x, y, t))
        assert abs(got - ref) < 1e-8 * max(abs(ref), 1.0)


# ---------------------------------------------------------------- the choice rule


@pytest.mark.parametrize("n", [9, 13, 21])
def test_the_recommendation_rule_covers_the_whole_table(n):
    for s in np.linspace(0.0, n - 1.0, 61):
        assert ei.best_formula_for(float(s), n) in ei.FORMULAS


@pytest.mark.parametrize("n", [9, 13])
def test_the_rule_picks_a_diagonal_formula_at_the_ends_and_a_central_one_in_the_middle(n):
    assert ei.best_formula_for(0.3, n) == "gregory-newton forward"
    assert ei.best_formula_for(n - 1.4, n) == "gregory-newton backward"
    assert ei.best_formula_for(n / 2.0, n) in ("stirling", "bessel")


def test_the_rule_rejects_a_degenerate_table():
    with pytest.raises(ValueError):
        ei.best_formula_for(0.5, 1)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_the_position_report_runs_and_names_a_winner_everywhere(order):
    """At a fixed truncation order the formulas genuinely differ, which is the only regime in
    which the classical advice has content."""
    rep = ei.error_by_position(np.exp, 11, 0.0, 1.0, order=order, n_probe=21)
    assert set(rep.errors) == set(ei.FORMULAS)
    assert len(rep.best_at) == 21
    for name in rep.best_at.values():
        assert name in ei.FORMULAS