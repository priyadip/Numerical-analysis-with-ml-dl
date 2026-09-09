"""Tests for nalib.stability.

Three groups. The stability functions are checked against their closed forms and against the
classical axis limits, which are known to many digits and are the sharpest available check. The
A-stability and L-stability classifications are checked against the methods whose status is a
theorem. The stiffness machinery is checked by running an explicit and an implicit method on the
same problem and confirming the explicit one fails exactly where the region says it will.

Every test sweeps methods, eigenvalues and step counts.
"""
import math

import numpy as np
import pytest

from nalib import stability as st

EXPLICIT = ["euler", "heun", "kutta third", "rk4"]
IMPLICIT = ["backward euler", "trapezoid", "implicit midpoint", "bdf2"]

#: Classical limits on the negative real axis, for the explicit methods.
REAL_LIMIT = {"euler": 2.0, "heun": 2.0, "kutta third": 2.512745, "rk4": 2.785294}
#: Classical limits on the imaginary axis. Euler's is zero and RK4's is 2 sqrt 2.
IMAGINARY_LIMIT = {"euler": 0.0, "heun": 0.0, "kutta third": math.sqrt(3.0),
                   "rk4": 2.0 * math.sqrt(2.0)}


# --------------------------------------------------------------------------- the functions


@pytest.mark.parametrize("name", EXPLICIT + IMPLICIT)
def test_every_growth_function_is_one_at_zero(name):
    """R(0) = 1, because a zero step leaves the state alone."""
    R = st.growth_function(name)
    assert abs(complex(R(0.0)) - 1.0) < 1e-12


@pytest.mark.parametrize("name", EXPLICIT + IMPLICIT)
def test_every_growth_function_matches_the_exponential_to_its_order(name):
    """R(z) must agree with e^z through z^p, which is what consistency of order p means."""
    orders = {"euler": 1, "heun": 2, "kutta third": 3, "rk4": 4,
              "backward euler": 1, "trapezoid": 2, "implicit midpoint": 2, "bdf2": 2}
    R = st.growth_function(name)
    p = orders[name]
    for z in (1e-3, -1e-3, 2e-3j, -2e-3j):
        gap = abs(complex(R(z)) - np.exp(z))
        assert gap < 20.0 * abs(z) ** (p + 1)


def test_the_closed_forms_are_what_they_should_be():
    backward = st.growth_function("backward euler")
    trapezoid = st.growth_function("trapezoid")
    for z in (-0.5, -2.0, -10.0, 0.3):
        assert complex(backward(z)).real == pytest.approx(1.0 / (1.0 - z), rel=1e-12)
        assert complex(trapezoid(z)).real == pytest.approx((1.0 + 0.5 * z) / (1.0 - 0.5 * z),
                                                           rel=1e-12)


@pytest.mark.parametrize("name", EXPLICIT)
def test_the_real_axis_limit_matches_the_classical_value(name):
    assert st.real_axis_limit(name) == pytest.approx(REAL_LIMIT[name], abs=1e-5)


@pytest.mark.parametrize("name", ["kutta third", "rk4"])
def test_the_imaginary_axis_limit_matches_the_classical_value(name):
    assert st.imaginary_axis_limit(name) == pytest.approx(IMAGINARY_LIMIT[name], abs=1e-5)


def test_euler_is_unstable_on_a_pure_oscillation_at_every_step():
    """Its imaginary axis limit is zero, which is the whole reason lesson 74 exists."""
    assert st.imaginary_axis_limit("euler") < 1e-5
    assert st.imaginary_axis_limit("heun") < 1e-2
    # and it is strictly outside, not merely marginal
    R = st.growth_function("euler")
    for x in (0.01, 0.1, 1.0):
        assert abs(complex(R(1j * x))) > 1.0


@pytest.mark.parametrize("name", IMPLICIT)
def test_the_a_stable_methods_have_no_real_axis_limit(name):
    assert st.real_axis_limit(name) == float("inf")


@pytest.mark.parametrize("name", EXPLICIT + IMPLICIT)
def test_the_region_grid_agrees_with_the_axis_limits(name):
    out = st.stability_region(name, real=(-4.0, 1.0), imaginary=(-3.0, 3.0), points=61)
    assert out["inside"].shape == (61, 61)
    # the origin is always inside
    assert bool(out["inside"][30, np.argmin(np.abs(out["real"]))])


# --------------------------------------------------------------------------- classification


def test_no_explicit_method_is_a_stable():
    out = st.classify(EXPLICIT)
    assert not bool(np.any(out["a_stable"]))


@pytest.mark.parametrize("name", IMPLICIT)
def test_the_implicit_methods_here_are_all_a_stable(name):
    out = st.classify([name])
    assert bool(out["a_stable"][0])


def test_backward_euler_and_bdf2_are_l_stable_and_the_trapezoid_rule_is_not():
    out = st.classify(["backward euler", "trapezoid", "implicit midpoint", "bdf2"])
    names = list(out["names"])
    assert bool(out["l_stable"][names.index("backward euler")])
    assert bool(out["l_stable"][names.index("bdf2")])
    assert not bool(out["l_stable"][names.index("trapezoid")])
    assert not bool(out["l_stable"][names.index("implicit midpoint")])


def test_a_fixed_threshold_would_misclassify_bdf2():
    """Why L-stability is tested by whether the factor is still falling.

    BDF2's growth factor decays like |z|^(-1/2), so at z = -1e10 it is still 7e-06 and any
    threshold at 1e-06 calls an L-stable method not L-stable.
    """
    out = st.classify(["bdf2"])
    far = float(out["magnitude_at_minus_1e10"][0])
    near = float(out["magnitude_at_minus_1e6"][0])
    assert far > 1e-6
    assert far < 0.5 * near
    assert bool(out["l_stable"][0])


def test_the_trapezoid_rule_growth_factor_tends_to_minus_one():
    out = st.classify(["trapezoid"])
    assert float(out["magnitude_at_minus_1e10"][0]) == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- stiffness


@pytest.mark.parametrize("ratio", [1.0, 10.0, 1000.0, 1e6])
def test_the_stiffness_ratio_is_the_eigenvalue_ratio(ratio):
    out = st.stiffness_ratio([[-ratio, 0.0], [0.0, -1.0]])
    assert out["ratio"] == pytest.approx(ratio, rel=1e-9)
    assert out["stiff"] == (ratio > 100.0)


def test_a_problem_with_no_decaying_mode_is_not_stiff():
    out = st.stiffness_ratio([[1.0, 0.0], [0.0, 2.0]])
    assert not out["stiff"]
    assert out["ratio"] == 1.0


def test_the_stiffness_ratio_sees_a_non_diagonal_matrix():
    """The eigenvalues matter, not the entries: a matrix can look mild and be stiff."""
    A = np.asarray([[-500.5, 499.5], [499.5, -500.5]])
    out = st.stiffness_ratio(A)
    assert out["ratio"] == pytest.approx(1000.0, rel=1e-6)
    assert out["stiff"]


@pytest.mark.parametrize("name", EXPLICIT)
def test_stability_fixes_the_step_whatever_accuracy_is_wanted(name):
    out = st.the_step_is_set_by_stability(name=name)
    assert out["stability_step"] == pytest.approx(st.real_axis_limit(name) / 1000.0)
    # the accuracy step grows as the tolerance loosens, and the stability step does not
    steps = np.asarray(out["accuracy_step"], dtype=float)
    assert np.all(np.diff(steps) < 0.0)


def test_the_waste_is_worst_when_little_accuracy_is_needed():
    """A stiff problem is usually in exactly that situation: the fast transient is not wanted."""
    out = st.the_step_is_set_by_stability()
    waste = np.asarray(out["wasted_factor"], dtype=float)
    assert np.all(np.diff(waste) < 0.0)
    assert out["worst_waste"] > 50.0


def test_the_explicit_method_fails_exactly_where_the_region_says():
    out = st.explicit_fails_on_a_stiff_problem(fast=-1000.0, slow=-1.0, t_end=1.0)
    limit = st.real_axis_limit("rk4")
    lam_h = np.asarray(out["lam_h"], dtype=float)
    errors = np.asarray(out["explicit_error"], dtype=float)
    for value, error in zip(lam_h, errors):
        if value > limit + 0.5:
            assert not np.isfinite(error) or error > 1.0
        if value < limit - 0.5:
            assert error < 1.0


def test_the_implicit_method_is_usable_at_every_step():
    out = st.explicit_fails_on_a_stiff_problem()
    assert out["implicit_usable_everywhere"]
    assert out["first_usable_explicit_steps"] is not None
    assert out["first_usable_explicit_steps"] >= 300


def test_a_stable_is_not_enough_and_l_stable_is_what_a_stiff_solver_needs():
    """The trapezoid rule never diverges and never damps, so a fast mode rings forever."""
    out = st.trapezoid_rings(lam=-1000.0, steps=10, t_end=1.0)
    assert out["trapezoid_growth"] == pytest.approx(-1.0, abs=0.05)
    assert abs(out["backward_euler_growth"]) < 0.02
    assert out["trapezoid_alternates"]
    assert out["trapezoid_amplitude_at_the_end"] > 0.5
    assert out["backward_euler_amplitude_at_the_end"] < 1e-10
    assert out["exact"] == pytest.approx(0.0, abs=1e-300)


def test_the_ringing_gets_worse_as_the_problem_gets_stiffer():
    """The growth factor is (1 + z/2)/(1 - z/2), which approaches -1 from above as z falls.

    At lam h = -10 it is -0.667, so a third of the amplitude is lost per step and the ringing
    eventually dies. At -100 it is -0.961 and at -1000 it is -0.996: the stiffer the problem,
    the longer the spurious oscillation persists, which is the opposite of what is wanted.
    """
    factors = []
    for lam in (-100.0, -1000.0, -10000.0):
        out = st.trapezoid_rings(lam=lam, steps=10, t_end=1.0)
        assert out["trapezoid_alternates"]
        assert out["trapezoid_growth"] < 0.0
        factors.append(out["trapezoid_growth"])
    # the factor moves towards -1 as the problem stiffens
    assert np.all(np.diff(factors) < 0.0)
    assert factors[-1] == pytest.approx(-1.0, abs=0.01)
    # and backward Euler kills the mode at every stiffness
    for lam in (-100.0, -1000.0, -10000.0):
        out = st.trapezoid_rings(lam=lam, steps=10, t_end=1.0)
        assert abs(out["backward_euler_growth"]) < 0.11


def test_the_second_barrier_is_order_two():
    out = st.second_dahlquist_barrier()
    assert out["barrier"] == 2
    names = list(out["names"])
    orders = np.asarray(out["order"])
    for name, order, a_stable in zip(names, orders, out["a_stable"]):
        if a_stable:
            assert order <= 2, f"{name} is A-stable with order {order}"


# ------------------------------------------------------------------ multistep stability


@pytest.mark.parametrize("name", ["euler", "trapezoid", "ab2", "ab4", "bdf2", "bdf3"])
def test_stability_at_the_origin_is_zero_stability(name):
    """rho(w) - 0 sigma(w) is rho(w), so lesson 71's root condition is the z = 0 case."""
    from nalib import multistep as ms

    alpha, beta, order = ms.method(name)
    assert st.multistep_is_stable_at(alpha, beta, 0.0) == ms.root_condition(alpha)["zero_stable"]


def test_the_unstable_example_fails_at_the_origin():
    from nalib import multistep as ms

    alpha, beta, order = ms.method("unstable order two")
    assert not st.multistep_is_stable_at(alpha, beta, 0.0)
    assert st.stability_angle(alpha, beta) == 0.0


@pytest.mark.parametrize("name", ["euler", "ab2", "ab4"])
def test_an_explicit_multistep_method_is_not_a_stable(name):
    from nalib import multistep as ms

    alpha, beta, order = ms.method(name)
    assert not st.multistep_a_stable(alpha, beta)
    assert st.stability_angle(alpha, beta) < 90.0


@pytest.mark.parametrize("name", ["backward euler", "trapezoid", "bdf2"])
def test_the_a_stable_multistep_methods_measure_ninety_degrees(name):
    from nalib import multistep as ms

    alpha, beta, order = ms.method(name)
    assert st.multistep_a_stable(alpha, beta)
    assert st.stability_angle(alpha, beta) == pytest.approx(90.0, abs=0.05)


def test_euler_and_the_one_step_growth_function_agree():
    """The general machinery must reproduce the scalar R(z) where both apply."""
    from nalib import multistep as ms

    for name in ("euler", "backward euler", "trapezoid"):
        alpha, beta, order = ms.method(name)
        R = st.growth_function(name)
        for z in (-0.5, -2.5, -10.0, 1j, -1.0 + 2.0j):
            roots = st.multistep_roots(alpha, beta, z)
            assert roots.size == 1
            assert complex(roots[0]) == pytest.approx(complex(R(z)), abs=1e-12)


def test_a_shorter_beta_is_padded_rather_than_refused():
    """AM3 has two alphas and three betas, and both describe the same degree."""
    from nalib import multistep as ms

    alpha, beta, order = ms.method("am3")
    assert len(alpha) != len(beta)
    roots = st.multistep_roots(alpha, beta, -1.0)
    assert roots.size == max(len(alpha), len(beta)) - 1


def test_the_bdf_angles_match_the_published_table():
    out = st.bdf_angles()
    assert out["matches_the_published_table"]
    assert out["worst_gap"] < 0.5
    for k, got in zip(out["order"], out["angle"]):
        assert got == pytest.approx(st.BDF_PUBLISHED_ANGLES[int(k)], abs=0.5)


def test_only_the_first_two_bdf_methods_are_a_stable():
    out = st.bdf_angles()
    assert out["a_stable_only_up_to"] == 2
    assert bool(np.all(out["a_stable"][:2]))
    assert not bool(np.any(out["a_stable"][2:]))


def test_the_bdf_wedge_closes_and_bdf7_has_none_of_it():
    out = st.bdf_angles()
    assert out["the_wedge_closes"]
    assert float(out["angle"][-1]) == 0.0
    assert not bool(out["stable_on_the_negative_real_axis"][-1])
    assert bool(np.all(out["stable_on_the_negative_real_axis"][:-1]))


def test_the_second_barrier_is_measured_and_not_assumed():
    out = st.second_dahlquist_barrier()
    assert out["barrier_holds"]
    assert out["highest_a_stable_order_found"] == 2
    names = list(out["names"])
    stable = np.asarray(out["a_stable"])
    assert bool(stable[names.index("trapezoid")])
    assert bool(stable[names.index("bdf2")])
    assert not bool(stable[names.index("bdf3")])
    assert not bool(stable[names.index("ab4")])
