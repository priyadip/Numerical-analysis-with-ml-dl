"""Tests for nalib.symplectic.

Three groups. The steps are checked against their closed forms and against the exact solution of
the harmonic oscillator, because a symplectic method is still an ordinary method and must have
the order it claims. The **structure** is checked against the definition, which is that the one
step map has Jacobian determinant 1: that is an identity a symplectic method satisfies for every
step and every state, and a non symplectic one misses by O(h^2). The long run behaviour is checked
for the property that actually separates the methods, which is the presence of a **trend** rather
than the size of the error.

The most useful test here is the last group's: at equal cost RK4 has a *smaller* energy band than
Stormer-Verlet and is still the worse method for a long integration, because its band moves and
Verlet's does not. Any test that only compared error sizes would rank them the wrong way round.
"""
import math

import numpy as np
import pytest

from nalib import symplectic as sy

NAMES = ["explicit euler", "symplectic euler", "stormer verlet"]
SYMPLECTIC = ["symplectic euler", "stormer verlet"]


# --------------------------------------------------------------------------- the steps


def test_the_pair_helper_refuses_mismatched_shapes():
    with pytest.raises(ValueError):
        sy.as_pair([1.0, 2.0], [1.0])


@pytest.mark.parametrize("size", [1, 2, 3, 5])
def test_the_pair_helper_keeps_the_dimension(size):
    q, p = sy.as_pair(np.arange(size), np.ones(size))
    assert q.shape == (size,) and p.shape == (size,)
    assert q.dtype == float and p.dtype == float


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError):
        sy.step_named("verlet-ish")


@pytest.mark.parametrize("h", [0.5, 0.1, 0.01])
def test_symplectic_euler_uses_the_new_momentum_and_explicit_euler_the_old(h):
    """The two methods differ in one place, and this is that place."""
    force, energy, exact = sy.harmonic_oscillator(omega=1.3)
    q, p = np.asarray([0.7]), np.asarray([-0.4])
    sq, sp = sy.symplectic_euler_step(force, q, p, h)
    eq, ep = sy.explicit_euler_step(force, q, p, h)
    # the momenta agree exactly, because both use force(q_old)
    assert float(sp[0]) == pytest.approx(float(ep[0]), rel=1e-15)
    # the positions differ by exactly h times the momentum change
    assert float(sq[0]) - float(eq[0]) == pytest.approx(h * (float(sp[0]) - float(p[0])),
                                                        rel=1e-13)


@pytest.mark.parametrize("h", [0.4, 0.1, 0.02])
def test_stormer_verlet_is_time_reversible(h):
    """Stepping forward then backward with the same h returns to the start, to roundoff.

    Time reversibility is a separate structural property from symplecticity and neither implies
    the other, but Verlet has both and explicit Euler has neither.
    """
    force, energy = sy.pendulum()
    q, p = np.asarray([0.9]), np.asarray([0.3])
    fq, fp = sy.stormer_verlet_step(force, q, p, h)
    bq, bp = sy.stormer_verlet_step(force, fq, fp, -h)
    assert float(bq[0]) == pytest.approx(float(q[0]), abs=1e-13)
    assert float(bp[0]) == pytest.approx(float(p[0]), abs=1e-13)


def test_explicit_euler_is_not_time_reversible():
    force, energy = sy.pendulum()
    q, p = np.asarray([0.9]), np.asarray([0.3])
    fq, fp = sy.explicit_euler_step(force, q, p, 0.4)
    bq, bp = sy.explicit_euler_step(force, fq, fp, -0.4)
    assert abs(float(bq[0]) - float(q[0])) > 1e-3


@pytest.mark.parametrize("name", NAMES)
def test_every_method_has_the_order_it_claims(name):
    out = sy.order_of(name)
    assert out["matches"], f"{name} fitted {out['fitted_order']} claimed {out['claimed_order']}"


@pytest.mark.parametrize("name", NAMES)
def test_a_zero_step_count_is_refused(name):
    force, energy, exact = sy.harmonic_oscillator()
    with pytest.raises(ValueError):
        sy.integrate(force, [1.0], [0.0], 1.0, 0, name)


@pytest.mark.parametrize("size", [1, 2, 4])
def test_the_integrator_handles_any_number_of_degrees_of_freedom(size):
    rng = np.random.default_rng(42)

    def force(q):
        return -np.asarray(q, dtype=float)

    q0 = rng.normal(size=size)
    p0 = rng.normal(size=size)
    out = sy.integrate(force, q0, p0, 2.0, 200, "stormer verlet")
    assert out["q"].shape == (201, size)
    assert out["p"].shape == (201, size)
    # each component is an independent oscillator, so each must track its own closed form
    for j in range(size):
        want = q0[j] * math.cos(2.0) + p0[j] * math.sin(2.0)
        assert float(out["q"][-1, j]) == pytest.approx(want, abs=1e-4)


# --------------------------------------------------------------------------- the structure


@pytest.mark.parametrize("name", SYMPLECTIC)
@pytest.mark.parametrize("h", [0.05, 0.2, 0.5])
def test_a_symplectic_step_preserves_phase_space_area_at_every_step_size(name, h):
    """This is the definition, and it holds for a large step as well as a small one."""
    out = sy.area_is_preserved(name, h=h, corners=60)
    assert out["preserves_area"]
    assert out["worst_departure_from_one"] < 1e-6


@pytest.mark.parametrize("h", [0.05, 0.2, 0.5])
def test_explicit_euler_inflates_the_area_by_exactly_h_squared(h):
    """The determinant of its one step map on the oscillator is 1 + h^2, which is why it gains
    energy: every step stretches phase space by the same factor."""
    out = sy.area_is_preserved("explicit euler", h=h, corners=60)
    assert not out["preserves_area"]
    assert out["worst_departure_from_one"] == pytest.approx(h ** 2, rel=1e-4)
    assert out["expected_for_explicit_euler"] == pytest.approx(1.0 + h ** 2, rel=1e-15)


def test_the_area_error_is_far_above_the_differencing_noise():
    """The measurement uses finite differences, so the gap between the two must be large."""
    good = sy.area_is_preserved("stormer verlet", h=0.2, corners=60)
    bad = sy.area_is_preserved("explicit euler", h=0.2, corners=60)
    assert bad["worst_departure_from_one"] > 1e5 * good["worst_departure_from_one"]


# --------------------------------------------------------------------------- the long run


def test_the_energy_is_bounded_for_the_symplectic_methods_and_not_for_euler():
    out = sy.drift_against_bounded()
    names = list(out["names"])
    assert not bool(out["bounded"][names.index("explicit euler")])
    for name in SYMPLECTIC:
        assert bool(out["bounded"][names.index(name)]), name


def test_explicit_euler_gains_energy_without_limit():
    out = sy.drift_against_bounded()
    names = list(out["names"])
    i = names.index("explicit euler")
    assert float(out["trend_per_unit_time"][i]) > 0.0
    assert float(out["final_relative_error"][i]) > 1.0


def test_the_verlet_band_is_much_narrower_than_the_symplectic_euler_band():
    """Both are bounded, and the second order method's band is the smaller one."""
    out = sy.drift_against_bounded()
    names = list(out["names"])
    narrow = float(out["band"][names.index("stormer verlet")])
    wide = float(out["band"][names.index("symplectic euler")])
    assert narrow < 0.05 * wide


@pytest.mark.parametrize("t_end", [50.0, 100.0, 200.0])
def test_the_band_does_not_grow_with_the_length_of_the_run(t_end):
    """A bounded error stays bounded, which is the only claim the theory makes."""
    force, energy = sy.pendulum()
    steps = int(t_end * 100)
    out = sy.energy_over_time(force, energy, [1.0], [0.0], t_end, steps, "stormer verlet")
    assert out["bounded"]
    assert out["band"] < 1e-3


def test_the_band_shrinks_like_the_square_of_the_step():
    out = sy.the_band_shrinks_with_the_step()
    assert out["shrinks_like_h_squared"]
    assert np.all(np.abs(out["ratio_per_halving"] - 4.0) < 0.2)


def test_the_bands_are_nested_so_a_step_change_does_not_jump_out_of_one():
    """Why the naive picture of disjoint bands is wrong, and why the drift test is needed."""
    out = sy.the_band_shrinks_with_the_step()
    assert out["bands_are_nested"]
    assert np.all(np.abs(out["band_high"]) < 1e-15)


def test_only_an_irregular_step_sequence_destroys_the_bound():
    """A repeating two step pattern is itself a symplectic map, so it keeps its own band."""
    out = sy.changing_the_step_destroys_the_bound()
    policies = list(out["policy"])
    assert bool(out["bounded"][policies.index("fixed")])
    assert bool(out["bounded"][policies.index("alternating")])
    assert not bool(out["bounded"][policies.index("random")])
    assert out["only_the_irregular_one_drifts"]


def test_the_random_step_run_has_a_trend_the_others_do_not():
    out = sy.changing_the_step_destroys_the_bound()
    policies = list(out["policy"])
    trends = np.abs(np.asarray(out["trend_per_unit_time"], dtype=float))
    assert trends[policies.index("random")] > 100.0 * trends[policies.index("fixed")]
    assert trends[policies.index("random")] > 100.0 * trends[policies.index("alternating")]


def test_at_equal_cost_rk4_is_more_accurate_and_still_the_worse_long_run_method():
    """The result the whole lesson is for: a smaller error that moves beats a larger one that
    does not only until the run is long enough."""
    out = sy.against_runge_kutta()
    assert out["verlet_evaluations"] == out["rk4_evaluations"]
    assert out["rk4_band"] < out["verlet_band"]
    assert out["verlet_is_bounded"]
    assert not out["rk4_is_bounded"]
    assert out["verlet_drift"] < 1e-3 * out["rk4_drift"]


def test_a_long_kepler_run_keeps_the_shape_of_the_ellipse():
    out = sy.kepler_orbit_stays_closed()
    assert out["shape_is_held"]
    assert out["perihelion_late"] == pytest.approx(out["perihelion_early"], rel=1e-6)
    assert out["aphelion_late"] == pytest.approx(out["aphelion_early"], rel=1e-6)
    assert out["angular_momentum_drift"] < 1e-10


@pytest.mark.parametrize("e", [0.0, 0.3, 0.6])
def test_the_kepler_start_is_at_perihelion_whatever_the_eccentricity(e):
    out = sy.kepler_orbit_stays_closed(eccentricity=e, periods=5.0, steps=5000)
    assert out["perihelion_early"] == pytest.approx(1.0 - e, rel=1e-4)
    assert out["aphelion_early"] == pytest.approx(1.0 + e, rel=1e-3)
