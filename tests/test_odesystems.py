"""Tests for nalib.odesystems.

Four groups. The reduction to first order is checked against its defining identity rather than
against a solved trajectory, because the identity holds exactly and the trajectory does not. The
named problems are checked against their **exact invariants**: a Kepler start has an energy of
exactly -1/2 and an angular momentum of exactly sqrt(1 - e^2) whatever the eccentricity, and the
Lorenz fixed points are known in closed form. The measurements are checked for the property each
one exists to show. The last group is the three ways a system can defeat a solver that is working
perfectly: energy drift, chaos, and a parameter that changes the answer.

Every test sweeps orders, eccentricities, methods or step counts.
"""
import math

import numpy as np
import pytest

from nalib import odesystems as od

METHODS = ["euler", "heun", "kutta third", "rk4"]
ORDERS = [1, 2, 3, 4, 5]
ECCENTRICITIES = [0.0, 0.3, 0.6, 0.9]


# --------------------------------------------------------------------------- the reduction


@pytest.mark.parametrize("m", ORDERS)
def test_the_reduction_shifts_the_state_up_by_one(m):
    """Whatever the equation, the first m-1 components of f are the last m-1 of the state."""
    rng = np.random.default_rng(42)

    def g(t, *state):
        return sum((k + 1) * v for k, v in enumerate(state)) + t

    f = od.to_first_order(g, m)
    for _ in range(20):
        state = rng.normal(size=m)
        out = f(0.7, state)
        assert out.size == m
        assert np.array_equal(out[:-1], state[1:])
        assert float(out[-1]) == pytest.approx(float(g(0.7, *state)), rel=1e-14)


@pytest.mark.parametrize("m", ORDERS)
def test_the_reduction_rejects_a_state_of_the_wrong_length(m):
    f = od.to_first_order(lambda t, *s: 0.0, m)
    with pytest.raises(ValueError):
        f(0.0, np.zeros(m + 1))


def test_an_order_below_one_is_refused():
    with pytest.raises(ValueError):
        od.to_first_order(lambda t, y: 0.0, 0)


@pytest.mark.parametrize("n", [40, 80, 160, 320])
def test_the_reduced_equation_reproduces_the_closed_form(n):
    """The general reduction applied to y'' = -y must give cos(t), to the solver's accuracy."""
    from nalib.ivp import integrate
    from nalib.rungekutta import named_step

    f = od.to_first_order(lambda t, y, dy: -y, 2)
    out = integrate(f, 0.0, [1.0, 0.0], 2.0, n, named_step("rk4"))
    want = np.cos(out["t"])
    assert float(np.max(np.abs(out["y"][:, 0] - want))) < 200.0 * (2.0 / n) ** 4


def test_the_reduction_identity_is_exact_at_every_state():
    out = od.reduction_is_faithful()
    assert out["identity_is_exact"]
    assert float(np.max(out["reduction_identity_residual"])) == 0.0


def test_the_reduction_does_not_cost_an_order():
    out = od.reduction_is_faithful()
    ratios = out["position_error"][:-1] / out["position_error"][1:]
    assert np.all(ratios > 8.0)


@pytest.mark.parametrize("name", METHODS)
def test_a_system_does_not_change_a_method_order(name):
    out = od.order_on_a_system(name=name)
    assert out["matches"], f"{name} fitted {out['fitted_order']} claimed {out['claimed_order']}"


# --------------------------------------------------------------------------- the problems


def test_the_small_angle_pendulum_solution_solves_its_own_equation():
    """f(t, exact(t)) must equal the derivative of exact(t), which is known in closed form."""
    f, exact, omega = od.small_angle_pendulum(length=0.7, gravity=9.81)
    for t in np.linspace(0.0, 3.0, 25):
        state = exact(float(t))
        got = f(float(t), state)
        want = np.asarray([-omega * math.sin(omega * t),
                           -omega ** 2 * math.cos(omega * t)])
        assert np.allclose(got, want, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("length", [0.5, 1.0, 2.5])
def test_the_pendulum_energy_is_stationary_at_the_bottom(length):
    f, energy = od.pendulum(length=length)
    assert float(energy([0.0, 0.0])[0]) == pytest.approx(0.0, abs=1e-15)
    assert np.allclose(f(0.0, [0.0, 0.0]), 0.0)


def test_damping_removes_energy_and_no_damping_does_not():
    from nalib.ivp import integrate
    from nalib.rungekutta import named_step

    for damping, falls in ((0.0, False), (0.3, True)):
        f, energy = od.pendulum(damping=damping)
        out = integrate(f, 0.0, [1.0, 0.0], 20.0, 20000, named_step("rk4"))
        values = np.asarray(energy(out["y"]), dtype=float)
        assert (float(values[-1]) < 0.05 * float(values[0])) == falls


@pytest.mark.parametrize("e", ECCENTRICITIES)
def test_every_kepler_start_has_energy_minus_one_half(e):
    """Every ellipse with semi-major axis 1 has the same energy, whatever its shape."""
    f, energy, angular_momentum = od.two_body()
    y0 = od.elliptical_orbit_start(e)
    assert float(energy(y0)[0]) == pytest.approx(-0.5, abs=1e-14)
    assert float(angular_momentum(y0)[0]) == pytest.approx(math.sqrt(1.0 - e * e), rel=1e-14)


@pytest.mark.parametrize("radius", [0.5, 1.0, 2.5, 10.0])
def test_a_circular_start_has_the_energy_the_virial_theorem_gives(radius):
    f, energy, angular_momentum = od.two_body(mass=1.0)
    y0 = od.circular_orbit_start(radius)
    assert float(energy(y0)[0]) == pytest.approx(-1.0 / (2.0 * radius), rel=1e-14)
    assert float(angular_momentum(y0)[0]) == pytest.approx(math.sqrt(radius), rel=1e-14)


def test_the_two_body_field_points_at_the_centre():
    f, energy, angular_momentum = od.two_body()
    rng = np.random.default_rng(42)
    for _ in range(20):
        state = np.concatenate([rng.normal(size=2) * 2.0, rng.normal(size=2)])
        got = f(0.0, state)
        # the acceleration is antiparallel to the position, so their cross product vanishes
        assert abs(state[0] * got[3] - state[1] * got[2]) < 1e-12 * float(np.linalg.norm(got))


@pytest.mark.parametrize("rho", [14.0, 28.0, 40.0])
def test_the_lorenz_fixed_points_are_where_the_closed_form_says(rho):
    sigma, beta = 10.0, 8.0 / 3.0
    f = od.lorenz(sigma=sigma, rho=rho, beta=beta)
    assert np.allclose(f(0.0, [0.0, 0.0, 0.0]), 0.0)
    c = math.sqrt(beta * (rho - 1.0))
    for sign in (1.0, -1.0):
        assert np.allclose(f(0.0, [sign * c, sign * c, rho - 1.0]), 0.0, atol=1e-12)


def test_the_resting_neuron_has_its_gates_exactly_at_equilibrium():
    """The gating variables are set to alpha/(alpha+beta), so their derivatives are exactly 0."""
    f, y0 = od.hodgkin_huxley(current=0.0)
    got = f(0.0, y0)
    assert np.allclose(got[1:], 0.0, atol=1e-15)
    # the voltage is not exactly at rest, because -65 mV is the classical value and not the root
    assert abs(float(got[0])) < 0.01


def test_the_bridge_at_rest_stays_at_rest_with_no_wind():
    f = od.tacoma_narrows(wind=0.0)
    assert np.allclose(f(0.0, [0.0, 0.0, 0.0, 0.0]), 0.0, atol=1e-15)


# --------------------------------------------------------------------------- energy


@pytest.mark.parametrize("name", ["euler", "heun", "rk4"])
def test_the_energy_drift_shrinks_with_the_method_order(name):
    f, energy = od.pendulum()
    out = od.energy_drift(f, energy, [1.0, 0.0], 200.0, 20000, name)
    limits = {"euler": (1.0, 100.0), "heun": (1e-4, 1e-1), "rk4": (0.0, 1e-5)}
    low, high = limits[name]
    assert low <= out["worst"] <= high


def test_euler_gains_energy_and_every_method_here_has_a_trend():
    """The point of lesson 74: a small error with a trend is still unbounded."""
    f, energy = od.pendulum()
    for name in ("euler", "heun", "rk4"):
        out = od.energy_drift(f, energy, [1.0, 0.0], 200.0, 20000, name)
        assert out["drifts_one_way"], name
    gains = od.energy_drift(f, energy, [1.0, 0.0], 200.0, 20000, "euler")
    assert gains["trend_per_unit_time"] > 0.0


@pytest.mark.parametrize("steps", [10000, 20000, 40000])
def test_refining_the_step_shrinks_the_drift_and_does_not_remove_it(steps):
    f, energy = od.pendulum()
    out = od.energy_drift(f, energy, [1.0, 0.0], 100.0, steps, "rk4")
    assert out["worst"] > 0.0
    assert abs(out["trend_per_unit_time"]) > 0.0


def test_a_circular_orbit_closes_and_an_eccentric_one_is_much_harder():
    circle = od.orbit_closes(eccentricity=0.0, periods=10.0, steps=4000)
    ellipse = od.orbit_closes(eccentricity=0.6, periods=10.0, steps=4000)
    assert circle["closure_gap"] < 1e-5
    assert ellipse["closure_gap"] > 100.0 * circle["closure_gap"]
    # the circle keeps its radius and the ellipse sweeps 1-e to 1+e, which is a range of 2e
    assert circle["radius_range"] < 1e-6
    assert ellipse["radius_range"] == pytest.approx(1.2, abs=1e-2)


@pytest.mark.parametrize("steps", [2000, 4000, 8000])
def test_a_finer_step_closes_the_orbit_better(steps):
    out = od.orbit_closes(eccentricity=0.0, periods=10.0, steps=steps)
    assert out["closure_gap"] < 1e-3
    assert out["angular_momentum_drift"] < 1e-6


def test_angular_momentum_survives_where_energy_does_not():
    """A method can conserve one invariant and not the other, so both are worth measuring."""
    out = od.orbit_closes(eccentricity=0.6, periods=10.0, steps=4000)
    assert out["angular_momentum_drift"] < out["energy_drift"]


# --------------------------------------------------------------------------- the hard cases


def test_the_lorenz_separation_rate_is_the_lyapunov_exponent():
    out = od.divergence_is_the_problem()
    assert out["fitted_lyapunov"] == pytest.approx(0.9, abs=0.15)
    assert 10.0 < out["time_to_order_one"] < 30.0


@pytest.mark.parametrize("steps", [20000, 30000, 45000])
def test_the_separation_rate_does_not_depend_on_the_solver(steps):
    """The exponent is in the equations, so refining the step must not change it."""
    out = od.divergence_is_the_problem(steps=steps)
    assert out["fitted_lyapunov"] == pytest.approx(0.9, abs=0.15)


def test_accuracy_buys_only_the_logarithm_of_itself():
    out = od.divergence_is_the_problem()
    assert out["extra_time_per_halving"] == pytest.approx(math.log(2.0)
                                                          / out["fitted_lyapunov"], rel=1e-9)
    assert out["extra_time_per_halving"] < 1.0


def test_the_neuron_stiffness_changes_by_more_than_an_order_of_magnitude():
    out = od.stiffness_along_the_path()
    assert out["varies_by"] > 10.0
    assert out["largest"] > 100.0
    assert out["spike_voltage"] > 0.0


def test_the_bridge_growth_crosses_one_and_is_not_monotone():
    """There is no single critical speed here, only a resonance the sweep passes through."""
    out = od.a_parameter_crossing_changes_the_answer()
    assert out["first_wind_that_grows"] is not None
    assert out["crossings_of_one"] >= 1
    assert not out["growth_is_monotone"]
    assert float(np.max(out["growth"])) > 5.0


def test_a_wind_below_the_crossing_damps_the_twist():
    out = od.a_parameter_crossing_changes_the_answer(winds=[40.0, 60.0])
    assert np.all(out["growth"] < 1.0)
    assert out["first_wind_that_grows"] is None
