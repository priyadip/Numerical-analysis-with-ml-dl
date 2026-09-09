# Numerical Verification Report

Generated 2026-09-09 04:47 UTC.

Executing without raising is not the same as being right. This report covers the
second question: are the numbers correct?

Two mechanisms are used.

1. **Assertions inside the lessons.** Every notebook checks its own claims, so a
   wrong number becomes an execution failure rather than something a reader has to
   notice. Tolerances come from machine epsilon and the condition number of the
   problem, never from whatever value made the check pass.
2. **Independent re-checks.** The identities below are recomputed here, outside the
   notebooks, so a mistake in a lesson cannot hide itself.

## Assertions executed per lesson

| Lesson | Assertions |
|---|---|
| `01_what_is_numerical_analysis` | 7 |
| `02_number_systems_and_representation` | 9 |
| `03_floating_point_arithmetic` | 11 |
| `04_error_types_and_propagation` | 6 |
| `05_loss_of_significance_and_cancellation` | 16 |
| `06_conditioning_and_stability` | 8 |
| `07_taylor_series_and_convergence_rates` | 4 |
| `08_algorithm_cost_and_complexity` | 6 |
| `09_bracketing_methods` | 18 |
| `10_fixed_point_iteration` | 10 |
| `11_newton_and_secant_methods` | 14 |
| `12_convergence_theory_and_sensitivity` | 16 |
| `13_polynomial_root_finding` | 16 |
| `14_nonlinear_systems_of_equations` | 10 |
| `15_vectors_matrices_and_norms` | 18 |
| `16_orthogonality_and_projectors` | 20 |
| `17_gaussian_elimination_and_lu` | 11 |
| `18_pivoting_and_pa_lu` | 5 |
| `19_conditioning_of_linear_systems` | 4 |
| `20_symmetric_positive_definite_and_cholesky` | 5 |
| `21_banded_sparse_and_structured_systems` | 6 |
| `22_condition_estimation_and_iterative_refinement` | 7 |
| `23_classical_iterative_methods` | 5 |
| `24_conjugate_gradient` | 8 |
| `25_preconditioning` | 7 |
| `26_krylov_subspaces_arnoldi_and_lanczos` | 7 |
| `27_gmres_and_nonsymmetric_solvers` | 12 |
| `28_multigrid_methods` | 3 |
| `29_least_squares_and_normal_equations` | 9 |
| `30_gram_schmidt_and_qr` | 6 |
| `31_householder_and_givens_qr` | 6 |
| `32_solving_least_squares_in_practice` | 4 |
| `33_rank_revealing_qr_and_total_least_squares` | 17 |
| `34_nonlinear_least_squares` | 8 |
| `35_eigenvalue_theory_and_localization` | 9 |
| `36_power_methods` | 6 |
| `37_qr_algorithm` | 4 |
| `38_symmetric_eigenvalue_problem` | 6 |
| `39_krylov_methods_for_eigenvalues` | 4 |
| `40_generalized_eigenvalue_problem` | 4 |
| `41_svd_theory` | 5 |
| `42_computing_the_svd` | 5 |
| `43_svd_applications_and_low_rank` | 5 |
| `44_polynomial_interpolation_forms` | 5 |
| `45_divided_differences` | 8 |
| `46_interpolation_error_and_runge_phenomenon` | 8 |
| `47_chebyshev_interpolation` | 10 |
| `48_finite_difference_operators_and_tables` | 8 |
| `49_equal_interval_interpolation_formulas` | 6 |
| `50_hermite_and_piecewise_interpolation` | 9 |
| `51_cubic_splines` | 8 |
| `52_bezier_and_bspline_curves` | 8 |
| `53_bivariate_interpolation` | 7 |
| `54_function_approximation_fundamentals` | 16 |
| `55_orthogonal_polynomials` | 11 |
| `56_chebyshev_approximation_and_economization` | 11 |
| `57_pade_rational_approximation` | 6 |
| `58_trigonometric_interpolation_and_the_dft` | 15 |
| `59_the_fft_and_signal_processing` | 13 |
| `60_dct_and_data_compression` | 14 |
| `61_numerical_differentiation` | 7 |
| `62_newton_cotes_quadrature` | 8 |
| `63_richardson_romberg_and_euler_maclaurin` | 9 |
| `64_adaptive_quadrature` | 3 |
| `65_gaussian_quadrature` | 6 |
| `66_improper_and_multiple_integrals` | 4 |
| `67_ivp_theory_and_eulers_method` | 5 |
| `68_taylor_and_picard_methods` | 2 |
| `69_runge_kutta_methods` | 5 |
| `70_adaptive_step_size_control` | 3 |
| `71_multistep_methods` | 2 |
| `72_stability_and_stiff_equations` | 5 |
| `73_systems_and_higher_order_equations` | 3 |
| `74_geometric_and_symplectic_integrators` | 3 |
| `75_boundary_value_problems` | 4 |
| `76_collocation_and_finite_elements` | 5 |
| `77_pde_classification_and_stencils` | 21 |
| `78_parabolic_equations` | 22 |
| `79_consistency_convergence_and_stability` | 15 |
| `80_multidimensional_parabolic_and_adi` | 11 |
| `81_hyperbolic_equations_and_cfl` | 14 |
| `82_elliptic_equations` | 13 |
| `83_nonlinear_pdes_and_method_of_lines` | 14 |
| `84_optimization_fundamentals` | 21 |
| `85_derivative_free_optimization` | 14 |
| `86_gradient_and_newton_methods` | 15 |
| `87_quasi_newton_and_trust_region` | 15 |
| `88_constrained_optimization` | 13 |
| `89_stochastic_optimization` | 13 |
| `90_proximal_and_composite_optimization` | 13 |
| `91_random_number_generation` | 14 |
| `92_monte_carlo_methods` | 16 |
| `93_brownian_motion_and_sdes` | 18 |
| `94_numerical_linear_algebra_in_machine_learning` | 30 |
| `95_optimization_for_machine_learning` | 27 |
| `96_floating_point_in_deep_learning` | 24 |
| `97_automatic_differentiation` | 28 |
| `98_scientific_machine_learning_and_inverse_problems` | 25 |
| **total** | **1000** |

## Independent identity checks

Each row was recomputed by `verification/run_all.py`, not read from a notebook.

| Area | Identity | Measured | Tolerance | Result |
|---|---|---|---|---|
| floating point | machine_epsilon() equals 2^-52 | 0.000e+00 | 0.000e+00 | PASS |
| floating point | standard model |delta| <= u at 24 mantissa bits | 2.958e-08 | 2.980e-08 | PASS |
| summation | Neumaier relative error on the hard case | 0.000e+00 | 1.000e-15 | PASS |
| summation | Kahan relative error on the hard case | 0.000e+00 | 1.000e-15 | PASS |
| number systems | exact round trips that failed | 0.000e+00 | 0.000e+00 | PASS |
| polynomials | |horner - numpy.polyval|, worst over 3000 cases | 0.000e+00 | 0.000e+00 | PASS |
| polynomials | remainder theorem: |remainder - p(r)| relative | 0.000e+00 | 1.000e-09 | PASS |
| conditioning | kappa of sqrt at 4.0 against analytic | 5.220e-12 | 1.000e-04 | PASS |
| conditioning | kappa of exp at 20.0 against analytic | 2.439e-09 | 1.000e-04 | PASS |
| conditioning | (x-2)^5: backward error of numpy.roots, in units of u | 8.000e+00 | 2.000e+01 | PASS |
| conditioning | (x-2)^5: forward error is LARGE (problem is ill conditioned) | 2.266e-03 | 1.000e-04 | PASS |
| convergence | Newton measured order against 2 | 2.333e-04 | 5.000e-02 | PASS |
| convergence | central difference measured order against 2 | 2.144e-04 | 5.000e-02 | PASS |

**13 of 13 identity checks passed.**

## How tolerances were chosen

| Kind of check | Tolerance | Reason |
|---|---|---|
| Same algorithm, two implementations | **exactly 0** | identical floating point operations must give identical results, so anything nonzero is a real difference |
| Correctly rounded quantity | 1 to 2 unit roundoffs | the best any method can do |
| Backward error of a stable algorithm | tens of unit roundoffs | the definition of backward stability |
| Measured convergence order | 0.05 absolute | the estimator itself has this much noise over a realistic number of steps |
| Timing exponents | wide, and explained | run time depends on the machine, so the claim is checked against a mechanism rather than a number |

No tolerance in this repository was widened to make a check pass. Where a measured
value did not match the prediction, the cause was found and either the code or the
prediction was corrected. Lesson 08 section 5 is the clearest example: the measured
exponent for matrix multiplication is 2.3 rather than 3, and the gap is explained
exactly by the growth in achieved Gflop/s, verified to three decimal places.

