# Notebook Execution Report

Generated 2026-09-09 04:34 UTC.

Every notebook is rebuilt from its authored source and executed end to end with
`nbclient`, the same engine `jupyter nbconvert --execute` uses. A notebook counts as
passing only when no cell raises, which includes every `assert` inside it.

## Summary

- Notebooks executed: **98**
- Passed: **98**
- Failed: **0**
- Planned but not yet written: **0**
- Total execution time: **1146.3 s**

## Results

| Lesson | Part | Status | Time (s) | Cells |
|---|---|---|---|---|
| `01_what_is_numerical_analysis` | `01_foundations` | **PASS** | 3.3 | 18 |
| `02_number_systems_and_representation` | `01_foundations` | **PASS** | 2.7 | 30 |
| `03_floating_point_arithmetic` | `01_foundations` | **PASS** | 2.7 | 32 |
| `04_error_types_and_propagation` | `01_foundations` | **PASS** | 4.1 | 22 |
| `05_loss_of_significance_and_cancellation` | `01_foundations` | **PASS** | 7.0 | 23 |
| `06_conditioning_and_stability` | `01_foundations` | **PASS** | 2.5 | 17 |
| `07_taylor_series_and_convergence_rates` | `01_foundations` | **PASS** | 2.9 | 17 |
| `08_algorithm_cost_and_complexity` | `01_foundations` | **PASS** | 9.4 | 19 |
| `09_bracketing_methods` | `02_root_finding` | **PASS** | 2.6 | 30 |
| `10_fixed_point_iteration` | `02_root_finding` | **PASS** | 2.9 | 20 |
| `11_newton_and_secant_methods` | `02_root_finding` | **PASS** | 5.2 | 26 |
| `12_convergence_theory_and_sensitivity` | `02_root_finding` | **PASS** | 2.6 | 32 |
| `13_polynomial_root_finding` | `02_root_finding` | **PASS** | 2.3 | 28 |
| `14_nonlinear_systems_of_equations` | `02_root_finding` | **PASS** | 2.8 | 27 |
| `15_vectors_matrices_and_norms` | `03_direct_linear_systems` | **PASS** | 4.0 | 42 |
| `16_orthogonality_and_projectors` | `03_direct_linear_systems` | **PASS** | 7.5 | 39 |
| `17_gaussian_elimination_and_lu` | `03_direct_linear_systems` | **PASS** | 24.9 | 33 |
| `18_pivoting_and_pa_lu` | `03_direct_linear_systems` | **PASS** | 20.5 | 22 |
| `19_conditioning_of_linear_systems` | `03_direct_linear_systems` | **PASS** | 4.2 | 25 |
| `20_symmetric_positive_definite_and_cholesky` | `03_direct_linear_systems` | **PASS** | 4.0 | 28 |
| `21_banded_sparse_and_structured_systems` | `03_direct_linear_systems` | **PASS** | 5.6 | 23 |
| `22_condition_estimation_and_iterative_refinement` | `03_direct_linear_systems` | **PASS** | 6.3 | 19 |
| `23_classical_iterative_methods` | `04_iterative_and_krylov` | **PASS** | 6.3 | 24 |
| `24_conjugate_gradient` | `04_iterative_and_krylov` | **PASS** | 3.0 | 23 |
| `25_preconditioning` | `04_iterative_and_krylov` | **PASS** | 3.5 | 18 |
| `26_krylov_subspaces_arnoldi_and_lanczos` | `04_iterative_and_krylov` | **PASS** | 3.0 | 22 |
| `27_gmres_and_nonsymmetric_solvers` | `04_iterative_and_krylov` | **PASS** | 8.9 | 33 |
| `28_multigrid_methods` | `04_iterative_and_krylov` | **PASS** | 12.5 | 36 |
| `29_least_squares_and_normal_equations` | `05_least_squares_and_qr` | **PASS** | 2.8 | 28 |
| `30_gram_schmidt_and_qr` | `05_least_squares_and_qr` | **PASS** | 2.7 | 26 |
| `31_householder_and_givens_qr` | `05_least_squares_and_qr` | **PASS** | 3.4 | 28 |
| `32_solving_least_squares_in_practice` | `05_least_squares_and_qr` | **PASS** | 2.9 | 30 |
| `33_rank_revealing_qr_and_total_least_squares` | `05_least_squares_and_qr` | **PASS** | 8.4 | 36 |
| `34_nonlinear_least_squares` | `05_least_squares_and_qr` | **PASS** | 2.9 | 32 |
| `35_eigenvalue_theory_and_localization` | `06_eigenvalues_and_svd` | **PASS** | 2.9 | 24 |
| `36_power_methods` | `06_eigenvalues_and_svd` | **PASS** | 3.4 | 28 |
| `37_qr_algorithm` | `06_eigenvalues_and_svd` | **PASS** | 3.8 | 22 |
| `38_symmetric_eigenvalue_problem` | `06_eigenvalues_and_svd` | **PASS** | 7.0 | 28 |
| `39_krylov_methods_for_eigenvalues` | `06_eigenvalues_and_svd` | **PASS** | 5.3 | 22 |
| `40_generalized_eigenvalue_problem` | `06_eigenvalues_and_svd` | **PASS** | 2.8 | 20 |
| `41_svd_theory` | `06_eigenvalues_and_svd` | **PASS** | 2.8 | 24 |
| `42_computing_the_svd` | `06_eigenvalues_and_svd` | **PASS** | 4.4 | 26 |
| `43_svd_applications_and_low_rank` | `06_eigenvalues_and_svd` | **PASS** | 17.2 | 24 |
| `44_polynomial_interpolation_forms` | `07_interpolation` | **PASS** | 2.4 | 22 |
| `45_divided_differences` | `07_interpolation` | **PASS** | 2.7 | 22 |
| `46_interpolation_error_and_runge_phenomenon` | `07_interpolation` | **PASS** | 2.7 | 18 |
| `47_chebyshev_interpolation` | `07_interpolation` | **PASS** | 2.9 | 26 |
| `48_finite_difference_operators_and_tables` | `07_interpolation` | **PASS** | 2.4 | 22 |
| `49_equal_interval_interpolation_formulas` | `07_interpolation` | **PASS** | 2.7 | 22 |
| `50_hermite_and_piecewise_interpolation` | `07_interpolation` | **PASS** | 2.6 | 18 |
| `51_cubic_splines` | `07_interpolation` | **PASS** | 2.6 | 20 |
| `52_bezier_and_bspline_curves` | `07_interpolation` | **PASS** | 2.5 | 36 |
| `53_bivariate_interpolation` | `07_interpolation` | **PASS** | 2.5 | 18 |
| `54_function_approximation_fundamentals` | `08_approximation_and_transforms` | **PASS** | 2.8 | 30 |
| `55_orthogonal_polynomials` | `08_approximation_and_transforms` | **PASS** | 2.7 | 28 |
| `56_chebyshev_approximation_and_economization` | `08_approximation_and_transforms` | **PASS** | 2.8 | 31 |
| `57_pade_rational_approximation` | `08_approximation_and_transforms` | **PASS** | 2.6 | 28 |
| `58_trigonometric_interpolation_and_the_dft` | `08_approximation_and_transforms` | **PASS** | 2.5 | 30 |
| `59_the_fft_and_signal_processing` | `08_approximation_and_transforms` | **PASS** | 16.8 | 29 |
| `60_dct_and_data_compression` | `08_approximation_and_transforms` | **PASS** | 2.4 | 29 |
| `61_numerical_differentiation` | `09_differentiation_and_integration` | **PASS** | 2.6 | 24 |
| `62_newton_cotes_quadrature` | `09_differentiation_and_integration` | **PASS** | 3.1 | 24 |
| `63_richardson_romberg_and_euler_maclaurin` | `09_differentiation_and_integration` | **PASS** | 2.3 | 24 |
| `64_adaptive_quadrature` | `09_differentiation_and_integration` | **PASS** | 18.3 | 22 |
| `65_gaussian_quadrature` | `09_differentiation_and_integration` | **PASS** | 2.4 | 32 |
| `66_improper_and_multiple_integrals` | `09_differentiation_and_integration` | **PASS** | 31.3 | 28 |
| `67_ivp_theory_and_eulers_method` | `10_ordinary_differential_equations` | **PASS** | 113.8 | 26 |
| `68_taylor_and_picard_methods` | `10_ordinary_differential_equations` | **PASS** | 2.8 | 18 |
| `69_runge_kutta_methods` | `10_ordinary_differential_equations` | **PASS** | 2.7 | 22 |
| `70_adaptive_step_size_control` | `10_ordinary_differential_equations` | **PASS** | 2.6 | 24 |
| `71_multistep_methods` | `10_ordinary_differential_equations` | **PASS** | 2.3 | 26 |
| `72_stability_and_stiff_equations` | `10_ordinary_differential_equations` | **PASS** | 7.6 | 29 |
| `73_systems_and_higher_order_equations` | `10_ordinary_differential_equations` | **PASS** | 14.1 | 20 |
| `74_geometric_and_symplectic_integrators` | `10_ordinary_differential_equations` | **PASS** | 9.1 | 24 |
| `75_boundary_value_problems` | `10_ordinary_differential_equations` | **PASS** | 16.6 | 33 |
| `76_collocation_and_finite_elements` | `10_ordinary_differential_equations` | **PASS** | 2.7 | 33 |
| `77_pde_classification_and_stencils` | `11_partial_differential_equations` | **PASS** | 2.2 | 25 |
| `78_parabolic_equations` | `11_partial_differential_equations` | **PASS** | 11.8 | 28 |
| `79_consistency_convergence_and_stability` | `11_partial_differential_equations` | **PASS** | 2.8 | 20 |
| `80_multidimensional_parabolic_and_adi` | `11_partial_differential_equations` | **PASS** | 7.9 | 17 |
| `81_hyperbolic_equations_and_cfl` | `11_partial_differential_equations` | **PASS** | 2.6 | 20 |
| `82_elliptic_equations` | `11_partial_differential_equations` | **PASS** | 59.8 | 22 |
| `83_nonlinear_pdes_and_method_of_lines` | `11_partial_differential_equations` | **PASS** | 3.4 | 20 |
| `84_optimization_fundamentals` | `12_optimization` | **PASS** | 2.6 | 26 |
| `85_derivative_free_optimization` | `12_optimization` | **PASS** | 2.6 | 22 |
| `86_gradient_and_newton_methods` | `12_optimization` | **PASS** | 8.0 | 24 |
| `87_quasi_newton_and_trust_region` | `12_optimization` | **PASS** | 61.2 | 24 |
| `88_constrained_optimization` | `12_optimization` | **PASS** | 208.9 | 22 |
| `89_stochastic_optimization` | `12_optimization` | **PASS** | 18.2 | 22 |
| `90_proximal_and_composite_optimization` | `12_optimization` | **PASS** | 54.9 | 24 |
| `91_random_number_generation` | `13_stochastic_methods` | **PASS** | 3.4 | 22 |
| `92_monte_carlo_methods` | `13_stochastic_methods` | **PASS** | 12.5 | 22 |
| `93_brownian_motion_and_sdes` | `13_stochastic_methods` | **PASS** | 22.3 | 20 |
| `94_numerical_linear_algebra_in_machine_learning` | `14_ml_and_ai_connections` | **PASS** | 3.2 | 28 |
| `95_optimization_for_machine_learning` | `14_ml_and_ai_connections` | **PASS** | 56.7 | 32 |
| `96_floating_point_in_deep_learning` | `14_ml_and_ai_connections` | **PASS** | 11.3 | 24 |
| `97_automatic_differentiation` | `14_ml_and_ai_connections` | **PASS** | 3.4 | 28 |
| `98_scientific_machine_learning_and_inverse_problems` | `14_ml_and_ai_connections` | **PASS** | 53.7 | 24 |

## Failures

None. Every notebook ran from the first cell to the last with no exception.

## Intentional failures

Some lessons demonstrate a method breaking down. Those failures are set up on
purpose, caught, and explained, so they do not stop the notebook. They are listed
here so nobody mistakes them for bugs.

| Lesson | What fails on purpose | How it is controlled |
|---|---|---|
| 01 | Expanded $(x-1)^6$ near $x=1$ returns negative values for a sixth power | asserted to be negative and asserted to exceed the true peak by 10x |
| 05 | The textbook quadratic formula loses every digit at $b = 10^8$ | asserted to have relative error above 0.1 |
| 05 | Naive summation discards 200000 small values entirely | asserted to return exactly 1.0 |
| 06 | `numpy.roots` returns complex roots for a real quintuple root | asserted forward error large, backward error under 20u |
| 06 | The forward recurrence for $I_n$ produces impossible negative integrals | asserted to break the analytic bound $0 < I_n \le 1/(5(n+1))$ |

## Environment note

On Windows, `zmq` emits a `RuntimeWarning` about the Proactor event loop when a
kernel starts. It is harmless, comes from the Jupyter transport layer rather than
from any lesson, and does not affect results.

