# Internal Coverage Checklist

This is a quality-control artifact. It is **not** part of the learner-facing
course. Learners should start at `README.md` and follow `LEARNING_PATH.md`.

It exists to answer one question with evidence rather than assumption:

> After finishing this repository, will the learner have learned everything
> important contained in the supplied books, even though the repository is
> organised differently from those books?

Regenerate with `python _planning/build_checklist.py`.

## Sources

- **S1** = Sauer, Numerical Analysis, 3rd ed. (Pearson, 2019)
- **S2** = Gupta, Numerical Methods: Fundamentals and Applications (Cambridge, 2019)
- **S3** = Trefethen and Bau, Numerical Linear Algebra (concept map, TrefethenBau.md)
- **SUP** = Supplementary / Advanced Material (not from any supplied source)

## Column meanings

| Column | Meaning |
|---|---|
| covered | The concept is explained in the named lesson, in both `.ipynb` and `.md` |
| implemented | There is working from-scratch code for it, where code makes sense |
| experimented | A numerical experiment in the lesson demonstrates it |
| tested | An automated test in `tests/` exercises it, where testable |
| verified | The result was checked against a library or an identity, and the maths was re-read |

Some concepts are definitional or theoretical, so `implemented` and `tested` stay
`no` for them by design. Those are marked with `n/a` once the lesson is written.

## Summary

- Total tracked concepts: **514**
  - S1: 186
  - S2: 178
  - SUP: 75
- Lessons in the final curriculum: **98**
- Concepts marked covered: **514 / 514** (100.0%)

## Concepts per lesson

| Lesson | Part | S1 | S2 | SUP | Total |
|---|---|---|---|---|---|
| `01_what_is_numerical_analysis` | `01_foundations` | 2 | 3 | 0 | 5 |
| `02_number_systems_and_representation` | `01_foundations` | 2 | 6 | 0 | 8 |
| `03_floating_point_arithmetic` | `01_foundations` | 5 | 3 | 0 | 9 |
| `04_error_types_and_propagation` | `01_foundations` | 0 | 9 | 0 | 9 |
| `05_loss_of_significance_and_cancellation` | `01_foundations` | 2 | 1 | 5 | 8 |
| `06_conditioning_and_stability` | `01_foundations` | 1 | 0 | 0 | 4 |
| `07_taylor_series_and_convergence_rates` | `01_foundations` | 1 | 1 | 1 | 3 |
| `08_algorithm_cost_and_complexity` | `01_foundations` | 1 | 0 | 2 | 3 |
| `09_bracketing_methods` | `02_root_finding` | 3 | 4 | 0 | 7 |
| `10_fixed_point_iteration` | `02_root_finding` | 4 | 2 | 0 | 6 |
| `11_newton_and_secant_methods` | `02_root_finding` | 5 | 4 | 0 | 9 |
| `12_convergence_theory_and_sensitivity` | `02_root_finding` | 2 | 3 | 0 | 5 |
| `13_polynomial_root_finding` | `02_root_finding` | 0 | 6 | 1 | 7 |
| `14_nonlinear_systems_of_equations` | `02_root_finding` | 3 | 3 | 1 | 7 |
| `15_vectors_matrices_and_norms` | `03_direct_linear_systems` | 2 | 0 | 0 | 8 |
| `16_orthogonality_and_projectors` | `03_direct_linear_systems` | 0 | 0 | 0 | 13 |
| `17_gaussian_elimination_and_lu` | `03_direct_linear_systems` | 5 | 7 | 0 | 14 |
| `18_pivoting_and_pa_lu` | `03_direct_linear_systems` | 4 | 2 | 0 | 8 |
| `19_conditioning_of_linear_systems` | `03_direct_linear_systems` | 4 | 1 | 0 | 5 |
| `20_symmetric_positive_definite_and_cholesky` | `03_direct_linear_systems` | 2 | 1 | 1 | 4 |
| `21_banded_sparse_and_structured_systems` | `03_direct_linear_systems` | 1 | 1 | 1 | 3 |
| `22_condition_estimation_and_iterative_refinement` | `03_direct_linear_systems` | 0 | 0 | 2 | 2 |
| `23_classical_iterative_methods` | `04_iterative_and_krylov` | 3 | 5 | 0 | 9 |
| `24_conjugate_gradient` | `04_iterative_and_krylov` | 1 | 0 | 0 | 3 |
| `25_preconditioning` | `04_iterative_and_krylov` | 1 | 0 | 0 | 4 |
| `26_krylov_subspaces_arnoldi_and_lanczos` | `04_iterative_and_krylov` | 1 | 0 | 1 | 8 |
| `27_gmres_and_nonsymmetric_solvers` | `04_iterative_and_krylov` | 2 | 0 | 0 | 7 |
| `28_multigrid_methods` | `04_iterative_and_krylov` | 0 | 0 | 1 | 1 |
| `29_least_squares_and_normal_equations` | `05_least_squares_and_qr` | 6 | 3 | 0 | 10 |
| `30_gram_schmidt_and_qr` | `05_least_squares_and_qr` | 3 | 0 | 0 | 6 |
| `31_householder_and_givens_qr` | `05_least_squares_and_qr` | 1 | 0 | 0 | 7 |
| `32_solving_least_squares_in_practice` | `05_least_squares_and_qr` | 0 | 0 | 1 | 6 |
| `33_rank_revealing_qr_and_total_least_squares` | `05_least_squares_and_qr` | 0 | 0 | 2 | 2 |
| `34_nonlinear_least_squares` | `05_least_squares_and_qr` | 4 | 0 | 0 | 4 |
| `35_eigenvalue_theory_and_localization` | `06_eigenvalues_and_svd` | 1 | 5 | 0 | 10 |
| `36_power_methods` | `06_eigenvalues_and_svd` | 4 | 3 | 0 | 8 |
| `37_qr_algorithm` | `06_eigenvalues_and_svd` | 3 | 1 | 0 | 4 |
| `38_symmetric_eigenvalue_problem` | `06_eigenvalues_and_svd` | 0 | 5 | 1 | 6 |
| `39_krylov_methods_for_eigenvalues` | `06_eigenvalues_and_svd` | 0 | 0 | 1 | 3 |
| `40_generalized_eigenvalue_problem` | `06_eigenvalues_and_svd` | 0 | 0 | 1 | 1 |
| `41_svd_theory` | `06_eigenvalues_and_svd` | 3 | 0 | 0 | 5 |
| `42_computing_the_svd` | `06_eigenvalues_and_svd` | 1 | 0 | 0 | 3 |
| `43_svd_applications_and_low_rank` | `06_eigenvalues_and_svd` | 3 | 0 | 2 | 7 |
| `44_polynomial_interpolation_forms` | `07_interpolation` | 3 | 7 | 2 | 12 |
| `45_divided_differences` | `07_interpolation` | 1 | 4 | 0 | 5 |
| `46_interpolation_error_and_runge_phenomenon` | `07_interpolation` | 3 | 1 | 0 | 4 |
| `47_chebyshev_interpolation` | `07_interpolation` | 3 | 0 | 1 | 4 |
| `48_finite_difference_operators_and_tables` | `07_interpolation` | 0 | 11 | 0 | 11 |
| `49_equal_interval_interpolation_formulas` | `07_interpolation` | 0 | 9 | 0 | 9 |
| `50_hermite_and_piecewise_interpolation` | `07_interpolation` | 0 | 2 | 0 | 2 |
| `51_cubic_splines` | `07_interpolation` | 2 | 2 | 0 | 4 |
| `52_bezier_and_bspline_curves` | `07_interpolation` | 2 | 2 | 1 | 5 |
| `53_bivariate_interpolation` | `07_interpolation` | 0 | 2 | 1 | 3 |
| `54_function_approximation_fundamentals` | `08_approximation_and_transforms` | 0 | 1 | 2 | 3 |
| `55_orthogonal_polynomials` | `08_approximation_and_transforms` | 0 | 0 | 2 | 2 |
| `56_chebyshev_approximation_and_economization` | `08_approximation_and_transforms` | 0 | 1 | 0 | 1 |
| `57_pade_rational_approximation` | `08_approximation_and_transforms` | 0 | 1 | 0 | 1 |
| `58_trigonometric_interpolation_and_the_dft` | `08_approximation_and_transforms` | 5 | 0 | 0 | 5 |
| `59_the_fft_and_signal_processing` | `08_approximation_and_transforms` | 4 | 0 | 1 | 5 |
| `60_dct_and_data_compression` | `08_approximation_and_transforms` | 10 | 0 | 0 | 10 |
| `61_numerical_differentiation` | `09_differentiation_and_integration` | 4 | 4 | 1 | 9 |
| `62_newton_cotes_quadrature` | `09_differentiation_and_integration` | 4 | 8 | 0 | 12 |
| `63_richardson_romberg_and_euler_maclaurin` | `09_differentiation_and_integration` | 1 | 3 | 0 | 4 |
| `64_adaptive_quadrature` | `09_differentiation_and_integration` | 2 | 0 | 0 | 2 |
| `65_gaussian_quadrature` | `09_differentiation_and_integration` | 1 | 4 | 0 | 8 |
| `66_improper_and_multiple_integrals` | `09_differentiation_and_integration` | 0 | 1 | 1 | 2 |
| `67_ivp_theory_and_eulers_method` | `10_ordinary_differential_equations` | 4 | 4 | 0 | 8 |
| `68_taylor_and_picard_methods` | `10_ordinary_differential_equations` | 1 | 2 | 0 | 3 |
| `69_runge_kutta_methods` | `10_ordinary_differential_equations` | 2 | 2 | 1 | 5 |
| `70_adaptive_step_size_control` | `10_ordinary_differential_equations` | 2 | 0 | 0 | 2 |
| `71_multistep_methods` | `10_ordinary_differential_equations` | 3 | 2 | 1 | 6 |
| `72_stability_and_stiff_equations` | `10_ordinary_differential_equations` | 1 | 3 | 1 | 5 |
| `73_systems_and_higher_order_equations` | `10_ordinary_differential_equations` | 6 | 1 | 0 | 7 |
| `74_geometric_and_symplectic_integrators` | `10_ordinary_differential_equations` | 0 | 0 | 1 | 1 |
| `75_boundary_value_problems` | `10_ordinary_differential_equations` | 5 | 3 | 0 | 8 |
| `76_collocation_and_finite_elements` | `10_ordinary_differential_equations` | 2 | 0 | 0 | 2 |
| `77_pde_classification_and_stencils` | `11_partial_differential_equations` | 0 | 3 | 0 | 3 |
| `78_parabolic_equations` | `11_partial_differential_equations` | 3 | 5 | 0 | 8 |
| `79_consistency_convergence_and_stability` | `11_partial_differential_equations` | 1 | 5 | 1 | 7 |
| `80_multidimensional_parabolic_and_adi` | `11_partial_differential_equations` | 0 | 2 | 0 | 2 |
| `81_hyperbolic_equations_and_cfl` | `11_partial_differential_equations` | 2 | 2 | 0 | 4 |
| `82_elliptic_equations` | `11_partial_differential_equations` | 3 | 2 | 0 | 5 |
| `83_nonlinear_pdes_and_method_of_lines` | `11_partial_differential_equations` | 2 | 0 | 1 | 3 |
| `84_optimization_fundamentals` | `12_optimization` | 1 | 0 | 2 | 3 |
| `85_derivative_free_optimization` | `12_optimization` | 3 | 0 | 0 | 3 |
| `86_gradient_and_newton_methods` | `12_optimization` | 2 | 0 | 1 | 3 |
| `87_quasi_newton_and_trust_region` | `12_optimization` | 2 | 0 | 2 | 4 |
| `88_constrained_optimization` | `12_optimization` | 0 | 0 | 2 | 2 |
| `89_stochastic_optimization` | `12_optimization` | 0 | 0 | 4 | 4 |
| `90_proximal_and_composite_optimization` | `12_optimization` | 0 | 0 | 3 | 3 |
| `91_random_number_generation` | `13_stochastic_methods` | 2 | 0 | 1 | 3 |
| `92_monte_carlo_methods` | `13_stochastic_methods` | 3 | 0 | 1 | 4 |
| `93_brownian_motion_and_sdes` | `13_stochastic_methods` | 5 | 0 | 0 | 5 |
| `94_numerical_linear_algebra_in_machine_learning` | `14_ml_and_ai_connections` | 0 | 0 | 3 | 3 |
| `95_optimization_for_machine_learning` | `14_ml_and_ai_connections` | 0 | 0 | 3 | 3 |
| `96_floating_point_in_deep_learning` | `14_ml_and_ai_connections` | 0 | 0 | 4 | 4 |
| `97_automatic_differentiation` | `14_ml_and_ai_connections` | 0 | 0 | 3 | 3 |
| `98_scientific_machine_learning_and_inverse_problems` | `14_ml_and_ai_connections` | 0 | 0 | 4 | 4 |

## Full source-to-lesson map

### S1 - Sauer, Numerical Analysis, 3rd ed. (Pearson, 2019)


**0 Fundamentals**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 0.1 | Evaluating a polynomial, nested (Horner) form | `01_what_is_numerical_analysis` | yes | yes | yes | yes | yes |
| 0.1 | Operation counting for polynomial evaluation | `08_algorithm_cost_and_complexity` | yes | yes | yes | yes | yes |
| 0.2 | Decimal to binary conversion | `02_number_systems_and_representation` | yes | yes | yes | yes | yes |
| 0.2 | Binary to decimal conversion | `02_number_systems_and_representation` | yes | yes | yes | yes | yes |
| 0.3 | IEEE 754 floating point formats (single, double, long double) | `03_floating_point_arithmetic` | yes | no | yes | yes | yes |
| 0.3 | Normalized machine representation, left-justified mantissa | `03_floating_point_arithmetic` | yes | no | yes | yes | yes |
| 0.3 | Machine epsilon definition | `03_floating_point_arithmetic` | yes | no | yes | yes | yes |
| 0.3 | Rounding versus chopping | `03_floating_point_arithmetic` | yes | no | yes | yes | yes |
| 0.3 | Addition of floating point numbers | `03_floating_point_arithmetic` | yes | no | yes | yes | yes |
| 0.4 | Loss of significance, catastrophic cancellation | `05_loss_of_significance_and_cancellation` | yes | yes | yes | yes | yes |
| 0.4 | Rewriting formulas to avoid cancellation (quadratic formula) | `05_loss_of_significance_and_cancellation` | yes | yes | yes | yes | yes |
| 0.5 | Review of calculus: IVT, MVT, Rolle, Taylor with remainder | `07_taylor_series_and_convergence_rates` | yes | yes | yes | yes | yes |

**1 Solving Equations**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 1.1 | Bisection method, bracketing a root | `09_bracketing_methods` | yes | yes | yes | yes | yes |
| 1.1 | Bisection accuracy bound and iteration count | `09_bracketing_methods` | yes | yes | yes | yes | yes |
| 1.2 | Fixed points of a function | `10_fixed_point_iteration` | yes | yes | yes | yes | yes |
| 1.2 | Geometry of fixed point iteration (cobweb diagram) | `10_fixed_point_iteration` | yes | yes | yes | yes | yes |
| 1.2 | Linear convergence of fixed point iteration | `10_fixed_point_iteration` | yes | yes | yes | yes | yes |
| 1.2 | Stopping criteria and their failure modes | `10_fixed_point_iteration` | yes | yes | yes | yes | yes |
| 1.3 | Forward and backward error | `06_conditioning_and_stability` | yes | yes | yes | yes | yes |
| 1.3 | The Wilkinson polynomial | `12_convergence_theory_and_sensitivity` | yes | yes | yes | yes | yes |
| 1.3 | Sensitivity of root finding, condition number of a root | `12_convergence_theory_and_sensitivity` | yes | yes | yes | yes | yes |
| 1.4 | Newton's method derivation | `11_newton_and_secant_methods` | yes | yes | yes | yes | yes |
| 1.4 | Quadratic convergence of Newton's method | `11_newton_and_secant_methods` | yes | yes | yes | yes | yes |
| 1.4 | Linear convergence at multiple roots, modified Newton | `11_newton_and_secant_methods` | yes | yes | yes | yes | yes |
| 1.5 | Secant method and its order of convergence | `11_newton_and_secant_methods` | yes | yes | yes | yes | yes |
| 1.5 | Secant variants (false position, Illinois, inverse quadratic) | `09_bracketing_methods` | yes | yes | yes | yes | yes |
| 1.5 | Brent's method | `11_newton_and_secant_methods` | yes | yes | yes | yes | yes |
| RC1 | Kinematics of the Stewart platform | `14_nonlinear_systems_of_equations` | yes | yes | yes | yes | yes |

**2 Systems of Equations**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 2.1 | Naive Gaussian elimination | `17_gaussian_elimination_and_lu` | yes | yes | yes | yes | yes |
| 2.1 | Operation counts for Gaussian elimination | `17_gaussian_elimination_and_lu` | yes | yes | yes | yes | yes |
| 2.2 | Matrix form of Gaussian elimination, LU factorization | `17_gaussian_elimination_and_lu` | yes | yes | yes | yes | yes |
| 2.2 | Back substitution with the LU factorization | `17_gaussian_elimination_and_lu` | yes | yes | yes | yes | yes |
| 2.2 | Complexity of the LU factorization | `17_gaussian_elimination_and_lu` | yes | yes | yes | yes | yes |
| 2.3 | Infinity norm of a vector | `15_vectors_matrices_and_norms` | yes | yes | yes | yes | yes |
| 2.3 | Residual, backward error and forward error for Ax=b | `19_conditioning_of_linear_systems` | yes | yes | yes | yes | yes |
| 2.3 | Error magnification factor and matrix condition number | `19_conditioning_of_linear_systems` | yes | yes | yes | yes | yes |
| 2.3 | Hilbert matrix as an ill-conditioned example | `19_conditioning_of_linear_systems` | yes | yes | yes | yes | yes |
| 2.3 | Swamping | `18_pivoting_and_pa_lu` | yes | yes | yes | yes | yes |
| 2.4 | Partial pivoting | `18_pivoting_and_pa_lu` | yes | yes | yes | yes | yes |
| 2.4 | Permutation matrices | `18_pivoting_and_pa_lu` | yes | yes | yes | yes | yes |
| 2.4 | PA = LU factorization | `18_pivoting_and_pa_lu` | yes | yes | yes | yes | yes |
| RC2 | The Euler-Bernoulli beam | `19_conditioning_of_linear_systems` | yes | yes | yes | yes | yes |
| 2.5 | Jacobi method | `23_classical_iterative_methods` | yes | no | yes | yes | yes |
| 2.5 | Gauss-Seidel method and SOR | `23_classical_iterative_methods` | yes | no | yes | yes | yes |
| 2.5 | Convergence of iterative methods, spectral radius | `23_classical_iterative_methods` | yes | no | yes | yes | yes |
| 2.5 | Sparse matrix computations | `21_banded_sparse_and_structured_systems` | yes | yes | yes | yes | yes |
| 2.6 | Symmetric positive definite matrices | `20_symmetric_positive_definite_and_cholesky` | yes | yes | yes | yes | yes |
| 2.6 | Cholesky factorization | `20_symmetric_positive_definite_and_cholesky` | yes | yes | yes | yes | yes |
| 2.6 | Conjugate Gradient method, A-inner product, conjugacy | `24_conjugate_gradient` | yes | yes | yes | yes | yes |
| 2.6 | Preconditioning, Jacobi and SSOR preconditioners | `25_preconditioning` | yes | yes | yes | yes | yes |
| 2.7 | Multivariate Newton's method and the Jacobian | `14_nonlinear_systems_of_equations` | yes | yes | yes | yes | yes |
| 2.7 | Broyden's method | `14_nonlinear_systems_of_equations` | yes | yes | yes | yes | yes |

**3 Interpolation**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 3.1 | Lagrange interpolation | `44_polynomial_interpolation_forms` | yes | no | yes | yes | yes |
| 3.1 | Newton's divided differences | `45_divided_differences` | yes | no | yes | yes | yes |
| 3.1 | Uniqueness of the degree-d interpolating polynomial | `44_polynomial_interpolation_forms` | yes | no | yes | yes | yes |
| 3.1 | Representing functions by approximating polynomials | `44_polynomial_interpolation_forms` | yes | no | yes | yes | yes |
| 3.2 | Interpolation error formula | `46_interpolation_error_and_runge_phenomenon` | yes | no | yes | yes | yes |
| 3.2 | Proof of the Newton form and the error formula | `46_interpolation_error_and_runge_phenomenon` | yes | no | yes | yes | yes |
| 3.2 | Runge phenomenon | `46_interpolation_error_and_runge_phenomenon` | yes | no | yes | yes | yes |
| 3.3 | Chebyshev's theorem on node placement | `47_chebyshev_interpolation` | yes | no | yes | yes | yes |
| 3.3 | Chebyshev polynomials and their recurrence | `47_chebyshev_interpolation` | yes | no | yes | yes | yes |
| 3.3 | Change of interval for Chebyshev nodes | `47_chebyshev_interpolation` | yes | no | yes | yes | yes |
| 3.4 | Cubic spline properties and the tridiagonal system | `51_cubic_splines` | yes | no | yes | yes | yes |
| 3.4 | Spline endpoint conditions (natural, clamped, parabolic, not-a-knot) | `51_cubic_splines` | yes | no | yes | yes | yes |
| 3.5 | Bezier curves | `52_bezier_and_bspline_curves` | yes | no | yes | yes | yes |
| RC3 | Fonts from Bezier curves | `52_bezier_and_bspline_curves` | yes | no | yes | yes | yes |

**4 Least Squares**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 4.1 | Inconsistent systems and the least squares idea | `29_least_squares_and_normal_equations` | yes | yes | yes | yes | yes |
| 4.1 | Normal equations and projection geometry | `29_least_squares_and_normal_equations` | yes | yes | yes | yes | yes |
| 4.1 | Fitting models to data | `29_least_squares_and_normal_equations` | yes | yes | yes | yes | yes |
| 4.1 | Conditioning of least squares, squared condition number | `29_least_squares_and_normal_equations` | yes | yes | yes | yes | yes |
| 4.2 | Periodic data models | `29_least_squares_and_normal_equations` | yes | yes | yes | yes | yes |
| 4.2 | Data linearization for exponential and power models | `29_least_squares_and_normal_equations` | yes | yes | yes | yes | yes |
| 4.3 | Classical Gram-Schmidt orthogonalization and least squares | `30_gram_schmidt_and_qr` | yes | yes | yes | yes | yes |
| 4.3 | Reduced and full QR factorization | `30_gram_schmidt_and_qr` | yes | yes | yes | yes | yes |
| 4.3 | Modified Gram-Schmidt and loss of orthogonality | `30_gram_schmidt_and_qr` | yes | yes | yes | yes | yes |
| 4.3 | Householder reflectors and Householder QR | `31_householder_and_givens_qr` | yes | yes | yes | yes | yes |
| 4.4 | Krylov methods and the Krylov subspace | `26_krylov_subspaces_arnoldi_and_lanczos` | yes | yes | yes | yes | yes |
| 4.4 | GMRES algorithm | `27_gmres_and_nonsymmetric_solvers` | yes | no | yes | yes | yes |
| 4.4 | Preconditioned GMRES | `27_gmres_and_nonsymmetric_solvers` | yes | no | yes | yes | yes |
| 4.5 | Gauss-Newton method | `34_nonlinear_least_squares` | yes | yes | yes | yes | yes |
| 4.5 | Models with nonlinear parameters | `34_nonlinear_least_squares` | yes | yes | yes | yes | yes |
| 4.5 | Levenberg-Marquardt method | `34_nonlinear_least_squares` | yes | yes | yes | yes | yes |
| RC4 | GPS, conditioning and nonlinear least squares | `34_nonlinear_least_squares` | yes | yes | yes | yes | yes |

**5 Differentiation and Integration**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 5.1 | Finite difference formulas for derivatives | `61_numerical_differentiation` | yes | yes | yes | yes | yes |
| 5.1 | Rounding versus truncation error, optimal step size | `61_numerical_differentiation` | yes | yes | yes | yes | yes |
| 5.1 | Extrapolation for derivatives | `61_numerical_differentiation` | yes | yes | yes | yes | yes |
| 5.1 | Symbolic differentiation and integration | `61_numerical_differentiation` | yes | yes | yes | yes | yes |
| 5.2 | Trapezoid rule | `62_newton_cotes_quadrature` | yes | yes | yes | yes | yes |
| 5.2 | Simpson's rule | `62_newton_cotes_quadrature` | yes | yes | yes | yes | yes |
| 5.2 | Composite Newton-Cotes formulas | `62_newton_cotes_quadrature` | yes | yes | yes | yes | yes |
| 5.2 | Open Newton-Cotes methods | `62_newton_cotes_quadrature` | yes | yes | yes | yes | yes |
| 5.3 | Romberg integration | `63_richardson_romberg_and_euler_maclaurin` | yes | yes | yes | yes | yes |
| 5.4 | Adaptive quadrature | `64_adaptive_quadrature` | yes | yes | yes | yes | yes |
| 5.5 | Gaussian quadrature, Legendre nodes and weights | `65_gaussian_quadrature` | yes | yes | yes | yes | yes |
| RC5 | Motion control in computer-aided modeling | `64_adaptive_quadrature` | yes | yes | yes | yes | yes |

**6 Ordinary Differential Equations**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 6.1 | Initial value problems, Euler's method | `67_ivp_theory_and_eulers_method` | yes | yes | yes | yes | yes |
| 6.1 | Existence, uniqueness and continuity, Lipschitz condition | `67_ivp_theory_and_eulers_method` | yes | yes | yes | yes | yes |
| 6.1 | First order linear equations | `67_ivp_theory_and_eulers_method` | yes | yes | yes | yes | yes |
| 6.2 | Local and global truncation error | `67_ivp_theory_and_eulers_method` | yes | yes | yes | yes | yes |
| 6.2 | Explicit trapezoid method | `69_runge_kutta_methods` | yes | yes | yes | yes | yes |
| 6.2 | Taylor methods for ODEs | `68_taylor_and_picard_methods` | yes | yes | yes | yes | yes |
| 6.3 | Systems of ODEs, higher order equations | `73_systems_and_higher_order_equations` | yes | yes | yes | yes | yes |
| 6.3 | Computer simulation: the pendulum | `73_systems_and_higher_order_equations` | yes | yes | yes | yes | yes |
| 6.3 | Computer simulation: orbital mechanics | `73_systems_and_higher_order_equations` | yes | yes | yes | yes | yes |
| 6.4 | The Runge-Kutta family, RK4 | `69_runge_kutta_methods` | yes | yes | yes | yes | yes |
| 6.4 | Computer simulation: Hodgkin-Huxley neuron | `73_systems_and_higher_order_equations` | yes | yes | yes | yes | yes |
| 6.4 | Computer simulation: Lorenz equations | `73_systems_and_higher_order_equations` | yes | yes | yes | yes | yes |
| RC6 | The Tacoma Narrows Bridge | `73_systems_and_higher_order_equations` | yes | yes | yes | yes | yes |
| 6.5 | Embedded Runge-Kutta pairs | `70_adaptive_step_size_control` | yes | yes | yes | yes | yes |
| 6.5 | Order 4/5 methods (RKF45, Dormand-Prince) | `70_adaptive_step_size_control` | yes | yes | yes | yes | yes |
| 6.6 | Implicit methods and stiff equations | `72_stability_and_stiff_equations` | yes | yes | yes | yes | yes |
| 6.7 | Generating multistep methods | `71_multistep_methods` | yes | yes | yes | yes | yes |
| 6.7 | Explicit multistep methods (Adams-Bashforth) | `71_multistep_methods` | yes | yes | yes | yes | yes |
| 6.7 | Implicit multistep methods (Adams-Moulton) | `71_multistep_methods` | yes | yes | yes | yes | yes |

**7 Boundary Value Problems**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 7.1 | Solutions of boundary value problems | `75_boundary_value_problems` | yes | yes | yes | yes | yes |
| 7.1 | Shooting method implementation | `75_boundary_value_problems` | yes | yes | yes | yes | yes |
| RC7 | Buckling of a circular ring | `75_boundary_value_problems` | yes | yes | yes | yes | yes |
| 7.2 | Finite difference method for linear BVPs | `75_boundary_value_problems` | yes | yes | yes | yes | yes |
| 7.2 | Finite difference method for nonlinear BVPs | `75_boundary_value_problems` | yes | yes | yes | yes | yes |
| 7.3 | Collocation | `76_collocation_and_finite_elements` | yes | yes | yes | yes | yes |
| 7.3 | Finite elements and the Galerkin method | `76_collocation_and_finite_elements` | yes | yes | yes | yes | yes |

**8 Partial Differential Equations**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 8.1 | Forward difference method for parabolic equations | `78_parabolic_equations` | yes | yes | yes | yes | yes |
| 8.1 | Stability analysis of the forward difference method | `79_consistency_convergence_and_stability` | yes | yes | yes | yes | yes |
| 8.1 | Backward difference method | `78_parabolic_equations` | yes | yes | yes | yes | yes |
| 8.1 | Crank-Nicolson method | `78_parabolic_equations` | yes | yes | yes | yes | yes |
| 8.2 | The wave equation | `81_hyperbolic_equations_and_cfl` | yes | yes | yes | yes | yes |
| 8.2 | The CFL condition | `81_hyperbolic_equations_and_cfl` | yes | yes | yes | yes | yes |
| 8.3 | Finite difference method for elliptic equations | `82_elliptic_equations` | yes | yes | yes | yes | yes |
| RC8 | Heat distribution on a cooling fin | `82_elliptic_equations` | yes | yes | yes | yes | yes |
| 8.3 | Finite element method for elliptic equations | `82_elliptic_equations` | yes | yes | yes | yes | yes |
| 8.4 | Implicit Newton solver for nonlinear PDEs | `83_nonlinear_pdes_and_method_of_lines` | yes | yes | yes | yes | yes |
| 8.4 | Nonlinear equations in two space dimensions | `83_nonlinear_pdes_and_method_of_lines` | yes | yes | yes | yes | yes |

**9 Random Numbers**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 9.1 | Pseudo-random numbers, linear congruential generators | `91_random_number_generation` | yes | yes | yes | yes | yes |
| 9.1 | Exponential and normal random numbers | `91_random_number_generation` | yes | yes | yes | yes | yes |
| 9.2 | Monte Carlo simulation | `92_monte_carlo_methods` | yes | yes | yes | yes | yes |
| 9.2 | Power laws for Monte Carlo estimation | `92_monte_carlo_methods` | yes | yes | yes | yes | yes |
| 9.2 | Quasi-random numbers (Halton sequences) | `92_monte_carlo_methods` | yes | yes | yes | yes | yes |
| 9.3 | Random walks | `93_brownian_motion_and_sdes` | yes | yes | yes | yes | yes |
| 9.3 | Continuous Brownian motion | `93_brownian_motion_and_sdes` | yes | yes | yes | yes | yes |
| 9.4 | Adding noise to differential equations | `93_brownian_motion_and_sdes` | yes | yes | yes | yes | yes |
| 9.4 | Numerical methods for SDEs (Euler-Maruyama, Milstein) | `93_brownian_motion_and_sdes` | yes | yes | yes | yes | yes |
| RC9 | The Black-Scholes formula | `93_brownian_motion_and_sdes` | yes | yes | yes | yes | yes |

**10 Trigonometric Interpolation and the FFT**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 10.1 | Complex arithmetic and roots of unity | `58_trigonometric_interpolation_and_the_dft` | yes | no | yes | yes | yes |
| 10.1 | Discrete Fourier Transform | `58_trigonometric_interpolation_and_the_dft` | yes | no | yes | yes | yes |
| 10.1 | Fast Fourier Transform | `59_the_fft_and_signal_processing` | yes | no | yes | yes | yes |
| 10.2 | DFT Interpolation Theorem | `58_trigonometric_interpolation_and_the_dft` | yes | no | yes | yes | yes |
| 10.2 | Efficient evaluation of trigonometric functions | `58_trigonometric_interpolation_and_the_dft` | yes | no | yes | yes | yes |
| 10.3 | Orthogonality and interpolation | `58_trigonometric_interpolation_and_the_dft` | yes | no | yes | yes | yes |
| 10.3 | Least squares fitting with trigonometric functions | `59_the_fft_and_signal_processing` | yes | no | yes | yes | yes |
| 10.3 | Sound, noise and filtering | `59_the_fft_and_signal_processing` | yes | no | yes | yes | yes |
| RC10 | The Wiener filter | `59_the_fft_and_signal_processing` | yes | no | yes | yes | yes |

**11 Compression**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 11.1 | One-dimensional Discrete Cosine Transform | `60_dct_and_data_compression` | yes | no | yes | yes | yes |
| 11.1 | The DCT and least squares approximation | `60_dct_and_data_compression` | yes | no | yes | yes | yes |
| 11.2 | Two-dimensional DCT | `60_dct_and_data_compression` | yes | no | yes | yes | yes |
| 11.2 | Image compression | `60_dct_and_data_compression` | yes | no | yes | yes | yes |
| 11.2 | Quantization | `60_dct_and_data_compression` | yes | no | yes | yes | yes |
| 11.3 | Information theory and coding | `60_dct_and_data_compression` | yes | no | yes | yes | yes |
| 11.3 | Huffman coding for the JPEG format | `60_dct_and_data_compression` | yes | no | yes | yes | yes |
| 11.4 | Modified Discrete Cosine Transform | `60_dct_and_data_compression` | yes | no | yes | yes | yes |
| 11.4 | Bit quantization | `60_dct_and_data_compression` | yes | no | yes | yes | yes |
| RC11 | A simple audio codec | `60_dct_and_data_compression` | yes | no | yes | yes | yes |

**12 Eigenvalues and Singular Values**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 12.1 | Power iteration | `36_power_methods` | yes | yes | yes | yes | yes |
| 12.1 | Convergence of power iteration | `36_power_methods` | yes | yes | yes | yes | yes |
| 12.1 | Inverse power iteration | `36_power_methods` | yes | yes | yes | yes | yes |
| 12.1 | Rayleigh quotient iteration | `36_power_methods` | yes | yes | yes | yes | yes |
| 12.2 | Simultaneous iteration and normalized simultaneous iteration | `37_qr_algorithm` | yes | no | yes | yes | yes |
| 12.2 | Real Schur form and the QR algorithm | `37_qr_algorithm` | yes | no | yes | yes | yes |
| 12.2 | Upper Hessenberg form | `37_qr_algorithm` | yes | no | yes | yes | yes |
| RC12 | How search engines rate page quality (PageRank) | `43_svd_applications_and_low_rank` | yes | no | yes | yes | yes |
| 12.3 | Geometry of the SVD (unit sphere to hyperellipse) | `41_svd_theory` | yes | no | yes | yes | yes |
| 12.3 | Finding the SVD in general | `41_svd_theory` | yes | no | yes | yes | yes |
| 12.4 | Properties of the SVD | `41_svd_theory` | yes | no | yes | yes | yes |
| 12.4 | Dimension reduction | `43_svd_applications_and_low_rank` | yes | no | yes | yes | yes |
| 12.4 | Compression with the SVD | `43_svd_applications_and_low_rank` | yes | no | yes | yes | yes |
| 12.4 | Calculating the SVD, the condition squaring warning | `42_computing_the_svd` | yes | no | yes | yes | yes |

**13 Optimization**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 13.1 | Golden section search | `85_derivative_free_optimization` | yes | yes | yes | yes | yes |
| 13.1 | Successive parabolic interpolation | `85_derivative_free_optimization` | yes | yes | yes | yes | yes |
| 13.1 | Nelder-Mead search | `85_derivative_free_optimization` | yes | yes | yes | yes | yes |
| 13.2 | Newton's method for optimization | `86_gradient_and_newton_methods` | yes | yes | yes | yes | yes |
| 13.2 | Steepest descent | `86_gradient_and_newton_methods` | yes | yes | yes | yes | yes |
| 13.2 | Conjugate gradient search for nonlinear problems | `87_quasi_newton_and_trust_region` | yes | yes | yes | yes | yes |
| RC13 | Molecular conformation and numerical optimization | `87_quasi_newton_and_trust_region` | yes | yes | yes | yes | yes |

**Appendix A**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| A.1-A.3 | Matrix fundamentals, linear systems, block multiplication | `15_vectors_matrices_and_norms` | yes | yes | yes | yes | yes |
| A.4-A.5 | Eigenvalues and eigenvectors, symmetric matrices | `35_eigenvalue_theory_and_localization` | yes | yes | yes | yes | yes |
| A.6 | Vector calculus: gradient, Jacobian, Hessian | `84_optimization_fundamentals` | yes | yes | yes | yes | yes |

**Appendix B**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| B.1-B.7 | Programming environment basics, translated to NumPy conventions | `01_what_is_numerical_analysis` | yes | yes | yes | yes | yes |

### S2 - Gupta, Numerical Methods: Fundamentals and Applications (Cambridge, 2019)


**1 Number Systems**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 1.2 | Representation of integers in binary, octal and hexadecimal | `02_number_systems_and_representation` | yes | yes | yes | yes | yes |
| 1.2.1 | Conversion from any base to decimal | `02_number_systems_and_representation` | yes | yes | yes | yes | yes |
| 1.2.2 | Conversion between binary, octal and hexadecimal | `02_number_systems_and_representation` | yes | yes | yes | yes | yes |
| 1.2.3 | Conversion from decimal to any other base | `02_number_systems_and_representation` | yes | yes | yes | yes | yes |
| 1.2.4 | Conversion from one base to any other base | `02_number_systems_and_representation` | yes | yes | yes | yes | yes |
| 1.3 | Representation of fractions | `02_number_systems_and_representation` | yes | yes | yes | yes | yes |

**2 Error Analysis**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 2.1 | Absolute, relative and percentage errors | `04_error_types_and_propagation` | yes | yes | yes | yes | yes |
| 2.2.1 | Modeling error | `04_error_types_and_propagation` | yes | yes | yes | yes | yes |
| 2.2.2 | Inherent error (error in the original data) | `04_error_types_and_propagation` | yes | yes | yes | yes | yes |
| 2.2.3 | Blunder | `04_error_types_and_propagation` | yes | yes | yes | yes | yes |
| 2.3.1 | Round-off error | `03_floating_point_arithmetic` | yes | no | yes | yes | yes |
| 2.3.2 | Overflow and underflow | `03_floating_point_arithmetic` | yes | no | yes | yes | yes |
| 2.3.3.1 | Propagated error in arithmetic operations | `04_error_types_and_propagation` | yes | yes | yes | yes | yes |
| 2.3.3.2 | Error propagation in a function of one variable | `04_error_types_and_propagation` | yes | yes | yes | yes | yes |
| 2.3.3.3 | Error propagation in a function of several variables | `04_error_types_and_propagation` | yes | yes | yes | yes | yes |
| 2.3.4 | Truncation error | `04_error_types_and_propagation` | yes | yes | yes | yes | yes |
| 2.3.5 | Machine epsilon | `03_floating_point_arithmetic` | yes | no | yes | yes | yes |
| 2.3.7 | Loss of significance, condition and stability | `05_loss_of_significance_and_cancellation` | yes | yes | yes | yes | yes |
| 2.4 | How error accumulates over many operations | `04_error_types_and_propagation` | yes | yes | yes | yes | yes |

**3 Nonlinear Equations**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 3.1 | Polynomial and transcendental equations | `09_bracketing_methods` | yes | yes | yes | yes | yes |
| 3.2 | Direct, graphical, trial-and-error and iterative approaches | `09_bracketing_methods` | yes | yes | yes | yes | yes |
| 3.3 | Bisection (Bolzano, interval halving) method | `09_bracketing_methods` | yes | yes | yes | yes | yes |
| 3.4 | Fixed-point (successive approximation) method | `10_fixed_point_iteration` | yes | yes | yes | yes | yes |
| 3.5 | Newton-Raphson method | `11_newton_and_secant_methods` | yes | yes | yes | yes | yes |
| 3.6 | Regula falsi (false position) method | `09_bracketing_methods` | yes | yes | yes | yes | yes |
| 3.7 | Secant method | `11_newton_and_secant_methods` | yes | yes | yes | yes | yes |
| 3.8.1-3.8.5 | Convergence proofs for all five root-finding methods | `12_convergence_theory_and_sensitivity` | yes | yes | yes | yes | yes |
| 3.9.1-3.9.5 | Order of convergence for all five methods | `12_convergence_theory_and_sensitivity` | yes | yes | yes | yes | yes |
| 3.10 | Muller method | `11_newton_and_secant_methods` | yes | yes | yes | yes | yes |
| 3.11 | Chebyshev method (third order) | `11_newton_and_secant_methods` | yes | yes | yes | yes | yes |
| 3.12 | Aitken delta-squared acceleration | `10_fixed_point_iteration` | yes | yes | yes | yes | yes |
| 3.13 | Summary and comparison of root-finding methods | `12_convergence_theory_and_sensitivity` | yes | yes | yes | yes | yes |

**4 Nonlinear Systems and Polynomials**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 4.1 | Fixed-point method for systems | `14_nonlinear_systems_of_equations` | yes | yes | yes | yes | yes |
| 4.2 | Seidel iteration method for systems | `14_nonlinear_systems_of_equations` | yes | yes | yes | yes | yes |
| 4.3 | Newton-Raphson method for systems | `14_nonlinear_systems_of_equations` | yes | yes | yes | yes | yes |
| 4.4 | Complex roots | `13_polynomial_root_finding` | yes | yes | yes | yes | yes |
| 4.5.1 | Descartes rule of signs | `13_polynomial_root_finding` | yes | yes | yes | yes | yes |
| 4.5.2 | Sturm sequence for counting real roots | `13_polynomial_root_finding` | yes | yes | yes | yes | yes |
| 4.6 | Birge-Vieta (Horner) method | `13_polynomial_root_finding` | yes | yes | yes | yes | yes |
| 4.7 | Lin-Bairstow method for quadratic factors | `13_polynomial_root_finding` | yes | yes | yes | yes | yes |
| 4.8 | Graeffe root squaring method | `13_polynomial_root_finding` | yes | yes | yes | yes | yes |

**5 Systems of Linear Equations**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 5.2 | Cramer rule and why it is impractical | `17_gaussian_elimination_and_lu` | yes | yes | yes | yes | yes |
| 5.3 | Matrix inversion method | `17_gaussian_elimination_and_lu` | yes | yes | yes | yes | yes |
| 5.4.1 | Doolittle LU method | `17_gaussian_elimination_and_lu` | yes | yes | yes | yes | yes |
| 5.4.2 | Crout LU method | `17_gaussian_elimination_and_lu` | yes | yes | yes | yes | yes |
| 5.4.3 | Cholesky method | `20_symmetric_positive_definite_and_cholesky` | yes | yes | yes | yes | yes |
| 5.5 | Gauss elimination method | `17_gaussian_elimination_and_lu` | yes | yes | yes | yes | yes |
| 5.5.1 | Operational counts for Gauss elimination | `17_gaussian_elimination_and_lu` | yes | yes | yes | yes | yes |
| 5.5.2 | Thomas algorithm for tridiagonal systems | `21_banded_sparse_and_structured_systems` | yes | yes | yes | yes | yes |
| 5.6 | Gauss-Jordan method | `17_gaussian_elimination_and_lu` | yes | yes | yes | yes | yes |
| 5.7 | Comparison of direct methods | `18_pivoting_and_pa_lu` | yes | yes | yes | yes | yes |
| 5.8 | Pivoting strategies | `18_pivoting_and_pa_lu` | yes | yes | yes | yes | yes |
| 5.9-5.10 | Iterative methods and the Jacobi method | `23_classical_iterative_methods` | yes | no | yes | yes | yes |
| 5.11 | Gauss-Seidel method | `23_classical_iterative_methods` | yes | no | yes | yes | yes |
| 5.12 | Relaxation method (SOR) | `23_classical_iterative_methods` | yes | no | yes | yes | yes |
| 5.13 | Convergence criteria for iterative methods | `23_classical_iterative_methods` | yes | no | yes | yes | yes |
| 5.14 | Matrix forms and convergence of iterative methods | `23_classical_iterative_methods` | yes | no | yes | yes | yes |
| 5.16 | Applications of linear systems | `19_conditioning_of_linear_systems` | yes | yes | yes | yes | yes |

**6 Eigenvalues and Eigenvectors**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 6.2 | Eigenvalues and eigenvectors, real and complex | `35_eigenvalue_theory_and_localization` | yes | yes | yes | yes | yes |
| 6.2.3-6.2.4 | Distinct and repeated eigenvalues, independent and dependent eigenvectors | `35_eigenvalue_theory_and_localization` | yes | yes | yes | yes | yes |
| 6.3.1 | Gerschgorin theorem | `35_eigenvalue_theory_and_localization` | yes | yes | yes | yes | yes |
| 6.3.2 | Brauer (Cassini oval) theorem | `35_eigenvalue_theory_and_localization` | yes | yes | yes | yes | yes |
| 6.4 | Rayleigh power method | `36_power_methods` | yes | yes | yes | yes | yes |
| 6.4.1 | Inverse power method | `36_power_methods` | yes | yes | yes | yes | yes |
| 6.4.2 | Shifted power method | `36_power_methods` | yes | yes | yes | yes | yes |
| 6.5 | Rutishauser (LR) decomposition method | `37_qr_algorithm` | yes | no | yes | yes | yes |

**7 Symmetric Eigenproblem**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 7.1.1 | Similarity transformations | `35_eigenvalue_theory_and_localization` | yes | yes | yes | yes | yes |
| 7.1.2 | Orthogonal transformations | `38_symmetric_eigenvalue_problem` | yes | no | yes | yes | yes |
| 7.2 | Jacobi rotation method | `38_symmetric_eigenvalue_problem` | yes | no | yes | yes | yes |
| 7.3 | Sturm sequence for real symmetric tridiagonal matrices | `38_symmetric_eigenvalue_problem` | yes | no | yes | yes | yes |
| 7.4 | Givens method (tridiagonalization by rotations) | `38_symmetric_eigenvalue_problem` | yes | no | yes | yes | yes |
| 7.5 | Householder method (tridiagonalization by reflections) | `38_symmetric_eigenvalue_problem` | yes | no | yes | yes | yes |

**8 Interpolation**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 8.2.1 | Power form of a polynomial | `44_polynomial_interpolation_forms` | yes | no | yes | yes | yes |
| 8.2.2 | Shifted power form | `44_polynomial_interpolation_forms` | yes | no | yes | yes | yes |
| 8.2.3 | Newton form | `44_polynomial_interpolation_forms` | yes | no | yes | yes | yes |
| 8.2.4 | Nested Newton form | `44_polynomial_interpolation_forms` | yes | no | yes | yes | yes |
| 8.2.5 | Recursive algorithm for the nested Newton form | `44_polynomial_interpolation_forms` | yes | no | yes | yes | yes |
| 8.2.6 | Change of center in the Newton form | `44_polynomial_interpolation_forms` | yes | no | yes | yes | yes |
| 8.3 | Lagrange method | `44_polynomial_interpolation_forms` | yes | no | yes | yes | yes |
| 8.4 | Newton divided difference method | `45_divided_differences` | yes | no | yes | yes | yes |
| 8.4.1 | Proof for higher order divided differences | `45_divided_differences` | yes | no | yes | yes | yes |
| 8.4.2 | Advantages of divided differences over Lagrange | `45_divided_differences` | yes | no | yes | yes | yes |
| 8.4.3 | Properties of divided differences | `45_divided_differences` | yes | no | yes | yes | yes |
| 8.5 | Error in the interpolating polynomial | `46_interpolation_error_and_runge_phenomenon` | yes | no | yes | yes | yes |
| 8.7 | Hermite interpolation | `50_hermite_and_piecewise_interpolation` | yes | no | yes | yes | yes |
| 8.8 | Piecewise interpolation | `50_hermite_and_piecewise_interpolation` | yes | no | yes | yes | yes |
| 8.9 | Weierstrass approximation theorem | `54_function_approximation_fundamentals` | yes | no | yes | yes | yes |

**9 Finite Operators**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 9.2.1 | Forward difference operator | `48_finite_difference_operators_and_tables` | yes | no | yes | yes | yes |
| 9.2.2 | Backward difference operator | `48_finite_difference_operators_and_tables` | yes | no | yes | yes | yes |
| 9.2.3 | Central difference operator | `48_finite_difference_operators_and_tables` | yes | no | yes | yes | yes |
| 9.3.1 | Mean (average) operator | `48_finite_difference_operators_and_tables` | yes | no | yes | yes | yes |
| 9.3.2 | Shift operator E | `48_finite_difference_operators_and_tables` | yes | no | yes | yes | yes |
| 9.3.3 | Differential operator D | `48_finite_difference_operators_and_tables` | yes | no | yes | yes | yes |
| 9.4.1 | Linearity and commutativity of finite operators | `48_finite_difference_operators_and_tables` | yes | no | yes | yes | yes |
| 9.4.2 | Interrelations of the finite operators | `48_finite_difference_operators_and_tables` | yes | no | yes | yes | yes |
| 9.5 | Operators applied to standard functions | `48_finite_difference_operators_and_tables` | yes | no | yes | yes | yes |
| 9.6 | Newton divided differences and other finite differences | `48_finite_difference_operators_and_tables` | yes | no | yes | yes | yes |
| 9.7 | Finite difference tables and error propagation in them | `48_finite_difference_operators_and_tables` | yes | no | yes | yes | yes |

**10 Equal Interval Interpolation**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 10.1 | Gregory-Newton forward difference formula and its error | `49_equal_interval_interpolation_formulas` | yes | no | yes | yes | yes |
| 10.2 | Gregory-Newton backward difference formula and its error | `49_equal_interval_interpolation_formulas` | yes | no | yes | yes | yes |
| 10.3 | Central difference formulas | `49_equal_interval_interpolation_formulas` | yes | no | yes | yes | yes |
| 10.4 | Gauss forward central difference formula | `49_equal_interval_interpolation_formulas` | yes | no | yes | yes | yes |
| 10.5 | Gauss backward central difference formula | `49_equal_interval_interpolation_formulas` | yes | no | yes | yes | yes |
| 10.6 | Stirling formula | `49_equal_interval_interpolation_formulas` | yes | no | yes | yes | yes |
| 10.7 | Bessel formula | `49_equal_interval_interpolation_formulas` | yes | no | yes | yes | yes |
| 10.8 | Everett formula | `49_equal_interval_interpolation_formulas` | yes | no | yes | yes | yes |
| 10.9 | Steffensen formula | `49_equal_interval_interpolation_formulas` | yes | no | yes | yes | yes |
| 10.10.1 | Lagrange bivariate interpolation | `53_bivariate_interpolation` | yes | no | yes | yes | yes |
| 10.10.2 | Newton bivariate interpolation for equi-spaced points | `53_bivariate_interpolation` | yes | no | yes | yes | yes |

**11 Splines and Curve Fitting**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 11.2.1 | Cubic spline interpolation | `51_cubic_splines` | yes | no | yes | yes | yes |
| 11.2.2 | Cubic spline for equi-spaced points | `51_cubic_splines` | yes | no | yes | yes | yes |
| 11.3 | Bezier curve | `52_bezier_and_bspline_curves` | yes | no | yes | yes | yes |
| 11.4 | B-spline curve | `52_bezier_and_bspline_curves` | yes | no | yes | yes | yes |
| 11.5.1 | Straight line least squares fitting | `29_least_squares_and_normal_equations` | yes | yes | yes | yes | yes |
| 11.5.2 | Nonlinear curve fitting by linearization | `29_least_squares_and_normal_equations` | yes | yes | yes | yes | yes |
| 11.5.3 | Quadratic curve fitting | `29_least_squares_and_normal_equations` | yes | yes | yes | yes | yes |
| 11.6 | Chebyshev polynomial approximation | `56_chebyshev_approximation_and_economization` | yes | no | yes | yes | yes |
| 11.7 | Pade approximation by rational functions | `57_pade_rational_approximation` | yes | no | yes | yes | yes |

**12 Numerical Differentiation**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 12.2 | Differentiation formulas derived from interpolation | `61_numerical_differentiation` | yes | yes | yes | yes | yes |
| 12.2 | Summary table of differentiation formulas | `61_numerical_differentiation` | yes | yes | yes | yes | yes |

**13 Numerical Integration**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 13.1 | Newton-Cotes quadrature derived via Lagrange | `62_newton_cotes_quadrature` | yes | yes | yes | yes | yes |
| 13.1.1 | Trapezoidal rule | `62_newton_cotes_quadrature` | yes | yes | yes | yes | yes |
| 13.1.2 | Simpson 1/3 rule | `62_newton_cotes_quadrature` | yes | yes | yes | yes | yes |
| 13.1.3 | Simpson 3/8 rule | `62_newton_cotes_quadrature` | yes | yes | yes | yes | yes |
| 13.1.4 | Boole rule | `62_newton_cotes_quadrature` | yes | yes | yes | yes | yes |
| 13.1.5 | Weddle rule | `62_newton_cotes_quadrature` | yes | yes | yes | yes | yes |
| 13.2 | Composite Newton-Cotes rules | `62_newton_cotes_quadrature` | yes | yes | yes | yes | yes |
| 13.3 | Error terms for all Newton-Cotes rules | `62_newton_cotes_quadrature` | yes | yes | yes | yes | yes |
| 13.4.1 | Gauss-Legendre formula | `65_gaussian_quadrature` | yes | yes | yes | yes | yes |
| 13.4.2 | Gauss-Chebyshev formula | `65_gaussian_quadrature` | yes | yes | yes | yes | yes |
| 13.4.3 | Gauss-Laguerre formula | `65_gaussian_quadrature` | yes | yes | yes | yes | yes |
| 13.4.4 | Gauss-Hermite formula | `65_gaussian_quadrature` | yes | yes | yes | yes | yes |
| 13.5 | Euler-Maclaurin formula | `63_richardson_romberg_and_euler_maclaurin` | yes | yes | yes | yes | yes |
| 13.6 | Richardson extrapolation | `63_richardson_romberg_and_euler_maclaurin` | yes | yes | yes | yes | yes |
| 13.7 | Romberg integration | `63_richardson_romberg_and_euler_maclaurin` | yes | yes | yes | yes | yes |
| 13.8 | Double integrals by trapezoidal and Simpson rules | `66_improper_and_multiple_integrals` | yes | yes | yes | yes | yes |

**14 First Order ODE IVPs**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 14.1 | Classification and terminology of differential equations | `67_ivp_theory_and_eulers_method` | yes | yes | yes | yes | yes |
| 14.1.8 | Existence and uniqueness of solutions | `67_ivp_theory_and_eulers_method` | yes | yes | yes | yes | yes |
| 14.2 | Picard method of successive approximations | `68_taylor_and_picard_methods` | yes | yes | yes | yes | yes |
| 14.3 | Taylor series method | `68_taylor_and_picard_methods` | yes | yes | yes | yes | yes |
| 14.4 | Euler method | `67_ivp_theory_and_eulers_method` | yes | yes | yes | yes | yes |
| 14.5 | Modified and improved Euler (Heun) method | `69_runge_kutta_methods` | yes | yes | yes | yes | yes |
| 14.6 | Runge-Kutta methods | `69_runge_kutta_methods` | yes | yes | yes | yes | yes |
| 14.7 | Milne (Milne-Simpson) method | `71_multistep_methods` | yes | yes | yes | yes | yes |
| 14.8 | Adams-Bashforth predictor and Adams-Moulton corrector | `71_multistep_methods` | yes | yes | yes | yes | yes |
| 14.9 | Errors in numerical methods for ODEs | `67_ivp_theory_and_eulers_method` | yes | yes | yes | yes | yes |
| 14.10 | Order and stability of numerical methods | `72_stability_and_stiff_equations` | yes | yes | yes | yes | yes |
| 14.11 | Stability analysis of the linear system y-prime = Ay | `72_stability_and_stiff_equations` | yes | yes | yes | yes | yes |
| 14.12 | Backward Euler method | `72_stability_and_stiff_equations` | yes | yes | yes | yes | yes |

**15 ODE Systems and BVPs**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 15.1-15.4 | Picard, Taylor, Euler and RK4 for systems of ODEs | `73_systems_and_higher_order_equations` | yes | yes | yes | yes | yes |
| 15.5 | Shooting method for boundary value problems | `75_boundary_value_problems` | yes | yes | yes | yes | yes |
| 15.6.1 | Finite difference approximations for first derivatives | `61_numerical_differentiation` | yes | yes | yes | yes | yes |
| 15.6.2 | Finite difference approximations for second derivatives | `61_numerical_differentiation` | yes | yes | yes | yes | yes |
| 15.7 | Finite difference method for boundary value problems | `75_boundary_value_problems` | yes | yes | yes | yes | yes |
| 15.8 | Finite difference approximations for unequal intervals | `75_boundary_value_problems` | yes | yes | yes | yes | yes |

**16 Partial Differential Equations**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| 16.1 | Classification of second-order quasi-linear PDEs | `77_pde_classification_and_stencils` | yes | yes | yes | yes | yes |
| 16.2 | Initial and boundary conditions | `77_pde_classification_and_stencils` | yes | yes | yes | yes | yes |
| 16.3 | Finite difference approximations for partial derivatives | `77_pde_classification_and_stencils` | yes | yes | yes | yes | yes |
| 16.4.1 | Bender-Schmidt explicit scheme | `78_parabolic_equations` | yes | yes | yes | yes | yes |
| 16.4.2 | Crank-Nicolson scheme | `78_parabolic_equations` | yes | yes | yes | yes | yes |
| 16.4.3 | General implicit (weighted) scheme | `78_parabolic_equations` | yes | yes | yes | yes | yes |
| 16.4.4 | Richardson scheme and its instability | `78_parabolic_equations` | yes | yes | yes | yes | yes |
| 16.4.5 | Du-Fort and Frankel scheme | `78_parabolic_equations` | yes | yes | yes | yes | yes |
| 16.5.1-16.5.3 | Consistency, convergence and order of a scheme | `79_consistency_convergence_and_stability` | yes | yes | yes | yes | yes |
| 16.5.4 | Stability of a finite difference scheme | `79_consistency_convergence_and_stability` | yes | yes | yes | yes | yes |
| 16.5.5-16.5.6 | Matrix method for stability (explicit and Crank-Nicolson) | `79_consistency_convergence_and_stability` | yes | yes | yes | yes | yes |
| 16.5.7-16.5.8 | Von Neumann method for stability (explicit and Crank-Nicolson) | `79_consistency_convergence_and_stability` | yes | yes | yes | yes | yes |
| 16.6.1-16.6.2 | Two dimensional heat conduction, explicit and CN schemes | `80_multidimensional_parabolic_and_adi` | yes | yes | yes | yes | yes |
| 16.6.3 | Alternating Direction Implicit (ADI) scheme | `80_multidimensional_parabolic_and_adi` | yes | yes | yes | yes | yes |
| 16.7.1 | Laplace equation by finite differences | `82_elliptic_equations` | yes | yes | yes | yes | yes |
| 16.7.2 | Poisson equation by finite differences | `82_elliptic_equations` | yes | yes | yes | yes | yes |
| 16.8.1 | Wave equation, explicit scheme | `81_hyperbolic_equations_and_cfl` | yes | yes | yes | yes | yes |
| 16.8.2 | Wave equation, implicit scheme | `81_hyperbolic_equations_and_cfl` | yes | yes | yes | yes | yes |
| 16.9 | Building your own finite difference scheme | `79_consistency_convergence_and_stability` | yes | yes | yes | yes | yes |

**Appendix A**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| A | Comparison of analytical and numerical techniques | `01_what_is_numerical_analysis` | yes | yes | yes | yes | yes |

**Appendix B**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| B | Numerical techniques and the computer | `01_what_is_numerical_analysis` | yes | yes | yes | yes | yes |

**Appendix C**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| C | Taylor series reference | `07_taylor_series_and_convergence_rates` | yes | yes | yes | yes | yes |

**Appendix D**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| D | Linear and nonlinear problems | `01_what_is_numerical_analysis` | yes | yes | yes | yes | yes |

### SUP - Supplementary / Advanced Material (not from any supplied source)


**-**

| Section | Concept | Lesson | cov | impl | exp | test | ver |
|---|---|---|---|---|---|---|---|
| - | Compensated (Kahan) summation | `05_loss_of_significance_and_cancellation` | yes | yes | yes | yes | yes |
| - | Neumaier summation, fixing Kahan's weak case | `05_loss_of_significance_and_cancellation` | yes | yes | yes | yes | yes |
| - | Pairwise summation and its log n error bound | `05_loss_of_significance_and_cancellation` | yes | yes | yes | yes | yes |
| - | Sterbenz lemma: when subtraction is exact | `05_loss_of_significance_and_cancellation` | yes | yes | yes | yes | yes |
| - | Rewriting formulas to remove cancellation (conjugate, half angle) | `05_loss_of_significance_and_cancellation` | yes | yes | yes | yes | yes |
| - | Measuring observed order of convergence numerically | `07_taylor_series_and_convergence_rates` | yes | yes | yes | yes | yes |
| - | Flop counting, memory cost and BLAS levels 1, 2 and 3 | `08_algorithm_cost_and_complexity` | yes | yes | yes | yes | yes |
| - | Why flop counts alone mislead: memory layout and blocking | `08_algorithm_cost_and_complexity` | yes | yes | yes | yes | yes |
| - | Companion matrix approach to polynomial roots | `13_polynomial_root_finding` | yes | yes | yes | yes | yes |
| - | Damped Newton and line search for nonlinear systems | `14_nonlinear_systems_of_equations` | yes | yes | yes | yes | yes |
| - | LDL-transpose factorization | `20_symmetric_positive_definite_and_cholesky` | yes | yes | yes | yes | yes |
| - | Sparse storage formats (CSR, CSC, COO), fill-in and reordering | `21_banded_sparse_and_structured_systems` | yes | yes | yes | yes | yes |
| - | Condition number estimation without forming the inverse | `22_condition_estimation_and_iterative_refinement` | yes | yes | yes | yes | yes |
| - | Iterative refinement and mixed precision refinement | `22_condition_estimation_and_iterative_refinement` | yes | yes | yes | yes | yes |
| - | Multigrid: smoothing, coarse grid correction and the V-cycle | `28_multigrid_methods` | yes | yes | yes | yes | yes |
| - | Tikhonov regularization and the L-curve | `32_solving_least_squares_in_practice` | yes | yes | yes | yes | yes |
| - | QR with column pivoting and numerical rank | `33_rank_revealing_qr_and_total_least_squares` | yes | yes | yes | yes | yes |
| - | Total least squares via the SVD | `33_rank_revealing_qr_and_total_least_squares` | yes | yes | yes | yes | yes |
| - | Divide and conquer for the symmetric tridiagonal eigenproblem | `38_symmetric_eigenvalue_problem` | yes | no | yes | yes | yes |
| - | Lanczos and Arnoldi as eigenvalue solvers, implicit restarting | `39_krylov_methods_for_eigenvalues` | yes | no | yes | yes | yes |
| - | Generalized eigenvalue problem, Cholesky reduction and the QZ idea | `40_generalized_eigenvalue_problem` | yes | no | yes | yes | yes |
| - | Randomized range finder and randomized SVD | `43_svd_applications_and_low_rank` | yes | no | yes | yes | yes |
| - | Conditioning of the Vandermonde matrix | `44_polynomial_interpolation_forms` | yes | no | yes | yes | yes |
| - | Barycentric form of Lagrange interpolation | `44_polynomial_interpolation_forms` | yes | no | yes | yes | yes |
| - | Lebesgue constant and the near-minimax property of Chebyshev nodes | `47_chebyshev_interpolation` | yes | no | yes | yes | yes |
| - | de Casteljau algorithm and B-spline basis functions | `52_bezier_and_bspline_curves` | yes | no | yes | yes | yes |
| - | Curse of dimensionality in multivariate interpolation | `53_bivariate_interpolation` | yes | no | yes | yes | yes |
| - | Norms on function spaces and existence of best approximations | `54_function_approximation_fundamentals` | yes | no | yes | yes | yes |
| - | Minimax versus least squares approximation, equioscillation | `54_function_approximation_fundamentals` | yes | no | yes | yes | yes |
| - | Gram-Schmidt on functions and three-term recurrences | `55_orthogonal_polynomials` | yes | no | yes | yes | yes |
| - | Legendre, Chebyshev, Laguerre and Hermite polynomial families | `55_orthogonal_polynomials` | yes | no | yes | yes | yes |
| - | Radix-2 FFT implemented from scratch | `59_the_fft_and_signal_processing` | yes | no | yes | yes | yes |
| - | Complex-step differentiation | `61_numerical_differentiation` | yes | yes | yes | yes | yes |
| - | Improper integrals, variable transformation and singularity handling | `66_improper_and_multiple_integrals` | yes | yes | yes | yes | yes |
| - | Butcher tableau notation and Runge-Kutta order conditions | `69_runge_kutta_methods` | yes | yes | yes | yes | yes |
| - | Root condition, zero-stability and the Dahlquist barriers | `71_multistep_methods` | yes | yes | yes | yes | yes |
| - | Regions of absolute stability, A-stability and L-stability | `72_stability_and_stiff_equations` | yes | yes | yes | yes | yes |
| - | Symplectic Euler and Stormer-Verlet, energy behaviour over long times | `74_geometric_and_symplectic_integrators` | yes | yes | yes | yes | yes |
| - | Lax equivalence theorem | `79_consistency_convergence_and_stability` | yes | yes | yes | yes | yes |
| - | Method of lines linking PDEs to ODE solvers | `83_nonlinear_pdes_and_method_of_lines` | yes | yes | yes | yes | yes |
| - | Optimality conditions, convexity and the Hessian test | `84_optimization_fundamentals` | yes | yes | yes | yes | yes |
| - | Conditioning of an optimization problem | `84_optimization_fundamentals` | yes | yes | yes | yes | yes |
| - | Wolfe conditions and backtracking line search | `86_gradient_and_newton_methods` | yes | yes | yes | yes | yes |
| - | Secant condition, BFGS and L-BFGS | `87_quasi_newton_and_trust_region` | yes | yes | yes | yes | yes |
| - | Trust region methods | `87_quasi_newton_and_trust_region` | yes | yes | yes | yes | yes |
| - | Lagrange multipliers and the KKT conditions | `88_constrained_optimization` | yes | yes | yes | yes | yes |
| - | Penalty, barrier and projected gradient methods | `88_constrained_optimization` | yes | yes | yes | yes | yes |
| - | Mersenne Twister and PCG64 generators, statistical tests | `91_random_number_generation` | yes | yes | yes | yes | yes |
| - | Variance reduction: antithetic variates and control variates | `92_monte_carlo_methods` | yes | yes | yes | yes | yes |
| - | PCA as a truncated SVD, whitening | `94_numerical_linear_algebra_in_machine_learning` | yes | yes | yes | yes | yes |
| - | Ridge regression as regularized least squares | `94_numerical_linear_algebra_in_machine_learning` | yes | yes | yes | yes | yes |
| - | Low-rank adaptation as a low-rank matrix factorization | `94_numerical_linear_algebra_in_machine_learning` | yes | yes | yes | yes | yes |
| - | Gradient descent as an ODE discretization | `95_optimization_for_machine_learning` | yes | yes | yes | yes | yes |
| - | Stochastic and minibatch gradient descent, and gradient noise | `89_stochastic_optimization` | yes | yes | yes | yes | yes |
| - | Momentum, Nesterov acceleration and the heavy ball method | `89_stochastic_optimization` | yes | yes | yes | yes | yes |
| - | Adaptive step sizes: AdaGrad, RMSProp, Adam and AdamW | `89_stochastic_optimization` | yes | yes | yes | yes | yes |
| - | Learning rate schedules and the convergence rate of SGD | `89_stochastic_optimization` | yes | yes | yes | yes | yes |
| - | The proximal operator and proximal gradient descent | `90_proximal_and_composite_optimization` | yes | yes | yes | yes | yes |
| - | ISTA and FISTA acceleration for composite problems | `90_proximal_and_composite_optimization` | yes | yes | yes | yes | yes |
| - | L1 regularization, LASSO and the soft thresholding operator | `90_proximal_and_composite_optimization` | yes | yes | yes | yes | yes |
| - | Sharp and flat minima, batch size and generalization | `95_optimization_for_machine_learning` | yes | yes | yes | yes | yes |
| - | Natural gradient and Gauss-Newton in machine learning | `95_optimization_for_machine_learning` | yes | yes | yes | yes | yes |
| - | float16 and bfloat16, exponent and mantissa trade-offs | `96_floating_point_in_deep_learning` | yes | yes | yes | yes | yes |
| - | The logsumexp trick and numerically stable softmax | `96_floating_point_in_deep_learning` | yes | yes | yes | yes | yes |
| - | Loss scaling and higher-precision accumulation | `96_floating_point_in_deep_learning` | yes | yes | yes | yes | yes |
| - | Non-determinism from non-associative floating point reduction | `96_floating_point_in_deep_learning` | yes | yes | yes | yes | yes |
| - | Forward mode automatic differentiation with dual numbers | `97_automatic_differentiation` | yes | yes | yes | yes | yes |
| - | Reverse mode automatic differentiation with a tape | `97_automatic_differentiation` | yes | yes | yes | yes | yes |
| - | Cost of automatic differentiation versus finite differences | `97_automatic_differentiation` | yes | yes | yes | yes | yes |
| - | Inverse problems and regularization | `98_scientific_machine_learning_and_inverse_problems` | yes | yes | yes | yes | yes |
| - | Physics-informed neural networks viewed as collocation | `98_scientific_machine_learning_and_inverse_problems` | yes | yes | yes | yes | yes |
| - | Neural ODEs | `98_scientific_machine_learning_and_inverse_problems` | yes | yes | yes | yes | yes |
| - | Krylov and low-rank methods inside large-scale learning | `98_scientific_machine_learning_and_inverse_problems` | yes | yes | yes | yes | yes |
| - | Matrix-free operators: Krylov methods need only a matvec | `26_krylov_subspaces_arnoldi_and_lanczos` | yes | yes | yes | yes | yes |
| - | Sketching for least squares | `43_svd_applications_and_low_rank` | yes | no | yes | yes | yes |

