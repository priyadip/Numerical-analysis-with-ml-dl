# Course Map

Every lesson in the course, in order. Each row is one lesson, and every lesson is
a pair of files with the same name: a runnable notebook (`.ipynb`) and a readable
lesson (`.md`).

**14 parts, 98 lessons, all written.**

Lesson names in **bold with a link** are written and verified. Names in plain text
are planned and not yet written, so they have nothing to link to.

If you are new here, read `README.md` first, then follow `LEARNING_PATH.md`.
For how topics depend on each other, see `COURSE_DEPENDENCIES.md`.

## Parts at a glance

| Part | Folder | Lessons | Status |
|---|---|---|---|
| Foundations of Numerical Computing | [`01_foundations/`](01_foundations/) | 01 to 08 (8) | complete |
| Nonlinear Equations and Root Finding | [`02_root_finding/`](02_root_finding/) | 09 to 14 (6) | complete |
| Direct Methods for Linear Systems | [`03_direct_linear_systems/`](03_direct_linear_systems/) | 15 to 22 (8) | complete |
| Iterative and Krylov Subspace Methods | [`04_iterative_and_krylov/`](04_iterative_and_krylov/) | 23 to 28 (6) | complete |
| Orthogonality, QR and Least Squares | [`05_least_squares_and_qr/`](05_least_squares_and_qr/) | 29 to 34 (6) | complete |
| Eigenvalue Problems and the Singular Value Decomposition | [`06_eigenvalues_and_svd/`](06_eigenvalues_and_svd/) | 35 to 43 (9) | complete |
| Interpolation | [`07_interpolation/`](07_interpolation/) | 44 to 53 (10) | complete |
| Approximation Theory and Transforms | [`08_approximation_and_transforms/`](08_approximation_and_transforms/) | 54 to 60 (7) | complete |
| Numerical Differentiation and Integration | [`09_differentiation_and_integration/`](09_differentiation_and_integration/) | 61 to 66 (6) | complete |
| Ordinary Differential Equations | [`10_ordinary_differential_equations/`](10_ordinary_differential_equations/) | 67 to 76 (10) | complete |
| Partial Differential Equations | [`11_partial_differential_equations/`](11_partial_differential_equations/) | 77 to 83 (7) | complete |
| Numerical Optimization | [`12_optimization/`](12_optimization/) | 84 to 90 (7) | complete |
| Stochastic and Monte Carlo Methods | [`13_stochastic_methods/`](13_stochastic_methods/) | 91 to 93 (3) | complete |
| Numerical Analysis in Machine Learning and AI | [`14_ml_and_ai_connections/`](14_ml_and_ai_connections/) | 94 to 98 (5) | complete |

## Foundations of Numerical Computing

How computers represent numbers, where error comes from, and how to tell a hard problem from a bad algorithm.

Folder: [`01_foundations/`](01_foundations/)

| # | Lesson | What it covers |
|---|---|---|
| 01 | **[01_what_is_numerical_analysis](01_foundations/01_what_is_numerical_analysis.md)** | Problem versus algorithm, approximation, nested polynomial evaluation, why efficiency is a mathematical question |
| 02 | **[02_number_systems_and_representation](01_foundations/02_number_systems_and_representation.md)** | Binary, octal, decimal and hexadecimal; integers and fractions; conversion in every direction |
| 03 | **[03_floating_point_arithmetic](01_foundations/03_floating_point_arithmetic.md)** | IEEE 754 layout, normalization, machine epsilon, rounding, subnormals, overflow and underflow, the fl model |
| 04 | **[04_error_types_and_propagation](01_foundations/04_error_types_and_propagation.md)** | Absolute and relative error, the full error taxonomy, and how error moves through arithmetic and functions |
| 05 | **[05_loss_of_significance_and_cancellation](01_foundations/05_loss_of_significance_and_cancellation.md)** | Catastrophic cancellation, rewriting unstable formulas, and summation done properly |
| 06 | **[06_conditioning_and_stability](01_foundations/06_conditioning_and_stability.md)** | Forward and backward error, condition number of a problem, backward stability of an algorithm |
| 07 | **[07_taylor_series_and_convergence_rates](01_foundations/07_taylor_series_and_convergence_rates.md)** | Taylor with remainder, the calculus theorems we keep using, order notation and measuring observed order |
| 08 | **[08_algorithm_cost_and_complexity](01_foundations/08_algorithm_cost_and_complexity.md)** | Counting flops, memory cost, why flop counts alone mislead, and the effect of memory layout |

## Nonlinear Equations and Root Finding

Solving f(x) = 0. The cheapest place to meet convergence order, sensitivity and backward error.

Folder: [`02_root_finding/`](02_root_finding/)

| # | Lesson | What it covers |
|---|---|---|
| 09 | **[09_bracketing_methods](02_root_finding/09_bracketing_methods.md)** | Bisection, regula falsi and the Illinois fix, with guaranteed error bounds |
| 10 | **[10_fixed_point_iteration](02_root_finding/10_fixed_point_iteration.md)** | Fixed points, the contraction mapping theorem, cobweb geometry, stopping criteria, Aitken and Steffensen |
| 11 | **[11_newton_and_secant_methods](02_root_finding/11_newton_and_secant_methods.md)** | Newton and its quadratic convergence, multiple roots, secant, Muller, the Chebyshev third-order method, Brent |
| 12 | **[12_convergence_theory_and_sensitivity](02_root_finding/12_convergence_theory_and_sensitivity.md)** | Order of convergence for every method side by side, plus the conditioning of a root and the Wilkinson polynomial |
| 13 | **[13_polynomial_root_finding](02_root_finding/13_polynomial_root_finding.md)** | Horner, Birge-Vieta, deflation, Descartes rule, Sturm sequences, Lin-Bairstow, Graeffe, companion matrices |
| 14 | **[14_nonlinear_systems_of_equations](02_root_finding/14_nonlinear_systems_of_equations.md)** | Multivariate fixed point and Seidel iteration, Newton with the Jacobian, Broyden, damping and line search |

## Direct Methods for Linear Systems

Solving Ax = b by elimination and factorization, and knowing when the answer can be trusted.

Folder: [`03_direct_linear_systems/`](03_direct_linear_systems/)

| # | Lesson | What it covers |
|---|---|---|
| 15 | **[15_vectors_matrices_and_norms](03_direct_linear_systems/15_vectors_matrices_and_norms.md)** | Vector and matrix norms, induced norms, equivalence, submultiplicativity, spectral radius |
| 16 | **[16_orthogonality_and_projectors](03_direct_linear_systems/16_orthogonality_and_projectors.md)** | Orthonormal bases, orthogonal and unitary matrices, why they cannot amplify error, projectors, and the geometry of projection |
| 17 | **[17_gaussian_elimination_and_lu](03_direct_linear_systems/17_gaussian_elimination_and_lu.md)** | Naive elimination, substitution, operation counts, LU, Doolittle and Crout, Gauss-Jordan, why Cramer fails |
| 18 | **[18_pivoting_and_pa_lu](03_direct_linear_systems/18_pivoting_and_pa_lu.md)** | Swamping, partial and complete pivoting, permutation matrices, PA=LU, the growth factor |
| 19 | **[19_conditioning_of_linear_systems](03_direct_linear_systems/19_conditioning_of_linear_systems.md)** | Residual versus error, the matrix condition number, error magnification, Hilbert matrices |
| 20 | **[20_symmetric_positive_definite_and_cholesky](03_direct_linear_systems/20_symmetric_positive_definite_and_cholesky.md)** | Positive definiteness and how to test it, Cholesky factorization, LDL, when Cholesky beats LU |
| 21 | **[21_banded_sparse_and_structured_systems](03_direct_linear_systems/21_banded_sparse_and_structured_systems.md)** | Tridiagonal systems and the Thomas algorithm, banded solvers, sparse storage formats, fill-in and ordering |
| 22 | **[22_condition_estimation_and_iterative_refinement](03_direct_linear_systems/22_condition_estimation_and_iterative_refinement.md)** | Estimating the condition number without inverting, and recovering accuracy with iterative refinement |

## Iterative and Krylov Subspace Methods

Solving Ax = b when A is large and sparse, by improving a guess instead of factorizing.

Folder: [`04_iterative_and_krylov/`](04_iterative_and_krylov/)

| # | Lesson | What it covers |
|---|---|---|
| 23 | **[23_classical_iterative_methods](04_iterative_and_krylov/23_classical_iterative_methods.md)** | Matrix splitting, Jacobi, Gauss-Seidel, SOR and relaxation, convergence by spectral radius |
| 24 | **[24_conjugate_gradient](04_iterative_and_krylov/24_conjugate_gradient.md)** | The quadratic form view, why steepest descent zigzags, the A-inner product, conjugacy, CG and its convergence bound |
| 25 | **[25_preconditioning](04_iterative_and_krylov/25_preconditioning.md)** | What a preconditioner does to the spectrum: Jacobi, SSOR, incomplete Cholesky, preconditioned CG |
| 26 | **[26_krylov_subspaces_arnoldi_and_lanczos](04_iterative_and_krylov/26_krylov_subspaces_arnoldi_and_lanczos.md)** | Krylov subspaces, why the naive basis is useless, Arnoldi, Lanczos, loss of orthogonality, Ritz values |
| 27 | **[27_gmres_and_nonsymmetric_solvers](04_iterative_and_krylov/27_gmres_and_nonsymmetric_solvers.md)** | GMRES as least squares over a Krylov space, Givens rotations, restarting, MINRES and BiCGSTAB |
| 28 | **[28_multigrid_methods](04_iterative_and_krylov/28_multigrid_methods.md)** | Why smoothers stall on low frequencies, coarse grid correction, the V-cycle, and mesh-independent convergence |

## Orthogonality, QR and Least Squares

Overdetermined systems, projection, orthogonalization, and the numerically sound way to fit data.

Folder: [`05_least_squares_and_qr/`](05_least_squares_and_qr/)

| # | Lesson | What it covers |
|---|---|---|
| 29 | **[29_least_squares_and_normal_equations](05_least_squares_and_qr/29_least_squares_and_normal_equations.md)** | Inconsistent systems, normal equations, projection geometry, model fitting, and the squared condition number |
| 30 | **[30_gram_schmidt_and_qr](05_least_squares_and_qr/30_gram_schmidt_and_qr.md)** | Projection, classical and modified Gram-Schmidt, reduced versus full QR, the loss-of-orthogonality experiment |
| 31 | **[31_householder_and_givens_qr](05_least_squares_and_qr/31_householder_and_givens_qr.md)** | Householder reflectors and the sign trick, Householder QR, Givens rotations, stability comparison |
| 32 | **[32_solving_least_squares_in_practice](05_least_squares_and_qr/32_solving_least_squares_in_practice.md)** | Normal equations versus QR versus SVD on the same hard problem, pseudoinverse, Tikhonov regularization |
| 33 | **[33_rank_revealing_qr_and_total_least_squares](05_least_squares_and_qr/33_rank_revealing_qr_and_total_least_squares.md)** | QR with column pivoting, numerical rank, subset selection, and total least squares when both sides have error |
| 34 | **[34_nonlinear_least_squares](05_least_squares_and_qr/34_nonlinear_least_squares.md)** | Gauss-Newton, Levenberg-Marquardt, models with nonlinear parameters, GPS positioning worked end to end |

## Eigenvalue Problems and the Singular Value Decomposition

Finding eigenvalues, eigenvectors and singular values, from the power method up to shifted QR and Krylov eigensolvers.

Folder: [`06_eigenvalues_and_svd/`](06_eigenvalues_and_svd/)

| # | Lesson | What it covers |
|---|---|---|
| 35 | **[35_eigenvalue_theory_and_localization](06_eigenvalues_and_svd/35_eigenvalue_theory_and_localization.md)** | Similarity, diagonalizability, defective matrices, Schur form, Gerschgorin and Brauer bounds, eigenvalue conditioning |
| 36 | **[36_power_methods](06_eigenvalues_and_svd/36_power_methods.md)** | Power iteration and its rate, inverse power iteration, shifts, the Rayleigh quotient and its cubic convergence |
| 37 | **[37_qr_algorithm](06_eigenvalues_and_svd/37_qr_algorithm.md)** | Simultaneous iteration, the unshifted QR algorithm, Hessenberg reduction, shifts, deflation, real Schur form, the LR method |
| 38 | **[38_symmetric_eigenvalue_problem](06_eigenvalues_and_svd/38_symmetric_eigenvalue_problem.md)** | Jacobi rotations, Givens and Householder tridiagonalization, Sturm sequence bisection, divide and conquer |
| 39 | **[39_krylov_methods_for_eigenvalues](06_eigenvalues_and_svd/39_krylov_methods_for_eigenvalues.md)** | Lanczos and Arnoldi for eigenvalues, Ritz value convergence, restarting, solving huge sparse eigenproblems |
| 40 | **[40_generalized_eigenvalue_problem](06_eigenvalues_and_svd/40_generalized_eigenvalue_problem.md)** | Ax = lambda Bx, reduction by Cholesky, the QZ idea, and where generalized problems come from |
| 41 | **[41_svd_theory](06_eigenvalues_and_svd/41_svd_theory.md)** | The unit sphere mapped to a hyperellipse, existence and uniqueness, the four fundamental subspaces, pseudoinverse |
| 42 | **[42_computing_the_svd](06_eigenvalues_and_svd/42_computing_the_svd.md)** | Why forming A-transpose-A is wrong, Golub-Kahan bidiagonalization, implicit QR sweeps, one-sided Jacobi |
| 43 | **[43_svd_applications_and_low_rank](06_eigenvalues_and_svd/43_svd_applications_and_low_rank.md)** | Numerical rank, truncated SVD, Eckart-Young-Mirsky, image compression, PageRank, randomized SVD |

## Interpolation

Building a function that passes exactly through given data, and knowing how wrong it is between the points.

Folder: [`07_interpolation/`](07_interpolation/)

| # | Lesson | What it covers |
|---|---|---|
| 44 | **[44_polynomial_interpolation_forms](07_interpolation/44_polynomial_interpolation_forms.md)** | Existence and uniqueness, power, shifted power, Newton and nested Newton forms, Lagrange, the Vandermonde system |
| 45 | **[45_divided_differences](07_interpolation/45_divided_differences.md)** | Newton divided differences, the recursion and its proof, properties, relation to derivatives, adding a point cheaply |
| 46 | **[46_interpolation_error_and_runge_phenomenon](07_interpolation/46_interpolation_error_and_runge_phenomenon.md)** | The error formula with proof, how error behaves as nodes are added, the Runge phenomenon, node placement |
| 47 | **[47_chebyshev_interpolation](07_interpolation/47_chebyshev_interpolation.md)** | Chebyshev points, the minimax theorem for the node polynomial, Chebyshev recurrence, Lebesgue constants |
| 48 | **[48_finite_difference_operators_and_tables](07_interpolation/48_finite_difference_operators_and_tables.md)** | Forward, backward, central, mean, shift and differential operators, their interrelations, difference tables and error spread |
| 49 | **[49_equal_interval_interpolation_formulas](07_interpolation/49_equal_interval_interpolation_formulas.md)** | Gregory-Newton forward and backward, Gauss central formulas, Stirling, Bessel, Everett, Steffensen, and when to use each |
| 50 | **[50_hermite_and_piecewise_interpolation](07_interpolation/50_hermite_and_piecewise_interpolation.md)** | Interpolating derivative data, osculating polynomials, piecewise linear error, the Weierstrass theorem |
| 51 | **[51_cubic_splines](07_interpolation/51_cubic_splines.md)** | Spline conditions, the tridiagonal system, natural, clamped, parabolic and not-a-knot ends, the equi-spaced case |
| 52 | **[52_bezier_and_bspline_curves](07_interpolation/52_bezier_and_bspline_curves.md)** | Bezier curves and control points, de Casteljau, B-spline basis functions, local control, fonts |
| 53 | **[53_bivariate_interpolation](07_interpolation/53_bivariate_interpolation.md)** | Lagrange and Newton interpolation on a grid, tensor product structure, and what breaks in higher dimensions |

## Approximation Theory and Transforms

Fitting data and functions when exact interpolation is the wrong goal, plus the Fourier and cosine transforms.

Folder: [`08_approximation_and_transforms/`](08_approximation_and_transforms/)

| # | Lesson | What it covers |
|---|---|---|
| 54 | **[54_function_approximation_fundamentals](08_approximation_and_transforms/54_function_approximation_fundamentals.md)** | Norms on function spaces, best approximation, existence and uniqueness, minimax versus least squares, Weierstrass |
| 55 | **[55_orthogonal_polynomials](08_approximation_and_transforms/55_orthogonal_polynomials.md)** | Gram-Schmidt on functions, three-term recurrences, Legendre, Chebyshev, Laguerre and Hermite families |
| 56 | **[56_chebyshev_approximation_and_economization](08_approximation_and_transforms/56_chebyshev_approximation_and_economization.md)** | Chebyshev series, near-minimax behaviour, economization of power series, comparison with truncated Taylor |
| 57 | **[57_pade_rational_approximation](08_approximation_and_transforms/57_pade_rational_approximation.md)** | Pade approximants, the linear system for the coefficients, why rational wins near poles |
| 58 | **[58_trigonometric_interpolation_and_the_dft](08_approximation_and_transforms/58_trigonometric_interpolation_and_the_dft.md)** | Roots of unity, the DFT, orthogonality of the trigonometric basis, the DFT interpolation theorem |
| 59 | **[59_the_fft_and_signal_processing](08_approximation_and_transforms/59_the_fft_and_signal_processing.md)** | Radix-2 FFT from scratch, the n log n argument, least squares with trigonometric models, filtering and the Wiener filter |
| 60 | **[60_dct_and_data_compression](08_approximation_and_transforms/60_dct_and_data_compression.md)** | The 1D and 2D discrete cosine transform, quantization, Huffman coding, the modified DCT and an audio codec |

## Numerical Differentiation and Integration

Derivatives and integrals from discrete values, with honest error control.

Folder: [`09_differentiation_and_integration/`](09_differentiation_and_integration/)

| # | Lesson | What it covers |
|---|---|---|
| 61 | **[61_numerical_differentiation](09_differentiation_and_integration/61_numerical_differentiation.md)** | Difference formulas from Taylor and from interpolation, higher derivatives, the roundoff and truncation trade-off, complex-step |
| 62 | **[62_newton_cotes_quadrature](09_differentiation_and_integration/62_newton_cotes_quadrature.md)** | Trapezoid, Simpson 1/3 and 3/8, Boole, Weddle, degree of precision, composite and open rules with error terms |
| 63 | **[63_richardson_romberg_and_euler_maclaurin](09_differentiation_and_integration/63_richardson_romberg_and_euler_maclaurin.md)** | Richardson extrapolation as a general device, the Euler-Maclaurin formula, the Romberg table, and where it fails |
| 64 | **[64_adaptive_quadrature](09_differentiation_and_integration/64_adaptive_quadrature.md)** | Error estimation by interval halving, recursive adaptive Simpson, peaks and singularities, cost comparison |
| 65 | **[65_gaussian_quadrature](09_differentiation_and_integration/65_gaussian_quadrature.md)** | Nodes as roots of orthogonal polynomials, degree of precision 2n-1, Golub-Welsch, Legendre, Chebyshev, Laguerre, Hermite |
| 66 | **[66_improper_and_multiple_integrals](09_differentiation_and_integration/66_improper_and_multiple_integrals.md)** | Infinite ranges and integrable singularities, double integrals by tensor rules, and the curse of dimensionality |

## Ordinary Differential Equations

Marching an initial value problem forward in time, and solving two-point boundary value problems.

Folder: [`10_ordinary_differential_equations/`](10_ordinary_differential_equations/)

| # | Lesson | What it covers |
|---|---|---|
| 67 | **[67_ivp_theory_and_eulers_method](10_ordinary_differential_equations/67_ivp_theory_and_eulers_method.md)** | Classification, the Lipschitz condition, existence and uniqueness, Euler, local and global truncation error |
| 68 | **[68_taylor_and_picard_methods](10_ordinary_differential_equations/68_taylor_and_picard_methods.md)** | Picard successive approximations, Taylor series methods of order k, and the cost of needing derivatives of f |
| 69 | **[69_runge_kutta_methods](10_ordinary_differential_equations/69_runge_kutta_methods.md)** | Heun and midpoint, deriving order conditions, classical RK4, Butcher tableaux, order verified by refinement |
| 70 | **[70_adaptive_step_size_control](10_ordinary_differential_equations/70_adaptive_step_size_control.md)** | Embedded pairs, RKF45 and Dormand-Prince, the step-size control law, work versus accuracy against fixed steps |
| 71 | **[71_multistep_methods](10_ordinary_differential_equations/71_multistep_methods.md)** | Adams-Bashforth and Adams-Moulton from quadrature, predictor-corrector, Milne-Simpson, zero-stability, Dahlquist barriers |
| 72 | **[72_stability_and_stiff_equations](10_ordinary_differential_equations/72_stability_and_stiff_equations.md)** | The test equation, regions of absolute stability, A-stability and L-stability, stiffness, backward Euler and implicit solvers |
| 73 | **[73_systems_and_higher_order_equations](10_ordinary_differential_equations/73_systems_and_higher_order_equations.md)** | Reduction to first order, systems by every method, the pendulum, orbital mechanics, Hodgkin-Huxley, Lorenz |
| 74 | **[74_geometric_and_symplectic_integrators](10_ordinary_differential_equations/74_geometric_and_symplectic_integrators.md)** | Why standard methods lose energy, symplectic Euler, Stormer-Verlet, and long-time behaviour on Hamiltonian systems |
| 75 | **[75_boundary_value_problems](10_ordinary_differential_equations/75_boundary_value_problems.md)** | The shooting method driven by a root finder, finite differences for linear and nonlinear BVPs, unequal intervals |
| 76 | **[76_collocation_and_finite_elements](10_ordinary_differential_equations/76_collocation_and_finite_elements.md)** | Collocation with a basis, the weak form, Galerkin, hat functions, assembling the stiffness matrix |

## Partial Differential Equations

Finite difference and finite element methods for parabolic, hyperbolic and elliptic problems, and the stability theory behind them.

Folder: [`11_partial_differential_equations/`](11_partial_differential_equations/)

| # | Lesson | What it covers |
|---|---|---|
| 77 | **[77_pde_classification_and_stencils](11_partial_differential_equations/77_pde_classification_and_stencils.md)** | Elliptic, parabolic and hyperbolic classification, initial and boundary conditions, finite difference stencils |
| 78 | **[78_parabolic_equations](11_partial_differential_equations/78_parabolic_equations.md)** | Bender-Schmidt explicit, backward implicit, Crank-Nicolson, the weighted scheme, Richardson and Du-Fort-Frankel |
| 79 | **[79_consistency_convergence_and_stability](11_partial_differential_equations/79_consistency_convergence_and_stability.md)** | Truncation error and consistency, order, convergence, the Lax equivalence idea, the matrix method and von Neumann analysis |
| 80 | **[80_multidimensional_parabolic_and_adi](11_partial_differential_equations/80_multidimensional_parabolic_and_adi.md)** | Two dimensional heat conduction, the cost problem with implicit schemes, and the alternating direction implicit method |
| 81 | **[81_hyperbolic_equations_and_cfl](11_partial_differential_equations/81_hyperbolic_equations_and_cfl.md)** | The wave equation explicit and implicit, the CFL condition, domains of dependence, and what violation looks like |
| 82 | **[82_elliptic_equations](11_partial_differential_equations/82_elliptic_equations.md)** | Laplace and Poisson by finite differences, the resulting sparse system and how to solve it, finite elements |
| 83 | **[83_nonlinear_pdes_and_method_of_lines](11_partial_differential_equations/83_nonlinear_pdes_and_method_of_lines.md)** | Implicit Newton solvers for nonlinear PDEs, two space dimensions, and the method of lines linking PDEs back to ODE solvers |

## Numerical Optimization

Finding minima with and without derivatives, from golden section search to quasi-Newton and constrained problems.

Folder: [`12_optimization/`](12_optimization/)

| # | Lesson | What it covers |
|---|---|---|
| 84 | **[84_optimization_fundamentals](12_optimization/84_optimization_fundamentals.md)** | Optimality conditions, convexity, the Hessian, conditioning of an optimization problem, test functions |
| 85 | **[85_derivative_free_optimization](12_optimization/85_derivative_free_optimization.md)** | Unimodality, golden section search, successive parabolic interpolation, Nelder-Mead simplex |
| 86 | **[86_gradient_and_newton_methods](12_optimization/86_gradient_and_newton_methods.md)** | Steepest descent and its dependence on the condition number, Newton for optimization, line search and Wolfe conditions |
| 87 | **[87_quasi_newton_and_trust_region](12_optimization/87_quasi_newton_and_trust_region.md)** | The secant condition, BFGS and L-BFGS, nonlinear conjugate gradient, trust region methods |
| 88 | **[88_constrained_optimization](12_optimization/88_constrained_optimization.md)** | Lagrange multipliers, KKT conditions, penalty and barrier methods, projected gradient |
| 89 | **[89_stochastic_optimization](12_optimization/89_stochastic_optimization.md)** | Stochastic and minibatch gradient descent, gradient noise, momentum and Nesterov, AdaGrad, RMSProp, Adam and AdamW, learning rate schedules |
| 90 | **[90_proximal_and_composite_optimization](12_optimization/90_proximal_and_composite_optimization.md)** | The proximal operator, proximal gradient descent, ISTA and FISTA, L1 regularization and LASSO, soft thresholding |

## Stochastic and Monte Carlo Methods

Random number generation, Monte Carlo estimation, Brownian motion and stochastic differential equations.

Folder: [`13_stochastic_methods/`](13_stochastic_methods/)

| # | Lesson | What it covers |
|---|---|---|
| 91 | **[91_random_number_generation](13_stochastic_methods/91_random_number_generation.md)** | Linear congruential generators and how they fail, Mersenne Twister and PCG64, inverse transform, Box-Muller, testing a generator |
| 92 | **[92_monte_carlo_methods](13_stochastic_methods/92_monte_carlo_methods.md)** | Monte Carlo integration, the one-over-root-n law, variance reduction, quasi-random Halton sequences and their better rate |
| 93 | **[93_brownian_motion_and_sdes](13_stochastic_methods/93_brownian_motion_and_sdes.md)** | Random walks, Brownian motion, stochastic differential equations, Euler-Maruyama, Milstein, strong and weak order, Black-Scholes |

## Numerical Analysis in Machine Learning and AI

Where the core material shows up in modern machine learning. Connections built on the mathematics, not replacements for it.

Folder: [`14_ml_and_ai_connections/`](14_ml_and_ai_connections/)

| # | Lesson | What it covers |
|---|---|---|
| 94 | **[94_numerical_linear_algebra_in_machine_learning](14_ml_and_ai_connections/94_numerical_linear_algebra_in_machine_learning.md)** | PCA as a truncated SVD, whitening, ridge regression and conditioning, low-rank adapters, embedding spectra |
| 95 | **[95_optimization_for_machine_learning](14_ml_and_ai_connections/95_optimization_for_machine_learning.md)** | What the optimizers of lesson 89 do to a learning problem: generalization, sharp and flat minima, batch size, and why second order methods stay rare |
| 96 | **[96_floating_point_in_deep_learning](14_ml_and_ai_connections/96_floating_point_in_deep_learning.md)** | float32, float16 and bfloat16, overflow in softmax and the logsumexp fix, loss scaling, non-deterministic reductions |
| 97 | **[97_automatic_differentiation](14_ml_and_ai_connections/97_automatic_differentiation.md)** | Forward and reverse mode from scratch with dual numbers and a tape, cost analysis, and how it differs from finite differences |
| 98 | **[98_scientific_machine_learning_and_inverse_problems](14_ml_and_ai_connections/98_scientific_machine_learning_and_inverse_problems.md)** | Inverse problems and regularization, physics-informed networks as collocation, neural ODEs, Krylov methods at scale |

