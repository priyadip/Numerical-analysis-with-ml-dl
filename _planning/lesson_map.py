"""The final curriculum: 14 parts, 96 lessons.

LESSONS maps lesson id -> (part folder, file stem, one-line description).
PARTS   maps part folder -> (display name, one-line description).

This is the single source of truth for lesson numbering and file names. Everything else
(coverage checklist, COURSE_MAP.md, LEARNING_PATH.md, notebook scaffolding) is generated
from it, so the names can never drift apart.
"""

PARTS = {
    "01_foundations": (
        "Foundations of Numerical Computing",
        "How computers represent numbers, where error comes from, and how to tell a hard "
        "problem from a bad algorithm.",
    ),
    "02_root_finding": (
        "Nonlinear Equations and Root Finding",
        "Solving f(x) = 0. The cheapest place to meet convergence order, sensitivity and "
        "backward error.",
    ),
    "03_direct_linear_systems": (
        "Direct Methods for Linear Systems",
        "Solving Ax = b by elimination and factorization, and knowing when the answer can "
        "be trusted.",
    ),
    "04_iterative_and_krylov": (
        "Iterative and Krylov Subspace Methods",
        "Solving Ax = b when A is large and sparse, by improving a guess instead of "
        "factorizing.",
    ),
    "05_least_squares_and_qr": (
        "Orthogonality, QR and Least Squares",
        "Overdetermined systems, projection, orthogonalization, and the numerically sound "
        "way to fit data.",
    ),
    "06_eigenvalues_and_svd": (
        "Eigenvalue Problems and the Singular Value Decomposition",
        "Finding eigenvalues, eigenvectors and singular values, from the power method up "
        "to shifted QR and Krylov eigensolvers.",
    ),
    "07_interpolation": (
        "Interpolation",
        "Building a function that passes exactly through given data, and knowing how wrong "
        "it is between the points.",
    ),
    "08_approximation_and_transforms": (
        "Approximation Theory and Transforms",
        "Fitting data and functions when exact interpolation is the wrong goal, plus the "
        "Fourier and cosine transforms.",
    ),
    "09_differentiation_and_integration": (
        "Numerical Differentiation and Integration",
        "Derivatives and integrals from discrete values, with honest error control.",
    ),
    "10_ordinary_differential_equations": (
        "Ordinary Differential Equations",
        "Marching an initial value problem forward in time, and solving two-point boundary "
        "value problems.",
    ),
    "11_partial_differential_equations": (
        "Partial Differential Equations",
        "Finite difference and finite element methods for parabolic, hyperbolic and "
        "elliptic problems, and the stability theory behind them.",
    ),
    "12_optimization": (
        "Numerical Optimization",
        "Finding minima with and without derivatives, from golden section search to "
        "quasi-Newton and constrained problems.",
    ),
    "13_stochastic_methods": (
        "Stochastic and Monte Carlo Methods",
        "Random number generation, Monte Carlo estimation, Brownian motion and stochastic "
        "differential equations.",
    ),
    "14_ml_and_ai_connections": (
        "Numerical Analysis in Machine Learning and AI",
        "Where the core material shows up in modern machine learning. Connections built on "
        "the mathematics, not replacements for it.",
    ),
}

LESSONS = {
    # ---------------- Part 1: Foundations of Numerical Computing
    "01": ("01_foundations", "01_what_is_numerical_analysis",
           "Problem versus algorithm, approximation, nested polynomial evaluation, why efficiency is a mathematical question"),
    "02": ("01_foundations", "02_number_systems_and_representation",
           "Binary, octal, decimal and hexadecimal; integers and fractions; conversion in every direction"),
    "03": ("01_foundations", "03_floating_point_arithmetic",
           "IEEE 754 layout, normalization, machine epsilon, rounding, subnormals, overflow and underflow, the fl model"),
    "04": ("01_foundations", "04_error_types_and_propagation",
           "Absolute and relative error, the full error taxonomy, and how error moves through arithmetic and functions"),
    "05": ("01_foundations", "05_loss_of_significance_and_cancellation",
           "Catastrophic cancellation, rewriting unstable formulas, and summation done properly"),
    "06": ("01_foundations", "06_conditioning_and_stability",
           "Forward and backward error, condition number of a problem, backward stability of an algorithm"),
    "07": ("01_foundations", "07_taylor_series_and_convergence_rates",
           "Taylor with remainder, the calculus theorems we keep using, order notation and measuring observed order"),
    "08": ("01_foundations", "08_algorithm_cost_and_complexity",
           "Counting flops, memory cost, why flop counts alone mislead, and the effect of memory layout"),

    # ---------------- Part 2: Nonlinear Equations and Root Finding
    "09": ("02_root_finding", "09_bracketing_methods",
           "Bisection, regula falsi and the Illinois fix, with guaranteed error bounds"),
    "10": ("02_root_finding", "10_fixed_point_iteration",
           "Fixed points, the contraction mapping theorem, cobweb geometry, stopping criteria, Aitken and Steffensen"),
    "11": ("02_root_finding", "11_newton_and_secant_methods",
           "Newton and its quadratic convergence, multiple roots, secant, Muller, the Chebyshev third-order method, Brent"),
    "12": ("02_root_finding", "12_convergence_theory_and_sensitivity",
           "Order of convergence for every method side by side, plus the conditioning of a root and the Wilkinson polynomial"),
    "13": ("02_root_finding", "13_polynomial_root_finding",
           "Horner, Birge-Vieta, deflation, Descartes rule, Sturm sequences, Lin-Bairstow, Graeffe, companion matrices"),
    "14": ("02_root_finding", "14_nonlinear_systems_of_equations",
           "Multivariate fixed point and Seidel iteration, Newton with the Jacobian, Broyden, damping and line search"),

    # ---------------- Part 3: Direct Methods for Linear Systems
    "15": ("03_direct_linear_systems", "15_vectors_matrices_and_norms",
           "Vector and matrix norms, induced norms, equivalence, submultiplicativity, spectral radius"),
    "16": ("03_direct_linear_systems", "16_orthogonality_and_projectors",
           "Orthonormal bases, orthogonal and unitary matrices, why they cannot amplify error, projectors, and the geometry of projection"),
    "17": ("03_direct_linear_systems", "17_gaussian_elimination_and_lu",
           "Naive elimination, substitution, operation counts, LU, Doolittle and Crout, Gauss-Jordan, why Cramer fails"),
    "18": ("03_direct_linear_systems", "18_pivoting_and_pa_lu",
           "Swamping, partial and complete pivoting, permutation matrices, PA=LU, the growth factor"),
    "19": ("03_direct_linear_systems", "19_conditioning_of_linear_systems",
           "Residual versus error, the matrix condition number, error magnification, Hilbert matrices"),
    "20": ("03_direct_linear_systems", "20_symmetric_positive_definite_and_cholesky",
           "Positive definiteness and how to test it, Cholesky factorization, LDL, when Cholesky beats LU"),
    "21": ("03_direct_linear_systems", "21_banded_sparse_and_structured_systems",
           "Tridiagonal systems and the Thomas algorithm, banded solvers, sparse storage formats, fill-in and ordering"),
    "22": ("03_direct_linear_systems", "22_condition_estimation_and_iterative_refinement",
           "Estimating the condition number without inverting, and recovering accuracy with iterative refinement"),

    # ---------------- Part 4: Iterative and Krylov Subspace Methods
    "23": ("04_iterative_and_krylov", "23_classical_iterative_methods",
           "Matrix splitting, Jacobi, Gauss-Seidel, SOR and relaxation, convergence by spectral radius"),
    "24": ("04_iterative_and_krylov", "24_conjugate_gradient",
           "The quadratic form view, why steepest descent zigzags, the A-inner product, conjugacy, CG and its convergence bound"),
    "25": ("04_iterative_and_krylov", "25_preconditioning",
           "What a preconditioner does to the spectrum: Jacobi, SSOR, incomplete Cholesky, preconditioned CG"),
    "26": ("04_iterative_and_krylov", "26_krylov_subspaces_arnoldi_and_lanczos",
           "Krylov subspaces, why the naive basis is useless, Arnoldi, Lanczos, loss of orthogonality, Ritz values"),
    "27": ("04_iterative_and_krylov", "27_gmres_and_nonsymmetric_solvers",
           "GMRES as least squares over a Krylov space, Givens rotations, restarting, MINRES and BiCGSTAB"),
    "28": ("04_iterative_and_krylov", "28_multigrid_methods",
           "Why smoothers stall on low frequencies, coarse grid correction, the V-cycle, and mesh-independent convergence"),

    # ---------------- Part 5: Orthogonality, QR and Least Squares
    "29": ("05_least_squares_and_qr", "29_least_squares_and_normal_equations",
           "Inconsistent systems, normal equations, projection geometry, model fitting, and the squared condition number"),
    "30": ("05_least_squares_and_qr", "30_gram_schmidt_and_qr",
           "Projection, classical and modified Gram-Schmidt, reduced versus full QR, the loss-of-orthogonality experiment"),
    "31": ("05_least_squares_and_qr", "31_householder_and_givens_qr",
           "Householder reflectors and the sign trick, Householder QR, Givens rotations, stability comparison"),
    "32": ("05_least_squares_and_qr", "32_solving_least_squares_in_practice",
           "Normal equations versus QR versus SVD on the same hard problem, pseudoinverse, Tikhonov regularization"),
    "33": ("05_least_squares_and_qr", "33_rank_revealing_qr_and_total_least_squares",
           "QR with column pivoting, numerical rank, subset selection, and total least squares when both sides have error"),
    "34": ("05_least_squares_and_qr", "34_nonlinear_least_squares",
           "Gauss-Newton, Levenberg-Marquardt, models with nonlinear parameters, GPS positioning worked end to end"),

    # ---------------- Part 6: Eigenvalue Problems and the SVD
    "35": ("06_eigenvalues_and_svd", "35_eigenvalue_theory_and_localization",
           "Similarity, diagonalizability, defective matrices, Schur form, Gerschgorin and Brauer bounds, eigenvalue conditioning"),
    "36": ("06_eigenvalues_and_svd", "36_power_methods",
           "Power iteration and its rate, inverse power iteration, shifts, the Rayleigh quotient and its cubic convergence"),
    "37": ("06_eigenvalues_and_svd", "37_qr_algorithm",
           "Simultaneous iteration, the unshifted QR algorithm, Hessenberg reduction, shifts, deflation, real Schur form, the LR method"),
    "38": ("06_eigenvalues_and_svd", "38_symmetric_eigenvalue_problem",
           "Jacobi rotations, Givens and Householder tridiagonalization, Sturm sequence bisection, divide and conquer"),
    "39": ("06_eigenvalues_and_svd", "39_krylov_methods_for_eigenvalues",
           "Lanczos and Arnoldi for eigenvalues, Ritz value convergence, restarting, solving huge sparse eigenproblems"),
    "40": ("06_eigenvalues_and_svd", "40_generalized_eigenvalue_problem",
           "Ax = lambda Bx, reduction by Cholesky, the QZ idea, and where generalized problems come from"),
    "41": ("06_eigenvalues_and_svd", "41_svd_theory",
           "The unit sphere mapped to a hyperellipse, existence and uniqueness, the four fundamental subspaces, pseudoinverse"),
    "42": ("06_eigenvalues_and_svd", "42_computing_the_svd",
           "Why forming A-transpose-A is wrong, Golub-Kahan bidiagonalization, implicit QR sweeps, one-sided Jacobi"),
    "43": ("06_eigenvalues_and_svd", "43_svd_applications_and_low_rank",
           "Numerical rank, truncated SVD, Eckart-Young-Mirsky, image compression, PageRank, randomized SVD"),

    # ---------------- Part 7: Interpolation
    "44": ("07_interpolation", "44_polynomial_interpolation_forms",
           "Existence and uniqueness, power, shifted power, Newton and nested Newton forms, Lagrange, the Vandermonde system"),
    "45": ("07_interpolation", "45_divided_differences",
           "Newton divided differences, the recursion and its proof, properties, relation to derivatives, adding a point cheaply"),
    "46": ("07_interpolation", "46_interpolation_error_and_runge_phenomenon",
           "The error formula with proof, how error behaves as nodes are added, the Runge phenomenon, node placement"),
    "47": ("07_interpolation", "47_chebyshev_interpolation",
           "Chebyshev points, the minimax theorem for the node polynomial, Chebyshev recurrence, Lebesgue constants"),
    "48": ("07_interpolation", "48_finite_difference_operators_and_tables",
           "Forward, backward, central, mean, shift and differential operators, their interrelations, difference tables and error spread"),
    "49": ("07_interpolation", "49_equal_interval_interpolation_formulas",
           "Gregory-Newton forward and backward, Gauss central formulas, Stirling, Bessel, Everett, Steffensen, and when to use each"),
    "50": ("07_interpolation", "50_hermite_and_piecewise_interpolation",
           "Interpolating derivative data, osculating polynomials, piecewise linear error, the Weierstrass theorem"),
    "51": ("07_interpolation", "51_cubic_splines",
           "Spline conditions, the tridiagonal system, natural, clamped, parabolic and not-a-knot ends, the equi-spaced case"),
    "52": ("07_interpolation", "52_bezier_and_bspline_curves",
           "Bezier curves and control points, de Casteljau, B-spline basis functions, local control, fonts"),
    "53": ("07_interpolation", "53_bivariate_interpolation",
           "Lagrange and Newton interpolation on a grid, tensor product structure, and what breaks in higher dimensions"),

    # ---------------- Part 8: Approximation Theory and Transforms
    "54": ("08_approximation_and_transforms", "54_function_approximation_fundamentals",
           "Norms on function spaces, best approximation, existence and uniqueness, minimax versus least squares, Weierstrass"),
    "55": ("08_approximation_and_transforms", "55_orthogonal_polynomials",
           "Gram-Schmidt on functions, three-term recurrences, Legendre, Chebyshev, Laguerre and Hermite families"),
    "56": ("08_approximation_and_transforms", "56_chebyshev_approximation_and_economization",
           "Chebyshev series, near-minimax behaviour, economization of power series, comparison with truncated Taylor"),
    "57": ("08_approximation_and_transforms", "57_pade_rational_approximation",
           "Pade approximants, the linear system for the coefficients, why rational wins near poles"),
    "58": ("08_approximation_and_transforms", "58_trigonometric_interpolation_and_the_dft",
           "Roots of unity, the DFT, orthogonality of the trigonometric basis, the DFT interpolation theorem"),
    "59": ("08_approximation_and_transforms", "59_the_fft_and_signal_processing",
           "Radix-2 FFT from scratch, the n log n argument, least squares with trigonometric models, filtering and the Wiener filter"),
    "60": ("08_approximation_and_transforms", "60_dct_and_data_compression",
           "The 1D and 2D discrete cosine transform, quantization, Huffman coding, the modified DCT and an audio codec"),

    # ---------------- Part 9: Numerical Differentiation and Integration
    "61": ("09_differentiation_and_integration", "61_numerical_differentiation",
           "Difference formulas from Taylor and from interpolation, higher derivatives, the roundoff and truncation trade-off, complex-step"),
    "62": ("09_differentiation_and_integration", "62_newton_cotes_quadrature",
           "Trapezoid, Simpson 1/3 and 3/8, Boole, Weddle, degree of precision, composite and open rules with error terms"),
    "63": ("09_differentiation_and_integration", "63_richardson_romberg_and_euler_maclaurin",
           "Richardson extrapolation as a general device, the Euler-Maclaurin formula, the Romberg table, and where it fails"),
    "64": ("09_differentiation_and_integration", "64_adaptive_quadrature",
           "Error estimation by interval halving, recursive adaptive Simpson, peaks and singularities, cost comparison"),
    "65": ("09_differentiation_and_integration", "65_gaussian_quadrature",
           "Nodes as roots of orthogonal polynomials, degree of precision 2n-1, Golub-Welsch, Legendre, Chebyshev, Laguerre, Hermite"),
    "66": ("09_differentiation_and_integration", "66_improper_and_multiple_integrals",
           "Infinite ranges and integrable singularities, double integrals by tensor rules, and the curse of dimensionality"),

    # ---------------- Part 10: Ordinary Differential Equations
    "67": ("10_ordinary_differential_equations", "67_ivp_theory_and_eulers_method",
           "Classification, the Lipschitz condition, existence and uniqueness, Euler, local and global truncation error"),
    "68": ("10_ordinary_differential_equations", "68_taylor_and_picard_methods",
           "Picard successive approximations, Taylor series methods of order k, and the cost of needing derivatives of f"),
    "69": ("10_ordinary_differential_equations", "69_runge_kutta_methods",
           "Heun and midpoint, deriving order conditions, classical RK4, Butcher tableaux, order verified by refinement"),
    "70": ("10_ordinary_differential_equations", "70_adaptive_step_size_control",
           "Embedded pairs, RKF45 and Dormand-Prince, the step-size control law, work versus accuracy against fixed steps"),
    "71": ("10_ordinary_differential_equations", "71_multistep_methods",
           "Adams-Bashforth and Adams-Moulton from quadrature, predictor-corrector, Milne-Simpson, zero-stability, Dahlquist barriers"),
    "72": ("10_ordinary_differential_equations", "72_stability_and_stiff_equations",
           "The test equation, regions of absolute stability, A-stability and L-stability, stiffness, backward Euler and implicit solvers"),
    "73": ("10_ordinary_differential_equations", "73_systems_and_higher_order_equations",
           "Reduction to first order, systems by every method, the pendulum, orbital mechanics, Hodgkin-Huxley, Lorenz"),
    "74": ("10_ordinary_differential_equations", "74_geometric_and_symplectic_integrators",
           "Why standard methods lose energy, symplectic Euler, Stormer-Verlet, and long-time behaviour on Hamiltonian systems"),
    "75": ("10_ordinary_differential_equations", "75_boundary_value_problems",
           "The shooting method driven by a root finder, finite differences for linear and nonlinear BVPs, unequal intervals"),
    "76": ("10_ordinary_differential_equations", "76_collocation_and_finite_elements",
           "Collocation with a basis, the weak form, Galerkin, hat functions, assembling the stiffness matrix"),

    # ---------------- Part 11: Partial Differential Equations
    "77": ("11_partial_differential_equations", "77_pde_classification_and_stencils",
           "Elliptic, parabolic and hyperbolic classification, initial and boundary conditions, finite difference stencils"),
    "78": ("11_partial_differential_equations", "78_parabolic_equations",
           "Bender-Schmidt explicit, backward implicit, Crank-Nicolson, the weighted scheme, Richardson and Du-Fort-Frankel"),
    "79": ("11_partial_differential_equations", "79_consistency_convergence_and_stability",
           "Truncation error and consistency, order, convergence, the Lax equivalence idea, the matrix method and von Neumann analysis"),
    "80": ("11_partial_differential_equations", "80_multidimensional_parabolic_and_adi",
           "Two dimensional heat conduction, the cost problem with implicit schemes, and the alternating direction implicit method"),
    "81": ("11_partial_differential_equations", "81_hyperbolic_equations_and_cfl",
           "The wave equation explicit and implicit, the CFL condition, domains of dependence, and what violation looks like"),
    "82": ("11_partial_differential_equations", "82_elliptic_equations",
           "Laplace and Poisson by finite differences, the resulting sparse system and how to solve it, finite elements"),
    "83": ("11_partial_differential_equations", "83_nonlinear_pdes_and_method_of_lines",
           "Implicit Newton solvers for nonlinear PDEs, two space dimensions, and the method of lines linking PDEs back to ODE solvers"),

    # ---------------- Part 12: Numerical Optimization
    "84": ("12_optimization", "84_optimization_fundamentals",
           "Optimality conditions, convexity, the Hessian, conditioning of an optimization problem, test functions"),
    "85": ("12_optimization", "85_derivative_free_optimization",
           "Unimodality, golden section search, successive parabolic interpolation, Nelder-Mead simplex"),
    "86": ("12_optimization", "86_gradient_and_newton_methods",
           "Steepest descent and its dependence on the condition number, Newton for optimization, line search and Wolfe conditions"),
    "87": ("12_optimization", "87_quasi_newton_and_trust_region",
           "The secant condition, BFGS and L-BFGS, nonlinear conjugate gradient, trust region methods"),
    "88": ("12_optimization", "88_constrained_optimization",
           "Lagrange multipliers, KKT conditions, penalty and barrier methods, projected gradient"),

    "89": ("12_optimization", "89_stochastic_optimization",
           "Stochastic and minibatch gradient descent, gradient noise, momentum and Nesterov, AdaGrad, RMSProp, Adam and AdamW, learning rate schedules"),
    "90": ("12_optimization", "90_proximal_and_composite_optimization",
           "The proximal operator, proximal gradient descent, ISTA and FISTA, L1 regularization and LASSO, soft thresholding"),

    # ---------------- Part 13: Stochastic and Monte Carlo Methods
    "91": ("13_stochastic_methods", "91_random_number_generation",
           "Linear congruential generators and how they fail, Mersenne Twister and PCG64, inverse transform, Box-Muller, testing a generator"),
    "92": ("13_stochastic_methods", "92_monte_carlo_methods",
           "Monte Carlo integration, the one-over-root-n law, variance reduction, quasi-random Halton sequences and their better rate"),
    "93": ("13_stochastic_methods", "93_brownian_motion_and_sdes",
           "Random walks, Brownian motion, stochastic differential equations, Euler-Maruyama, Milstein, strong and weak order, Black-Scholes"),

    # ---------------- Part 14: Numerical Analysis in Machine Learning and AI
    "94": ("14_ml_and_ai_connections", "94_numerical_linear_algebra_in_machine_learning",
           "PCA as a truncated SVD, whitening, ridge regression and conditioning, low-rank adapters, embedding spectra"),
    "95": ("14_ml_and_ai_connections", "95_optimization_for_machine_learning",
           "What the optimizers of lesson 89 do to a learning problem: generalization, sharp and flat minima, batch size, and why second order methods stay rare"),
    "96": ("14_ml_and_ai_connections", "96_floating_point_in_deep_learning",
           "float32, float16 and bfloat16, overflow in softmax and the logsumexp fix, loss scaling, non-deterministic reductions"),
    "97": ("14_ml_and_ai_connections", "97_automatic_differentiation",
           "Forward and reverse mode from scratch with dual numbers and a tape, cost analysis, and how it differs from finite differences"),
    "98": ("14_ml_and_ai_connections", "98_scientific_machine_learning_and_inverse_problems",
           "Inverse problems and regularization, physics-informed networks as collocation, neural ODEs, Krylov methods at scale"),
}

assert len(LESSONS) == 98, len(LESSONS)
assert sorted(LESSONS) == [f"{i:02d}" for i in range(1, 99)]
for _lid, (_part, _stem, _desc) in LESSONS.items():
    assert _part in PARTS, _part
    assert _stem.startswith(_lid + "_"), _stem
