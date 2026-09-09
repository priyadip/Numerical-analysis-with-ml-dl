"""nalib: the from-scratch numerical analysis library for this course.

Every algorithm here is written in plain NumPy so you can read it. Nothing is a thin
wrapper around SciPy. Where a library equivalent exists, the docstring names it so you can
cross-check.

Each algorithm appears twice in this course on purpose:

1. In the notebook that teaches it, spelled out step by step next to the mathematics.
2. Here, as a clean tested function that later lessons import.

The two are compared numerically in the lesson where the algorithm is introduced.

Modules
-------
numbersystems   base conversion for integers and fractions
floatingpoint   IEEE 754 inspection, simulated precision, summation algorithms
errors          absolute, relative, forward and backward error, condition numbers
convergence     measuring observed order of convergence
polynomials     Horner evaluation, synthetic division, nested forms
cost            timing harness and empirical complexity fitting
roots           bracketing, fixed point, Newton and hybrid root finders
polyroots       Descartes, Sturm, Birge-Vieta, Bairstow, Graeffe, companion matrix
nlsystems       Newton, damped Newton and Broyden for F(x) = 0 in n dimensions
linalg          vector and matrix norms, and the three views of a matrix product
orthogonality   orthogonal matrices, projectors, and the geometry of projection
lu              Gaussian elimination, LU factorization and triangular solves
pivoting        partial and complete pivoting, PA = LU, and the growth factor
linsys          conditioning and error diagnosis for Ax = b, plus standard test matrices
cholesky        symmetric positive definite matrices, Cholesky and LDL factorizations
banded          tridiagonal and banded solvers, sparse storage, fill-in and reordering
refinement      condition estimation in O(n^2), and iterative refinement
iterative       Jacobi, damped Jacobi, Gauss-Seidel and SOR, and the spectral radius that
                governs them
krylov          conjugate gradient, Arnoldi, Lanczos, and matrix-free linear operators
nonsymmetric    GMRES, MINRES, BiCG and BiCGSTAB, and why eigenvalues stop predicting
multigrid       smoothers, grid transfers, V and W cycles, and O(n) solves
leastsquares    the normal equations, projection, model fitting, and the squared kappa
qr              Gram-Schmidt, Householder reflectors and Givens rotations
rrqr            column pivoted QR, subset selection and total least squares
nlls            Gauss-Newton, Levenberg-Marquardt and nonlinear least squares
eigen           eigenvalue localization, conditioning and the Schur form
power           power iteration, inverse iteration and the Rayleigh quotient
qralg           the QR algorithm, Hessenberg reduction, shifts and deflation
symeig          Jacobi, tridiagonalization, Sturm bisection, divide and conquer
krylov_eig      Rayleigh-Ritz, restarted Arnoldi, and matrix-free eigenvalues
geneig          the pencil A x = lambda B x, Cholesky reduction and QZ
svd             the singular value decomposition, subspaces and perturbation
svdcompute      bidiagonalization, implicit sweeps and one-sided Jacobi
lowrank         Eckart-Young, truncation, randomised SVD and PageRank
interp          the five forms of the interpolating polynomial, and barycentric evaluation
divdiff         divided differences, their properties and where their accuracy goes
interperror     the error formula, its three factors, and the Runge phenomenon
chebyshev       Chebyshev polynomials, minimax nodes and Lebesgue constants
finitediff      the finite difference operator algebra and difference tables
equalinterval   Gregory-Newton, Gauss, Stirling, Bessel, Everett and Steffensen
hermite         Hermite and osculating interpolation, and the piecewise alternative
splines         cubic splines, the tridiagonal system and the four end conditions
bezier          Bezier curves, de Casteljau, B-splines and local support
bivariate       tensor product interpolation, the curse, and Mairhuber's theorem
approx          function norms, best approximation, Remez, equioscillation, Weierstrass
orthopoly       orthogonal polynomial families, three-term recurrences, Golub-Welsch
chebapprox      Chebyshev series, near-minimax truncation, and economization
pade            Pade approximants, the Toeplitz system, poles and Froissart doublets
dft             roots of unity, the discrete Fourier transform and its interpolation theorem
fft             radix-2 and Bluestein transforms, trigonometric fitting, filtering, Wiener
dct             the cosine transform, quantization, Huffman coding, JPEG and the MDCT
differentiation finite difference weights, step size, Richardson, the complex step
newtoncotes     equally spaced quadrature, degree of precision, composite rules
romberg         Euler-Maclaurin, the Romberg table, and the periodic trapezoid rule
adaptive        adaptive Simpson, local error estimates, and where they lie
gaussquad       Gauss and Gauss-Kronrod rules by Golub-Welsch and Laurie
multiquad       improper integrals, tanh-sinh, tensor products and Monte Carlo
ivp             initial value problems, the Lipschitz condition, Euler and its limits
taylorode       Picard iteration and Taylor series methods, and what they cost
rungekutta      Butcher tableaux, order conditions, and the classical explicit methods
adaptivestep    embedded pairs, the step control law, and what adaptivity buys
multistep       Adams methods, predictor-corrector, zero-stability and the barriers
stability       stability regions, A-stability and L-stability, and stiffness
odesystems      higher order equations as systems, and the classical named problems
symplectic      geometric integrators that keep the structure rather than the digits
bvp             two point boundary value problems by shooting and finite differences
femode          collocation, the weak form, Galerkin, hat functions and assembly
pdeclass        classifying a second order PDE, and the stencils that discretize one
parabolic       the heat equation: explicit, backward, Crank-Nicolson and the warnings
pdestability    consistency, the two stability tests, and the Lax equivalence theorem
adi             two dimensional parabolic problems and alternating direction implicit
hyperbolic      the wave equation, the Courant condition and domains of dependence
elliptic        Laplace and Poisson in two dimensions, and the system they produce
nonlinearpde    Newton on a discrete PDE, bifurcation, and the method of lines
optimize        what a minimum is, how to recognise one, and what it costs to find
derivfree       golden section, parabolic interpolation and the Nelder-Mead simplex
gradient        steepest descent, Newton for optimization, and line searches
quasinewton     BFGS, limited memory BFGS, nonlinear CG and trust regions
constrained     KKT conditions, penalty and barrier methods, and projection
stochastic      SGD, momentum, and the adaptive methods when the gradient is a sample
proximal        the proximal operator, ISTA and FISTA, soft thresholding and LASSO
rng             linear congruential generators, how they fail, and how to test one
montecarlo      Monte Carlo integration, variance reduction and quasi-random points
sde             Brownian motion, Euler-Maruyama and Milstein, strong and weak order
mllinalg        PCA as a truncated SVD, whitening, ridge, and low-rank adapters
mlopt           gradient descent as a flow, batch size, sharpness, natural gradient
mixedprecision  float16 and bfloat16, logsumexp, loss scaling, reduction order
autodiff        dual numbers, a tape, both modes, and what they cost
sciml           inverse problems, collocation with a trained basis, neural ODEs
"""

__version__ = "0.1.0"

__all__ = [
    "numbersystems",
    "floatingpoint",
    "errors",
    "convergence",
    "polynomials",
    "cost",
    "roots",
    "polyroots",
    "nlsystems",
    "linalg",
    "orthogonality",
    "lu",
    "pivoting",
    "linsys",
    "cholesky",
    "banded",
    "refinement",
    "iterative",
    "krylov",
    "nonsymmetric",
    "multigrid",
    "leastsquares",
    "qr",
    "rrqr",
    "nlls",
    "eigen",
    "power",
    "qralg",
    "symeig",
    "krylov_eig",
    "geneig",
    "svd",
    "svdcompute",
    "lowrank",
    "interp",
    "divdiff",
    "interperror",
    "chebyshev",
    "finitediff",
    "equalinterval",
    "hermite",
    "splines",
    "bezier",
    "bivariate",
    "approx",
    "orthopoly",
    "chebapprox",
    "pade",
    "dft",
    "fft",
    "dct",
    "differentiation",
    "newtoncotes",
    "romberg",
    "adaptive",
    "gaussquad",
    "multiquad",
    "ivp",
    "taylorode",
    "rungekutta",
    "adaptivestep",
    "multistep",
    "stability",
    "odesystems",
    "symplectic",
    "bvp",
    "femode",
    "pdeclass",
    "parabolic",
    "pdestability",
    "adi",
    "hyperbolic",
    "elliptic",
    "nonlinearpde",
    "optimize",
    "derivfree",
    "gradient",
    "quasinewton",
    "constrained",
    "stochastic",
    "proximal",
    "rng",
    "montecarlo",
    "sde",
    "mllinalg",
    "mlopt",
    "mixedprecision",
    "autodiff",
    "sciml",
]
