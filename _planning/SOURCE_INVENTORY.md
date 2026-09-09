# Source Inventory

This file records what is actually in the supplied folder. It is a quality-control
artifact. Learners do not need it.

Folder inspected: `d:\Numerical analysis`

Recursive scan at Phase 1 found 2 files, both PDF. A third source, `TrefethenBau.md`, was
supplied afterwards and is recorded as S3 below.

---

## Source S1

| Field | Value |
|---|---|
| File | `numericalanalysis_book1.pdf` (102 MB) |
| Title | Numerical Analysis |
| Author | Timothy Sauer (George Mason University) |
| Edition | Third Edition |
| Publisher / Year | Pearson, 2019 |
| ISBN | 978-0-13-469645-4 (LCCN 2017028491) |
| Pages | 690 PDF pages, book pages i to about 673 |
| Text layer | None. Scanned images only (678 JPEG page scans, 0 embedded fonts) |
| How it was read | Rendered page by page with Ghostscript, then read visually |
| Page offset | PDF page = printed book page + 17 (verified: PDF 18 = book page 1) |

### Structure

- **Chapter 0 Fundamentals** (p1): 0.1 Evaluating a Polynomial; 0.2 Binary Numbers
  (decimal to binary, binary to decimal); 0.3 Floating Point Representation of Real
  Numbers (floating point formats, machine representation, addition of floating point
  numbers); 0.4 Loss of Significance; 0.5 Review of Calculus
- **Chapter 1 Solving Equations** (p26): 1.1 Bisection Method (bracketing a root, how
  accurate and how fast); 1.2 Fixed-Point Iteration (fixed points, geometry, linear
  convergence, stopping criteria); 1.3 Limits of Accuracy (forward and backward error,
  the Wilkinson polynomial, sensitivity of root-finding); 1.4 Newton's Method (quadratic
  convergence, linear convergence); 1.5 Root-Finding without Derivatives (secant method
  and variants, Brent's method); Reality Check 1 Kinematics of the Stewart platform
- **Chapter 2 Systems of Equations** (p74): 2.1 Gaussian Elimination (naive GE,
  operation counts); 2.2 The LU Factorization (matrix form of GE, back substitution,
  complexity); 2.3 Sources of Error (error magnification and condition number,
  swamping); 2.4 The PA=LU Factorization (partial pivoting, permutation matrices);
  2.5 Iterative Methods (Jacobi, Gauss-Seidel and SOR, convergence, sparse matrix
  computations); 2.6 Methods for symmetric positive-definite matrices (SPD, Cholesky,
  Conjugate Gradient, preconditioning); 2.7 Nonlinear Systems of Equations (multivariate
  Newton, Broyden); Reality Check 2 The Euler-Bernoulli Beam
- **Chapter 3 Interpolation** (p144): 3.1 Data and Interpolating Functions (Lagrange,
  Newton's divided differences, how many degree d polynomials pass through n points,
  code, representing functions by approximating polynomials); 3.2 Interpolation Error
  (error formula, proof of Newton form and error formula, Runge phenomenon);
  3.3 Chebyshev Interpolation (Chebyshev's theorem, Chebyshev polynomials, change of
  interval); 3.4 Cubic Splines (properties, endpoint conditions); 3.5 Bezier Curves;
  Reality Check 3 Fonts from Bezier curves
- **Chapter 4 Least Squares** (p196): 4.1 Least Squares and the Normal Equations
  (inconsistent systems, fitting models to data, conditioning of least squares);
  4.2 A Survey of Models (periodic data, data linearization); 4.3 QR Factorization
  (Gram-Schmidt orthogonalization and least squares, modified Gram-Schmidt, Householder
  reflectors); 4.4 GMRES (Krylov methods, preconditioned GMRES); 4.5 Nonlinear Least
  Squares (Gauss-Newton, models with nonlinear parameters, Levenberg-Marquardt);
  Reality Check 4 GPS, Conditioning, and Nonlinear Least Squares
- **Chapter 5 Numerical Differentiation and Integration** (p253): 5.1 Numerical
  Differentiation (finite difference formulas, rounding error, extrapolation, symbolic
  differentiation and integration); 5.2 Newton-Cotes Formulas (trapezoid, Simpson,
  composite, open Newton-Cotes); 5.3 Romberg Integration; 5.4 Adaptive Quadrature;
  5.5 Gaussian Quadrature; Reality Check 5 Motion Control in Computer-Aided Modeling
- **Chapter 6 Ordinary Differential Equations** (p293): 6.1 Initial Value Problems
  (Euler's method, existence uniqueness and continuity, first-order linear equations);
  6.2 Analysis of IVP Solvers (local and global truncation error, explicit trapezoid
  method, Taylor methods); 6.3 Systems of ODEs (higher order equations, pendulum,
  orbital mechanics); 6.4 Runge-Kutta Methods and Applications (RK family,
  Hodgkin-Huxley neuron, Lorenz equations); 6.5 Variable Step-Size Methods (embedded
  RK pairs, order 4/5 methods); 6.6 Implicit Methods and Stiff Equations;
  6.7 Multistep Methods (generating, explicit, implicit); Reality Check 6 The Tacoma
  Narrows Bridge
- **Chapter 7 Boundary Value Problems** (p366): 7.1 Shooting Method; 7.2 Finite
  Difference Methods (linear and nonlinear BVPs); 7.3 Collocation and the Finite Element
  Method (collocation, finite elements and the Galerkin method); Reality Check 7
  Buckling of a Circular Ring
- **Chapter 8 Partial Differential Equations** (p394): 8.1 Parabolic Equations (forward
  difference method, stability analysis, backward difference method, Crank-Nicolson);
  8.2 Hyperbolic Equations (wave equation, CFL condition); 8.3 Elliptic Equations
  (finite difference, finite element method); 8.4 Nonlinear PDEs (implicit Newton
  solver, two space dimensions); Reality Check 8 Heat Distribution on a Cooling Fin
- **Chapter 9 Random Numbers and Applications** (p453): 9.1 Random Numbers
  (pseudo-random numbers, exponential and normal random numbers); 9.2 Monte Carlo
  Simulation (power laws for Monte Carlo estimation, quasi-random numbers); 9.3 Discrete
  and Continuous Brownian Motion (random walks, continuous Brownian motion); 9.4
  Stochastic Differential Equations (adding noise, numerical methods for SDEs);
  Reality Check 9 The Black-Scholes Formula
- **Chapter 10 Trigonometric Interpolation and the FFT** (p489): 10.1 The Fourier
  Transform (complex arithmetic, DFT, FFT); 10.2 Trigonometric Interpolation (DFT
  Interpolation Theorem, efficient evaluation); 10.3 The FFT and Signal Processing
  (orthogonality and interpolation, least squares fitting with trigonometric functions,
  sound noise and filtering); Reality Check 10 The Wiener Filter
- **Chapter 11 Compression** (p518): 11.1 The Discrete Cosine Transform (1D DCT, DCT and
  least squares approximation); 11.2 Two-Dimensional DCT and Image Compression (2D DCT,
  image compression, quantization); 11.3 Huffman Coding (information theory and coding,
  Huffman coding for the JPEG format); 11.4 Modified DCT and Audio Compression (MDCT,
  bit quantization); Reality Check 11 A Simple Audio Codec
- **Chapter 12 Eigenvalues and Singular Values** (p556): 12.1 Power Iteration Methods
  (power iteration, convergence, inverse power iteration, Rayleigh quotient iteration);
  12.2 QR Algorithm (simultaneous iteration, real Schur form and the QR algorithm, upper
  Hessenberg form); 12.3 Singular Value Decomposition (geometry of the SVD, finding the
  SVD in general); 12.4 Applications of the SVD (properties, dimension reduction,
  compression, calculating the SVD); Reality Check 12 How Search Engines Rate Page
  Quality
- **Chapter 13 Optimization** (p593): 13.1 Unconstrained Optimization without
  Derivatives (golden section search, successive parabolic interpolation, Nelder-Mead);
  13.2 Unconstrained Optimization with Derivatives (Newton's method, steepest descent,
  conjugate gradient search); Reality Check 13 Molecular Conformation and Numerical
  Optimization
- **Appendix A Matrix Algebra** (p612): fundamentals, systems of linear equations, block
  multiplication, eigenvalues and eigenvectors, symmetric matrices, vector calculus
- **Appendix B Introduction to Matlab** (p620)
- Answers to Selected Exercises (p630), Bibliography (p646), Index (p652)

Every section carries Exercises and Computer Problems. Code in the book is MATLAB.

### Pages read directly to confirm depth

| PDF page | Book page | What was confirmed |
|---|---|---|
| 3 | copyright | Title, author, edition, ISBN |
| 4 to 10 | iii to ix | Full table of contents |
| 18 | 1 | Chapter 0 opening, page offset calibration |
| 26 | 9 | IEEE 754 field widths, normalized form, machine epsilon definition |
| 106 | 89 | Section 2.3, infinity norm, backward error, forward error, residual |
| 128 | 111 | Conjugate Gradient algorithm box, A-inner product, conjugacy |
| 245 | 228 | Classical vs modified Gram-Schmidt loss of orthogonality experiment, Householder reflectors |
| 252 | 235 | GMRES with Krylov space and built-in orthogonalization |
| 583 | 566 | Normalized simultaneous iteration, unshifted QR algorithm derivation |
| 596 | 579 | SVD geometry, unit circle mapped to ellipse, singular vectors |
| 608 | 591 | Computing the SVD via Hessenberg/tridiagonal form, condition squaring warning |

---

## Source S2

| Field | Value |
|---|---|
| File | `numericalanalysis_book2.pdf` (15 MB) |
| Title | Numerical Methods: Fundamentals and Applications |
| Author | Rajesh Kumar Gupta (Central University of Haryana / Central University of Punjab) |
| Publisher / Year | Cambridge University Press, 2019 |
| ISBN | 978-1-108-71600-0 |
| Pages | 829 PDF pages, book pages to 791 (Index) |
| Text layer | Yes, full text extractable (1.17 MB of text) |
| How it was read | `pdftotext` extraction of the whole book |
| Code | C programs, hosted externally at cambridge.org, not inside the PDF |

### Structure

- **Ch 1 Number Systems** (p1): representation of integers, conversion between binary,
  octal, decimal, hexadecimal in both directions, representation of fractions
- **Ch 2 Error Analysis** (p13): 2.1 absolute, relative and percentage errors;
  2.2 errors in modeling (modeling error, inherent error, blunder); 2.3 errors in
  implementation (round-off, overflow and underflow, floating point arithmetic and error
  propagation, propagated error in arithmetic operations, error propagation in functions
  of one and several variables, truncation error, machine epsilon, loss of significance:
  condition and stability); 2.4 some interesting facts about error
- **Ch 3 Nonlinear Equations** (p47): polynomial and transcendental equations; direct,
  graphical, trial-and-error and iterative methods; bisection; fixed-point; Newton-
  Raphson; regula falsi; secant; convergence criteria for each of the five; order of
  convergence for each; Muller method; Chebyshev method; Aitken delta-squared process
- **Ch 4 Nonlinear Systems and Polynomial Equations** (p153): fixed-point for systems;
  Seidel iteration; Newton-Raphson for systems; complex roots; polynomial equations;
  Descartes rule of signs; Sturm sequence; Birge-Vieta (Horner) method; Lin-Bairstow
  method; Graeffe root squaring method
- **Ch 5 Systems of Linear Equations** (p177): Cramer rule; matrix inversion method;
  LU decomposition (Doolittle, Crout, Cholesky); Gauss elimination with operation counts;
  Thomas algorithm for tridiagonal systems; Gauss-Jordan; comparison of direct methods;
  pivoting strategies; iterative methods; Jacobi; Gauss-Seidel; relaxation method;
  convergence criteria for iterative methods; matrix forms and convergence; applications
- **Ch 6 Eigenvalues and Eigenvectors** (p268): real and complex eigenvalues; distinct
  and repeated eigenvalues; linearly independent and dependent eigenvectors; bounds on
  eigenvalues (Gerschgorin theorem, Brauer theorem); Rayleigh power method; inverse power
  method; shifted power method; Rutishauser (LU decomposition) method
- **Ch 7 Eigenvalues and Eigenvectors of Real Symmetric Matrices** (p299): similarity
  transformations; orthogonal transformations; Jacobi method; Sturm sequence for real
  symmetric tridiagonal matrices; Givens method; Householder method
- **Ch 8 Interpolation** (p331): polynomial forms (power, shifted power, Newton, nested
  Newton, recursive algorithm, change of center); Lagrange method; Newton divided
  differences with proof and properties; error in interpolating polynomial; Hermite
  interpolation; piecewise interpolation; Weierstrass approximation theorem
- **Ch 9 Finite Operators** (p364): forward, backward and central difference operators;
  mean/average operator; shift operator; differential operator; linearity and
  commutativity; interrelations between operators; operators on some functions; Newton
  divided differences and other finite differences; finite difference tables and error
  propagation in them
- **Ch 10 Interpolation for Equal Intervals and Bivariate Interpolation** (p389):
  Gregory-Newton forward and backward difference formulas with errors; central difference
  formulas; Gauss forward and backward central difference formulas; Stirling; Bessel;
  Everett; Steffensen; bivariate interpolation (Lagrange and Newton)
- **Ch 11 Splines, Curve Fitting, and Other Approximating Curves** (p445): spline
  interpolation; cubic spline (general and equi-spaced); Bezier curve; B-spline curve;
  least squares curve (straight line, linearization of nonlinear curves, quadratic);
  Chebyshev polynomials approximation; Pade approximation
- **Ch 12 Numerical Differentiation** (p495): numerical differentiation formulas derived
  from interpolation, with a summary table
- **Ch 13 Numerical Integration** (p509): Newton-Cotes quadrature via Lagrange
  (trapezoidal, Simpson 1/3, Simpson 3/8, Boole, Weddle); composite rules; errors in each
  rule; Gauss quadrature (Gauss-Legendre, Gauss-Chebyshev, Gauss-Laguerre, Gauss-Hermite);
  Euler-Maclaurin formula; Richardson extrapolation; Romberg integration; double integrals
- **Ch 14 First Order ODEs: Initial Value Problems** (p576): classification and
  terminology; existence and uniqueness; Picard method; Taylor series method; Euler;
  modified/improved Euler (Heun); Runge-Kutta methods; Milne (Milne-Simpson); Adams
  (Adams-Bashforth predictor and Adams-Moulton corrector); errors in numerical methods;
  order and stability; stability analysis of y-prime = Ay; backward Euler
- **Ch 15 Systems of First Order ODEs and Higher Order ODEs: IVPs and BVPs** (p642):
  Picard, Taylor, Euler and RK4 for systems; shooting method; finite difference
  approximations for first and second derivatives; finite difference method for BVPs;
  finite differences for unequal intervals
- **Ch 16 Partial Differential Equations: Finite Difference Methods** (p676):
  classification of second-order quasi-linear PDEs; initial and boundary conditions;
  finite difference approximations for partial derivatives; parabolic equations
  (Bender-Schmidt explicit, Crank-Nicolson, general implicit, Richardson, Du-Fort and
  Frankel); consistency, convergence, order and stability (matrix method and von Neumann
  method for both explicit and CN schemes); 2D heat conduction (explicit, CN, ADI);
  elliptic equations (Laplace, Poisson); hyperbolic equation (explicit, implicit);
  creating your own scheme
- **Appendices A-F**: comparison of analytical and numerical techniques; numerical
  techniques and computer; Taylor series; linear and nonlinear; graphs of standard
  functions; Greek letters

Every chapter ends with an Exercise set (38 exercise blocks found in total).

---

## Source S3

| Field | Value |
|---|---|
| File | `TrefethenBau.md` (19 KB, in the parent folder) |
| Title | Numerical Linear Algebra |
| Authors | Lloyd N. Trefethen and David Bau III |
| Publisher / Year | SIAM, 1997 |
| Form supplied | A structured concept map of all 40 lectures, in Markdown, not the book text |
| Added | 2026-08-20, after the initial course design was complete |
| How it was read | Read in full, directly. No extraction needed. |

### Structure

Six parts, 40 lectures: Fundamentals (1 to 5), QR Factorization and Least Squares (6 to 11),
Conditioning and Stability (12 to 19), Systems of Equations (20 to 23), Eigenvalues (24 to 31),
Iterative Methods (32 to 40). The document also records the book's major conceptual threads and
its algorithm inventory.

The concept list is deliberately **not** reproduced anywhere in this repository.
`TrefethenBau.md` is the authoritative checklist, and the mapping from it into the course is
[`TrefethenBau_Course_Mapping.md`](TrefethenBau_Course_Mapping.md).

### Effect on the course

The integration audit resolved all 40 lectures with nothing unaccounted for:

| Status | Lectures |
|---|---|
| Already fully covered by an existing lesson plan | 11 |
| Integrated, with a named missing piece added | 16 |
| Requiring substantial strengthening | 11 |
| Requiring a new lesson | 2 |

The single structural change, one inserted lesson, is recorded in
[`CURRICULUM_CHANGE_LOG.md`](CURRICULUM_CHANGE_LOG.md).

Because this source arrived after Part 1 was written, **19 concepts previously tracked as
supplementary were reattributed to S3**. They had only been supplementary because the source
had not been supplied yet.

---

## Note on the original Phase 1 finding

The Phase 1 discovery pass recorded that Trefethen and Bau was **not** in the folder, and the
course was designed on that basis, with the linear algebra depth delivered from Sauer plus
clearly labelled supplementary material.

That finding was correct when it was made. The reference document was supplied afterwards, and
this inventory and the course architecture have both been updated to match.

One useful consequence: most of the Trefethen and Bau material now has **two independent
sources** behind it. The concept map says what must be covered, and the Sauer pages read
directly during Phase 1 (listed in the table above) confirm the mathematical content.
