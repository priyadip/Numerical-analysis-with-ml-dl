# Source Coverage Report

**This file is generated.** `_planning/make_coverage_report.py` writes it from
`coverage_data.py` and `coverage_status.json`, and `verification/run_all.py` runs
both. Nothing in it is typed by hand, so it cannot fall behind the repository.

It answers one question with evidence rather than assumption:

> Does the finished part of this course actually contain the material it claims to,
> including the material from the source books that it reorganised?

The full concept-by-concept table is in
[`_planning/INTERNAL_COVERAGE_CHECKLIST.md`](../_planning/INTERNAL_COVERAGE_CHECKLIST.md).

---

## Overall position

| Measure | Count | Of total | Notes |
|---|---:|---:|---|
| Lessons written and verified | 98 | 98 | notebook present and executing |
| Concepts marked covered | 514 | 514 | 100.0 percent |
| Parts complete | 14 | 14 | every concept in the part covered |
| **Concepts outstanding in written lessons** | **0** | | nothing written is incomplete |

## By source

| Source | Covered | Of that source's total |
|---|---:|---:|
| S1, Sauer, *Numerical Analysis* 3rd ed. | 186 | 186 |
| S2, Gupta, *Numerical Methods* | 178 | 178 |
| S3, Trefethen and Bau, *Numerical Linear Algebra* | 75 | 75 |
| SUP, supplementary material | 75 | 75 |
| **Total** | **514** | **514** |

## By part

| Part | Folder | Covered | Assigned | Status |
|---|---|---:|---:|---|
| Foundations of Numerical Computing | `01_foundations/` | 49 | 49 | **complete** |
| Nonlinear Equations and Root Finding | `02_root_finding/` | 41 | 41 | **complete** |
| Direct Methods for Linear Systems | `03_direct_linear_systems/` | 57 | 57 | **complete** |
| Iterative and Krylov Subspace Methods | `04_iterative_and_krylov/` | 32 | 32 | **complete** |
| Orthogonality, QR and Least Squares | `05_least_squares_and_qr/` | 35 | 35 | **complete** |
| Eigenvalue Problems and the Singular Value Decomposition | `06_eigenvalues_and_svd/` | 47 | 47 | **complete** |
| Interpolation | `07_interpolation/` | 59 | 59 | **complete** |
| Approximation Theory and Transforms | `08_approximation_and_transforms/` | 27 | 27 | **complete** |
| Numerical Differentiation and Integration | `09_differentiation_and_integration/` | 37 | 37 | **complete** |
| Ordinary Differential Equations | `10_ordinary_differential_equations/` | 47 | 47 | **complete** |
| Partial Differential Equations | `11_partial_differential_equations/` | 32 | 32 | **complete** |
| Numerical Optimization | `12_optimization/` | 22 | 22 | **complete** |
| Stochastic and Monte Carlo Methods | `13_stochastic_methods/` | 12 | 12 | **complete** |
| Numerical Analysis in Machine Learning and AI | `14_ml_and_ai_connections/` | 17 | 17 | **complete** |

## Source chapters fully absorbed

A chapter is listed here only when **every** concept tracked from it is covered.

| Source | Chapter | Concepts | Lands in |
|---|---|---:|---|
| S1 | 0 Fundamentals | 12 | lessons 01 to 03, 05, 07 to 08 |
| S1 | 1 Solving Equations | 16 | lessons 06, 09 to 12, 14 |
| S1 | 10 Trigonometric Interpolation and the FFT | 9 | lessons 58 to 59 |
| S1 | 11 Compression | 10 | lesson 60 |
| S1 | 12 Eigenvalues and Singular Values | 14 | lessons 36 to 37, 41 to 43 |
| S1 | 13 Optimization | 7 | lessons 85 to 87 |
| S1 | 2 Systems of Equations | 24 | lessons 14 to 15, 17 to 21, 23 to 25 |
| S1 | 3 Interpolation | 14 | lessons 44 to 47, 51 to 52 |
| S1 | 4 Least Squares | 17 | lessons 26 to 27, 29 to 31, 34 |
| S1 | 5 Differentiation and Integration | 12 | lessons 61 to 65 |
| S1 | 6 Ordinary Differential Equations | 19 | lessons 67 to 73 |
| S1 | 7 Boundary Value Problems | 7 | lessons 75 to 76 |
| S1 | 8 Partial Differential Equations | 11 | lessons 78 to 79, 81 to 83 |
| S1 | 9 Random Numbers | 10 | lessons 91 to 93 |
| S1 | Appendix A | 3 | lessons 15, 35, 84 |
| S1 | Appendix B | 1 | lesson 01 |
| S2 | 1 Number Systems | 6 | lesson 02 |
| S2 | 10 Equal Interval Interpolation | 11 | lessons 49, 53 |
| S2 | 11 Splines and Curve Fitting | 9 | lessons 29, 51 to 52, 56 to 57 |
| S2 | 12 Numerical Differentiation | 2 | lesson 61 |
| S2 | 13 Numerical Integration | 16 | lessons 62 to 63, 65 to 66 |
| S2 | 14 First Order ODE IVPs | 13 | lessons 67 to 69, 71 to 72 |
| S2 | 15 ODE Systems and BVPs | 6 | lessons 61, 73, 75 |
| S2 | 16 Partial Differential Equations | 19 | lessons 77 to 82 |
| S2 | 2 Error Analysis | 13 | lessons 03 to 05 |
| S2 | 3 Nonlinear Equations | 13 | lessons 09 to 12 |
| S2 | 4 Nonlinear Systems and Polynomials | 9 | lessons 13 to 14 |
| S2 | 5 Systems of Linear Equations | 17 | lessons 17 to 21, 23 |
| S2 | 6 Eigenvalues and Eigenvectors | 8 | lessons 35 to 37 |
| S2 | 7 Symmetric Eigenproblem | 6 | lessons 35, 38 |
| S2 | 8 Interpolation | 15 | lessons 44 to 46, 50, 54 |
| S2 | 9 Finite Operators | 11 | lesson 48 |
| S2 | Appendix A | 1 | lesson 01 |
| S2 | Appendix B | 1 | lesson 01 |
| S2 | Appendix C | 1 | lesson 07 |
| S2 | Appendix D | 1 | lesson 01 |
| S3 | L1 Matrix-Vector Multiplication | 2 | lesson 15 |
| S3 | L10 Householder | 3 | lesson 31 |
| S3 | L11 Least Squares | 1 | lesson 29 |
| S3 | L12 Conditioning | 1 | lesson 06 |
| S3 | L13 | 1 | lesson 03 |
| S3 | L14 | 1 | lesson 06 |
| S3 | L15 More on Stability | 1 | lesson 06 |
| S3 | L16 Stability of Householder | 2 | lesson 31 |
| S3 | L17 Stability of Back Substitution | 2 | lesson 17 |
| S3 | L18 Conditioning of Least Squares | 2 | lesson 32 |
| S3 | L19 Stability of Least Squares Algorithms | 2 | lesson 32 |
| S3 | L2 Orthogonal Vectors and Matrices | 6 | lesson 16 |
| S3 | L21, L22 | 1 | lesson 18 |
| S3 | L22 Stability of Gaussian Elimination | 1 | lesson 18 |
| S3 | L24 | 1 | lesson 35 |
| S3 | L24, L29 | 1 | lesson 35 |
| S3 | L25 Overview of Eigenvalue Algorithms | 2 | lesson 35 |
| S3 | L27 Rayleigh Quotient | 1 | lesson 36 |
| S3 | L3 | 3 | lesson 15 |
| S3 | L3 Norms | 1 | lesson 15 |
| S3 | L31 | 2 | lesson 42 |
| S3 | L32 Overview of Iterative Methods | 1 | lesson 23 |
| S3 | L33 | 1 | lesson 26 |
| S3 | L33 Arnoldi Iteration | 1 | lesson 26 |
| S3 | L34 | 1 | lesson 26 |
| S3 | L34 How Arnoldi Locates Eigenvalues | 2 | lesson 39 |
| S3 | L35 | 1 | lesson 31 |
| S3 | L35 GMRES | 2 | lesson 27 |
| S3 | L35, L39 | 1 | lesson 27 |
| S3 | L36 | 1 | lesson 26 |
| S3 | L36 Lanczos Iteration | 2 | lesson 26 |
| S3 | L37 | 1 | lesson 65 |
| S3 | L37 Lanczos to Gauss Quadrature | 2 | lesson 65 |
| S3 | L38 Conjugate Gradients | 2 | lesson 24 |
| S3 | L39 Biorthogonalization | 2 | lesson 27 |
| S3 | L4 Singular Value Decomposition | 2 | lesson 41 |
| S3 | L4, L11 | 1 | lesson 32 |
| S3 | L40 | 1 | lesson 25 |
| S3 | L40 Preconditioning | 2 | lesson 25 |
| S3 | L5 | 1 | lesson 43 |
| S3 | L5 More on the SVD | 1 | lesson 43 |
| S3 | L6 Projectors | 7 | lesson 16 |
| S3 | L7 QR Factorization | 1 | lesson 30 |
| S3 | L8 Gram-Schmidt | 2 | lesson 30 |
| SUP | - | 75 | lessons 05, 07 to 08, 13 to 14, 20 to 22, 26, 28, 32 to 33, 38 to 40, 43 to 44, 47, 52 to 55, 59, 61, 66, 69, 71 to 72, 74, 79, 83 to 84, 86 to 92, 94 to 98 |

## Evidence behind the flags

The five flags are set by `_planning/mark_covered.py`, which inspects the built
artifacts. No flag is set by hand, and `run_all.py` re-derives all of them on every
run.

| Flag | How it is decided |
|---|---|
| `covered` | Both `.ipynb` and `.md` exist for the lesson |
| `implemented` | The executed notebook contains at least one `def`, meaning from-scratch code |
| `experimented` | The executed notebook produced at least one figure |
| `tested` | A module in `tests/` exercises the `nalib` module the lesson builds on |
| `verified` | The notebook executed with no error **and** contains at least one `assert` |

Flag totals across all tracked concepts:

```text
covered         :  514 of 514
implemented     :  374 of 514
experimented    :  514 of 514
tested          :  514 of 514
verified        :  514 of 514
```

## What is deliberately not claimed

- 0 of 514 concepts remain, all assigned to a named lesson
  so that nothing can be quietly dropped.
- A `covered` flag means the material is present, executed and asserted. It does not
  claim the treatment is the deepest possible one.
- Chapters are listed as absorbed on the concept list this course tracks, which is a
  reading of the source rather than the source itself.
