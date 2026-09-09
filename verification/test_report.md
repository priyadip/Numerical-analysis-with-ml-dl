# Test Report

Generated 2026-09-09 04:47 UTC.

The suite covers `src/nalib`, the from-scratch library that the lessons build and
then import. Every module is tested against known analytic results, against NumPy
or SciPy where an equivalent exists, on edge cases, and on seeded random input.

## Summary

- Tests passed: **9606**
- Tests failed: **0**
- Test modules: **89**

## Modules

| Module | Covers |
|---|---|
| `test_numbersystems.py` | base conversion, Theorem 2.3 on terminating expansions |
| `test_floatingpoint.py` | IEEE field decomposition, ulp spacing, the standard model, summation algorithms, the Sterbenz lemma |
| `test_errors.py` | error measures, condition numbers against analytic values, propagation rules, the governing inequality |
| `test_convergence.py` | observed order on sequences of known order, linear rate fitting, refinement studies |
| `test_polynomials.py` | Horner against `numpy.polyval` bit for bit, synthetic division, the remainder theorem, measured flop counts |
| `test_cost.py` | the operation counter against hand-computed counts, exponent fitting |

## What these tests actually catch

Two real defects were found by this suite while Part 1 was being written, and both
are worth recording because they are the kind of thing that silently survives
otherwise:

1. **`np.longdouble` is not higher precision on this platform.** On Windows with the
   Microsoft compiler it is an alias for `float64`. Two lessons had been using it as
   a high precision reference, which meant they were comparing a value against
   itself. Both now use `math.fsum`, which is exactly rounded, or
   `fractions.Fraction`, which is exact. The Sterbenz test that exposed this would
   have passed vacuously forever.
2. **An off-by-one in a flop count.** The naive polynomial evaluator performs n+1
   additions, not n, because the running total starts at zero. The measured count
   disagreed with the derived formula and the formula was wrong.

## Raw pytest output

```text
........................................................................ [ 79%]
........................................................................ [ 80%]
........................................................................ [ 80%]
........................................................................ [ 81%]
........................................................................ [ 82%]
........................................................................ [ 83%]
........................................................................ [ 83%]
........................................................................ [ 84%]
........................................................................ [ 85%]
........................................................................ [ 86%]
........................................................................ [ 86%]
........................................................................ [ 87%]
........................................................................ [ 88%]
........................................................................ [ 89%]
........................................................................ [ 89%]
...................ss................................................... [ 90%]
........................................................................ [ 91%]
....s................................................................... [ 92%]
........................................................................ [ 92%]
........................................................................ [ 93%]
........................................................................ [ 94%]
........................................................................ [ 95%]
........................................................................ [ 95%]
........................................................................ [ 96%]
........................................................................ [ 97%]
........................................................................ [ 98%]
........................................................................ [ 98%]
........................................................................ [ 99%]
.....................................                                    [100%]
9606 passed, 7 skipped in 723.64s (0:12:03)
```

