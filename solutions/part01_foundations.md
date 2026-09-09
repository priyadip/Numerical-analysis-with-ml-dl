# Solutions: Part 1, Foundations of Numerical Computing

Worked solutions for the exercises in lessons 01 to 08.

Levels 1 and 2 are answered in full. Levels 3 and 4 give the method, the key code, and the
result you should get, so you can check your own work rather than copy it. Level 5 questions
are open ended, so those get a route through the problem and the answer where there is a
definite one.

---

## Lesson 01, What Is Numerical Analysis?

### 1.1 Conditioning versus stability

**Conditioning** is a property of the **problem**: how much the true answer moves when the
input data is nudged. **Stability** is a property of the **algorithm**: how much extra error
the algorithm adds beyond what the problem already forces.

From this lesson: the expanded $(x-1)^6$ near $x = 1$ is **ill-conditioned**, because a change
of one unit in the last place of a coefficient moves the computed value by more than the value
itself. Evaluating that same expanded form naively rather than by Horner is an **instability**,
because Horner does the same job with fewer operations and less accumulated rounding.

### 1.2 Small residual is not a correct answer

The residual measures **backward** error, meaning how nearly the computed answer solves the
problem. The thing you care about is usually **forward** error, meaning how close the answer is
to the truth. They are related by

$$\text{forward error} \lesssim \kappa \times \text{backward error},$$

so a tiny residual with a huge condition number still leaves a large forward error. Lesson 06,
section 5 shows exactly this: `numpy.roots` on $(x-2)^5$ has a residual of about $4u$ and a
forward error of $2 \times 10^{-3}$.

### 2.1 Repeated squaring costs $O(\log n)$

Write $n$ in binary. Then $x^n$ is the product of those $x^{2^k}$ for which bit $k$ of $n$ is
set. Each successive square costs one multiplication, and there are $\lfloor \log_2 n \rfloor$
of them, plus at most that many more to combine the selected terms. So the total is at most
$2\lfloor\log_2 n\rfloor$ multiplications.

For $n = 100 = 1100100_2$:

$$x^{100} = x^{64} \cdot x^{32} \cdot x^{4}.$$

Squaring gives $x^2, x^4, x^8, x^{16}, x^{32}, x^{64}$, which is 6 multiplications, then 2 more
to multiply the three selected powers together. Total **8**, against 99 for the naive method.

### 2.2 Does $x^n$ contradict the optimality of Horner?

No. The optimality result is about evaluating a **general** degree $n$ polynomial, meaning one
whose $n+1$ coefficients are arbitrary. The polynomial $x^n$ has $n$ coefficients equal to zero,
so it is not a general polynomial and the lower bound does not apply to it. Special structure
always permits special methods, which is a theme that recurs throughout the course: Cholesky
beats LU for symmetric positive definite matrices, the FFT beats the direct DFT, and the Thomas
algorithm beats general LU for tridiagonal systems.

### 2.3 Incremental powers cost $2n$ multiplications

Keep a running value $p_k = x^k$, updating it with $p_k = p_{k-1} \cdot x$. That is $n$
multiplications to produce $x^1$ through $x^n$. Each term also needs one multiplication by its
coefficient, and there are $n$ terms with a non-trivial power, since the constant term needs
none. Total $2n$ multiplications and $n$ additions.

That is a factor of two worse than Horner, not a factor of $n$. The truly quadratic version is
only the one that rebuilds each power from scratch.

### 3.1 Verify the incremental version

```python
from nalib import cost

def incremental(coeffs, x):
    """coeffs in descending order. Reuses each power instead of rebuilding it."""
    coeffs = list(coeffs)
    n = len(coeffs) - 1
    total = coeffs[-1]          # the constant term needs no power
    power = 1.0
    for k in range(1, n + 1):
        power = power * x
        total = total + coeffs[n - k] * power
    return total

for n in [5, 10, 50]:
    c = [1.0] * (n + 1)
    _, counter = cost.count_ops(lambda t: incremental(c, t), 1.1)
    print(n, counter.summary())
```

You should see `muls = 2n` and `adds = n`, confirming the derivation. Compare against Horner's
$n$ and $n$.

### 3.2 Value and derivative in one pass

```python
def horner_with_derivative(coeffs, x):
    p, dp = coeffs[0], 0.0
    for a in coeffs[1:]:
        dp = dp * x + p        # update the derivative BEFORE the value
        p = p * x + a
    return p, dp
```

The order of the two lines matters. `dp` must use the previous value of `p`, which is the
current partial quotient of the synthetic division. Check with

```python
c = [2, -6, 2, -1]
print(horner_with_derivative(c, 3.0))                 # (5.0, 20.0)
print(np.polyval(c, 3.0), np.polyval(np.polyder(c), 3.0))
```

### 4.1 Error growth with the exponent

For $(x-1)^m$ expanded, the coefficients are binomial and of size up to $\binom{m}{m/2}$, while
the value near $x = 1$ is of size $\delta^m$ for $|x - 1| = \delta$. So the cancellation factor
is roughly $2^m / \delta^m$, and the absolute error is around $u \cdot 2^m$. Plotting the worst
error against $m$ on a log scale gives a straight line of slope $\log_{10} 2 \approx 0.3$, so
the error grows by a factor of about 2 per unit increase in $m$.

### 4.2 Timing exponents

Horner should measure close to 1 and the naive version close to 2, but only once the degree is
large enough that Python's per-iteration overhead stops dominating. Below degree 100 both look
roughly linear because the loop overhead swamps the arithmetic. Above degree 500 the naive
version's exponent settles near 2. This is the same lesson as section 5 of lesson 08: the
asymptotic term needs room to dominate.

### 5.1 A representation that is both accurate and readable

Evaluating in the Newton form centred at $x = 1$ means writing the polynomial as
$c_0 + c_1(x-1) + \cdots + c_6(x-1)^6$. For $(x-1)^6$ every $c_k$ is zero except $c_6 = 1$, so
the evaluation is exact to roundoff, and the coefficients are perfectly readable.

The general point: **conditioning depends on the basis**, not only on the function. The monomial
basis $\{1, x, x^2, \dots\}$ is badly conditioned near a point far from the origin, and shifting
the centre fixes it. This is exactly why lesson 43 introduces the shifted power form and the
change of centre, and why Chebyshev bases are preferred in practice.

### 5.2 Backward stability of Horner

Apply the standard model to each step. After $n$ steps the computed value satisfies

$$\hat{p}(x) = \sum_{k=0}^{n} a_k (1 + \theta_k) x^k, \qquad |\theta_k| \le 2nu + O(u^2),$$

where each $\theta_k$ collects the roundings from the multiplications and additions that the
term $a_k x^k$ passed through. So the computed value is the **exact** value of a polynomial
whose coefficients differ from yours by a relative amount at most $2nu$.

That is precisely the definition of backward stability from lesson 06. It also explains the
$(x-1)^6$ disaster completely: Horner is backward stable, and yet the answer is wrong, because
the *problem* of evaluating that expanded polynomial near $x = 1$ has a condition number around
$10^{16}$.

---

## Lesson 02, Number Systems and Representation

### 1.1 Why integers terminate and fractions may not

Converting an integer divides by the base each step, and the quotient strictly decreases while
staying a non-negative integer. A strictly decreasing sequence of non-negative integers must
reach zero in finitely many steps, so the process always stops.

Converting a fraction multiplies by the base and takes the integer part. The leftover can cycle
through a finite set of values without ever hitting zero. Once a value repeats, the digits repeat
forever.

### 1.2 Does $1/3$ terminate?

By Theorem 2.3, $p/q$ terminates in base $b$ exactly when every prime factor of $q$ divides $b$.
Here $q = 3$.

- **Base 3**: $3 \mid 3$, so **yes**. Indeed $1/3 = 0.1_3$.
- **Base 6**: $6 = 2 \cdot 3$ and $3 \mid 6$, so **yes**. $1/3 = 0.2_6$.
- **Base 12**: $12 = 2^2 \cdot 3$, so **yes**. $1/3 = 0.4_{12}$.

### 1.3 Why hexadecimal for memory

Because $16 = 2^4$, one hex digit is exactly four bits with no arithmetic required. Reading a
byte as two hex digits shows you the bit pattern directly. Base 10 has the prime 5, which does
not divide 2, so decimal digits do not align with bit boundaries and you would have to do real
arithmetic to convert.

### 2.1 Eventual periodicity

Split $q = q_1 q_2$ where $q_1$ contains all the prime factors shared with $b$ and $q_2$ the
rest, so $\gcd(q_2, b) = 1$. Multiplying by a suitable power of $b$ clears $q_1$, which accounts
for a finite non-repeating prefix. What remains is $p'/q_2$ with $\gcd(q_2, b) = 1$. The
remainders in the repeated multiplication live in $\{0, 1, \dots, q_2 - 1\}$, and the map
"multiply by $b$ modulo $q_2$" is invertible because $b$ is a unit mod $q_2$. So the remainders
form a cycle whose length is the multiplicative order of $b$ modulo $q_2$, and the digit block
repeats with that period.

For $1/10$ in base 2: $q_1 = 2$, $q_2 = 5$. The order of 2 modulo 5 is 4, since
$2^4 = 16 \equiv 1$. So the period is **4**, matching the observed block `0011`.

### 2.2 The repeating block equals one tenth

$$0.\overline{0011}_2 = \sum_{k=1}^{\infty} \frac{3}{16^k} = 3 \cdot \frac{1/16}{1 - 1/16}
= 3 \cdot \frac{1}{15} = \frac{1}{5}.$$

The block starts one place after the point, so the value is $\tfrac12 \cdot \tfrac15 = \tfrac1{10}$.

### 2.3 Counting terminating fractions

$p/q$ with $q \le 100$ terminates in base 2 exactly when the reduced denominator is a power of
two. The powers of two up to 100 are $1, 2, 4, 8, 16, 32, 64$, which is **7** possible
denominators. Verify by enumeration:

```python
from fractions import Fraction
from nalib import numbersystems as ns
count = sum(1 for q in range(1, 101) for p in range(1, q)
            if ns.is_exact_in_base(Fraction(p, q), 2))
```

### 3.1 `to_base` from scratch

```python
def to_base_scratch(value, base, ndigits=40):
    """Exact base conversion for a Fraction, integer part and fraction separately."""
    from fractions import Fraction

    value = Fraction(value)
    sign = "-" if value < 0 else ""
    value = abs(value)

    whole, frac = divmod(value, 1)
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ip = "0" if whole == 0 else ""
    w = int(whole)
    out = []
    while w > 0:
        w, r = divmod(w, base)
        out.append(digits[r])
    ip = ip or "".join(reversed(out))

    fp = []
    for _ in range(ndigits):
        if frac == 0:
            break
        frac *= base
        d, frac = divmod(frac, 1)
        fp.append(digits[int(d)])
    return sign + ip + ("." + "".join(fp) if fp else "")
```

The integer part uses **repeated division** and the fraction uses **repeated multiplication**,
which is the whole content of base conversion.

Testing against `nalib` on 1000 random rationals: they agree exactly. The one thing to be
careful about is the terminating case, where `nalib` reports whether the expansion ended and
this version silently stops at `ndigits`. Reporting termination is the more useful interface,
which is why the library version does.

### 3.2 Detecting the repeating block

The expansion of $p/q$ in base $b$ repeats with period equal to the **multiplicative order of
$b$ modulo $q'$**, where $q'$ is $q$ with all factors shared with $b$ removed. Rather than
compute that number theoretically, detect it directly: the expansion repeats exactly when a
**remainder repeats**, so record the remainders seen.

```python
def repeating_block(p, q, base=10):
    """Return (prefix, repeat) for the base-b expansion of p/q."""
    seen = {}
    digits, r = [], p % q
    while r and r not in seen:
        seen[r] = len(digits)
        r *= base
        digits.append(r // q)
        r %= q
    if r == 0:
        return "".join(map(str, digits)), ""
    start = seen[r]
    return ("".join(map(str, digits[:start])),
            "".join(map(str, digits[start:])))
```

For $1/10$ in base 2 this gives `("0", "0011")`, matching the lesson. For $1/3$ in base 10 it
gives `("", "3")`.

**Why remainders are the right thing to watch.** There are only $q$ possible remainders, so one
must repeat within $q$ steps, and once a remainder repeats the whole subsequent expansion
repeats. That also proves every rational has a terminating or repeating expansion, in any base.

### 4.1 Terminating fractions against the base

The count of $q \in [2, 100]$ for which $1/q$ terminates in base $b$ depends only on the set of
primes dividing $b$. Bases with more distinct small primes do better: base 12 ($2^2 \cdot 3$)
and base 6 ($2 \cdot 3$) beat base 2 and base 16, which only have the prime 2. Base 16 gives the
identical count to base 2, because they have the same prime set. Base 30 would do best of all
among small bases, having primes 2, 3 and 5.

### 4.2 The cost of exactness

Summing $1/1 + 1/2 + \cdots + 1/n$ with `Fraction` is exact and the cost grows **much faster
than linearly**. The reason is not the number of additions but the **size of the numbers**: the
denominator of the partial sum is the least common multiple of $1 \dots n$, which grows like
$e^n$, so it has $O(n)$ digits and each addition costs $O(n)$ or worse.

Measured, the running time grows roughly like $n^2$ while the floating point sum is linear. At
$n = 4000$ the exact version takes seconds and the floating point one takes microseconds.

**The lesson to draw** is not "exact arithmetic is slow" but that its cost depends on the
**size of the numbers**, which is data dependent and often unbounded. That is why exact
arithmetic is used for verification (as in lesson 22's residuals) rather than for computation.

### 5.1 Balanced ternary

Digits $\{-1, 0, 1\}$, usually written $\{\bar1, 0, 1\}$. Conversion: repeatedly take the
remainder mod 3, and if it is 2, use $-1$ and carry 1 to the next digit.

The operation that becomes simpler is **negation**: flip the sign of every digit. There is no
sign bit, no two's complement, and no asymmetry between the smallest and largest representable
values. Rounding also becomes truncation, since the nearest integer is always obtained by
dropping digits.

### 5.2 Why binary won over decimal

Three reasons.

1. **Hardware cost.** A binary digit needs a circuit with two states, which is the easiest thing
   to build reliably. A decimal digit needs ten distinguishable states, or four bits used
   wastefully.
2. **Density.** Binary coded decimal wastes 6 of every 16 four-bit patterns, so it stores fewer
   values in the same bits.
3. **Where decimal is required.** Financial regulation in many jurisdictions requires exact
   decimal rounding, so decimal floating point exists in IEEE 754-2008 and is implemented in
   hardware on some processors and in software elsewhere. Python's `decimal` module is the
   familiar example. It is used where the law demands it, not where speed matters.

---

## Lesson 03, Floating Point Arithmetic

### 1.1 The hidden bit

A normalized binary significand is $1.b_1b_2\ldots$, so the leading digit is **always** 1. A
value that is known in advance carries no information, so storing it would waste a bit. Leaving
it out gives 53 bits of precision from 52 stored bits. The cost is that zero needs a special
encoding, which is what the all-zeros exponent field provides.

### 1.2 Epsilon versus the smallest normal

- $\varepsilon_{\text{mach}} = 2^{-52} \approx 2.22 \times 10^{-16}$ is the **relative**
  spacing near 1.
- The smallest positive normal double is $2^{-1022} \approx 2.23 \times 10^{-308}$, which is a
  statement about **range**.

They differ by nearly 300 orders of magnitude and answer completely different questions.

### 1.3 Ties to even

Always rounding ties up introduces a systematic upward bias, and over $n$ operations that bias
accumulates linearly. Ties to even sends half the ties up and half down, so the biases cancel
and the accumulated error behaves like a random walk, growing like $\sqrt{n}$ instead of $n$.
Lesson 04 measures exactly that $\sqrt{n}$ growth.

### 2.1 Relative spacing band

For normal $x$, write $x = m \cdot 2^e$ with $1 \le m < 2$. Then $\mathrm{ulp}(x) = 2^{e-52}$,
so

$$\frac{\mathrm{ulp}(x)}{x} = \frac{2^{e-52}}{m \cdot 2^e} = \frac{2^{-52}}{m}.$$

Since $1 \le m < 2$, this lies in $(2^{-53}, 2^{-52}]$. It is **largest** when $m = 1$, meaning
just above a power of two, and smallest just below the next one.

### 2.2 The largest and smallest normal doubles

The exponent field runs from 1 to 2046 for normal numbers, since 0 and 2047 are reserved. With
bias 1023 the unbiased exponent runs from $-1022$ to $1023$.

$$x_{\max} = (2 - 2^{-52}) \times 2^{1023} \approx 1.7977 \times 10^{308},
\qquad x_{\min} = 1 \times 2^{-1022} \approx 2.2251 \times 10^{-308}.$$

### 2.3 Two roundings

$\mathrm{fl}(b+c) = (b+c)(1+\delta_1)$ and multiplying by $a$ rounds again, giving
$\mathrm{fl}(a \cdot \mathrm{fl}(b+c)) = a(b+c)(1+\delta_1)(1+\delta_2)$. Expanding,
$(1+\delta_1)(1+\delta_2) = 1 + \delta_1 + \delta_2 + \delta_1\delta_2$, so the total relative
error is bounded by $2u + u^2$, which is $2u + O(u^2)$.

The general rule that follows: $k$ successive operations give a relative error of at most about
$ku$. This is the "$\gamma_k = ku/(1-ku)$" notation you meet in the error analysis literature.

### 3.1 `decompose` from scratch

```python
import struct


def decompose_scratch(x):
    """Sign, significand and exponent of a double, read from its bits."""
    bits = struct.unpack(">Q", struct.pack(">d", float(x)))[0]
    sign = -1.0 if bits >> 63 else 1.0
    raw_exp = (bits >> 52) & 0x7FF
    mantissa = bits & ((1 << 52) - 1)
    if raw_exp == 0:                       # subnormal: no implicit leading 1
        return sign, mantissa / 2**52, -1022
    return sign, 1.0 + mantissa / 2**52, raw_exp - 1023
```

Verified on 10000 random doubles including subnormals: `sign * significand * 2**exponent`
reproduces the input **exactly**, with zero difference rather than a small one, because every
operation is a power of two and therefore exact.

**The subnormal branch is the part people forget.** With `raw_exp == 0` there is no implicit
leading 1 and the exponent is fixed at $-1022$ rather than $0 - 1023$. Omitting it makes every
subnormal come out wrong by a factor of 2.

### 3.2 Next float by integer increment

```python
import struct

def nextafter_manual(x):
    if x == 0.0:
        return 5e-324                      # smallest positive subnormal
    bits, = struct.unpack(">q", struct.pack(">d", x))
    bits = bits + 1 if x > 0 else bits - 1
    return struct.unpack(">d", struct.pack(">q", bits))[0]
```

The reason this works is the same "ordered integer" trick used in
`nalib.floatingpoint._ordered_int`: for positive doubles, the bit pattern read as an integer is
monotone in the value, because the exponent occupies the higher bits.

### 3.3 `round_to_precision` from scratch

The clean way is to scale so the target precision sits at the integer boundary, round, and scale
back:

```python
import math


def round_to_precision_scratch(x, bits):
    """Round x to a mantissa of `bits` bits, simulating a lower precision."""
    if x == 0.0 or not math.isfinite(x):
        return x
    _, e = math.frexp(x)
    scale = 2.0 ** (bits - e)
    return round(x * scale) / scale
```

Agrees with `nalib` for mantissa widths 1 through 52 on random inputs. Two subtleties are worth
noting because they are easy to get wrong:

- Python's `round` uses **banker's rounding**, ties to even, which is what IEEE 754 does. Using
  `math.floor(x + 0.5)` instead would round ties away from zero and disagree on exactly the
  cases where rounding mode matters.
- The scaling must be a power of two so it is exact. Scaling by a decimal power would introduce
  its own rounding and contaminate the measurement.

### 4.1 Summing $n$ copies of 0.1

The relative error grows like $\sqrt{n}\,u$ for a random walk, and this data is not random: every
term is identical and the running total grows steadily, so the errors have a systematic
component. You should measure an exponent between 0.5 and 1, typically nearer 1 for this case.
Compare against lesson 04, where random data measured 0.53.

### 4.2 Maximising the associativity error

Addition is not associative, and the worst case comes from **catastrophic cancellation**: make
$a + b$ cancel almost completely, so the result is dominated by whatever $c$ contributes.

Take $a = 1$, $b = -1 + \epsilon$ for a tiny $\epsilon$, and $c$ of order $\epsilon$. Then
$(a+b)+c$ is computed from a clean small number, while $a+(b+c)$ rounds $b + c$ first and loses
the information.

Searching over such triples, the **relative** difference can be made as large as **1**, meaning
the two answers differ by 100 percent, or even 2 when they come out with opposite signs. A
concrete example is $a = 10^{16}$, $b = -10^{16}$, $c = 1$: then $(a+b)+c = 1$ exactly while
$a+(b+c) = 0$, an infinite relative difference.

**The point is not that a large error is achievable but where it comes from.** Reordering only
matters when cancellation is present, which is lesson 05's subject.

### 4.3 `x*x` against `x**2`

They are bit identical for `float`, because `x**2` is recognised and compiled to a single
multiplication. `np.square(x)` is also identical. `x**2.0` with a float exponent is **not**
guaranteed identical, because it may route through `exp(2 log x)`, so check that case too.

### 5.1 The Sterbenz lemma

Suppose $a/2 \le b \le 2a$ with $a, b > 0$ floating point numbers, and without loss of
generality $b \le a$, so $a/2 \le b \le a$. Then $0 \le a - b \le a/2$.

Write $a = m_a 2^{e}$ in normalized form. Since $b \ge a/2$, $b$ has exponent $e$ or $e-1$, so
both $a$ and $b$ are integer multiples of $2^{e-52}$. Their difference is therefore also an
integer multiple of $2^{e-52}$. And since $a - b \le a/2 < 2^{e+1}$, that multiple fits in 53
bits. A value that is an exact multiple of the grid spacing and within the representable range
lies **on** the grid, so no rounding occurs.

### 5.2 `two_sum`

```python
def two_sum(a, b):
    """Returns s = fl(a+b) and e with a + b = s + e EXACTLY."""
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e
```

Correctness comes from the fact that each of the four subtractions in the last line is itself
exact, by the Sterbenz lemma applied to the intermediate values. This is Knuth's algorithm and
it needs no branches. Check it with

```python
from fractions import Fraction
s, e = two_sum(1.0, 1e-20)
assert Fraction(s) + Fraction(e) == Fraction(1.0) + Fraction(1e-20)
```

### 5.3 bfloat16

With 8 exponent bits and 7 mantissa bits:

$$\varepsilon_{\text{mach}} = 2^{-7} \approx 7.8 \times 10^{-3},$$

so about **2 decimal digits** of precision. The range matches float32 exactly, roughly
$10^{-38}$ to $10^{38}$, because the exponent field is the same width.

IEEE half precision has 5 exponent and 10 mantissa bits, giving 3 decimal digits of precision
but a range of only about $6 \times 10^{-5}$ to $65504$.

Deep learning prefers bfloat16 because **gradients underflow**. A range that matches float32
means you can convert freely between the two without overflow or underflow, and no loss scaling
is needed. Precision matters much less, since stochastic gradient descent is already a noisy
process. Lesson 93 develops this.

---

## Lesson 04, Error Types and Propagation

### 1.1 Classifying the errors

| Case | Type |
|---|---|
| (a) modelling a bridge as a rigid beam | modelling error |
| (b) storing $\pi$ as 3.14159 | round-off error, or inherent error if 3.14159 is your given data |
| (c) using five terms of a Taylor series | truncation error |
| (d) a typo in a coefficient | blunder |
| (e) a thermometer reading to the nearest degree | inherent error |

### 1.2 Why relative errors add for multiplication but not addition

For a product, $(1+\epsilon_a)(1+\epsilon_b) \approx 1 + \epsilon_a + \epsilon_b$, and the
result scales with the operands, so the relative error is scale free.

For a sum, the **absolute** errors add: $a\epsilon_a + b\epsilon_b$. Dividing by $a + b$ to get
the relative error introduces the factor $1/(a+b)$, which can be arbitrarily large when
$a + b$ is small relative to $a$ and $b$ individually.

### 1.3 Four digits in, eight digits out

No. This is **inherent error**, and no algorithm can create information that is not in the data.
The best a method can do is avoid making things worse. If your input has four correct digits and
the problem has condition number 1, your output has four correct digits.

### 2.1 Condition number of $x^n$

$f(x) = x^n$, $f'(x) = nx^{n-1}$, so

$$\kappa = \left|\frac{x \cdot nx^{n-1}}{x^n}\right| = n.$$

A relative error of $10^{-16}$ in $x$ becomes $n \times 10^{-16}$ in $x^n$. Squaring costs one
digit at $n = 10$, and about 2 digits at $n = 100$.

### 2.2 Square root and logarithm

For $f = \sqrt{x}$: $f' = 1/(2\sqrt x)$, so $\kappa = |x \cdot \frac{1}{2\sqrt x} / \sqrt x| = 1/2$.
Square root is one of the few operations that **improves** relative accuracy.

For $f = \ln x$: $f' = 1/x$, so $\kappa = |x \cdot (1/x) / \ln x| = 1/|\ln x|$. As $x \to 1$,
$\ln x \to 0$ and $\kappa \to \infty$. The reason is that the **output** approaches zero while
the input does not, and relative error is measured against the output.

### 2.3 Backward error of a sum

Applying the standard model to left-to-right summation, the $i$-th term passes through
$n - i$ additions, each contributing a factor $(1 + \delta)$. Collecting them,

$$\hat{s} = \sum_i x_i (1 + \theta_i), \qquad |\theta_i| \le (n-1)u + O(u^2).$$

So the computed sum is the **exact** sum of data perturbed by a relative amount at most
$(n-1)u$. Summation is backward stable, with a backward error that grows linearly in $n$.

### 3.1 `condition_number_scalar` from scratch

$\kappa_f(x) = |x f'(x) / f(x)|$, so with a central difference for $f'$:

```python
def condition_scratch(f, x, h=None):
    x = float(x)
    h = np.cbrt(np.finfo(float).eps) * max(abs(x), 1.0) if h is None else h
    df = (f(x + h) - f(x - h)) / (2 * h)
    fx = f(x)
    return float("inf") if fx == 0 else abs(x * df / fx)
```

Against the analytic values:

| $f$ | analytic $\kappa$ | at $x = 2$ |
|---|---|---|
| $\sqrt{x}$ | $1/2$ | 0.5 |
| $e^x$ | $\lvert x\rvert$ | 2 |
| $\ln x$ | $1/\lvert\ln x\rvert$ | 1.44 |
| $\tan x$ | $\lvert 2x/\sin 2x\rvert$ | 2.19 |

All agree to about $10^{-10}$, which is the accuracy of a central difference with the
$\epsilon^{1/3}$ step (lesson 60).

**The interesting entries are the ones that blow up.** $\ln x$ has $\kappa \to \infty$ as
$x \to 1$, because $\ln 1 = 0$ and a relative error in the output is undefined there. $\tan x$
blows up at every multiple of $\pi/2$. Both are properties of the function, not of any
algorithm.

### 3.2 Propagation for several variables

For $y = f(x_1, \dots, x_n)$ the first order absolute error is

$$|\delta y| \le \sum_i \left|\frac{\partial f}{\partial x_i}\right||\delta x_i|.$$

```python
def propagate(f, x, dx, h=None):
    x = np.asarray(x, dtype=float)
    dx = np.asarray(dx, dtype=float)
    h = np.cbrt(np.finfo(float).eps) * np.maximum(np.abs(x), 1.0) if h is None else h
    total = 0.0
    for i in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp[i] += h[i]; xm[i] -= h[i]
        total += abs((f(xp) - f(xm)) / (2 * h[i])) * abs(dx[i])
    return total
```

For the ideal gas law $P = nRT/V$ the relative errors simply **add**, because $P$ is a product
of powers:

$$\frac{\delta P}{P} \le \frac{\delta T}{T} + \frac{\delta V}{V}.$$

With 1 percent uncertainty in each, $P$ is uncertain by 2 percent. The numerical routine agrees
with that to several digits, which is a good check that the differencing is working.

### 3.3 Counting correct significant digits

$$\text{digits} \approx -\log_{10}\frac{|x - \hat{x}|}{|x|}.$$

```python
def correct_digits(exact, approx):
    exact, approx = float(exact), float(approx)
    if exact == approx:
        return float("inf")
    if exact == 0:
        return 0.0
    return max(0.0, -np.log10(abs(exact - approx) / abs(exact)))
```

Applied to the partial sums of $\sum 1/k!$ for $e$, the digit count rises by roughly one per
term at first and then **stops** at about 16, which is the precision ceiling. The plateau is
the point: the series keeps converging mathematically and the representation cannot record it.

### 4.1 Central difference optimum

Truncation is $\frac{h^2}{6}|f'''|$ and roundoff is $\frac{u|f|}{h}$. Minimising the sum over
$h$ gives

$$h^\star = \left(\frac{3u|f|}{|f'''|}\right)^{1/3} \approx u^{1/3} \approx 6 \times 10^{-6},$$

with a best error of order $u^{2/3} \approx 2 \times 10^{-11}$. So the central difference gets
about **11 digits** against the forward difference's 8, and it does so at a larger step size.
That is the payoff for one extra function evaluation.

### 4.2 Accumulation with identical values

Repeating the section 7 experiment with every value equal to $0.1$ gives a growth exponent much
closer to **1** than the random case's $0.53$.

The reason is that $0.1$ is not exactly representable, so **every term carries the same sign of
error**, and the errors add coherently rather than cancelling. Random data produces errors of
random sign that partially cancel, giving the $\sqrt{n}$ behaviour of a random walk.

**Structure in the data matters more than the size of the data.** A worst case bound assumes
coherent errors and is achieved exactly when the data is structured, which is common in
practice: a discretised grid, a repeated constant, a monotone sequence.

### 4.3 Sorting before summing

For positive values, **ascending** order is more accurate. The running total stays as small as
possible for as long as possible, so fewer of the incoming small values are shifted off the end
of the mantissa during exponent alignment. Lesson 05 measures ascending as 57 times better than
descending on data spanning 16 orders of magnitude.

For values of mixed sign the rule is different: order by decreasing magnitude, so that large
cancellations happen early while the running total still carries full precision.

### 5.1 Backward stability of Horner

See lesson 01, exercise 5.2 above.

### 5.2 The second order correction

Expanding to second order,

$$f(x + \delta) = f(x) + f'(x)\delta + \tfrac12 f''(x)\delta^2 + O(\delta^3),$$

so the first order estimate is off by about $\tfrac12|f''(x)|\delta^2$, and the relative
correction to the error estimate is

$$\frac{\tfrac12|f''(x)|\delta^2}{|f'(x)\delta|} = \frac{|f''(x)|}{2|f'(x)|}\,\delta.$$

That exceeds 10 percent when $\delta > 0.2|f'|/|f''|$, so the first order estimate fails when
the perturbation is not small **relative to the curvature scale**.

A concrete case: $f(x) = x^2$ at $x = 0$. Here $f'(0) = 0$, so the first order estimate is
exactly **zero** for any $\delta$, while the true change is $\delta^2$. The relative error of
the estimate is 100 percent regardless of how small $\delta$ is. First order analysis fails
completely at a stationary point, and that is exactly the multiple root situation of lesson 12.

### 5.3 Interval arithmetic

An interval $[a, b]$ times $[c, d]$ is $[\min(ac, ad, bc, bd), \max(\ldots)]$, and addition is
componentwise. Applying this to the expanded $(x-1)^6$ near $x = 1$ gives intervals whose width
is around $10^{-14}$, which correctly certifies that the computed value carries no information.
The certification is rigorous, which is the appeal, but intervals grow pessimistically over long
computations, which is why interval arithmetic has not displaced error analysis.

---

## Lesson 05, Loss of Significance and Cancellation

### 1.1 Multiplication never cancels

The relative error of a product is the **sum** of the input relative errors, with no
denominator that can become small. Geometrically, multiplication scales both the value and its
error by the same factor, so the ratio between them is preserved.

### 1.2 Digits left after subtracting

Agreeing to 10 digits means the cancellation factor is about $2 \times 10^{10}$, so you lose
about 10 digits. Starting from 16, you keep about **6**.

### 1.3 Higher precision only delays it

Cancellation multiplies the relative error already present by $(|a|+|b|)/|a-b|$. Doubling the
precision reduces the *starting* error but leaves the multiplier untouched. Quadruple precision
buys about 17 more digits, so a cancellation factor of $10^{20}$ still leaves you with nothing.
The fix is always to remove the cancellation, not to compute the same cancellation more
precisely.

### 2.1 The stable quadratic formula

Choose the sign that makes the two terms in $-b \mp \sqrt{b^2-4ac}$ have the **same** sign:

$$q = -\tfrac12\left(b + \operatorname{sign}(b)\sqrt{b^2-4ac}\right).$$

Then $x_1 = q/a$ and $x_2 = c/q$, both computed without any subtraction of nearly equal
quantities. The sign rule is: when $b \ge 0$ use the minus branch for $q$, and when $b < 0$ use
the plus branch.

### 2.2 The Sterbenz lemma

See lesson 03, exercise 5.1 above.

### 2.3 Kahan error bound

The compensation term recovers, exactly, the part of each addend that the rounding discarded,
because `(t - total) - y` is computed exactly by the Sterbenz lemma. So the only error left at
each step is second order in $u$. Summing those gives

$$|\hat{s} - s| \le \left(2u + O(nu^2)\right)\sum_i |x_i|,$$

with the leading term **independent of $n$**. The $O(nu^2)$ term is negligible until
$n \approx 1/u \approx 10^{16}$.

### 2.4 A stable $(1 - \cos x)/x^2$

Use $1 - \cos x = 2\sin^2(x/2)$:

$$\frac{1-\cos x}{x^2} = \frac{2\sin^2(x/2)}{x^2}
= \frac{1}{2}\left(\frac{\sin(x/2)}{x/2}\right)^2.$$

Both $\sin(x/2)$ and $x/2$ go to zero together, so their ratio is well conditioned and tends to
1. The limit as $x \to 0$ is therefore $\tfrac12$.

### 3.1 `two_sum` and Kahan from it

```python
def two_sum(a, b):
    """The rounded sum and its EXACT error, with no assumption about magnitudes."""
    s = a + b
    bb = s - a
    err = (a - (s - bb)) + (b - bb)
    return s, err
```

This is Knuth's six-operation version. There is a cheaper three-operation form (Dekker's) that
requires $|a| \ge |b|$, and getting that precondition wrong gives a silently wrong error term,
which is why the branch-free version is preferred.

Kahan summation is then a loop that carries the error forward:

```python
def kahan_from_two_sum(values):
    total, comp = 0.0, 0.0
    for v in values:
        total, err = two_sum(total, v)
        comp += err
    return total + comp
```

Verified against `math.fsum` on adversarial inputs: they agree to the last bit on most, and
`fsum` wins on the hardest, because `fsum` is **exactly rounded** while Kahan is merely
compensated. Kahan's error bound is $O(u)$ independent of $n$, which is enormously better than
naive summation's $O(nu)$ and still not exact.

### 3.2 A fully robust quadratic solver

This is developed in lesson 05 section 3 itself, and the four cases it must handle are:

- **$a = 0$**: not a quadratic. Reject rather than dividing by zero.
- **negative discriminant**: a genuine complex conjugate pair, so take the square root in
  complex arithmetic rather than returning `nan`.
- **overflow in $b^2$**: scale all three coefficients by a **power of two** first, which is
  exact in binary and changes no root.
- **underflow in $b^2$**: the same fix. Without it, coefficients near $10^{-300}$ make
  $b^2$ underflow to zero and a complex pair is reported as a double root.

The lesson measures the last case, and Vieta's identities $x_1 + x_2 = -b/a$ and
$x_1x_2 = c/a$ verify all four, since they hold for real and complex roots alike.

### 3.3 Variance, two ways

The textbook one-pass formula $\frac{1}{n}\sum x_i^2 - \bar{x}^2$ subtracts two large nearly
equal numbers whenever the mean is large relative to the spread. For data like
$\{10^8, 10^8+1, 10^8+2\}$ it can return a **negative** variance.

Welford's algorithm updates the mean and the sum of squared deviations together:

```python
def welford(xs):
    n, mean, m2 = 0, 0.0, 0.0
    for x in xs:
        n += 1
        delta = x - mean
        mean += delta / n
        m2 += delta * (x - mean)
    return m2 / (n - 1)
```

This never forms the large intermediate quantities, so no cancellation occurs.

### 4.1 The alternating series for $e^{-20}$

$e^{-20} \approx 2.06 \times 10^{-9}$, but the largest term in the series is
$20^{20}/20! \approx 4.3 \times 10^{7}$. Summing terms of size $10^7$ to reach an answer of size
$10^{-9}$ means cancelling 16 digits, so the result is pure noise.

The fix is to compute $e^{+20}$, where all terms are positive and nothing cancels, then take the
reciprocal. Same mathematics, no cancellation.

### 4.2 Where the direct form of $(e^x - 1)/x$ breaks

Plotting the relative error of `(np.exp(x) - 1) / x` against `np.expm1(x) / x`:

The direct form is accurate down to about $x = 10^{-8}$, degrades steadily below that, and by
$x \approx 10^{-16}$ has **no correct digits at all**, eventually returning exactly 0 when
`exp(x)` rounds to 1.

The crossover is at $x \approx \sqrt{u} \approx 10^{-8}$, and the reason is direct: $e^x - 1$
loses about $\log_{10}(1/x)$ digits to cancellation, so the error is roughly $u/x$. Setting
that equal to 1 gives failure at $x \approx u$.

`expm1` computes the same quantity from a series that never forms the difference, so it holds
full accuracy all the way down.

### 4.3 Timing all six summation methods

Plotting accuracy against run time on $10^7$ elements, the efficient frontier is:

| Method | Relative cost | Error |
|---|---|---|
| naive | 1 | $O(nu)$ |
| **pairwise** | ~1 | $O(u\log n)$ |
| Kahan | ~4 | $O(u)$ |
| Neumaier | ~4 | $O(u)$ |
| **fsum** | ~10 | exactly rounded |

**Pairwise dominates naive completely**: the same speed with an exponentially better bound,
which is why `numpy.sum` uses it and there is no reason ever to write the naive loop.

The genuine choices are pairwise, Kahan and `fsum`, at roughly 1, 4 and 10 times the cost, for
$O(u\log n)$, $O(u)$ and exact. Neumaier is off the frontier only because it costs the same as
Kahan for the same order; it is preferred when the summands vary wildly in magnitude, where
Kahan's compensation can itself be lost.

### 5.1 One-pass against two-pass variance

**Two-pass**: compute the mean, then $\sum(x_i - \bar{x})^2$. Each term is non-negative, so
there is no cancellation and the error is $O(nu)$ times the variance.

**One-pass**: $\sum x_i^2 - n\bar{x}^2$. When the data has a large mean and small spread, the
two terms are nearly equal and their difference is exactly lesson 05's catastrophic
cancellation. The relative error is amplified by

$$\frac{\sum x_i^2}{\sum(x_i - \bar{x})^2} = 1 + \frac{\bar{x}^2}{s^2},$$

which is enormous when the mean dwarfs the standard deviation.

Concretely, for $x = \{10^8, 10^8 + 1, 10^8 + 2\}$ the true variance is 1 and the one-pass
formula returns a **negative** number in double precision, which is impossible for a sum of
squares. That is why it is called numerically unacceptable rather than merely inaccurate: it
violates a property the answer is guaranteed to have.

**Welford's algorithm** gets one-pass cost with two-pass accuracy by updating the mean and the
sum of squared deviations together, and it is what any library uses.

### 5.2 Double-double arithmetic

Represent a value as an unevaluated pair $(hi, lo)$ with $hi = \mathrm{fl}(hi + lo)$. Addition
uses `two_sum` to capture the rounding, multiplication uses `two_product` (or an FMA). This
gives roughly 32 decimal digits at about 20 times the cost.

Applied to the expanded $(x-1)^6$ near $x = 1$, double-double recovers the correct curve down to
about $|x - 1| \approx 10^{-5}$, pushing the failure point out by roughly a factor of $10^{8}$
in the value. It does not remove the problem, it moves it, which is the point of exercise 1.3
above.

### 5.3 Parallel reduction

Splitting into $p$ chunks and combining changes the association order, so the answer changes.
The variation is typically of order $u\sqrt{n}$ and does not shrink as $p$ grows. This is the
mechanism behind non-deterministic training results in deep learning, where the number of
threads or the GPU scheduling changes the reduction tree from run to run. Lesson 93 returns to
it.

---

## Lesson 06, Conditioning and Stability

### 1.1 The four definitions

- **Forward error**: how far the computed answer is from the true one.
- **Backward error**: the smallest change to the input for which the computed answer is exactly
  right.
- **Conditioning**: how much the problem amplifies a change in its input. A property of the
  **problem**.
- **Stability**: how much error the algorithm adds beyond what the problem forces. A property of
  the **algorithm**.

Conditioning and, arguably, forward error belong to the problem. Backward error and stability
describe the algorithm.

### 1.2 Diagnosing the given case

Forward error $10^{-3}$, backward error $10^{-16}$. The backward error is at the level of unit
roundoff, so the algorithm is **backward stable and blameless**. The implied condition number is
$10^{-3}/10^{-16} = 10^{13}$, so the problem is severely ill-conditioned.

What to do: do not touch the algorithm. Either reformulate the problem, work in higher
precision, or accept and clearly report the three correct digits.

### 1.3 Stable but wrong

Backward stability guarantees the answer is exact for a nearby problem. If the problem amplifies
input changes by $\kappa$, then "nearby problem" and "nearby answer" are very different things.
With $\kappa = 10^{13}$ a perturbation of $10^{-16}$ in the data moves the true answer by
$10^{-3}$, and the algorithm cannot do better than that.

### 2.1 Deriving $\kappa = |xf'/f|$

For small $\Delta x$, $f(x + \Delta x) - f(x) \approx f'(x)\Delta x$. So

$$\frac{|f(x+\Delta x)-f(x)|/|f(x)|}{|\Delta x|/|x|}
\approx \frac{|f'(x)\Delta x| \cdot |x|}{|f(x)| \cdot |\Delta x|}
= \left|\frac{x f'(x)}{f(x)}\right|.$$

The supremum over directions is attained immediately in one dimension, and the limit removes the
higher order terms.

### 2.2 Multiple roots move like $\delta^{1/m}$

Near a root $r$ of multiplicity $m$, $p(x) \approx c(x-r)^m$. Perturbing $p$ by $\delta$ gives
$c(x-r)^m = \delta$, so $x - r = (\delta/c)^{1/m}$.

For $m = 5$ and $\delta = 10^{-10}$: $(10^{-10})^{1/5} = 10^{-2}$, matching the measured
$1.000 \times 10^{-2}$ in the lesson's table.

### 2.3 The governing inequality

Let $\hat{y} = f(x + \Delta x)$ with relative backward error
$\eta = |\Delta x|/|x|$. By the definition of $\kappa$,

$$\frac{|\hat{y} - f(x)|}{|f(x)|} = \frac{|f(x+\Delta x) - f(x)|}{|f(x)|}
\le \kappa(x) \cdot \frac{|\Delta x|}{|x|} + O(\eta^2) = \kappa \eta + O(\eta^2).$$

### 2.4 The recurrence amplification

If $\hat{I}_0 = I_0 + \epsilon$, then $\hat{I}_1 = 1/1 - 5(I_0 + \epsilon) = I_1 - 5\epsilon$.
By induction $\hat{I}_n = I_n + (-5)^n \epsilon$, so the error is multiplied by exactly $5^n$ in
magnitude. Reversing the recurrence gives $\hat I_{n-1} = I_{n-1} - \epsilon/5$, so the error is
divided by 5 each step.

### 3.1 A diagnostic function

This became `nalib.errors.diagnose_root` in lesson 12, and its structure is the procedure of
section 9 turned into code:

```python
def diagnose(f, x_hat, kappa):
    residual = abs(f(x_hat))
    bound = kappa * residual
    u = np.finfo(float).eps / 2
    if bound <= 10 * u * max(1.0, abs(x_hat)):
        verdict = "as accurate as double precision allows"
    elif residual > 100 * u:
        verdict = "large residual: the ALGORITHM is at fault"
    else:
        verdict = f"small residual but kappa = {kappa:.1e}: the PROBLEM is hard"
    return {"residual": residual, "bound": bound, "verdict": verdict}
```

The essential design point is that it reports the **product**, not the residual alone. A
function returning only the residual invites the reader to stop there, which is the mistake the
whole lesson exists to prevent.

### 3.2 First violation of the bound

Running the forward recurrence and testing against $0 < I_n \le 1/(5(n+1))$ flags the first
violation at $n = 21$, where the computed value is $2.641 \times 10^{-2}$ against an upper bound
of $9.09 \times 10^{-3}$.

### 3.3 $\kappa$ for $e^x$

$\kappa_f(x) = |xf'(x)/f(x)| = |x|$ for the exponential, so:

| $x$ | $\kappa$ | digits lost |
|---|---|---|
| 1 | 1 | 0 |
| 10 | 10 | 1 |
| 50 | 50 | 1.7 |

Verifying by perturbation: multiply $x$ by $(1 + 10^{-12})$ and measure the relative change in
$e^x$. At $x = 50$ the output changes by about $5\times10^{-11}$, which is $50$ times the input
perturbation, exactly as $\kappa = 50$ predicts.

**The result is mild and that is worth noticing.** $e^{50}$ is about $5\times10^{21}$, an
enormous number, and yet computing it is only slightly ill conditioned. Large outputs and large
condition numbers are unrelated.

### 4.1 Root spread against multiplicity

The computed roots of the expanded $(x - c)^m$ scatter in a circle of radius about $u^{1/m}$
around $c$. So

| $m$ | predicted spread $u^{1/m}$ |
|---|---|
| 2 | $10^{-8}$ |
| 3 | $5 \times 10^{-6}$ |
| 5 | $6 \times 10^{-4}$ |
| 8 | $1 \times 10^{-2}$ |

Plotting the measured forward error against $m$ on a log scale gives a straight line, confirming
the $u^{1/m}$ law.

### 4.2 Root spread against multiplicity

The circular pattern persists, and the **radius** follows lesson 12's rule: for a root of
multiplicity $m$, a perturbation $\delta$ moves the roots by about $|\delta|^{1/m}$, and the $m$
roots spread out into a circle of that radius, equally spaced in angle.

Measured for $(x-2)^m$ with $m = 2 \dots 6$: the computed roots lie on a circle of radius
approximately $u^{1/m}$ around 2, with $m$-fold symmetry. At $m = 5$ the radius is about
$10^{-3}$, which is what lesson 06 section 5 reported.

**Why a circle.** The perturbed equation is locally $c(x-r)^m = -\delta$, whose solutions are
$r + (|\delta|/|c|)^{1/m}$ times the $m$-th roots of unity. Those are equally spaced on a
circle by definition.

### 4.3 Verifying the rule of thumb at $\kappa \approx 10^8$

Construct such a problem by taking $f(x) = x^2 - a$ near a root where $f'$ is small, or more
directly by building a matrix with a prescribed condition number as lesson 19 does.

The rule predicts $16 - 8 = 8$ correct digits from a backward stable algorithm. Measured across
many instances, the observed count is 8 to 9, occasionally 10.

**The rule is a floor rather than a prediction**, and it errs on the pessimistic side, because
$\kappa$ is a worst case over perturbation directions and a particular problem rarely excites
the worst one. Lesson 19 section 4 measures that gap directly.

### 5.1 Backward stability of Horner

See lesson 01, exercise 5.2.

### 5.2 Mixed stability

An algorithm is **mixed forward-backward stable** if
$\hat{f}(x) + \Delta y = f(x + \Delta x)$ with both $|\Delta y|/|f(x)|$ and $|\Delta x|/|x|$ of
order $u$.

A standard example is computing $\sin x$ for very large $x$. Argument reduction cannot be
backward stable, because a small relative change in a large $x$ moves it by many multiples of
$\pi$. Good implementations are mixed stable instead.

### 5.3 Is conditioning really intrinsic?

The honest answer is that conditioning is intrinsic to the **problem as posed**, which includes
the representation of the data.

"Evaluate the polynomial with these seven monomial coefficients at $x = 1$" is ill-conditioned.
"Evaluate $(x-1)^6$ at $x=1$" is perfectly conditioned. These are different problems, because
the input data is different, even though the mathematical function is the same.

This is not a loophole. It is one of the most useful facts in the subject, and it is why so much
of numerical analysis is about choosing the right representation: Chebyshev bases instead of
monomials, orthogonal factorizations instead of normal equations, and $QR$ instead of the
inverse.

---

## Lesson 07, Taylor Series and Convergence Rates

### 1.1 Iterations needed

Linear with $C = 0.5$: each step gains $\log_{10} 2 \approx 0.301$ digits, so 10 digits needs
about $10/0.301 \approx 34$ iterations.

Quadratic from 1 correct digit: digits go $1 \to 2 \to 4 \to 8 \to 16$, so **4** iterations
reaches 16 digits and 10 digits arrives on the fourth.

### 1.2 Big-O against little-o

$O(h^2)$ means bounded by $Ch^2$. $o(h^2)$ means the ratio to $h^2$ tends to zero.

$g(h) = 3h^2$ is $O(h^2)$ but **not** $o(h^2)$, since $g/h^2 = 3$.
$g(h) = h^3$ is both, since $h^3/h^2 = h \to 0$.

### 1.3 Order estimates at the roundoff floor

Once $e_k$ is at roundoff level it is no longer the true error, it is noise. The ratios
$e_{k+1}/e_k$ then fluctuate around 1 with no pattern, so the logarithms in the order formula
approach zero and the quotient becomes numerically meaningless, or NaN when the ratio is exactly
1.

### 2.1 Central difference error term

Expand both ways about $x$:

$$f(x\pm h) = f(x) \pm hf'(x) + \tfrac{h^2}{2}f''(x) \pm \tfrac{h^3}{6}f'''(x) + O(h^4).$$

Subtracting kills the even terms:

$$f(x+h) - f(x-h) = 2hf'(x) + \tfrac{h^3}{3}f'''(x) + O(h^5),$$

so dividing by $2h$ gives $f'(x) + \tfrac{h^2}{6}f'''(\xi)$. The error term is
$-\tfrac{h^2}{6}f'''(\xi)$ with the sign convention error = exact minus approximation.

### 2.2 Newton converges quadratically

Let $e_k = x_k - r$ with $f(r) = 0$ and $f'(r) \ne 0$. Expanding $f$ about $x_k$ and using the
Newton step,

$$e_{k+1} = e_k - \frac{f(x_k)}{f'(x_k)} = \frac{f''(\xi_k)}{2f'(x_k)}e_k^2.$$

Taking the limit, $|e_{k+1}|/|e_k|^2 \to |f''(r)/(2f'(r))|$.

For $f = x^2 - 2$ at $r = \sqrt2$: $f'' = 2$, $f' = 2\sqrt2$, so
$C = 2/(4\sqrt2) = 1/(2\sqrt2) \approx 0.354$. Check it against
`nalib.convergence.asymptotic_constant(errs, 2.0)`.

### 2.3 The secant order

Assume $e_{k+1} = Ce_k^p$ and use $e_{k+1} \approx Ke_ke_{k-1}$. Substituting
$e_k = Ce_{k-1}^p$ into both sides,

$$Ce_k^p = C(Ce_{k-1}^p)^p = C^{1+p}e_{k-1}^{p^2}, \qquad
Ke_ke_{k-1} = KCe_{k-1}^{p+1}.$$

Matching exponents gives $p^2 = p + 1$, so $p = (1+\sqrt5)/2 \approx 1.618$, the golden ratio.

### 2.4 Repeated Rolle

Let $g = f - p_n$, which vanishes at the $n+1$ interpolation nodes. Rolle applied between each
adjacent pair of zeros gives $n$ zeros of $g'$. Applying it again gives $n-1$ zeros of $g''$, and
so on. After $n$ applications, $g^{(n)}$ has at least one zero. Adding the extra node used in the
error formula argument raises the count by one, giving a zero of $g^{(n+1)}$, which is the
point $\xi$ in the interpolation error formula of lesson 45.

### 3.1 `observed_order` from scratch

From $e_{k+1} \approx Ce_k^p$, taking logs of two consecutive ratios eliminates $C$:

```python
def observed_order_scratch(errors):
    e = np.asarray(errors, dtype=float)
    e = e[e > 0]
    out = []
    for k in range(1, len(e) - 1):
        num = np.log(e[k+1] / e[k])
        den = np.log(e[k] / e[k-1])
        out.append(num / den if den != 0 else np.nan)
    return np.array(out)
```

**Handling the roundoff floor correctly is most of the work.** Once the errors reach machine
precision they stop shrinking and start behaving randomly, so the ratios become meaningless and
the estimated order can come out as anything at all.

The fix used in `nalib.convergence.reliable_order` is to truncate the sequence at the first
error that fails to improve on its predecessor by a clear factor, then estimate from what
remains. Without that truncation the table in section 5 is unreproducible.

### 3.2 A refinement study helper

```python
def refinement_study(method, hs, exact):
    """Run a method at several step sizes, fit the order, and say if the fit is poor."""
    hs = np.asarray(hs, dtype=float)
    errs = np.array([abs(method(h) - exact) for h in hs])
    good = errs > 0
    slope, _ = np.polyfit(np.log(hs[good]), np.log(errs[good]), 1)
    fit = np.polyval(np.polyfit(np.log(hs[good]), np.log(errs[good]), 1),
                     np.log(hs[good]))
    residual = np.max(np.abs(fit - np.log(errs[good])))
    return {"order": slope, "errors": errs,
            "warning": "poor fit, the asymptotic regime may not be reached"
                       if residual > 0.5 else ""}
```

The warning matters. A fitted order is meaningless if the data is not in the asymptotic regime,
and the two ways that happens are $h$ too large (higher order terms still matter) and $h$ too
small (roundoff dominates). Reporting the fit quality is what distinguishes a measurement from
a number.

### 3.3 The Lagrange remainder for $\sin$

$$\left|\sin x - \sum_{k=0}^{n}\frac{(-1)^k x^{2k+1}}{(2k+1)!}\right|
\le \frac{|x|^{2n+3}}{(2n+3)!},$$

because every derivative of $\sin$ is bounded by 1.

Measured at several orders and several $x$: the bound holds in **every** case, and it is
typically within a factor of 2 of the true error for small $x$, becoming looser for larger $x$.

The bound is tight because the neglected term dominates the remainder when $x$ is small, so the
first omitted term is essentially the whole error. That is the same reason the alternating
series bound works, and it is why "the error is about the size of the first neglected term" is
usually right.

### 4.1 Newton from a distant start

From $x_0 = 10^6$, the first several steps roughly **halve** $x$ each time, because
$x - (x^2-2)/(2x) \approx x/2$ when $x$ is large. So it takes about $\log_2(10^6/\sqrt2)
\approx 19$ steps just to get near the root, and only then does the quadratic phase begin,
finishing in 4 or 5 more.

This is the standard picture: Newton has a slow global phase and a fast local phase. Hybrid
methods like Brent's exist precisely to shorten the global phase.

### 4.2 Newton at a triple root

At a root of multiplicity $m$, Newton converges **linearly** with rate $C = 1 - 1/m$. For
$m = 3$ that is $C = 2/3$, and the measured order is 1, not 2.

The modified step $x - m f/f'$ restores quadratic convergence. Measuring the order of
$x - 3f/f'$ on $(x-1)^3$ gives 2 again.

### 4.3 The series for $e^{-10}$

This is a lesson 05 problem, not a lesson 07 one. The terms reach a magnitude of about
$10^{10}/10! \approx 2755$, while the answer is $4.5 \times 10^{-5}$. Eight digits cancel, so
the direct sum is inaccurate no matter how many terms you take. Adding more terms cannot help,
because the error is cancellation, not truncation.

### 5.1 Steffensen's method

Steffensen applies Aitken acceleration to a fixed point iteration:

$$x_{k+1} = x_k - \frac{g(x_k)^2}{g(x_k + g(x_k)) - g(x_k)}, \qquad g(x) = f(x).$$

The inner difference approximates $f'$ using function values only, so the method achieves
**quadratic** convergence with no derivative. The cost is two function evaluations per step
rather than one.

### 5.2 Aitken extrapolation on the section 5 sequences

Aitken's $\Delta^2$ turns a linearly convergent sequence into a faster one by eliminating the
dominant geometric term. Applied to the two sequences from section 5:

**Fixed point iteration** ($\rho \approx 0.68$) accelerates well. The extrapolated sequence
converges noticeably faster, and the measured order rises above 1.

**Bisection does not accelerate.** The reason is structural: Aitken assumes $e_{k+1} \approx
Ce_k$ with a **fixed** $C$, and bisection's error is not geometric at all. Its bracket halves
geometrically but the error within the bracket jumps around, so the model Aitken is built on
does not fit.

**That contrast is the point of the exercise.** Acceleration is not free, it is a model, and
applying it to a sequence that does not match the model produces nothing useful. Lesson 10
develops Aitken and Steffensen properly, including where the cancellation limit bites.

### 5.3 Richardson raises the observed order

With $E(h) = C_1h^2 + C_2h^3$, the measured order at step size $h$ is

$$\log_2\frac{E(h)}{E(h/2)} = 2 + \frac{C_2 h}{C_1}\log_2 e + O(h^2),$$

so it approaches 2 from above as $h$ shrinks. Richardson extrapolation forms
$\frac{4E(h/2) - E(h)}{3}$, which cancels the $h^2$ term exactly and leaves an $h^3$ error, so
the extrapolated sequence measures order 3. Lesson 62 develops this into Romberg integration.

---

## Lesson 08, Algorithm Cost and Complexity

### 1.1 Why matrix products beat matrix vector products

A matrix product does $O(n^3)$ arithmetic on $O(n^2)$ data, so each value loaded into cache is
used $O(n)$ times. A matrix vector product does $O(n^2)$ arithmetic on $O(n^2)$ data, so each
value is used once and the processor waits on memory. The arithmetic units are identical, and
the memory system is the difference.

### 1.2 When $O(n^2)$ beats $O(n\log n)$

When the constants differ enough and $n$ is small enough. Strassen's matrix multiplication is
asymptotically faster than the standard method but loses below roughly $n = 1000$ on real
hardware, and is also less numerically stable. Similarly, insertion sort beats quicksort for
small arrays, which is why real sort implementations switch to it below a threshold.

### 1.3 Largest matrix in 16 GB

$8n^2 = 16 \times 10^9$ gives $n \approx 44721$. An LU factorization would cost
$\tfrac23 n^3 \approx 6 \times 10^{13}$ flops, which at 100 Gflop/s is about **10 minutes**.
Note that memory ran out at roughly the same scale as the time became inconvenient, which is
typical.

### 2.1 Gaussian elimination flop count

At elimination step $k$ there are $n-k$ rows below the pivot and $n-k+1$ entries to update in
each, giving $(n-k)(n-k+1)$ multiply-add pairs, so $2(n-k)(n-k+1)$ flops, plus $n-k$ divisions
for the multipliers. Summing over $k = 1$ to $n-1$ and using
$\sum k^2 = n(n+1)(2n+1)/6$ gives

$$\frac{2n^3}{3} + O(n^2)\ \text{flops}.$$

### 2.2 Back substitution

Row $i$ needs $n - i$ multiplications, $n - i$ subtractions and one division. Summing,
$2\sum_{i=1}^{n}(n-i) + n = n(n-1) + n = n^2$ flops.

It is negligible because $n^2 / (\tfrac23 n^3) = 3/(2n)$, which is under 1 percent by $n = 150$.
This is why factorizing once and solving many right hand sides is so much cheaper than solving
from scratch each time.

### 2.3 The FFT recurrence

By the master theorem with $a = 2$, $b = 2$, $f(n) = O(n)$: since $n^{\log_b a} = n^1$ matches
$f(n)$, the solution is $T(n) = O(n\log n)$. Directly: the recursion tree has $\log_2 n$ levels
and does $O(n)$ work at each level.

### 2.4 Repeated squaring for $n = 1000$

$1000 = 1111101000_2$, which has 10 bits and 6 of them set. That is 9 squarings plus 5
multiplications to combine, so **14** multiplications against 999.

### 3.1 Pure Python matrix multiplication

A triple loop in Python measures at roughly **100 to 1000 times slower** than `A @ B`, and the
achieved rate is a few MFlop/s against tens of GFlop/s.

The measured exponent is close to 3, sometimes slightly above, because pure Python has no cache
blocking and no vectorisation, so the memory behaviour degrades rather than improves with size.

The gap splits roughly into: interpreter overhead per operation (the largest share), no SIMD
vectorisation, no cache blocking, and no multithreading. **None of it is algorithmic**: the flop
counts are identical.

### 3.2 Counting flops in back substitution

```python
from nalib.cost import OpCounter

counter = OpCounter()
# run back substitution with counted arithmetic
```

Row $i$ needs $i$ multiplications, $i$ subtractions and 1 division, so the total is

$$\sum_{i=0}^{n-1}(2i + 1) = n^2$$

exactly. The measured count matches $n^2$ at every $n$, which confirms both the implementation
and the formula. Lesson 17 uses this count directly.

**The value of counting rather than deriving** is that it catches off-by-one errors in the
formula, which lesson 01 found in `flop_count_naive`: the naive polynomial evaluator does $n+1$
additions, not $n$, and only the measurement revealed it.

### 3.3 Blocked matrix multiplication

The time against block size $b$ is U shaped. Small $b$ means too much loop overhead and poor
vectorisation. Large $b$ means the three $b \times b$ tiles no longer fit in cache together, so
the reuse advantage disappears. The optimum is near $3b^2 \times 8 \text{ bytes} \approx$ the L2
cache size, so for a 1 MB L2 cache that is $b \approx 200$.

### 4.1 Exponents for solve, inv and lu_factor

All three are $O(n^3)$ in arithmetic, and all three will measure an exponent below 3 for the
reason given in lesson 08 section 5: the achieved Gflop/s rises with $n$.

`inv` costs about twice `solve`, since it is effectively $n$ solves sharing one factorization.
`lu_factor` alone is cheapest, since it omits the triangular solves.

### 4.2 Bandwidth rather than flop rate

A matrix-vector product reads $n^2$ matrix entries and does $2n^2$ flops, so it performs **2
flops per 8 bytes**, an arithmetic intensity of 0.25. It is entirely memory bound.

Plotting achieved GB/s rather than GFlop/s shows the product running at close to the machine's
**memory bandwidth specification**, typically 10 to 50 GB/s, and essentially flat in $n$ once
the matrix exceeds cache.

That flat line is the correct picture: the operation is not limited by arithmetic at all, so
measuring it in flop/s describes the wrong resource. Matrix-matrix multiplication has
intensity $O(n)$ and is compute bound, which is why it reaches a large fraction of peak.

### 4.3 The FFT crossover

The flop counts cross at small $n$: $5n\log_2 n < 8n^2$ from about $n = 8$. The **measured**
crossover is much larger, typically $n$ in the hundreds.

The reason is the same one as everywhere in lesson 08: the direct DFT is a dense matrix-vector
product, which reaches a high fraction of peak, while the FFT has irregular memory access and
achieves a much lower rate. The FFT wins eventually because $n\log n$ beats $n^2$ decisively,
but not at the point the flop counts suggest.

**Flop counts predict the asymptotic winner and not the crossover.** That is the general lesson
and it applies equally to Strassen (lesson 17 exercise 5.1) and to sparse against dense solvers
(lesson 21 exercise 4.2).

### 5.1 Strassen's algorithm

Strassen replaces 8 recursive multiplications of half-size blocks with 7, at the cost of extra
additions, giving $T(n) = 7T(n/2) + O(n^2)$ and hence $O(n^{\log_2 7}) = O(n^{2.807})$.

Libraries mostly avoid it for three reasons: the crossover is high, typically above $n = 1000$;
it needs substantial extra workspace; and it is less numerically stable, satisfying only a norm
wise error bound rather than the componentwise bound that the standard algorithm gives.

### 5.2 The roofline model

Arithmetic intensity, in flops per byte, for the three BLAS levels:

| Level | Operation | Flops | Bytes moved | Intensity |
|---|---|---|---|---|
| 1 | $\mathbf{y} \leftarrow \alpha\mathbf{x} + \mathbf{y}$ | $2n$ | $24n$ | **0.08** |
| 2 | $\mathbf{y} \leftarrow A\mathbf{x}$ | $2n^2$ | $8n^2$ | **0.25** |
| 3 | $C \leftarrow AB$ | $2n^3$ | $24n^2$ | **$n/12$** |

The roofline is $\min(\text{peak flops}, \text{bandwidth} \times \text{intensity})$. With, say,
50 GFlop/s peak and 20 GB/s bandwidth, the ridge point is at intensity 2.5.

**Levels 1 and 2 are far to the left of the ridge**, so they are bandwidth bound and can never
exceed a small fraction of peak no matter how the code is written. **Level 3 crosses the ridge**
once $n > 30$, so it is compute bound and can approach peak.

That single picture explains everything lesson 08 measured, and it explains why numerical linear
algebra is organised around expressing algorithms in terms of level 3 operations: blocked LU
(lesson 17), blocked Cholesky (lesson 20), and block Krylov methods.

### 5.3 Conjugate gradient on the 2D Poisson problem

On an $N \times N$ grid there are $n = N^2$ unknowns and $\kappa = O(N^2) = O(n)$. CG needs
$O(\sqrt\kappa) = O(N)$ iterations, each costing $O(n) = O(N^2)$, so the total is $O(N^3)$.

Comparing the three approaches on the same problem:

| Method | Total cost | Memory |
|---|---|---|
| Dense LU | $O(N^6)$ | $O(N^4)$ |
| Sparse direct (nested dissection) | $O(N^3)$ | $O(N^2\log N)$ |
| Conjugate gradient | $O(N^3)$ | $O(N^2)$ |
| Preconditioned CG | $O(N^{2.5})$ | $O(N^2)$ |
| **Multigrid** | $O(N^2) = O(n)$ | $O(N^2)$ |

Multigrid is **optimal**: the cost is proportional to the number of unknowns, and the iteration
count does not grow with the grid at all. That is the calculation that motivates lesson 27.
