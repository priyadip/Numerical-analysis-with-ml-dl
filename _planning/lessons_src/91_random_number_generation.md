# 91. Random Number Generation

**Part 13: Stochastic and Monte Carlo Methods**

## Learning objectives

By the end of this lesson you will be able to:

1. Write down a linear congruential generator and say when its period is full.
2. Explain the lattice structure every LCG has, and find the planes for a given one.
3. Turn uniform numbers into any distribution whose inverse is available, and into normals when it
   is not.
4. Apply tests that separate a good generator from a bad one, and say why the obvious tests do not.
5. Say what a modern generator gives you that an LCG does not.

## Prerequisites

Lesson 03 (floating point, because every generator produces integers and the conversion to a real
number in $[0,1)$ is where precision enters). Lesson 09 (fixed point iteration, which is what a
generator is). Nothing else; this lesson starts Part 13.

---

## 1. The recurrence

A random number generator is a deterministic recurrence chosen so that its output is hard to
distinguish from random. The oldest and simplest is the **linear congruential generator**,

$$
x_{k+1} = (a\,x_k + c) \bmod m ,
$$

with the output taken as $x_k/m$. Three integers define it: the multiplier $a$, the increment $c$
and the modulus $m$. Almost every system library used one of these from 1960 to 1995.

Two things are immediately true. The state is an integer below $m$, so the sequence must repeat
after at most $m$ steps. And once it repeats it repeats forever, because the next value depends only
on the current one.

```python
from nalib import rng as rg

print("the first ten values of each named generator, as integers:")
for name in sorted(rg.FAMOUS):
    a, c, m = rg.FAMOUS[name]
    values = rg.lcg(6, a, c, m, seed=1)
    print(f"  {name:>18}  a = {a:>10}  c = {c:>10}  m = {m:>10}")
    print(f"  {'':>18}  {values}")
```

The numbers look unremarkable, which is the difficulty. A bad generator does not produce obviously
patterned output; it produces output that fails a test you did not think to run.

## 2. The period

The period is at most $m$, and reaching $m$ needs conditions on $a$ and $c$. Hull and Dobell proved
they are exactly three:

- $c$ and $m$ are coprime,
- $a - 1$ is divisible by every prime factor of $m$,
- $a - 1$ is divisible by $4$ if $m$ is.

This is an equivalence, not a sufficient condition, so a single counterexample anywhere would
disprove it. At a small modulus every case can be checked.

```python
from nalib import rng as rg

out = rg.the_hull_dobell_conditions_are_exact()
print(f"modulus {out['modulus']}, {out['combinations']} combinations of a and c")
print(f"  agreements between the theorem and the measured period: {out['agreements']}")
print(f"  disagreements: {len(out['disagreements'])}")
print(f"  combinations with full period: {out['full_period_count']}")
print(f"  the conditions are exact: {out['the_conditions_are_exact']}")

assert out["the_conditions_are_exact"]
assert out["full_period_count"] > 0

print("\na few periods, to see the range:")
for a, c in ((1, 1), (5, 1), (5, 3), (13, 7), (13, 8), (64, 1)):
    print(f"  a = {a:>3}, c = {c:>2}: period {rg.period(a, c, 64):>3}, "
          f"full predicted {rg.hull_dobell(a, c, 64)}")
```

Every one of the $4096$ combinations agrees with the theorem.

A full period is necessary and nowhere near sufficient. RANDU has a period of $2^{29}$, which was
ample for the machines of its day, and it is the worst generator in wide use ever shipped.

## 3. The lattice, and what killed RANDU

Take the output $d$ at a time and plot the points. Every LCG's tuples lie on a family of parallel
hyperplanes, and the question is how many. Few planes means large gaps between them, and a
simulation that samples a $d$ dimensional region will systematically miss the gaps.

For RANDU, $a = 65539 = 2^{16} + 3$, and

$$
a^{2} = 2^{32} + 6\cdot 2^{16} + 9 \equiv 6\cdot 2^{16} + 9 = 6a - 9 \pmod{2^{31}} ,
$$

so every triple satisfies the exact integer identity

$$
x_{k+2} - 6\,x_{k+1} + 9\,x_k \equiv 0 \pmod{2^{31}} .
$$

The plane normal is $(9, -6, 1)$. The measurement below does not assume it: it searches for the
shortest integer relation the recurrence satisfies, which is what the spectral test does, and
reports what it finds.

```python
from nalib import rng as rg

out = rg.randu_lies_on_a_few_planes()
print(f"the shortest relation found for RANDU: {out['normal']}, length "
      f"{out['relation_length']:.3f}")
print(f"  it is an exact identity: {out['relation_is_exact']}, "
      f"worst residual {out['worst_residual']}")
print(f"  planes counted from the sample: {out['planes']}")
print(f"  PCG64 triples on those same planes: "
      f"{out['pcg64_triples_on_the_same_planes']} of "
      f"{out['pcg64_triples_tested']}")

print(f"\n{'generator':>20}{'shortest relation':>22}{'planes':>9}")
for row in out["rows"]:
    found = str(row["normal"]) if row["normal"] else "none in the box"
    print(f"{row['generator']:>20}{found:>22}"
          f"{(row['planes'] if row['planes'] else '-'):>9}")

assert out["planes"] == 15
assert out["worst_residual"] == 0
assert out["only_randu_has_a_short_relation"]
```

**Fifteen planes.** Three dimensional Monte Carlo done with RANDU samples fifteen sheets of a cube
and nothing between them, and a great deal of published work from the 1970s did exactly that.

The same bounded search finds no short relation for any of the other four generators, and none of
the thirty thousand PCG64 triples lies on RANDU's planes. So the search is finding a real property
of RANDU rather than reporting the same answer for everything.

## 4. The bits are not equally good

With a power of two modulus, the arithmetic mod $2^{j+1}$ closes on itself, so bit $j$ counted from
the bottom depends only on the bottom $j+1$ bits of the state. Its period is therefore at most
$2^{j+1}$, and it reaches that bound.

```python
from nalib import rng as rg

out = rg.the_low_bits_are_worse()
print(f"modulus {out['modulus']}, whose full period is {out['modulus']}")
print(f"{'bit':>5}{'predicted period':>19}{'measured period':>18}{'matches':>10}")
for row in out["rows"]:
    print(f"{row['bit']:>5}{row['predicted_period']:>19}"
          f"{row['measured_period']:>18}{str(row['matches']):>10}")

assert out["every_bit_matches"]
```

The bottom bit simply alternates. That is the reason `rand() % 6` was the wrong way to roll a die
for thirty years: a remainder reads the bottom bits, which have a period of $8$ rather than $2^{20}$,
so the die produces a repeating cycle of eight faces.

The repair is to use the **top** bits, which is what dividing by $m$ does, or to use a generator with
no such structure.

## 5. Testing a generator

The obvious tests are uniformity and correlation. Run both on every generator here.

```python
from nalib import rng as rg

out = rg.the_named_generators_compared()
print(f"{'generator':>20}{'chi-square / dof':>19}{'uniform':>10}"
      f"{'lag 1 correlation':>20}{'lag 2':>13}")
for row in out["rows"]:
    print(f"{row['generator']:>20}{row['chi_square_ratio']:>19.4f}"
          f"{str(row['passes_uniformity']):>10}"
          f"{row['lag_one_correlation']:>20.6f}{row['lag_two_correlation']:>13.6f}")

print(f"\nevery generator passes uniformity: "
      f"{out['every_generator_passes_uniformity']}")
print(f"worst LCG lag one correlation: {out['worst_lcg_correlation']:.6f}")
print(f"PCG64 lag one correlation:     {out['pcg64_correlation']:.6f}")

assert out["every_generator_passes_uniformity"]
```

**Every generator passes, including RANDU.** Its lag one correlation is $0.00049$ against PCG64's
$0.0016$, so on this test RANDU looks *better* than the modern generator. Both numbers are sampling
noise.

That is why RANDU survived a decade. The tests people ran were these, and these tests cannot see the
failure. Finding it needs a test in more than one dimension.

```python
from nalib import rng as rg

out = rg.one_test_is_not_enough()
print(f"{out['draws']} draws from RANDU")
print(f"  one dimension, 100 bins:      statistic {out['flat_statistic']:>10.2f}, "
      f"ratio {out['flat_ratio']:.4f}, passes {out['flat_passes']}")
print(f"  three dimensions, 1000 cells: statistic {out['cube_statistic']:>10.2f}, "
      f"ratio {out['cube_ratio']:.4f}")
print(f"  the same cube test on PCG64:  ratio {out['pcg64_cube_ratio']:.4f}")
print(f"\nRANDU fails in three dimensions: {out['randu_fails_in_three_dimensions']}")
print(f"PCG64 passes:                   {out['pcg64_passes_in_three_dimensions']}")

assert out["randu_fails_in_three_dimensions"]
assert out["pcg64_passes_in_three_dimensions"]
```

In one dimension RANDU's ratio is $0.907$, comfortably inside the pass band. In three dimensions it
is $2.256$ where PCG64 is $1.031$. The failure was always there; the one dimensional test could not
reach it.

Modern test batteries (Diehard, TestU01's BigCrush) are collections of a hundred or so tests of this
kind, precisely because no small set is enough.

## 6. From uniform to anything else

A generator produces uniform numbers on $[0,1)$. Every other distribution is built from them.

**Inverse transform.** If $F$ is a distribution function and $u$ is uniform, then $F^{-1}(u)$ has
distribution $F$. The proof is one line: $\Pr[F^{-1}(u) \le t] = \Pr[u \le F(t)] = F(t)$. For the
exponential, $F(t) = 1 - e^{-\lambda t}$ inverts to $-\log(1-u)/\lambda$.

**Box-Muller.** The normal has no closed form inverse, so a different identity is used. Two
independent uniforms give

$$
z_1 = \sqrt{-2\log u_1}\,\cos(2\pi u_2) , \qquad
z_2 = \sqrt{-2\log u_1}\,\sin(2\pi u_2) ,
$$

two independent standard normals. It is exact, and it costs a logarithm, a square root and two
trigonometric calls per pair.

```python
from nalib import rng as rg

out = rg.the_transforms_are_exact()
print(f"{out['draws']} samples through each transform")
print(f"{'transform':>26}{'KS gap':>13}{'gap times sqrt(n)':>20}{'passes':>9}")
print(f"{'inverse transform, exp':>26}{out['exponential_gap']:>13.3e}"
      f"{out['exponential_scaled']:>20.4f}{str(out['exponential_passes']):>9}")
print(f"{'Box-Muller, normal':>26}{out['normal_gap']:>13.3e}"
      f"{out['normal_scaled']:>20.4f}{str(out['normal_passes']):>9}")
print(f"\nthe normals have mean {out['normal_mean']:+.6f} and variance "
      f"{out['normal_variance']:.6f}")

assert out["both_pass"]
```

The Kolmogorov-Smirnov statistic is the largest gap between the empirical distribution and the true
one. Scaled by $\sqrt n$ it has a fixed distribution, so $0.42$ and $0.73$ at a million samples are
ordinary values and anything above $1.63$ would be a one per cent event. **Both transforms are
exact**, and the only thing the test can find is sampling error.

## 7. Rejection, and what it costs

Marsaglia's polar method replaces the trigonometry with a rejection loop: draw a point in the square
$[-1,1]^2$, keep it if it lands inside the unit disc, and build the normals from its radius. The
acceptance probability is the ratio of the two areas,

$$
\frac{\pi}{4} = 0.785398\ldots ,
$$

so a fifth of the draws are thrown away and two trigonometric calls are saved.

```python
from nalib import rng as rg

out = rg.the_polar_method_wastes_a_known_fraction()
print(f"{out['draws']} normals by the polar method")
print(f"  measured acceptance:  {out['acceptance']:.6f}")
print(f"  predicted, pi / 4:    {out['predicted_acceptance']:.6f}")
print(f"  ratio:                {out['ratio']:.6f}")
print(f"  and the output is normal: {out['passes']} "
      f"(scaled KS gap {out['scaled_gap']:.4f})")

assert out["matches_pi_over_four"]
assert out["passes"]
```

The measured acceptance is $0.785352$ against $0.785398$, agreeing to five digits. Whether the trade
is worth it depends entirely on the machine: on hardware with fast trigonometry Box-Muller wins, and
on hardware without it the polar method does. Neither is more accurate.

## 8. Seeing the planes

The failure is visible. Take the RANDU triples that lie near one plane of the family and look at the
cube edge on, along the normal; the points collapse onto a small number of lines. The same view of a
modern generator shows nothing.

```python
from nalib import rng as rg

fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4.0))

detail = rg.planes_in_three_dimensions("randu", draws=8000)
raw = rg.lcg(8000, *rg.FAMOUS["randu"], seed=1)
triples = np.column_stack([raw[:-2], raw[1:-1], raw[2:]]) / rg.FAMOUS["randu"][2]
normal = np.asarray(detail["normal"], dtype=float)
normal = normal / np.linalg.norm(normal)
# an orthonormal frame whose third axis is the plane normal
helper = np.array([1.0, 0.0, 0.0])
first = helper - (helper @ normal) * normal
first = first / np.linalg.norm(first)
second = np.cross(normal, first)
left.plot(triples @ first, triples @ normal, ".", ms=1.0, alpha=0.5)
left.set_xlabel("along the planes")
left.set_ylabel("across them, in units of the normal")
left.set_title(f"RANDU triples, {detail['planes']} planes")

modern = rng.random(triples.shape[0] * 3).reshape(-1, 3)
right.plot(modern @ first, modern @ normal, ".", ms=1.0, alpha=0.5)
right.set_xlabel("along the same direction")
right.set_ylabel("across it")
right.set_title("PCG64 triples, no structure")

for panel in (left, right):
    panel.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("../figures/91_generators.png", dpi=110); plt.close(fig)
print("saved ../figures/91_generators.png")
print(f"RANDU spans {detail['planes']} distinct plane indices; the modern points span a "
      f"continuum")
```

![RANDU's triples seen along the plane normal, against a modern generator](../figures/91_generators.png)

The left panel is banded and the right is not. Nothing about the one dimensional statistics of the
two samples differs, which is section 5's point drawn rather than tabulated.

## 9. From scratch

An LCG is one line, and the tests that catch RANDU are a few more. Writing them out makes clear that
nothing here is deep, and that the failure is a property of the arithmetic rather than of the code.

```python
import numpy as np


def my_lcg(count, a, c, m, seed=1):
    out = np.empty(count, dtype=np.int64)
    state = seed % m
    for k in range(count):
        state = (a * state + c) % m
        out[k] = state
    return out


def my_plane_count(values, normal, m):
    triples = np.column_stack([values[:-2], values[1:-1], values[2:]])
    combination = triples @ np.asarray(normal, dtype=np.int64)
    on_a_plane = combination % m == 0
    return int(np.unique(combination // m).size), bool(np.all(on_a_plane))


raw = my_lcg(20000, 65539, 0, 2 ** 31)
count, exact = my_plane_count(raw, [9, -6, 1], 2 ** 31)
print(f"from scratch: {count} planes, the relation is exact: {exact}")

library = rg.randu_lies_on_a_few_planes()
print(f"the library agrees: {count == library['planes']}")

assert count == library["planes"]
assert exact
```

Twelve lines reproduce the result that took RANDU out of service.

## 10. Exercises

**Level 1, understanding**

1.1 State the linear congruential recurrence and say what bounds its period.

1.2 State the Hull-Dobell conditions and say why each is needed.

1.3 Explain why an LCG's tuples lie on hyperplanes.

1.4 Say why `rand() % n` is a bad way to get a number below `n`, in two separate ways.

1.5 Say what the inverse transform needs, and what to do when it is not available.

**Level 2, derivation**

2.1 Prove the Hull-Dobell conditions are sufficient for full period.

2.2 Derive the RANDU relation $x_{k+2} = 6x_{k+1} - 9x_k \pmod{2^{31}}$ from $a = 2^{16}+3$.

2.3 Prove that $F^{-1}(u)$ has distribution $F$ when $u$ is uniform, including the case where $F$ is
not strictly increasing.

2.4 Derive Box-Muller from the polar form of the two dimensional normal density.

2.5 Show that bit $j$ of a power of two LCG has period at most $2^{j+1}$.

**Level 3, computational**

3.1 Implement the spectral test properly, by lattice reduction rather than a bounded search, and
compare its answer against the search for all five named generators.

3.2 Implement a lagged Fibonacci generator and find its failure, which is not the same as an LCG's.

3.3 Implement the ziggurat method for normals and compare its cost against Box-Muller and the polar
method.

3.4 Implement the gap test and the birthday spacings test, and find a generator each one catches
that the others do not.

3.5 Implement a counter based generator in the style of Philox and check that its streams are
independent.

**Level 4, experimental**

4.1 Measure the number of planes for every named generator in dimensions 2 through 8, and find where
each one first looks bad.

4.2 Measure how many samples a chi-square test needs to detect RANDU in three dimensions, as a
function of the number of cells.

4.3 Measure the cost per normal of Box-Muller, the polar method and `numpy`'s own generator, and
say what the difference is made of.

**Level 5, advanced**

5.1 **Why the period is not the point.** A generator with period $2^{19937}$ can still fail. Explain
what the Mersenne Twister's known weakness is and what fixed it.

5.2 **Independent streams.** Explain the two ways to give parallel workers independent randomness,
and say what goes wrong with the obvious third way.

5.3 **What "random" means here.** Explain the difference between a statistically good generator and
a cryptographically secure one, and say why Monte Carlo needs only the first.

## 11. Key takeaways

- **A generator is a recurrence, and its period is bounded by its state.** The Hull-Dobell
  conditions decide when an LCG reaches its bound, exactly, in all $4096$ cases swept.

- **A full period guarantees nothing.** RANDU has period $2^{29}$ and lies on **15 planes**.

- **The plane relation is an exact integer identity**, with residual $0$, found by search rather
  than assumed, and the same search finds nothing for the other four generators.

- **The low bits are much worse than the high bits**, with measured periods of $2, 4, 8, 16, 32, 64$
  for the bottom six.

- **The easy tests catch nothing.** Every generator here passes uniformity, and RANDU's lag one
  correlation of $0.00049$ is smaller than PCG64's $0.0016$.

- **A three dimensional test catches it at once**, at a ratio of $2.256$ against PCG64's $1.031$.

- **The transforms are exact.** Inverse transform and Box-Muller give scaled KS statistics of
  $0.42$ and $0.73$ at a million samples, where $1.63$ would be a one per cent event.

- **Rejection wastes a known fraction.** The polar method accepts $0.785352$ of its draws against a
  predicted $\pi/4 = 0.785398$.

## Where this goes next

Lesson 92 spends these numbers. Monte Carlo integration turns a sum of samples into an estimate of
an integral, and its error falls like $1/\sqrt N$ regardless of the dimension, which is either
terrible or wonderful depending on what the alternative was. The generator matters there in a way
this lesson can only hint at: an integral over a three dimensional region computed with RANDU is
computed on fifteen sheets.
