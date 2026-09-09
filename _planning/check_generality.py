"""Scan every lesson source and every library file for hardcoded dimensions.

The rule (COURSE_ARCHITECTURE.md section 7.1 and 7.2) is that code must derive every size,
shape, index and bound from its input rather than from a literal. Demonstration **data** may
be specific, and has to be. The **code that processes it** may not.

This scanner finds the patterns that usually mean the rule has been broken:

- ``range(<literal>)`` where the count should come from ``.shape`` or ``len``
- ``np.eye(<literal>)`` / ``np.zeros(<literal>)`` sitting next to a named matrix
- a magic constant that duplicates a dimension, such as ``sqrt(25) == 5.0``
- fixed indices such as ``s[2]`` or ``Vt[2]`` where ``[-1]`` is meant

It reports candidates, not certainties. Some literals are genuinely part of the mathematics:
a 2 by 2 rotation matrix is 2 by 2 by definition, a quadratic has 3 coefficients, and a plot of
the plane is 2-dimensional. Those are listed so a human can confirm them, and the confirmed
ones are recorded in ALLOWED below so the scan stays quiet on re-runs.

Run: python _planning/check_generality.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

#: Literals that are part of the mathematics, not an assumption about input size.
#: Each entry is (filename fragment, code fragment, why it is allowed).
ALLOWED = [
    ("_2d", "np.array([[c, -s], [s, c]])", "a plane rotation is 2 by 2 by definition"),
    ("_2d", "np.array([[c, s], [s, -c]])", "a plane reflection is 2 by 2 by definition"),
    ("05_loss", "roots_naive(a, b, c)", "a quadratic has exactly three coefficients"),
    ("05_loss", "roots_stable(a, b, c)", "a quadratic has exactly three coefficients"),
    ("05_loss", "roots_reference(a, b, c)", "a quadratic has exactly three coefficients"),
    ("15_vectors", "rng_r.standard_normal(6)",
     "defining a 6 by 4 rank-one example; this is DATA, and the code that reads it is general"),
    ("15_vectors", "rng_r.standard_normal(4)",
     "the other factor of the same rank-one example"),
    ("banded", "return np.zeros(0)",
     "the empty system has an empty solution; 0 is the answer, not an assumed size"),
    ("multigrid", "return np.zeros(0)",
     "a grid too small to coarsen restricts to nothing; 0 is the answer, not an assumption"),
    ("multigrid", "np.zeros(0)",
     "the same, in the conditional form of the injection transfer"),
    ("part04", "return np.zeros(0), 0",
     "a 2D grid too small to coarsen restricts to nothing; 0 is the answer, not a size"),
    ("odesystems", "v, m, h, n = s[0], s[1], s[2], s[3]",
     "the Hodgkin-Huxley neuron has exactly four state variables by definition, and naming "
     "them is clearer than indexing; the length does not depend on the input"),
    ("odesystems", "y, dy, theta, dtheta = s[0], s[1], s[2], s[3]",
     "the Tacoma deck model has exactly four state variables by definition, one displacement "
     "and one twist with their velocities; the length does not depend on the input"),
]

PATTERNS = [
    (re.compile(r"\brange\((\d+)\)"), "range with a literal count"),
    (re.compile(r"np\.eye\((\d+)\)"), "np.eye with a literal size"),
    # A literal 0 is excluded deliberately and generally: np.zeros(0) is the EMPTY array, which
    # is an answer ("this has no entries") rather than an assumption about the input's size. It
    # arose independently in banded, multigrid, qralg, symeig and krylov_eig, and allowlisting
    # it file by file was five entries saying the same thing.
    (re.compile(r"np\.zeros\(([1-9]\d*)\)"), "np.zeros with a literal length"),
    (re.compile(r"np\.ones\(([1-9]\d*)\)"), "np.ones with a literal length"),
    (re.compile(r"standard_normal\((\d+)\)"), "random vector of a literal length"),
    # == 0 is excluded for the same reason np.zeros(0) is: "is this empty?" is a question about
    # the answer, not an assumption about the input. Comparing a shape against 1, 2, 3 ... is a
    # real assumption and is still caught.
    (re.compile(r"\.shape\[\d\] *== *[1-9]\d*"), "shape compared against a literal"),
    # a fixed index of 2 or more usually means "the last one" was intended
    (re.compile(r"\b(?:s|sigma|Vt|U|ev|vals|angles)\[([2-9])\]"),
     "fixed index into a result whose length depends on the input"),
    # `sqrt(25) == 5.0` is a dimension written twice. `== 0.0` and `== 1.0` never are, so the
    # comparison arm only fires from 2 upwards, which keeps ordinary assertions out of the way.
    (re.compile(r"np\.sqrt\(\d{2,}\)|math\.sqrt\(\d{2,}\)|== *[2-9]\d*\.0 *#"),
     "a constant that may duplicate a dimension"),
]

#: Counts that are almost always a loop over trials or samples rather than a dimension.
SAMPLE_WORDS = ("trial", "sample", "_ in range", "repeat", "draw", "n_samples")

#: In a test, PICKING a size is the whole job. `og.inner(np.ones(3), np.ones(4))` is a test
#: deliberately passing mismatched lengths to check the error; `np.zeros(4)` is a test choosing
#: its input. Flagging those would bury the real findings in noise and teach everyone to ignore
#: this scanner, which is worse than not having one.
#:
#: Generality of the LIBRARY is guaranteed by `tests/test_dimension_independence.py`, which
#: sweeps sizes and magnitudes deliberately. That is the right tool for the job, and this
#: scanner does not duplicate it.
#:
#: What is NOT exempt in a test is a **fixed index**, such as `s[2]` where `s[-1]` was meant.
#: That is a bug wherever it appears, because the length of `s` depends on the input.
TEST_EXEMPT = {
    "random vector of a literal length",
    "np.eye with a literal size",
    "np.zeros with a literal length",
    "np.ones with a literal length",
    "range with a literal count",
}


def is_test_file(path: str) -> bool:
    return os.path.basename(path).startswith("test_")


def code_blocks(path: str):
    """Yield (line_number, line) for python code only, skipping prose and text fences.

    Three file kinds, because all three are deliverables and all three must obey the rule:

    - ``.py``    every line is code,
    - ``.md``    only the contents of python fences, not text fences and not prose,
    - ``.ipynb`` only the source of cells whose ``cell_type`` is ``code``.

    Scanning the built ``.ipynb`` matters as much as scanning the source it came from. The
    source is where a fix gets made, but the notebook is what a reader opens and runs, so the
    notebook is the thing that actually has to be correct.
    """
    if path.endswith(".ipynb"):
        with open(path, encoding="utf-8") as fh:
            notebook = json.load(fh)
        lineno = 0
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            for line in "".join(cell.get("source", [])).split("\n"):
                lineno += 1
                yield lineno, line
        return

    text = open(path, encoding="utf-8").read()
    if path.endswith(".py"):
        for i, line in enumerate(text.split("\n"), 1):
            yield i, line
        return

    inside = False
    for i, line in enumerate(text.split("\n"), 1):
        if line.startswith("```python"):
            inside = True
            continue
        if line.startswith("```"):
            inside = False
            continue
        if inside:
            yield i, line


def is_allowed(path: str, line: str) -> bool:
    for frag, code, _why in ALLOWED:
        if frag in path and code in line:
            return True
    return False


def scan(path: str) -> list[tuple[int, str, str]]:
    hits = []
    testing = is_test_file(path)
    for lineno, line in code_blocks(path):
        stripped = line.strip()
        if stripped.startswith("#") or is_allowed(path, line):
            continue
        for pattern, why in PATTERNS:
            if not pattern.search(line):
                continue
            # a bare `for _ in range(N)` is a trial count, not a dimension
            if any(w in line for w in SAMPLE_WORDS) and "range" in line:
                break
            # a test choosing the size of its own input is the point of a test
            if testing and why in TEST_EXEMPT:
                break
            hits.append((lineno, why, stripped))
            break
    return hits


def collect_targets(root: str = ".") -> list[str]:
    """Every file that must obey the rule.

    The authored source is scanned because that is where a fix has to be made. The built
    `.ipynb` and `.md` are scanned as well, because those are what a reader actually opens and
    runs, and a build could in principle drift from its source. The tests and the worked
    solutions are scanned for the same reason: they contain code a reader will copy.
    """
    patterns = [
        "_planning/lessons_src/*.md",     # authored source
        "src/nalib/*.py",                 # the library
        "tests/*.py",                     # the tests
        "[0-9][0-9]_*/*.ipynb",           # the built notebooks
        "[0-9][0-9]_*/*.md",              # the built markdown lessons
        "solutions/*.md",                 # worked solutions
    ]
    out: list[str] = []
    for pattern in patterns:
        out.extend(sorted(glob.glob(os.path.join(root, pattern))))
    return out


def main() -> int:
    targets = collect_targets()
    total = 0
    for path in targets:
        hits = scan(path)
        if hits:
            print(f"\n{os.path.basename(path)}")
            for lineno, why, line in hits:
                print(f"  line {lineno:>4}  [{why}]")
                print(f"            {line[:96]}")
            total += len(hits)
    print(f"\n{total} candidate(s) to review across {len(targets)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
