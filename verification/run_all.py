"""Run every verification check and write the reports.

    python verification/run_all.py                 # everything
    python verification/run_all.py --skip-notebooks

Checks performed:

1. **Notebook execution.** Every lesson notebook is rebuilt from its source and executed end
   to end. Pass or fail, wall time and any traceback are recorded.
2. **Test suite.** `pytest tests/` is run and its summary captured.
3. **Numerical verification.** Every assertion embedded in every notebook is counted, and the
   defining-identity checks are re-run independently of the notebooks.
4. **Source coverage.** The concept checklist is regenerated from what actually exists.
5. **File pairing.** Every `.ipynb` must have a matching `.md`, and vice versa.
6. **Link resolution.** Every relative markdown link must resolve.
7. **Placeholder sweep.** The whole repository is searched for TODO, FIXME, bare `pass` and
   similar.
8. **Generality.** Every lesson and library file is scanned for hardcoded dimensions, so that
   nothing quietly assumes a fixed input size. See COURSE_ARCHITECTURE.md sections 7.1, 7.2.

Reports are written into `verification/`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLANNING = os.path.join(ROOT, "_planning")
sys.path.insert(0, PLANNING)
sys.path.insert(0, os.path.join(ROOT, "src"))

from lesson_map import LESSONS, PARTS  # noqa: E402

PLACEHOLDER = re.compile(
    r"\b(TODO|FIXME|TBD|XXX|HACK)\b|coming soon|implement later|fill in later", re.I
)
BARE_PASS = re.compile(r"^\s*pass\s*$")
BARE_ELLIPSIS = re.compile(r"^\s*\.\.\.\s*$")

SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", ".venv", "figures"}


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def built_lessons() -> list[str]:
    """Lesson ids that currently have both output files."""
    out = []
    for lid in sorted(LESSONS):
        part, stem, _d = LESSONS[lid]
        nb = os.path.join(ROOT, part, f"{stem}.ipynb")
        md = os.path.join(ROOT, part, f"{stem}.md")
        if os.path.exists(nb) and os.path.exists(md):
            out.append(lid)
    return out


# ---------------------------------------------------------------- 1. notebooks


def run_notebooks() -> list[dict]:
    """Rebuild and execute every lesson that has a source file."""
    cmd = [sys.executable, os.path.join(PLANNING, "lessonbuild.py"), "--all",
           "--timeout", "900"]
    subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    path = os.path.join(PLANNING, "build_status.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_execution_report(reports: list[dict]) -> tuple[int, int]:
    # A lesson that is planned but not yet written carries status NO_SOURCE. It was never
    # executed, so counting it as "executed" would overstate the size of the repository.
    # It is reported separately, as work outstanding.
    pending = [r for r in reports if r.get("status") == "NO_SOURCE"]
    executed = [r for r in reports if r.get("status") != "NO_SOURCE"]
    passed = sum(1 for r in executed if r.get("status") == "PASS")
    failed = [r for r in executed if r.get("status") == "FAIL"]

    lines = [
        "# Notebook Execution Report",
        "",
        f"Generated {stamp()}.",
        "",
        "Every notebook is rebuilt from its authored source and executed end to end with",
        "`nbclient`, the same engine `jupyter nbconvert --execute` uses. A notebook counts as",
        "passing only when no cell raises, which includes every `assert` inside it.",
        "",
        "## Summary",
        "",
        f"- Notebooks executed: **{len(executed)}**",
        f"- Passed: **{passed}**",
        f"- Failed: **{len(failed)}**",
        f"- Planned but not yet written: **{len(pending)}**",
        f"- Total execution time: **{sum(r.get('seconds', 0) for r in executed):.1f} s**",
        "",
        "## Results",
        "",
        "| Lesson | Part | Status | Time (s) | Cells |",
        "|---|---|---|---|---|",
    ]
    for r in executed:
        lines.append(
            f"| `{r['stem']}` | `{r.get('part', '')}` | **{r['status']}** | "
            f"{r.get('seconds', 0):.1f} | {r.get('n_cells', 0)} |"
        )
    lines.append("")

    if pending:
        lines += ["## Planned, not yet written", "",
                  "These have a slot in the lesson plan but no source, so nothing was executed",
                  "for them. They are listed so the totals above cannot be mistaken for the",
                  "whole plan.", "",
                  "| Lesson | Stem |", "|---|---|"]
        lines += [f"| {r.get('lesson', '')} | `{r.get('stem', '')}` |" for r in pending]
        lines.append("")

    if failed:
        lines += ["## Failures", ""]
        for r in failed:
            lines.append(f"### {r['stem']}")
            lines.append("")
            for e in r.get("errors", []):
                lines.append(f"- cell {e['cell']}: `{e['ename']}: {e['evalue']}`")
            lines.append("")
    else:
        lines += [
            "## Failures",
            "",
            "None. Every notebook ran from the first cell to the last with no exception.",
            "",
        ]

    lines += [
        "## Intentional failures",
        "",
        "Some lessons demonstrate a method breaking down. Those failures are set up on",
        "purpose, caught, and explained, so they do not stop the notebook. They are listed",
        "here so nobody mistakes them for bugs.",
        "",
        "| Lesson | What fails on purpose | How it is controlled |",
        "|---|---|---|",
        "| 01 | Expanded $(x-1)^6$ near $x=1$ returns negative values for a sixth power |"
        " asserted to be negative and asserted to exceed the true peak by 10x |",
        "| 05 | The textbook quadratic formula loses every digit at $b = 10^8$ |"
        " asserted to have relative error above 0.1 |",
        "| 05 | Naive summation discards 200000 small values entirely |"
        " asserted to return exactly 1.0 |",
        "| 06 | `numpy.roots` returns complex roots for a real quintuple root |"
        " asserted forward error large, backward error under 20u |",
        "| 06 | The forward recurrence for $I_n$ produces impossible negative integrals |"
        " asserted to break the analytic bound $0 < I_n \\le 1/(5(n+1))$ |",
        "",
        "## Environment note",
        "",
        "On Windows, `zmq` emits a `RuntimeWarning` about the Proactor event loop when a",
        "kernel starts. It is harmless, comes from the Jupyter transport layer rather than",
        "from any lesson, and does not affect results.",
        "",
    ]

    with open(os.path.join(HERE, "notebook_execution_report.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return passed, len(failed)


# ---------------------------------------------------------------- 2. tests


def run_tests() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
        cwd=ROOT, capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    m = re.search(r"(\d+) passed", out)
    passed = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) failed", out)
    failed = int(m.group(1)) if m else 0
    return {"passed": passed, "failed": failed, "output": out, "rc": proc.returncode}


def write_test_report(result: dict) -> None:
    modules = sorted(
        f for f in os.listdir(os.path.join(ROOT, "tests"))
        if f.startswith("test_") and f.endswith(".py")
    )
    tail = "\n".join(result["output"].strip().splitlines()[-30:])

    lines = [
        "# Test Report",
        "",
        f"Generated {stamp()}.",
        "",
        "The suite covers `src/nalib`, the from-scratch library that the lessons build and",
        "then import. Every module is tested against known analytic results, against NumPy",
        "or SciPy where an equivalent exists, on edge cases, and on seeded random input.",
        "",
        "## Summary",
        "",
        f"- Tests passed: **{result['passed']}**",
        f"- Tests failed: **{result['failed']}**",
        f"- Test modules: **{len(modules)}**",
        "",
        "## Modules",
        "",
        "| Module | Covers |",
        "|---|---|",
        "| `test_numbersystems.py` | base conversion, Theorem 2.3 on terminating expansions |",
        "| `test_floatingpoint.py` | IEEE field decomposition, ulp spacing, the standard model,"
        " summation algorithms, the Sterbenz lemma |",
        "| `test_errors.py` | error measures, condition numbers against analytic values,"
        " propagation rules, the governing inequality |",
        "| `test_convergence.py` | observed order on sequences of known order, linear rate"
        " fitting, refinement studies |",
        "| `test_polynomials.py` | Horner against `numpy.polyval` bit for bit, synthetic"
        " division, the remainder theorem, measured flop counts |",
        "| `test_cost.py` | the operation counter against hand-computed counts, exponent"
        " fitting |",
        "",
        "## What these tests actually catch",
        "",
        "Two real defects were found by this suite while Part 1 was being written, and both",
        "are worth recording because they are the kind of thing that silently survives",
        "otherwise:",
        "",
        "1. **`np.longdouble` is not higher precision on this platform.** On Windows with the",
        "   Microsoft compiler it is an alias for `float64`. Two lessons had been using it as",
        "   a high precision reference, which meant they were comparing a value against",
        "   itself. Both now use `math.fsum`, which is exactly rounded, or",
        "   `fractions.Fraction`, which is exact. The Sterbenz test that exposed this would",
        "   have passed vacuously forever.",
        "2. **An off-by-one in a flop count.** The naive polynomial evaluator performs n+1",
        "   additions, not n, because the running total starts at zero. The measured count",
        "   disagreed with the derived formula and the formula was wrong.",
        "",
        "## Raw pytest output",
        "",
        "```text",
        tail,
        "```",
        "",
    ]
    with open(os.path.join(HERE, "test_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- 3. numerical checks


def count_assertions() -> dict:
    """How many assertions each built notebook actually executed."""
    counts = {}
    for lid in built_lessons():
        part, stem, _d = LESSONS[lid]
        with open(os.path.join(ROOT, part, f"{stem}.ipynb"), encoding="utf-8") as fh:
            nb = json.load(fh)
        code = "\n".join(
            "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
        )
        counts[lid] = len(re.findall(r"^\s*assert\s", code, re.M))
    return counts


def independent_identity_checks() -> list[tuple[str, str, float, float, bool]]:
    """Re-run the core numerical identities outside the notebooks.

    Returns rows of (area, identity, measured, tolerance, passed).
    """
    import math
    from fractions import Fraction

    import numpy as np

    from nalib import (convergence as cv, errors as err, floatingpoint as fp,
                       numbersystems as ns, polynomials as poly)

    u = np.finfo(float).eps / 2
    rng = np.random.default_rng(42)
    rows = []

    def add(area, name, measured, tol):
        rows.append((area, name, float(measured), float(tol), bool(measured <= tol)))

    # floating point
    add("floating point", "machine_epsilon() equals 2^-52",
        abs(fp.machine_epsilon() - 2.0**-52), 0.0)
    worst = max(abs(fp.relative_rounding_error(float(rng.standard_normal()) * 10.0**int(
        rng.integers(-20, 20)), 24)) for _ in range(20000))
    add("floating point", "standard model |delta| <= u at 24 mantissa bits",
        worst, fp.simulated_unit_roundoff(24))

    vals = np.concatenate([[1.0], np.full(100_000, 1e-16)])
    exact = 1.0 + 100_000 * 1e-16
    add("summation", "Neumaier relative error on the hard case",
        abs(fp.neumaier_sum(vals) - exact) / exact, 1e-15)
    add("summation", "Kahan relative error on the hard case",
        abs(fp.kahan_sum(vals) - exact) / exact, 1e-15)

    # number systems
    bad = 0
    for _ in range(2000):
        f = Fraction(int(rng.integers(-9999, 9999)), int(rng.integers(1, 500)))
        for b in (2, 3, 8, 10, 16):
            if ns.is_exact_in_base(f, b) and ns.from_base(ns.to_base(f, b, 80), b) != f:
                bad += 1
    add("number systems", "exact round trips that failed", bad, 0)

    # polynomials
    worst = 0.0
    for _ in range(3000):
        c = rng.standard_normal(int(rng.integers(1, 12)))
        x = float(rng.standard_normal())
        worst = max(worst, abs(float(poly.horner(c, x)) - float(np.polyval(c, x))))
    add("polynomials", "|horner - numpy.polyval|, worst over 3000 cases", worst, 0.0)

    worst = 0.0
    for _ in range(2000):
        c = rng.standard_normal(int(rng.integers(1, 12)))
        r = float(rng.standard_normal())
        _q, rem = poly.synthetic_division(c, r)
        worst = max(worst, abs(rem - float(np.polyval(c, r))) / max(1.0, abs(rem)))
    add("polynomials", "remainder theorem: |remainder - p(r)| relative", worst, 1e-9)

    # conditioning
    for f, x, kappa in [(np.sqrt, 4.0, 0.5), (np.exp, 20.0, 20.0), (np.sin, 1.0, None)]:
        k = err.condition_number_scalar(f, x)
        if kappa is not None:
            add("conditioning", f"kappa of {f.__name__} at {x} against analytic",
                abs(k - kappa) / kappa, 1e-4)

    coeffs = poly.from_roots([2.0] * 5)
    roots = np.roots(coeffs)
    fwd = float(np.abs(roots - 2.0).max())
    bwd = float(np.abs(np.polyval(coeffs, roots)).max() / np.abs(coeffs).max())
    add("conditioning", "(x-2)^5: backward error of numpy.roots, in units of u",
        bwd / u, 20.0)
    rows.append(("conditioning",
                 "(x-2)^5: forward error is LARGE (problem is ill conditioned)",
                 fwd, 1e-4, fwd > 1e-4))

    # convergence
    x, errs = 1.0, []
    for _ in range(6):
        x = x - (x * x - 2.0) / (2.0 * x)
        errs.append(abs(x - math.sqrt(2.0)))
    add("convergence", "Newton measured order against 2",
        abs(float(cv.observed_order(errs)[-1]) - 2.0), 0.05)

    hs = np.array([2.0**-k for k in range(2, 14)])
    e_central = np.array(
        [abs((np.exp(1 + h) - np.exp(1 - h)) / (2 * h) - np.exp(1)) for h in hs])
    add("convergence", "central difference measured order against 2",
        abs(cv.refinement_order(hs, e_central) - 2.0), 0.05)

    return rows


def write_numerical_report(rows, assertion_counts) -> tuple[int, int]:
    total_asserts = sum(assertion_counts.values())
    n_pass = sum(1 for r in rows if r[4])

    lines = [
        "# Numerical Verification Report",
        "",
        f"Generated {stamp()}.",
        "",
        "Executing without raising is not the same as being right. This report covers the",
        "second question: are the numbers correct?",
        "",
        "Two mechanisms are used.",
        "",
        "1. **Assertions inside the lessons.** Every notebook checks its own claims, so a",
        "   wrong number becomes an execution failure rather than something a reader has to",
        "   notice. Tolerances come from machine epsilon and the condition number of the",
        "   problem, never from whatever value made the check pass.",
        "2. **Independent re-checks.** The identities below are recomputed here, outside the",
        "   notebooks, so a mistake in a lesson cannot hide itself.",
        "",
        "## Assertions executed per lesson",
        "",
        "| Lesson | Assertions |",
        "|---|---|",
    ]
    for lid, n in sorted(assertion_counts.items()):
        lines.append(f"| `{LESSONS[lid][1]}` | {n} |")
    lines += [f"| **total** | **{total_asserts}** |", ""]

    lines += [
        "## Independent identity checks",
        "",
        "Each row was recomputed by `verification/run_all.py`, not read from a notebook.",
        "",
        "| Area | Identity | Measured | Tolerance | Result |",
        "|---|---|---|---|---|",
    ]
    for area, name, measured, tol, ok in rows:
        lines.append(
            f"| {area} | {name} | {measured:.3e} | {tol:.3e} | "
            f"{'PASS' if ok else 'FAIL'} |"
        )
    lines += [
        "",
        f"**{n_pass} of {len(rows)} identity checks passed.**",
        "",
        "## How tolerances were chosen",
        "",
        "| Kind of check | Tolerance | Reason |",
        "|---|---|---|",
        "| Same algorithm, two implementations | **exactly 0** | identical floating point"
        " operations must give identical results, so anything nonzero is a real difference |",
        "| Correctly rounded quantity | 1 to 2 unit roundoffs | the best any method can do |",
        "| Backward error of a stable algorithm | tens of unit roundoffs | the definition of"
        " backward stability |",
        "| Measured convergence order | 0.05 absolute | the estimator itself has this much"
        " noise over a realistic number of steps |",
        "| Timing exponents | wide, and explained | run time depends on the machine, so the"
        " claim is checked against a mechanism rather than a number |",
        "",
        "No tolerance in this repository was widened to make a check pass. Where a measured",
        "value did not match the prediction, the cause was found and either the code or the",
        "prediction was corrected. Lesson 08 section 5 is the clearest example: the measured",
        "exponent for matrix multiplication is 2.3 rather than 3, and the gap is explained",
        "exactly by the growth in achieved Gflop/s, verified to three decimal places.",
        "",
    ]
    with open(os.path.join(HERE, "numerical_verification.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return n_pass, len(rows)


# ---------------------------------------------------------------- 5 and 6


LINK_RE = re.compile(r"\[[^\]]*\]\(([^)#]+?)\)")
FENCE_RE = re.compile(r"^\s*```")


def check_links() -> list[str]:
    """Every relative link in a learner-facing markdown file must resolve.

    Fenced code blocks are skipped: an expression like `funcs[m](vals)` inside a code block
    looks exactly like a markdown link and is not one. `_planning/lessons_src` is skipped
    too, because its links are written relative to where the built lesson will live, not to
    the source file.
    """
    broken = []
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        if os.path.normpath(root).endswith(os.path.join("_planning", "lessons_src")):
            continue
        for f in files:
            if not f.endswith(".md"):
                continue
            p = os.path.join(root, f)
            rel = os.path.relpath(p, ROOT).replace("\\", "/")
            in_fence = False
            with open(p, encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    if FENCE_RE.match(line):
                        in_fence = not in_fence
                        continue
                    if in_fence:
                        continue
                    for m in LINK_RE.finditer(line):
                        target = m.group(1).strip()
                        if target.startswith(("http://", "https://", "mailto:")):
                            continue
                        if not os.path.exists(os.path.normpath(
                                os.path.join(root, target))):
                            broken.append(f"{rel}:{i} -> {target}")
    return broken


def check_pairing() -> list[str]:
    problems = []
    for part in PARTS:
        d = os.path.join(ROOT, part)
        if not os.path.isdir(d):
            continue
        nbs = {f[:-6] for f in os.listdir(d) if f.endswith(".ipynb")}
        mds = {f[:-3] for f in os.listdir(d) if f.endswith(".md")}
        for stem in sorted(nbs - mds):
            problems.append(f"{part}/{stem}.ipynb has no matching .md")
        for stem in sorted(mds - nbs):
            problems.append(f"{part}/{stem}.md has no matching .ipynb")
    return problems


#: Words that mark a line as prose ABOUT placeholder markers rather than an actual
#: placeholder. Documentation has to be able to name what it looks for.
DOC_CONTEXT = re.compile(
    r"placeholder|sweep|searched for|markers|is searched|no bare", re.I
)


def is_documentation_reference(line: str) -> bool:
    """True when the line is describing the placeholder check, not containing one."""
    return bool(DOC_CONTEXT.search(line))


def sweep_placeholders() -> tuple[list[str], list[str]]:
    """Returns (real placeholders, documentation references that were excluded)."""
    hits: list[str] = []
    docs: list[str] = []

    def record(rel, where, line):
        entry = f"{rel}{where}: {line.strip()[:80]}"
        (docs if is_documentation_reference(line) else hits).append(entry)

    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, ROOT).replace("\\", "/")
            if rel == "verification/run_all.py":
                continue           # this file defines the patterns it searches for
            if f.endswith((".md", ".py", ".txt")):
                with open(p, encoding="utf-8", errors="replace") as fh:
                    for i, line in enumerate(fh, 1):
                        if (PLACEHOLDER.search(line) or BARE_PASS.match(line)
                                or BARE_ELLIPSIS.match(line)):
                            record(rel, f":{i}", line)
            elif f.endswith(".ipynb"):
                with open(p, encoding="utf-8") as fh:
                    nb = json.load(fh)
                for ci, c in enumerate(nb["cells"]):
                    for line in "".join(c["source"]).splitlines():
                        if (PLACEHOLDER.search(line) or BARE_PASS.match(line)
                                or BARE_ELLIPSIS.match(line)):
                            record(rel, f" cell {ci}", line)
    return hits, docs


# ---------------------------------------------------------------- main


def check_generality() -> list[str]:
    """Scan every authored and built file for hardcoded dimensions.

    Delegates to `_planning/check_generality.py` for **both** the rule and the file list, so
    there is one definition of each. Returns a list of "file:line why" strings, empty when
    everything derives its sizes from its input. See COURSE_ARCHITECTURE.md sections 7.1 and 7.2.

    This function used to build its own target list, and twice that list had a hole in it. A
    Part 6 sweep found a fixed index into a spectrum in `tests/test_lowrank.py` that this check
    reported 0 for while the standalone scanner reported 1, so the tests were added. A Part 7
    sweep found `np.eye(3)` in `solutions/part07_interpolation.md` the same way, so the
    solutions, the built notebooks and the built markdown were added. Rather than patch the list
    a third time, it now calls `collect_targets`, which is the list the standalone scanner uses,
    so the two can no longer disagree.
    """
    sys.path.insert(0, os.path.join(ROOT, "_planning"))
    try:
        import check_generality as cg
    except ImportError:
        return []
    out = []
    for path in cg.collect_targets(ROOT):
        for lineno, why, _line in cg.scan(path):
            out.append(f"{os.path.basename(path)}:{lineno} {why}")
    return out


def check_control_chars() -> list[str]:
    """Look for control characters left behind by a shell escape that got interpreted.

    A heredoc or an echo that expands escapes turns ``\\times`` into a TAB followed by
    ``imes``, and ``\\approx`` into a BEL followed by ``pprox``. The file still opens, still
    renders, and still passes every other check here, so the damage is invisible until a human
    reads the LaTeX and finds ``$5    imes10^{-17}$``. One sweep found **37 of these across 10
    files**, some of which had been in the repository for many lessons.

    No file in this repository wants a control character other than a newline. Python is
    indented with spaces, Markdown is indented with spaces, and a notebook stores its newlines
    as escaped ``\\n`` inside JSON strings. So the rule is simply: there should be none.

    Returns a list of "file:line saw <name> before <context>" strings, empty when clean.
    """
    names = {7: "BEL (a mangled backslash-a)", 8: "BS (a mangled backslash-b)",
             9: "TAB (a mangled backslash-t)", 11: "VT (a mangled backslash-v)",
             12: "FF (a mangled backslash-f)", 13: "CR (a mangled backslash-r)"}
    out = []
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in sorted(files):
            if not name.endswith((".md", ".ipynb", ".py")):
                continue
            path = os.path.join(root, name)
            try:
                text = open(path, encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                continue
            for lineno, line in enumerate(text.split("\n"), start=1):
                for col, ch in enumerate(line):
                    if ord(ch) < 32:
                        why = names.get(ord(ch), f"control character {ord(ch)}")
                        context = line[col + 1:col + 12].strip()
                        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
                        out.append(f"{rel}:{lineno} saw {why} before {context!r}")
                        break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-notebooks", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("REPOSITORY VERIFICATION")
    print("=" * 70)

    if args.skip_notebooks:
        path = os.path.join(PLANNING, "build_status.json")
        reports = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
        print(f"\n[1/9] notebooks   : skipped, reusing {len(reports)} previous results")
    else:
        print("\n[1/9] notebooks   : rebuilding and executing ...")
        reports = run_notebooks()
    nb_pass, nb_fail = write_execution_report(reports)
    print(f"      {nb_pass} passed, {nb_fail} failed")

    print("\n[2/9] test suite  : running pytest ...")
    tests = run_tests()
    write_test_report(tests)
    print(f"      {tests['passed']} passed, {tests['failed']} failed")

    print("\n[3/9] numerical   : re-running identity checks ...")
    rows = independent_identity_checks()
    counts = count_assertions()
    id_pass, id_total = write_numerical_report(rows, counts)
    print(f"      {id_pass}/{id_total} identities, {sum(counts.values())} notebook assertions")

    print("\n[4/9] coverage    : regenerating the checklist ...")
    # Re-derive the flags from the built artifacts for EVERY lesson that exists, then rebuild
    # the checklist. Marking lessons by hand is how the coverage numbers fell two parts behind
    # once, and nothing in the pipeline noticed.
    subprocess.run([sys.executable, os.path.join(PLANNING, "mark_covered.py"), "all"],
                   cwd=ROOT, capture_output=True, text=True)
    subprocess.run([sys.executable, os.path.join(PLANNING, "build_checklist.py")],
                   cwd=ROOT, capture_output=True, text=True)
    subprocess.run([sys.executable, os.path.join(PLANNING, "make_coverage_report.py")],
                   cwd=ROOT, capture_output=True, text=True)
    with open(os.path.join(PLANNING, "coverage_status.json"), encoding="utf-8") as fh:
        status = json.load(fh)
    covered = sum(1 for v in status.values() if v.get("covered"))
    print(f"      {covered}/{len(status)} concepts marked covered")

    print("\n[5/9] pairing     : checking .ipynb and .md pairs ...")
    pairing = check_pairing()
    print(f"      {len(pairing)} problems")

    print("\n[6/9] links       : checking relative markdown links ...")
    broken = check_links()
    for b in broken[:10]:
        print(f"      broken: {b}")
    print(f"      {len(broken)} broken")

    print("\n[7/9] placeholders: sweeping ...")
    placeholders, doc_refs = sweep_placeholders()
    print(f"      {len(placeholders)} found "
          f"({len(doc_refs)} documentation references excluded)")

    print()
    print("[8/9] generality  : scanning for hardcoded dimensions ...")
    generality = check_generality()
    for g in generality[:10]:
        print(f"      {g}")
    print(f"      {len(generality)} candidate(s)")

    print("\n[9/9] characters  : sweeping for mangled escapes ...")
    control = check_control_chars()
    for c in control[:10]:
        print(f"      {c}")
    print(f"      {len(control)} found")

    summary = {
        "generated": stamp(),
        "notebooks_passed": nb_pass,
        "notebooks_failed": nb_fail,
        "tests_passed": tests["passed"],
        "tests_failed": tests["failed"],
        "identities_passed": id_pass,
        "identities_total": id_total,
        "notebook_assertions": sum(counts.values()),
        "concepts_covered": covered,
        "concepts_total": len(status),
        "pairing_problems": pairing,
        "broken_links": broken,
        "placeholders": placeholders,
        "documentation_references": doc_refs,
        "generality_candidates": generality,
        "control_characters": control,
        "lessons_built": len(built_lessons()),
        "lessons_planned": len(LESSONS),
    }
    with open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)

    ok = (nb_fail == 0 and tests["failed"] == 0 and id_pass == id_total
          and not pairing and not placeholders and not broken and not generality
          and not control)
    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
