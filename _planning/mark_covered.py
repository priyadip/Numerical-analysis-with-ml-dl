"""Mark the coverage flags for lessons that are finished.

Usage:

    python _planning/mark_covered.py 01 02 03 ...

For each named lesson this sets `covered` on every concept mapped to it, and sets the other
flags based on evidence that is actually checked rather than assumed:

    implemented   the lesson's notebook contains a `def ` (from-scratch code)
    experimented  the lesson produced at least one figure, either inline in the notebook or
                  saved to figures/ and referenced from the built markdown
    tested        every nalib module the lesson imports has a test module in tests/
    verified      the notebook executed cleanly AND contains at least one assert

Both `experimented` and `tested` are read off the lesson rather than from a table. An earlier
version used a hardcoded list of lesson to module mappings that stopped at lesson 08, so every
later lesson reported `tested` as False however many tests backed it, and required an inline
PNG, so every lesson that saves its figure and closes it reported `experimented` as False.

Nothing here is set by hand, so the checklist cannot drift away from the repository.
"""

from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from coverage_data import ROWS  # noqa: E402
from lesson_map import LESSONS  # noqa: E402

STATUS_PATH = os.path.join(HERE, "coverage_status.json")
BUILD_PATH = os.path.join(HERE, "build_status.json")

#: Matches every way a lesson can pull in a library module, so the `tested` flag is decided by
#: what the lesson actually imports rather than by a table that goes stale.
IMPORT_PATTERNS = (
    # each stops at the end of its line: a trailing \s in the character class would run on
    # into the next statement and capture whatever name started it
    re.compile(r"^[ \t]*from nalib import ([^\n]+)", re.M),
    re.compile(r"^[ \t]*from nalib\.(\w+) import", re.M),
    re.compile(r"\bnalib\.(\w+)\b"),
)


def module_is_tested(name: str) -> bool:
    """Does any test module exercise this library module?

    A file named `test_<module>.py` is the usual arrangement and not the only valid one:
    `pivoting` is tested inside `tests/test_lu.py`, which is where it belongs. So the question
    is whether any test file imports it, not whether one is named after it.
    """
    direct = os.path.join(ROOT, "tests", f"test_{name}.py")
    if os.path.exists(direct):
        return True
    tests = os.path.join(ROOT, "tests")
    if not os.path.isdir(tests):
        return False
    needle_a = f"from nalib import {name}"
    needle_b = f"from nalib.{name} import"
    needle_c = f"nalib.{name}"
    for entry in os.listdir(tests):
        if not (entry.startswith("test_") and entry.endswith(".py")):
            continue
        with open(os.path.join(tests, entry), encoding="utf-8") as fh:
            text = fh.read()
        if needle_a in text or needle_b in text or needle_c in text:
            return True
    return False


def modules_used(code: str) -> set:
    """The nalib modules a lesson's code imports, in any of the spellings used here."""
    found = set()
    for pattern in IMPORT_PATTERNS:
        for hit in pattern.findall(code):
            for piece in str(hit).split(","):
                name = piece.strip().split(" as ")[0].strip()
                if name and name.isidentifier():
                    found.add(name)
    return found


def row_key(row) -> str:
    source, chapter, section, concept, _lesson = row
    return f"{source}|{chapter}|{section}|{concept}"


def evidence(lesson_id: str) -> dict:
    """Inspect the built artifacts and report what is actually there."""
    part, stem, _desc = LESSONS[lesson_id]
    nb_path = os.path.join(ROOT, part, f"{stem}.ipynb")
    md_path = os.path.join(ROOT, part, f"{stem}.md")

    out = {f: False for f in
           ("covered", "implemented", "experimented", "tested", "verified")}
    if not (os.path.exists(nb_path) and os.path.exists(md_path)):
        return out
    out["covered"] = True

    with open(nb_path, encoding="utf-8") as fh:
        nb = json.load(fh)

    code = "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )
    has_error = any(
        o.get("output_type") == "error"
        for c in nb["cells"]
        for o in c.get("outputs", [])
    )
    has_image = any(
        "image/png" in o.get("data", {})
        for c in nb["cells"]
        for o in c.get("outputs", [])
    )

    # a figure counts whether it is shown inline or written to figures/ and linked from
    # the built markdown; both are ways of producing one, and only the first used to count
    with open(md_path, encoding="utf-8") as fh:
        built = fh.read()
    saves_figure = ("savefig(" in code) and ("figures/" in built)

    out["implemented"] = "def " in code
    out["experimented"] = has_image or saves_figure
    out["verified"] = (not has_error) and ("assert " in code)

    modules = modules_used(code)
    out["tested"] = bool(modules) and all(module_is_tested(m) for m in modules)
    return out


def main() -> int:
    ids = sys.argv[1:]
    if ids == ["all"]:
        # Every lesson whose notebook exists. This is what run_all uses, so the coverage
        # numbers can never drift behind the lessons: forgetting to mark a lesson by hand is
        # exactly how the report went two parts stale once.
        ids = sorted(lid for lid in LESSONS
                     if os.path.exists(os.path.join(ROOT, LESSONS[lid][0],
                                                    LESSONS[lid][1] + ".ipynb")))
    if not ids:
        print("usage: python _planning/mark_covered.py 01 02 ...   (or: all)")
        return 2

    with open(STATUS_PATH, encoding="utf-8") as fh:
        status = json.load(fh)

    for lid in ids:
        if lid not in LESSONS:
            print(f"unknown lesson id: {lid}")
            return 2
        ev = evidence(lid)
        n = 0
        for row in ROWS:
            if row[4] != lid:
                continue
            key = row_key(row)
            status.setdefault(key, {})
            for flag, value in ev.items():
                status[key][flag] = value
            n += 1
        flags = ", ".join(f for f, v in ev.items() if v) or "none"
        print(f"lesson {lid} ({LESSONS[lid][1]}): {n:3d} concepts -> {flags}")

    with open(STATUS_PATH, "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=1, sort_keys=True)
    print(f"\nwrote {os.path.relpath(STATUS_PATH, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
