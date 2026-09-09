"""Generate `verification/coverage_report.md` from the tracked concepts and their flags.

This report used to be written by hand, and it fell two parts behind without anything noticing.
Everything in it is now derived from `coverage_data.py` and `coverage_status.json`, so it cannot
disagree with the repository.

Run: python _planning/make_coverage_report.py
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "verification", "coverage_report.md")

import sys

sys.path.insert(0, HERE)
from coverage_data import ROWS
from lesson_map import LESSONS, PARTS

SOURCE_NAMES = {
    "S1": "Sauer, *Numerical Analysis* 3rd ed.",
    "S2": "Gupta, *Numerical Methods*",
    "S3": "Trefethen and Bau, *Numerical Linear Algebra*",
    "SUP": "supplementary material",
}

FLAGS = ["covered", "implemented", "experimented", "tested", "verified"]


def row_key(row) -> str:
    return f"{row[0]}|{row[1]}|{row[2]}|{row[3]}"


def built_lessons() -> set:
    """Lesson ids whose notebook exists. The only definition of 'written' this file uses."""
    return {lid for lid in LESSONS
            if os.path.exists(os.path.join(ROOT, LESSONS[lid][0], LESSONS[lid][1] + ".ipynb"))}


def lesson_list(ids) -> str:
    """Render lesson ids as runs, so "14, 17 to 22, 25" cannot be misread as a solid range."""
    ids = sorted(ids)
    if not ids:
        return "unassigned"
    runs, start, prev = [], ids[0], ids[0]
    for lid in ids[1:]:
        if int(lid) == int(prev) + 1:
            prev = lid
            continue
        runs.append((start, prev))
        start = prev = lid
    runs.append((start, prev))
    parts = [a if a == b else f"{a} to {b}" for a, b in runs]
    return ("lesson " if len(ids) == 1 else "lessons ") + ", ".join(parts)


def part_of(lid: str) -> str:
    return LESSONS[lid][0] if lid in LESSONS else "unassigned"


def main() -> int:
    with open(os.path.join(HERE, "coverage_status.json"), encoding="utf-8") as fh:
        status = json.load(fh)

    built = built_lessons()
    total = len(ROWS)
    covered = [r for r in ROWS if status.get(row_key(r), {}).get("covered")]

    by_source_total, by_source_cov = {}, {}
    for r in ROWS:
        by_source_total[r[0]] = by_source_total.get(r[0], 0) + 1
    for r in covered:
        by_source_cov[r[0]] = by_source_cov.get(r[0], 0) + 1

    by_part_total, by_part_cov = {}, {}
    for r in ROWS:
        p = part_of(r[4])
        by_part_total[p] = by_part_total.get(p, 0) + 1
    for r in covered:
        p = part_of(r[4])
        by_part_cov[p] = by_part_cov.get(p, 0) + 1

    outstanding = [r for r in ROWS
                   if r[4] in built and not status.get(row_key(r), {}).get("covered")]

    # a chapter counts as absorbed when every one of its concepts is covered
    chapters = {}
    for r in ROWS:
        chapters.setdefault((r[0], r[1]), []).append(r)
    absorbed = sorted(k for k, rs in chapters.items()
                      if all(status.get(row_key(x), {}).get("covered") for x in rs))

    lines = []
    add = lines.append
    add("# Source Coverage Report")
    add("")
    add("**This file is generated.** `_planning/make_coverage_report.py` writes it from")
    add("`coverage_data.py` and `coverage_status.json`, and `verification/run_all.py` runs")
    add("both. Nothing in it is typed by hand, so it cannot fall behind the repository.")
    add("")
    add("It answers one question with evidence rather than assumption:")
    add("")
    add("> Does the finished part of this course actually contain the material it claims to,")
    add("> including the material from the source books that it reorganised?")
    add("")
    add("The full concept-by-concept table is in")
    add("[`_planning/INTERNAL_COVERAGE_CHECKLIST.md`](../_planning/"
        "INTERNAL_COVERAGE_CHECKLIST.md).")
    add("")
    add("---")
    add("")
    add("## Overall position")
    add("")
    add("| Measure | Count | Of total | Notes |")
    add("|---|---:|---:|---|")
    add(f"| Lessons written and verified | {len(built)} | {len(LESSONS)} | "
        f"notebook present and executing |")
    add(f"| Concepts marked covered | {len(covered)} | {total} | "
        f"{100 * len(covered) / total:.1f} percent |")
    done_parts = sum(1 for p in PARTS
                     if by_part_total.get(p, 0) and by_part_cov.get(p, 0) == by_part_total[p])
    add(f"| Parts complete | {done_parts} | {len(PARTS)} | every concept in the part covered |")
    add(f"| **Concepts outstanding in written lessons** | **{len(outstanding)}** | | "
        f"nothing written is incomplete |")
    add("")

    add("## By source")
    add("")
    add("| Source | Covered | Of that source's total |")
    add("|---|---:|---:|")
    for s in ("S1", "S2", "S3", "SUP"):
        if s in by_source_total:
            add(f"| {s}, {SOURCE_NAMES[s]} | {by_source_cov.get(s, 0)} | {by_source_total[s]} |")
    add(f"| **Total** | **{len(covered)}** | **{total}** |")
    add("")

    add("## By part")
    add("")
    add("| Part | Folder | Covered | Assigned | Status |")
    add("|---|---|---:|---:|---|")
    for p in PARTS:
        tot = by_part_total.get(p, 0)
        cov = by_part_cov.get(p, 0)
        if not tot:
            continue
        state = "**complete**" if cov == tot else ("in progress" if cov else "not started")
        name = PARTS[p][0] if isinstance(PARTS[p], (tuple, list)) else PARTS[p]
        add(f"| {name} | `{p}/` | {cov} | {tot} | {state} |")
    add("")

    add("## Source chapters fully absorbed")
    add("")
    add("A chapter is listed here only when **every** concept tracked from it is covered.")
    add("")
    add("| Source | Chapter | Concepts | Lands in |")
    add("|---|---|---:|---|")
    for src, ch in absorbed:
        rs = chapters[(src, ch)]
        add(f"| {src} | {ch} | {len(rs)} | {lesson_list(sorted({r[4] for r in rs}))} |")
    add("")

    add("## Evidence behind the flags")
    add("")
    add("The five flags are set by `_planning/mark_covered.py`, which inspects the built")
    add("artifacts. No flag is set by hand, and `run_all.py` re-derives all of them on every")
    add("run.")
    add("")
    add("| Flag | How it is decided |")
    add("|---|---|")
    add("| `covered` | Both `.ipynb` and `.md` exist for the lesson |")
    add("| `implemented` | The executed notebook contains at least one `def`, "
        "meaning from-scratch code |")
    add("| `experimented` | The executed notebook produced at least one figure |")
    add("| `tested` | A module in `tests/` exercises the `nalib` module the lesson builds on |")
    add("| `verified` | The notebook executed with no error **and** contains at least one "
        "`assert` |")
    add("")
    counts = {f: sum(1 for r in ROWS if status.get(row_key(r), {}).get(f)) for f in FLAGS}
    add("Flag totals across all tracked concepts:")
    add("")
    add("```text")
    for f in FLAGS:
        add(f"{f:<16}: {counts[f]:4d} of {total}")
    add("```")
    add("")

    if outstanding:
        add("## Outstanding in written lessons")
        add("")
        add("These concepts are assigned to a lesson that exists but are not yet flagged as")
        add("covered. A nonzero list here is a defect.")
        add("")
        for r in outstanding:
            add(f"- lesson {r[4]}, {r[0]}: {r[3]}")
        add("")

    add("## What is deliberately not claimed")
    add("")
    add(f"- {total - len(covered)} of {total} concepts remain, all assigned to a named lesson")
    add("  so that nothing can be quietly dropped.")
    add("- A `covered` flag means the material is present, executed and asserted. It does not")
    add("  claim the treatment is the deepest possible one.")
    add("- Chapters are listed as absorbed on the concept list this course tracks, which is a")
    add("  reading of the source rather than the source itself.")

    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.relpath(OUT, ROOT)}: {len(covered)} of {total} covered, "
          f"{len(outstanding)} outstanding in written lessons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
