"""Build INTERNAL_COVERAGE_CHECKLIST.md and coverage_status.json from coverage_data.py.

Run from the repository root:

    python _planning/build_checklist.py

Re-running is safe. Existing status flags in coverage_status.json are preserved, so this
can be re-run after each phase to regenerate the checklist without losing progress.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from coverage_data import ROWS, SOURCE_NAMES  # noqa: E402
from lesson_map import LESSONS, PARTS  # noqa: E402

STATUS_PATH = os.path.join(HERE, "coverage_status.json")
CHECKLIST_PATH = os.path.join(HERE, "INTERNAL_COVERAGE_CHECKLIST.md")
FLAGS = ["covered", "implemented", "experimented", "tested", "verified"]


def row_key(row) -> str:
    source, chapter, section, concept, lesson = row
    return f"{source}|{chapter}|{section}|{concept}"


def load_status() -> dict:
    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_status(status: dict) -> None:
    with open(STATUS_PATH, "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=1, sort_keys=True)


def sync_status(status: dict) -> dict:
    """Add new rows with all flags False, drop rows no longer in the data file."""
    keys = {row_key(r) for r in ROWS}
    for r in ROWS:
        status.setdefault(row_key(r), {f: False for f in FLAGS})
        for f in FLAGS:
            status[row_key(r)].setdefault(f, False)
    for gone in set(status) - keys:
        del status[gone]
    return status


def tick(value: bool) -> str:
    return "yes" if value else "no"


def build_markdown(status: dict) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Internal Coverage Checklist")
    add("")
    add("This is a quality-control artifact. It is **not** part of the learner-facing")
    add("course. Learners should start at `README.md` and follow `LEARNING_PATH.md`.")
    add("")
    add("It exists to answer one question with evidence rather than assumption:")
    add("")
    add("> After finishing this repository, will the learner have learned everything")
    add("> important contained in the supplied books, even though the repository is")
    add("> organised differently from those books?")
    add("")
    add("Regenerate with `python _planning/build_checklist.py`.")
    add("")
    add("## Sources")
    add("")
    for code, name in SOURCE_NAMES.items():
        add(f"- **{code}** = {name}")
    add("")
    add("## Column meanings")
    add("")
    add("| Column | Meaning |")
    add("|---|---|")
    add("| covered | The concept is explained in the named lesson, in both `.ipynb` and `.md` |")
    add("| implemented | There is working from-scratch code for it, where code makes sense |")
    add("| experimented | A numerical experiment in the lesson demonstrates it |")
    add("| tested | An automated test in `tests/` exercises it, where testable |")
    add("| verified | The result was checked against a library or an identity, and the maths was re-read |")
    add("")
    add("Some concepts are definitional or theoretical, so `implemented` and `tested` stay")
    add("`no` for them by design. Those are marked with `n/a` once the lesson is written.")
    add("")

    # ---- summary
    total = len(ROWS)
    per_source = Counter(r[0] for r in ROWS)
    done = sum(1 for r in ROWS if status[row_key(r)]["covered"])
    add("## Summary")
    add("")
    add(f"- Total tracked concepts: **{total}**")
    for code in ("S1", "S2", "SUP"):
        add(f"  - {code}: {per_source.get(code, 0)}")
    add(f"- Lessons in the final curriculum: **{len(LESSONS)}**")
    add(f"- Concepts marked covered: **{done} / {total}** "
        f"({100.0 * done / total:.1f}%)")
    add("")

    # ---- per lesson index
    by_lesson: dict[str, list] = defaultdict(list)
    for r in ROWS:
        by_lesson[r[4]].append(r)

    add("## Concepts per lesson")
    add("")
    add("| Lesson | Part | S1 | S2 | SUP | Total |")
    add("|---|---|---|---|---|---|")
    for lid in sorted(LESSONS):
        part, stem, _desc = LESSONS[lid]
        rs = by_lesson.get(lid, [])
        c = Counter(r[0] for r in rs)
        add(f"| `{stem}` | `{part}` | {c.get('S1', 0)} | {c.get('S2', 0)} | "
            f"{c.get('SUP', 0)} | {len(rs)} |")
    add("")
    orphans = [lid for lid in LESSONS if lid not in by_lesson]
    if orphans:
        add(f"**Warning:** lessons with no mapped concept: {', '.join(sorted(orphans))}")
        add("")

    # ---- the full map, grouped by source then chapter
    add("## Full source-to-lesson map")
    add("")
    by_source: dict[str, list] = defaultdict(list)
    for r in ROWS:
        by_source[r[0]].append(r)

    for code in ("S1", "S2", "SUP"):
        rs = by_source.get(code, [])
        if not rs:
            continue
        add(f"### {code} - {SOURCE_NAMES[code]}")
        add("")
        current_chapter = None
        for source, chapter, section, concept, lid in rs:
            if chapter != current_chapter:
                current_chapter = chapter
                add("")
                add(f"**{chapter}**")
                add("")
                add("| Section | Concept | Lesson | cov | impl | exp | test | ver |")
                add("|---|---|---|---|---|---|---|---|")
            st = status[row_key((source, chapter, section, concept, lid))]
            stem = LESSONS[lid][1]
            add(f"| {section} | {concept} | `{stem}` | {tick(st['covered'])} | "
                f"{tick(st['implemented'])} | {tick(st['experimented'])} | "
                f"{tick(st['tested'])} | {tick(st['verified'])} |")
        add("")

    return "\n".join(lines) + "\n"


def main() -> None:
    status = sync_status(load_status())
    save_status(status)
    with open(CHECKLIST_PATH, "w", encoding="utf-8") as fh:
        fh.write(build_markdown(status))
    total = len(ROWS)
    done = sum(1 for r in ROWS if status[row_key(r)]["covered"])
    print(f"tracked concepts : {total}")
    print(f"lessons          : {len(LESSONS)}")
    print(f"covered          : {done} ({100.0 * done / total:.1f}%)")
    print(f"wrote            : {os.path.relpath(CHECKLIST_PATH, ROOT)}")
    print(f"wrote            : {os.path.relpath(STATUS_PATH, ROOT)}")


if __name__ == "__main__":
    main()
