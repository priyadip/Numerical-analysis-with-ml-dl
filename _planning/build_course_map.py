"""Generate COURSE_MAP.md from lesson_map.py.

Run from the repository root:

    python _planning/build_course_map.py
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from lesson_map import LESSONS, PARTS  # noqa: E402

OUT = os.path.join(ROOT, "COURSE_MAP.md")


def is_written(part: str, stem: str) -> bool:
    """A lesson counts as written only when both output files exist."""
    return (os.path.exists(os.path.join(ROOT, part, f"{stem}.ipynb"))
            and os.path.exists(os.path.join(ROOT, part, f"{stem}.md")))


def main() -> None:
    by_part: dict[str, list] = defaultdict(list)
    for lid in sorted(LESSONS):
        part, stem, desc = LESSONS[lid]
        by_part[part].append((lid, stem, desc))

    lines: list[str] = []
    add = lines.append

    add("# Course Map")
    add("")
    add("Every lesson in the course, in order. Each row is one lesson, and every lesson is")
    add("a pair of files with the same name: a runnable notebook (`.ipynb`) and a readable")
    add("lesson (`.md`).")
    add("")
    written = sum(1 for lid, (p, st, _d) in LESSONS.items() if is_written(p, st))
    if written == len(LESSONS):
        add(f"**{len(PARTS)} parts, {len(LESSONS)} lessons, all written.**")
    else:
        add(f"**{len(PARTS)} parts, {len(LESSONS)} lessons.** "
            f"{written} written so far, {len(LESSONS) - written} still to come.")
    add("")
    add("Lesson names in **bold with a link** are written and verified. Names in plain text")
    add("are planned and not yet written, so they have nothing to link to.")
    add("")
    add("If you are new here, read `README.md` first, then follow `LEARNING_PATH.md`.")
    add("For how topics depend on each other, see `COURSE_DEPENDENCIES.md`.")
    add("")

    # quick index
    add("## Parts at a glance")
    add("")
    add("| Part | Folder | Lessons | Status |")
    add("|---|---|---|---|")
    for part, (name, _d) in PARTS.items():
        ids = [lid for lid, _s, _x in by_part[part]]
        span = f"{ids[0]} to {ids[-1]}" if len(ids) > 1 else ids[0]
        done = sum(1 for lid, st, _x in by_part[part] if is_written(part, st))
        state = "complete" if done == len(ids) else (
            f"{done} of {len(ids)} written" if done else "not started")
        add(f"| {name} | [`{part}/`]({part}/) | {span} ({len(ids)}) | {state} |")
    add("")

    for part, (name, desc) in PARTS.items():
        add(f"## {name}")
        add("")
        add(desc)
        add("")
        add(f"Folder: [`{part}/`]({part}/)")
        add("")
        add("| # | Lesson | What it covers |")
        add("|---|---|---|")
        for lid, stem, ldesc in by_part[part]:
            if is_written(part, stem):
                label = f"**[{stem}]({part}/{stem}.md)**"
            else:
                label = f"{stem} *(not written yet)*"
            add(f"| {lid} | {label} | {ldesc} |")
        add("")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.relpath(OUT, ROOT)}  ({len(LESSONS)} lessons, {len(PARTS)} parts)")


if __name__ == "__main__":
    main()
