"""Build lesson notebooks and markdown lessons from a single authored source.

Why this exists
---------------
Every lesson has to ship as a matching pair: a runnable `.ipynb` and a readable `.md` that
state the same mathematics and reach the same conclusions. Writing the two by hand
guarantees they drift apart. So each lesson is authored once, in
`_planning/lessons_src/NN_stem.md`, and both outputs are generated from it.

Source format
-------------
Plain Markdown. Any fenced block tagged ```python becomes a code cell in the notebook.
Everything else, including ```text fences used for pseudocode, stays as markdown.

The standard setup cell is injected automatically after the first markdown block, so no
lesson has to repeat it.

Pipeline
--------
    source .md  ->  .ipynb  ->  execute  ->  .md with real outputs

The generated `.md` carries the outputs that the notebook actually produced, so the reading
version quotes real numbers rather than numbers someone typed in.

Usage
-----
    python _planning/lessonbuild.py 01 02 03      # named lessons
    python _planning/lessonbuild.py --part 01_foundations
    python _planning/lessonbuild.py --all
    python _planning/lessonbuild.py 01 --no-exec  # build only, do not run
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time

import nbformat
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_DIR = os.path.join(HERE, "lessons_src")
FIG_DIR = os.path.join(ROOT, "figures")
STATUS_PATH = os.path.join(HERE, "build_status.json")

sys.path.insert(0, HERE)
from lesson_map import LESSONS  # noqa: E402

PREAMBLE = '''\
# Standard setup, the same in every lesson of this course.
import sys, pathlib

_root = pathlib.Path.cwd()
while not (_root / "src" / "nalib").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root / "src"))

import numpy as np
import matplotlib.pyplot as plt

SEED = 42                                  # fixed so your numbers match the text
rng = np.random.default_rng(SEED)

np.set_printoptions(precision=6, linewidth=100, suppress=False)
plt.rcParams.update({
    "figure.figsize": (7.5, 4.5), "figure.dpi": 110,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
})
'''


# ---------------------------------------------------------------- source -> notebook


def split_source(text: str) -> list[tuple[str, str]]:
    """Split authored markdown into (kind, body) blocks, kind in {'markdown', 'code'}."""
    blocks: list[tuple[str, str]] = []
    buf: list[str] = []
    in_python = False
    in_other = False

    def flush(kind: str) -> None:
        body = "".join(buf).strip("\n")
        buf.clear()
        if body.strip():
            blocks.append((kind, body))

    for line in text.splitlines(keepends=True):
        s = line.rstrip("\n")
        if in_python:
            if s.strip() == "```":
                flush("code")
                in_python = False
            else:
                buf.append(line)
            continue
        if in_other:
            buf.append(line)
            if s.strip() == "```":
                in_other = False
            continue
        if s.lstrip().startswith("```"):
            lang = s.strip()[3:].strip().lower()
            if lang == "python":
                flush("markdown")
                in_python = True
            else:
                in_other = True
                buf.append(line)
            continue
        buf.append(line)

    if in_python:
        raise ValueError("source ends inside an unclosed ```python fence")
    if in_other:
        raise ValueError("source ends inside an unclosed fence")
    flush("markdown")
    return blocks


def build_notebook(source_text: str) -> nbformat.NotebookNode:
    """Turn authored markdown into a notebook, injecting the standard setup cell."""
    blocks = split_source(source_text)
    cells = []
    injected = False
    for kind, body in blocks:
        if kind == "markdown":
            cells.append(nbformat.v4.new_markdown_cell(body))
            if not injected:
                cells.append(nbformat.v4.new_code_cell(PREAMBLE.rstrip()))
                injected = True
        else:
            cells.append(nbformat.v4.new_code_cell(body))
    if not injected:
        cells.insert(0, nbformat.v4.new_code_cell(PREAMBLE.rstrip()))

    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": sys.version.split()[0]},
        }
    )
    return nb


# ---------------------------------------------------------------- execution


def execute_notebook(nb: nbformat.NotebookNode, workdir: str, timeout: int = 600) -> dict:
    """Run every cell. Returns a report dict. Does not raise on cell errors."""
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": workdir}},
        allow_errors=True,
    )
    t0 = time.perf_counter()
    client.execute()
    elapsed = time.perf_counter() - t0

    errors = []
    for i, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                errors.append(
                    {
                        "cell": i,
                        "ename": out.get("ename", ""),
                        "evalue": out.get("evalue", ""),
                        "traceback": "\n".join(out.get("traceback", []))[-2000:],
                    }
                )
    return {"seconds": round(elapsed, 2), "errors": errors, "n_cells": len(nb.cells)}


# ---------------------------------------------------------------- notebook -> markdown

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _clean(text: str, limit: int = 6000) -> str:
    text = _ANSI.sub("", text).rstrip()
    if len(text) > limit:
        head = text[: limit // 2]
        tail = text[-limit // 2 :]
        text = f"{head}\n... [output trimmed] ...\n{tail}"
    return text


def notebook_to_markdown(nb: nbformat.NotebookNode, stem: str) -> str:
    """Render an executed notebook as a readable markdown lesson with its real outputs."""
    os.makedirs(FIG_DIR, exist_ok=True)
    parts: list[str] = []
    fig_no = 0

    for cell in nb.cells:
        if cell.cell_type == "markdown":
            parts.append(cell.source.rstrip())
            continue

        src = cell.source.rstrip()
        if src:
            parts.append(f"```python\n{src}\n```")

        text_chunks: list[str] = []
        images: list[str] = []
        for out in cell.get("outputs", []):
            kind = out.get("output_type")
            if kind == "stream":
                text_chunks.append(_clean(out.get("text", "")))
            elif kind in ("execute_result", "display_data"):
                data = out.get("data", {})
                if "image/png" in data:
                    fig_no += 1
                    name = f"{stem}_fig{fig_no:02d}.png"
                    with open(os.path.join(FIG_DIR, name), "wb") as fh:
                        fh.write(base64.b64decode(data["image/png"]))
                    images.append(f"![Figure {fig_no} from {stem}](../figures/{name})")
                elif "text/plain" in data:
                    text_chunks.append(_clean(data["text/plain"]))
            elif kind == "error":
                tb = _clean("\n".join(out.get("traceback", [])))
                text_chunks.append(tb)

        body = "\n".join(c for c in text_chunks if c.strip())
        if body.strip():
            parts.append(f"*Output:*\n\n```text\n{body}\n```")
        parts.extend(images)

    return "\n\n".join(p for p in parts if p.strip()) + "\n"


# ---------------------------------------------------------------- driver


def build_one(lesson_id: str, execute: bool = True, timeout: int = 600) -> dict:
    part, stem, _desc = LESSONS[lesson_id]
    src_path = os.path.join(SRC_DIR, f"{stem}.md")
    if not os.path.exists(src_path):
        return {"lesson": lesson_id, "stem": stem, "status": "NO_SOURCE"}

    out_dir = os.path.join(ROOT, part)
    os.makedirs(out_dir, exist_ok=True)
    nb_path = os.path.join(out_dir, f"{stem}.ipynb")
    md_path = os.path.join(out_dir, f"{stem}.md")

    with open(src_path, encoding="utf-8") as fh:
        nb = build_notebook(fh.read())

    report = {"lesson": lesson_id, "stem": stem, "part": part}
    if execute:
        run = execute_notebook(nb, out_dir, timeout=timeout)
        report.update(run)
        report["status"] = "PASS" if not run["errors"] else "FAIL"
    else:
        report["status"] = "BUILT"

    with open(nb_path, "w", encoding="utf-8") as fh:
        nbformat.write(nb, fh)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(notebook_to_markdown(nb, stem))

    report["notebook"] = os.path.relpath(nb_path, ROOT).replace("\\", "/")
    report["markdown"] = os.path.relpath(md_path, ROOT).replace("\\", "/")
    return report


def save_status(reports: list[dict]) -> None:
    existing = {}
    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH, encoding="utf-8") as fh:
            existing = {r["lesson"]: r for r in json.load(fh)}
    for r in reports:
        existing[r["lesson"]] = r
    with open(STATUS_PATH, "w", encoding="utf-8") as fh:
        json.dump([existing[k] for k in sorted(existing)], fh, indent=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("lessons", nargs="*", help="lesson ids such as 01 02")
    ap.add_argument("--part", help="build every lesson in a part folder")
    ap.add_argument("--all", action="store_true", help="build every lesson with a source")
    ap.add_argument("--no-exec", action="store_true", help="build but do not execute")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    ids = list(args.lessons)
    if args.part:
        ids += [k for k, v in LESSONS.items() if v[0] == args.part]
    if args.all:
        ids = [
            k
            for k, v in LESSONS.items()
            if os.path.exists(os.path.join(SRC_DIR, f"{v[1]}.md"))
        ]
    ids = sorted(set(ids))
    if not ids:
        ap.error("nothing to build: pass lesson ids, --part, or --all")

    reports = []
    worst = 0
    for lid in ids:
        rep = build_one(lid, execute=not args.no_exec, timeout=args.timeout)
        reports.append(rep)
        status = rep["status"]
        secs = f"{rep.get('seconds', 0):6.1f}s" if "seconds" in rep else "       "
        print(f"[{status:9s}] {secs}  {rep['stem']}")
        for err in rep.get("errors", []):
            print(f"      cell {err['cell']}: {err['ename']}: {err['evalue']}")
            worst = 1
        if status == "NO_SOURCE":
            worst = 1

    save_status(reports)
    npass = sum(1 for r in reports if r["status"] == "PASS")
    print(f"\n{npass} / {len(reports)} passed")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
