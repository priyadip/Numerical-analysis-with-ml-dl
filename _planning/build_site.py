"""Generate the readable website from the repository.

Why this exists
---------------
The lessons, the library, the tests and the solutions are four views of the same material, and a
reader following a topic wants to move between them without hunting. Writing those cross links by
hand would guarantee they rot, so every one of them is generated here from the repository as it is.

What it produces
----------------
    docs/                     the site source, which mkdocs turns into HTML
      index.md                the landing page
      reference.md            every lesson, with its notebook, modules, tests and solutions
      library.md              every public function in nalib, and the lesson that teaches it
      <part>/<lesson>.md      the lesson, with a link bar added at the top
      solutions/*.md          the worked solutions
      verification/*.md       the audits
      figures/*.png           the figures the lessons refer to
    mkdocs.yml                the site configuration, including the whole navigation

Nothing here edits a lesson in place. The link bar is added to the copy under ``docs/``, so a
rebuild of the lessons by ``lessonbuild.py`` cannot be undone by a rebuild of the site, and the
files a reader browses on GitHub are untouched.

Run: python _planning/build_site.py
"""

from __future__ import annotations

import ast
import io
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, "docs")

sys.path.insert(0, HERE)
from lesson_map import LESSONS, PARTS  # noqa: E402
import site_config as cfg  # noqa: E402


def read(path: str) -> str:
    return io.open(path, encoding="utf-8").read()


def write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    io.open(path, "w", encoding="utf-8", newline="\n").write(text)


# ------------------------------------------------------------------ what each lesson uses


def modules_used(text: str) -> list[str]:
    """The nalib modules a lesson imports, in the order a reader meets them."""
    seen = []
    for name in re.findall(r"from nalib import ([a-z_0-9]+)", text):
        if name not in seen:
            seen.append(name)
    return seen


def part_solutions() -> dict[str, str]:
    """Map a part folder to its solutions file, by matching the part number."""
    out = {}
    for name in sorted(os.listdir(os.path.join(ROOT, "solutions"))):
        if not name.startswith("part") or not name.endswith(".md"):
            continue
        number = name[4:6]
        for part in PARTS:
            if part.startswith(number):
                out[part] = name
    return out


def link_bar(lesson_id: str, part: str, stem: str, text: str, solutions: dict[str, str]) -> str:
    """The block that goes at the top of a lesson page, linking everything that lesson touches."""
    notebook = f"{part}/{stem}.ipynb"
    pieces = [
        '!!! abstract "Everything for this lesson"',
        "",
        f"    **Notebook** &nbsp; [run it in Colab]({cfg.colab(notebook)})"
        f" &nbsp;·&nbsp; [view on GitHub]({cfg.blob(notebook)})"
        f" &nbsp;·&nbsp; [render in nbviewer]({cfg.nbviewer(notebook)})",
        "",
    ]
    used = modules_used(text)
    if used:
        library = " &nbsp;·&nbsp; ".join(
            f"[`nalib.{name}`]({cfg.blob('src/nalib/' + name + '.py')})" for name in used)
        pieces += [f"    **Library source** &nbsp; {library}", ""]
        tested = [name for name in used
                  if os.path.exists(os.path.join(ROOT, "tests", f"test_{name}.py"))]
        if tested:
            checks = " &nbsp;·&nbsp; ".join(
                f"[`test_{name}.py`]({cfg.blob('tests/test_' + name + '.py')})" for name in tested)
            pieces += [f"    **Tests** &nbsp; {checks}", ""]
    if part in solutions:
        target = solutions[part]
        pieces += [f"    **Worked solutions** &nbsp; "
                   f"[every exercise in this part](../solutions/{target[:-3]}.md)", ""]
    pieces += [f"    **Source of this page** &nbsp; "
               f"[`_planning/lessons_src/{stem}.md`]"
               f"({cfg.blob('_planning/lessons_src/' + stem + '.md')})", ""]
    return "\n".join(pieces)


def inject(text: str, bar: str) -> str:
    """Put the link bar after the title and the part line, where a reader will look for it."""
    lines = text.split("\n")
    cut = 1
    for index in range(1, min(len(lines), 8)):
        if lines[index].startswith("**Part "):
            cut = index + 1
            break
    return "\n".join(lines[:cut]) + "\n\n" + bar + "\n".join(lines[cut:])


# ------------------------------------------------------------------ the generated indexes


def reference_page(solutions: dict[str, str]) -> str:
    """One table per part: every lesson with its notebook, its modules, its tests, its solutions."""
    out = ["# Reference index", "",
           "Every lesson, and everything that belongs to it. The lesson column goes to the page you",
           "can read; the rest go to the code on GitHub.", ""]
    if not cfg.is_configured():
        out += ["!!! warning", "",
                "    The repository has not been set in `_planning/site_config.py`, so the code",
                "    links on this page do not point anywhere yet.", ""]
    for part, (name, description) in PARTS.items():
        ids = sorted(lid for lid, (p, _s, _d) in LESSONS.items() if p == part)
        if not ids:
            continue
        out += [f"## {name}", "", description, "",
                "| # | Lesson | Notebook | Library | Tests |",
                "|---|---|---|---|---|"]
        for lid in ids:
            _p, stem, _d = LESSONS[lid]
            text = read(os.path.join(ROOT, part, stem + ".md"))
            title = re.search(r"^# \d+\.\s*(.+)$", text, re.M).group(1).strip()
            used = modules_used(text)
            library = " ".join(f"[`{n}`]({cfg.blob('src/nalib/' + n + '.py')})" for n in used)
            tested = " ".join(
                f"[`{n}`]({cfg.blob('tests/test_' + n + '.py')})" for n in used
                if os.path.exists(os.path.join(ROOT, "tests", f"test_{n}.py")))
            out.append(f"| {lid} | [{title}]({part}/{stem}.md) "
                       f"| [notebook]({cfg.blob(f'{part}/{stem}.ipynb')}) "
                       f"| {library} | {tested} |")
        if part in solutions:
            out += ["", f"Worked solutions for this part: "
                        f"[{solutions[part][:-3]}](solutions/{solutions[part][:-3]}.md).", ""]
        else:
            out.append("")
    return "\n".join(out) + "\n"


def library_page() -> str:
    """Every public function in nalib, with the module it lives in and the lessons that use it."""
    modules = sorted(name[:-3] for name in os.listdir(os.path.join(ROOT, "src", "nalib"))
                     if name.endswith(".py") and not name.startswith("_"))
    teaches: dict[str, list[str]] = {name: [] for name in modules}
    for lid, (part, stem, _d) in sorted(LESSONS.items()):
        for name in modules_used(read(os.path.join(ROOT, part, stem + ".md"))):
            if name in teaches:
                teaches[name].append(lid)

    out = ["# Library index", "",
           "Every module in `nalib`, what it holds, and the lessons that use it. Nothing here is a",
           "wrapper around SciPy: each algorithm is written out in plain NumPy so it can be read.", "",
           "| Module | Public functions | Lines | Used by lessons |",
           "|---|---:|---:|---|"]
    total_functions = 0
    for name in modules:
        source = read(os.path.join(ROOT, "src", "nalib", name + ".py"))
        tree = ast.parse(source)
        public = [node.name for node in tree.body
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and not node.name.startswith("_")]
        total_functions += len(public)
        lessons = ", ".join(
            f"[{lid}]({LESSONS[lid][0]}/{LESSONS[lid][1]}.md)" for lid in teaches[name]) or "-"
        out.append(f"| [`{name}`]({cfg.blob('src/nalib/' + name + '.py')}) | {len(public)} "
                   f"| {source.count(chr(10))} | {lessons} |")
    out += ["", f"**{len(modules)} modules, {total_functions} public functions.**", "",
            "## Every public function", "",
            "| Function | Module | Lessons |", "|---|---|---|"]
    for name in modules:
        source = read(os.path.join(ROOT, "src", "nalib", name + ".py"))
        lessons = ", ".join(
            f"[{lid}]({LESSONS[lid][0]}/{LESSONS[lid][1]}.md)" for lid in teaches[name]) or "-"
        for node in ast.parse(source).body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue
            target = cfg.blob(f"src/nalib/{name}.py") + f"#L{node.lineno}"
            out.append(f"| [`{node.name}`]({target}) | `{name}` | {lessons} |")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ assembling docs/


def site_pages() -> set[str]:
    """Every path the site will contain, repository relative, so a link can be checked against it."""
    pages = {"index.md", "reference.md", "library.md", "COURSE_MAP.md", "LEARNING_PATH.md",
             "COURSE_ARCHITECTURE.md", "COURSE_DEPENDENCIES.md"}
    for _lid, (part, stem, _d) in LESSONS.items():
        pages.add(f"{part}/{stem}.md")
    for folder in ("solutions", "verification"):
        for name in os.listdir(os.path.join(ROOT, folder)):
            if name.endswith(".md"):
                pages.add(f"{folder}/{name}")
    for name in os.listdir(os.path.join(ROOT, "figures")):
        if name.endswith(".png"):
            pages.add(f"figures/{name}")
    return pages


def repoint(text: str, here: str, pages: set[str]) -> str:
    """Rewrite links so each one lands somewhere.

    ``here`` is the directory the page sits in, repository relative and "" at the root. A link that
    resolves to a page the site has keeps working as a relative link. A link to a directory goes to
    the reference index, because the site has no page for a folder. Everything else is a file with
    no page, so it goes to GitHub.
    """
    depth = here.count("/") + 1 if here else 0
    up = "../" * depth

    def resolve(target: str) -> str:
        base = here if here else "."
        return os.path.normpath(os.path.join(base, target)).replace(os.sep, "/")

    def fix(match: re.Match) -> str:
        label, target = match.group(1), match.group(2)
        if target.startswith(("http://", "https://", "mailto:", "#")):
            return match.group(0)
        anchor = ""
        if "#" in target:
            target, _, anchor = target.partition("#")
            anchor = "#" + anchor
        if not target:
            return match.group(0)
        # README points at a lesson folder, and the site has no page for a folder
        if target.endswith("/"):
            return f"[{label}]({up}reference.md)"
        landing = resolve(target)
        if landing in pages:
            relative = os.path.relpath(landing, here if here else ".").replace(os.sep, "/")
            return f"[{label}]({relative}{anchor})"
        return f"[{label}]({cfg.blob(landing)})"

    return re.sub(r"\[([^\]]*)\]\(([^)\s]+)\)", fix, text)


def build() -> dict:
    if os.path.isdir(DOCS):
        shutil.rmtree(DOCS)
    os.makedirs(DOCS)
    solutions = part_solutions()
    pages = site_pages()
    counts = {"lessons": 0, "solutions": 0, "reports": 0, "figures": 0}

    # the lessons, each with its link bar
    nav_parts = []
    for part, (name, _description) in PARTS.items():
        ids = sorted(lid for lid, (p, _s, _d) in LESSONS.items() if p == part)
        entries = []
        for lid in ids:
            _p, stem, _d = LESSONS[lid]
            text = read(os.path.join(ROOT, part, stem + ".md"))
            title = re.search(r"^# (\d+\.\s*.+)$", text, re.M).group(1).strip()
            page = inject(repoint(text, part, pages),
                          link_bar(lid, part, stem, text, solutions))
            write(os.path.join(DOCS, part, stem + ".md"), page)
            entries.append((title, f"{part}/{stem}.md"))
            counts["lessons"] += 1
        if entries:
            nav_parts.append((name, entries))

    # the solutions and the audits, copied with their links repointed
    for folder in ("solutions", "verification"):
        for name in sorted(os.listdir(os.path.join(ROOT, folder))):
            if not name.endswith(".md"):
                continue
            write(os.path.join(DOCS, folder, name),
                  repoint(read(os.path.join(ROOT, folder, name)), folder, pages))
            counts["solutions" if folder == "solutions" else "reports"] += 1

    # the top level documents
    for name in ("COURSE_MAP.md", "LEARNING_PATH.md", "COURSE_ARCHITECTURE.md",
                 "COURSE_DEPENDENCIES.md"):
        write(os.path.join(DOCS, name), repoint(read(os.path.join(ROOT, name)), "", pages))
    write(os.path.join(DOCS, "index.md"),
          repoint(read(os.path.join(ROOT, "README.md")), "", pages))
    write(os.path.join(DOCS, "reference.md"), reference_page(solutions))
    write(os.path.join(DOCS, "library.md"), library_page())

    # the figures the lessons point at
    source_figures = os.path.join(ROOT, "figures")
    target_figures = os.path.join(DOCS, "figures")
    os.makedirs(target_figures, exist_ok=True)
    for name in sorted(os.listdir(source_figures)):
        if name.endswith(".png"):
            shutil.copyfile(os.path.join(source_figures, name),
                            os.path.join(target_figures, name))
            counts["figures"] += 1

    write(os.path.join(DOCS, "javascript", "mathjax.js"), MATHJAX)
    write(os.path.join(ROOT, "mkdocs.yml"), mkdocs_yaml(nav_parts, solutions))
    return counts


#: Tell MathJax to read the delimiters this course actually writes, and to re-typeset after the
#: theme swaps a page in without a reload.
MATHJAX = r"""window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// the theme swaps pages in without a reload, so the maths has to be typeset again
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
"""


def mkdocs_yaml(nav_parts, solutions) -> str:
    """The site configuration, navigation included, so the two cannot disagree."""
    lines = [
        "# Generated by _planning/build_site.py. Edit that, or _planning/site_config.py, not this.",
        f"site_name: {cfg.SITE_NAME}",
        f"site_description: >-",
        f"  {cfg.SITE_DESCRIPTION}",
        f"site_url: {cfg.site_url()}",
        f"repo_url: https://github.com/{cfg.REPOSITORY}",
        f"repo_name: {cfg.REPOSITORY}",
        "edit_uri: \"\"",
        "docs_dir: docs",
        "",
        "theme:",
        "  name: material",
        "  features:",
        "    - navigation.instant",
        "    - navigation.tracking",
        "    - navigation.sections",
        "    - navigation.top",
        "    - navigation.indexes",
        "    - toc.follow",
        "    - search.suggest",
        "    - search.highlight",
        "    - content.code.copy",
        "  palette:",
        "    - media: \"(prefers-color-scheme: light)\"",
        "      scheme: default",
        "      primary: indigo",
        "      accent: indigo",
        "      toggle:",
        "        icon: material/weather-night",
        "        name: Switch to dark mode",
        "    - media: \"(prefers-color-scheme: dark)\"",
        "      scheme: slate",
        "      primary: indigo",
        "      accent: indigo",
        "      toggle:",
        "        icon: material/weather-sunny",
        "        name: Switch to light mode",
        "",
        "markdown_extensions:",
        "  - admonition",
        "  - attr_list",
        "  - md_in_html",
        "  - tables",
        "  - footnotes",
        "  - toc:",
        "      permalink: true",
        "      toc_depth: 3",
        "  - pymdownx.details",
        "  - pymdownx.superfences",
        "  - pymdownx.highlight:",
        "      anchor_linenums: true",
        "  - pymdownx.inlinehilite",
        "  - pymdownx.arithmatex:",
        "      generic: true",
        "",
        "extra_javascript:",
        "  - javascript/mathjax.js",
        "  - https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js",
        "",
        "plugins:",
        "  - search",
        "",
        "nav:",
        "  - Home: index.md",
        "  - Reference index: reference.md",
        "  - Library index: library.md",
        "  - Course map: COURSE_MAP.md",
        "  - Learning path: LEARNING_PATH.md",
        "  - Architecture: COURSE_ARCHITECTURE.md",
        "  - Dependencies: COURSE_DEPENDENCIES.md",
    ]
    for name, entries in nav_parts:
        lines.append(f"  - {name}:")
        for title, path in entries:
            safe = title.replace('"', "'")
            lines.append(f"      - \"{safe}\": {path}")
    lines.append("  - Solutions:")
    for part in PARTS:
        if part in solutions:
            name = solutions[part][:-3]
            lines.append(f"      - \"{PARTS[part][0]}\": solutions/{name}.md")
    lines.append("  - Verification:")
    for name in sorted(os.listdir(os.path.join(ROOT, "verification"))):
        if name.endswith(".md"):
            lines.append(f"      - \"{name[:-3]}\": verification/{name}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    if not cfg.is_configured():
        print("warning: set REPOSITORY in _planning/site_config.py, "
              "or every code link will be dead")
    made = build()
    print("wrote docs/ and mkdocs.yml")
    print("  %(lessons)d lesson pages, %(solutions)d solution pages, "
          "%(reports)d reports, %(figures)d figures" % made)
