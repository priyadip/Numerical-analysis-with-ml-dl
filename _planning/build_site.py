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
import json
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


#: The reading view. On a wide screen the contents of the page leave the flow and become a
#: drawer, and the list of lessons moves into the index popup, so the lesson itself has the whole
#: width. The drawer part is scoped to Material's own desktop breakpoint, because below it the
#: theme already does this. The popup works at every width.
FOCUS_CSS = """/* Generated by _planning/build_site.py */

/* The header buttons. They look the same. What differs is when they are shown: the contents
   button belongs to the wide screen reading view, and the index button works at every width.
   The media query at the end of this file is what turns the contents button on, and it has to
   come last: a media query adds no specificity, so an equally specific rule after it would win
   at every width. */
.na-toggle,
.na-index-toggle {
  align-items: center;
  justify-content: center;
  width: 2.2rem;
  height: 2.4rem;
  padding: 0;
  border: 0;
  cursor: pointer;
  color: currentColor;
  background: transparent;
  opacity: 0.8;
  transition: opacity 0.2s;
}
.na-toggle:hover,
.na-index-toggle:hover { opacity: 1; }
.na-toggle svg,
.na-index-toggle svg { width: 1.1rem; height: 1.1rem; fill: currentColor; }

.na-toggle { display: none; }
.na-index-toggle { display: flex; }

/* The dimmer behind the open drawer. Hidden until the drawer exists. */
.na-backdrop { display: none; }

/* The whole course index. It opens below the header rather than over it, so the button that
   opened it stays visible and can close it again. */
.na-index {
  display: none;
  position: fixed;
  top: 2.4rem;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 6;
  align-items: flex-start;
  justify-content: center;
  padding: 1rem;
  background: rgba(0, 0, 0, 0.5);
}
body[data-index="open"] .na-index { display: flex; }

.na-index-panel {
  display: flex;
  flex-direction: column;
  width: 100%;
  max-width: 62rem;
  max-height: 100%;
  overflow: hidden;
  border-radius: 0.15rem;
  background: var(--md-default-bg-color);
  box-shadow: 0 0.3rem 1.6rem rgba(0, 0, 0, 0.4);
}

.na-index-head {
  padding: 0.7rem 0.9rem;
  border-bottom: 0.05rem solid var(--md-default-fg-color--lightest);
}
.na-index-filter {
  width: 100%;
  padding: 0.4rem 0.6rem;
  font-family: inherit;
  font-size: 0.7rem;
  color: var(--md-default-fg-color);
  background: var(--md-default-fg-color--lightest);
  border: 0;
  border-radius: 0.1rem;
}
.na-index-filter:focus { outline: 0.05rem solid var(--md-accent-fg-color); }

.na-index-body {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(14rem, 1fr));
  gap: 0.2rem 1.2rem;
  align-content: start;
  overflow-y: auto;
  padding: 0.9rem;
}
.na-index-group { margin-bottom: 0.7rem; }
.na-index-group[hidden] { display: none; }
.na-index-group h3 {
  margin: 0 0 0.25rem;
  font-size: 0.58rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--md-default-fg-color--light);
}
.na-index-group a {
  display: block;
  padding: 0.16rem 0.3rem;
  font-size: 0.68rem;
  line-height: 1.35;
  border-radius: 0.1rem;
  color: var(--md-default-fg-color);
  text-decoration: none;
}
.na-index-group a[hidden] { display: none; }
.na-index-group a:hover { background: var(--md-default-fg-color--lightest); }
.na-index-group a.na-here { font-weight: 700; color: var(--md-typeset-a-color); }

.na-index-empty {
  grid-column: 1 / -1;
  padding: 0.4rem 0.3rem;
  font-size: 0.68rem;
  color: var(--md-default-fg-color--light);
}
.na-index-empty[hidden] { display: none; }

@media screen and (min-width: 76.25em) {

  /* the reading column, centred, wider than the theme default but still a comfortable
     line length rather than the whole monitor */
  .md-grid { max-width: 68rem; }

  /* the header sits above the drawer and the dimmer */
  .md-header { z-index: 7; }

  /* every lesson is in the index popup now, so the left sidebar has nothing left to say here */
  .md-sidebar--primary { display: none; }

  /* the contents of the page leave the flow, so the lesson expands to fill the row */
  .md-sidebar--secondary {
    position: fixed;
    /* the theme writes top and height inline on every scroll, so these have to win */
    top: 2.4rem !important;
    bottom: 0;
    height: calc(100vh - 2.4rem) !important;
    right: 0;
    width: 15rem;
    margin: 0;
    padding: 0;
    z-index: 6;
    background-color: var(--md-default-bg-color);
    box-shadow: 0 0 1rem rgba(0, 0, 0, 0.25);
    transform: translateX(100%);
    transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  }
  body[data-toc="open"] .md-sidebar--secondary { transform: translateX(0); }

  /* the scrolling area has to fill the drawer it now lives in */
  .md-sidebar--secondary .md-sidebar__scrollwrap {
    height: 100% !important;
    max-height: none !important;
    margin: 0;
    padding: 0.8rem 0.6rem 2rem;
    overflow-y: auto;
  }

  .na-backdrop {
    display: block;
    position: fixed;
    top: 2.4rem;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 5;
    background: rgba(0, 0, 0, 0.45);
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.25s;
  }
  body[data-toc="open"] .na-backdrop { opacity: 1; pointer-events: auto; }

  /* this rule is last on purpose, see the note at the top of the file */
  .na-toggle { display: flex; }
}
"""


#: The behaviour: a three dot button that opens the whole course index, a button for the contents
#: of the page, Escape to close, and both closing themselves once a link has been followed.
#: Re-attached on every page because the theme swaps pages in without a reload.
FOCUS_JS = """/* Generated by _planning/build_site.py */

const NA_ICONS = {
  toc: '<svg viewBox="0 0 24 24"><path d="M3 5h6v2H3V5m0 6h6v2H3v-2m0 6h6v2H3v-2m8-12h10v2H11V5m0 6h10v2H11v-2m0 6h10v2H11v-2Z"/></svg>',
  index: '<svg viewBox="0 0 24 24"><path d="M12 16a2 2 0 0 1 2 2 2 2 0 0 1-2 2 2 2 0 0 1-2-2 2 2 0 0 1 2-2m0-6a2 2 0 0 1 2 2 2 2 0 0 1-2 2 2 2 0 0 1-2-2 2 2 0 0 1 2-2m0-6a2 2 0 0 1 2 2 2 2 0 0 1-2 2 2 2 0 0 1-2-2 2 2 0 0 1 2-2Z"/></svg>'
};

function naClose() {
  delete document.body.dataset.toc;
  delete document.body.dataset.index;
}

// two paths name the same page when they differ only by a trailing slash
function naTrim(path) {
  return path.length > 1 && path.endsWith("/") ? path.slice(0, -1) : path;
}

// the panel itself, made once and then reused
function naPanel() {
  let panel = document.querySelector(".na-index");
  if (panel) return panel;

  panel = document.createElement("div");
  panel.className = "na-index";
  panel.innerHTML =
    '<div class="na-index-panel">' +
      '<div class="na-index-head">' +
        '<input class="na-index-filter" type="text" spellcheck="false" ' +
               'placeholder="Type to find a lesson" aria-label="Filter the index">' +
      '</div>' +
      '<div class="na-index-body"></div>' +
    '</div>';

  // a click on the dimmed area, but not on the panel, closes the index
  panel.addEventListener("click", event => {
    if (event.target === panel) naClose();
  });
  panel.querySelector(".na-index-filter").addEventListener("input", naFilter);
  document.body.appendChild(panel);
  return panel;
}

// The links are relative, and how deep the current page sits decides what they have to say, so
// they are written again every time the index opens. The logo already points at the site root
// from wherever the reader is, which is the only base needed.
function naFill() {
  const body = naPanel().querySelector(".na-index-body");
  body.innerHTML = "";
  if (typeof NA_INDEX === "undefined") return;

  // The theme writes the site root as a relative link, and not always the same way: "." on the
  // home page, "../.." deeper in, sometimes with a trailing slash. The URL parser sorts that out.
  const logo = document.querySelector(".md-header__button.md-logo");
  const root = logo ? logo.getAttribute("href") : ".";
  const base = new URL(root.endsWith("/") ? root : root + "/", location.href);
  const here = naTrim(location.pathname);

  NA_INDEX.forEach(group => {
    const block = document.createElement("section");
    block.className = "na-index-group";
    const heading = document.createElement("h3");
    heading.textContent = group.title;
    block.appendChild(heading);

    group.items.forEach(item => {
      const link = document.createElement("a");
      link.href = new URL(item.url, base).href;
      link.textContent = item.title;
      if (naTrim(link.pathname) === here) link.className = "na-here";
      link.addEventListener("click", naClose);
      block.appendChild(link);
    });
    body.appendChild(block);
  });

  const empty = document.createElement("p");
  empty.className = "na-index-empty";
  empty.textContent = "Nothing in the course matches that.";
  empty.hidden = true;
  body.appendChild(empty);
}

function naFilter() {
  const box = document.querySelector(".na-index-filter");
  const needle = box ? box.value.trim().toLowerCase() : "";
  let found = 0;

  document.querySelectorAll(".na-index-group").forEach(group => {
    let shown = 0;
    group.querySelectorAll("a").forEach(link => {
      const hit = needle === "" || link.textContent.toLowerCase().includes(needle);
      link.hidden = !hit;
      if (hit) shown += 1;
    });
    group.hidden = shown === 0;
    found += shown;
  });

  const empty = document.querySelector(".na-index-empty");
  if (empty) empty.hidden = found > 0;
}

// the same button opens and closes, so a reader who opened it by accident can undo that
function naToggleIndex() {
  const open = document.body.dataset.index === "open";
  naClose();
  if (open) return;
  naFill();
  document.body.dataset.index = "open";
  const box = document.querySelector(".na-index-filter");
  if (box) {
    box.value = "";
    naFilter();
    box.focus();
  }
}

function naToggleContents() {
  const open = document.body.dataset.toc === "open";
  naClose();
  if (!open) document.body.dataset.toc = "open";
}

function naSetup() {
  const header = document.querySelector(".md-header__inner");
  if (!header) return;

  if (!document.querySelector(".na-backdrop")) {
    const dimmer = document.createElement("div");
    dimmer.className = "na-backdrop";
    dimmer.addEventListener("click", naClose);
    document.body.appendChild(dimmer);
  }

  if (!header.querySelector(".na-toggle")) {
    const contents = document.createElement("button");
    contents.className = "na-toggle";
    contents.title = "Contents of this page";
    contents.setAttribute("aria-label", "Contents of this page");
    contents.innerHTML = NA_ICONS.toc;
    contents.addEventListener("click", naToggleContents);
    const source = header.querySelector(".md-header__source");
    if (source) header.insertBefore(contents, source);
    else header.appendChild(contents);
  }

  // last in the header, so it sits at the right hand end of it
  if (!header.querySelector(".na-index-toggle")) {
    const index = document.createElement("button");
    index.className = "na-index-toggle";
    index.title = "Every lesson in the course";
    index.setAttribute("aria-label", "Every lesson in the course");
    index.innerHTML = NA_ICONS.index;
    index.addEventListener("click", naToggleIndex);
    header.appendChild(index);
  }

  // following a link means the reader is done with the drawer
  document.querySelectorAll(".md-sidebar a").forEach(link => {
    if (link.dataset.naBound) return;
    link.dataset.naBound = "1";
    link.addEventListener("click", naClose);
  });

  naClose();
}

document.addEventListener("keydown", event => {
  if (event.key === "Escape") naClose();
});

if (typeof document$ !== "undefined") {
  document$.subscribe(naSetup);
} else {
  document.addEventListener("DOMContentLoaded", naSetup);
}
"""


def page_url(path: str) -> str:
    """The address mkdocs gives a page, from the source path used in the navigation."""
    if path == "index.md":
        return ""
    return path[:-len(".md")] + "/"


def index_groups(nav_parts, solutions) -> list:
    """The whole navigation, flattened into named groups for the index popup.

    Built from the same two arguments that build the navigation in ``mkdocs.yml``, so the popup
    and the sidebar are two views of one list and cannot drift apart.
    """
    groups = [("Start here", [
        ("Home", page_url("index.md")),
        ("Reference index", page_url("reference.md")),
        ("Library index", page_url("library.md")),
        ("Course map", page_url("COURSE_MAP.md")),
        ("Learning path", page_url("LEARNING_PATH.md")),
        ("Architecture", page_url("COURSE_ARCHITECTURE.md")),
        ("Dependencies", page_url("COURSE_DEPENDENCIES.md")),
    ])]

    for name, entries in nav_parts:
        groups.append((name, [(title, page_url(path)) for title, path in entries]))

    worked = [(PARTS[part][0], page_url("solutions/" + solutions[part]))
              for part in PARTS if part in solutions]
    if worked:
        groups.append(("Solutions", worked))

    reports = [(name[:-len(".md")], page_url("verification/" + name))
               for name in sorted(os.listdir(os.path.join(ROOT, "verification")))
               if name.endswith(".md")]
    if reports:
        groups.append(("Verification", reports))
    return groups


def index_js(groups) -> str:
    """The index as a JavaScript file, so the popup does not have to fetch anything."""
    payload = [{"title": title, "items": [{"title": a, "url": b} for a, b in items]}
               for title, items in groups]
    return ("/* Generated by _planning/build_site.py */\n\n"
            "const NA_INDEX = " + json.dumps(payload, indent=2) + ";\n")


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
    write(os.path.join(DOCS, "javascript", "index_data.js"),
          index_js(index_groups(nav_parts, solutions)))
    write(os.path.join(DOCS, "javascript", "focus.js"), FOCUS_JS)
    write(os.path.join(DOCS, "stylesheets", "focus.css"), FOCUS_CSS)
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
        "    - navigation.footer",
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
        "extra_css:",
        "  - stylesheets/focus.css",
        "",
        "extra_javascript:",
        "  - javascript/mathjax.js",
        "  - https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js",
        "  - javascript/index_data.js",
        "  - javascript/focus.js",
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
