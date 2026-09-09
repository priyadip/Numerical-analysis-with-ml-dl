"""Settings for the published website.

The only line anyone normally edits is ``REPOSITORY``. Everything else in the site is generated
from the repository itself by ``build_site.py``, so the navigation, the cross links and the
reference tables cannot drift away from the lessons the way a hand written index would.
"""

#: The GitHub repository, as "owner/name". Every link to a notebook, a source file or a test is
#: built from this, so it is the one setting that has to be right.
REPOSITORY = "priyadip/Numerical-analysis-with-ml-dl"

#: The branch the links point at.
BRANCH = "main"

#: What the site calls itself.
SITE_NAME = "Numerical Analysis"
SITE_DESCRIPTION = (
    "A graduate course in numerical analysis: 98 lessons, every algorithm written from scratch, "
    "every claim measured by code that runs."
)

#: Where the site is served from. GitHub Pages for a project repository serves at
#: https://OWNER.github.io/NAME/, which is what the default below builds.
SITE_URL = ""


def blob(path: str) -> str:
    """A link to a file in the repository, as GitHub renders it."""
    return f"https://github.com/{REPOSITORY}/blob/{BRANCH}/{path}"


def raw(path: str) -> str:
    """A link to the raw bytes of a file."""
    return f"https://raw.githubusercontent.com/{REPOSITORY}/{BRANCH}/{path}"


def colab(path: str) -> str:
    """A link that opens a notebook in Google Colab."""
    return f"https://colab.research.google.com/github/{REPOSITORY}/blob/{BRANCH}/{path}"


def nbviewer(path: str) -> str:
    """A link that renders a notebook in nbviewer, which handles large notebooks GitHub gives up on."""
    return f"https://nbviewer.org/github/{REPOSITORY}/blob/{BRANCH}/{path}"


def site_url() -> str:
    """The published address, derived from the repository unless one was set explicitly."""
    if SITE_URL:
        return SITE_URL
    owner, _, name = REPOSITORY.partition("/")
    return f"https://{owner.lower()}.github.io/{name}/"


def is_configured() -> bool:
    """Whether the repository has been filled in, so a caller can warn instead of emitting bad links."""
    return "/" in REPOSITORY and not REPOSITORY.startswith("OWNER")
