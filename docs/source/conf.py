# type: ignore
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

from pathlib import Path
from datetime import datetime

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys

from sphinx.application import Sphinx
from sphinx_gallery.gen_gallery import DEFAULT_GALLERY_CONF

import moscot

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "extensions"))

import utils  # noqa: E402

# -- Project information -----------------------------------------------------

project = moscot.__name__
author = moscot.__author__
version = moscot.__version__
copyright = f"{datetime.now():%Y}, Theislab"

github_org = "theislab"
github_repo = "moscot"
github_ref = "main"
github_nb_repo = "moscot_notebooks"
utils.fetch_notebooks(repo_url=f"https://github.com/{github_org}/{github_nb_repo}")

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx_gallery.load_style",
    "sphinxcontrib.bibtex",
    "nbsphinx",
    "typed_returns",
    "sphinx_design",
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "ott": ("https://ott-jax.readthedocs.io/en/latest/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
}
master_doc = "index"
pygments_style = "sphinx"

# bibliography
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "alpha"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
source_suffix = [".rst"]  # , ".ipynb"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
autosummary_generate = True
autodoc_member_order = "bysource"
typehints_fully_qualified = False
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]
todo_include_todos = False


# spelling
spelling_lang = "en_US"
spelling_warning = True
spelling_word_list_filename = "spelling_wordlist.txt"
spelling_add_pypi_package_names = True
spelling_show_suggestions = True
spelling_exclude_patterns = ["references.rst"]
# see: https://pyenchant.github.io/pyenchant/api/enchant.tokenize.html
spelling_filters = [
    "enchant.tokenize.URLFilter",
    "enchant.tokenize.EmailFilter",
    "enchant.tokenize.MentionFilter",
]

exclude_patterns = [
    "auto_*/**.ipynb",
    "auto_*/**.md5",
    "auto_*/**.py",
    "**.ipynb_checkpoints",
    "auto_examples/problems/**/index.rst",
    "auto_*/**/index.rst",
]  # ignore anything that isn't .rst or .ipynb

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "furo"
html_static_path = ["_static"]
html_logo = "_static/img/logo.png"
html_show_sphinx = False
html_show_sourcelink = False
html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#003262",
        "color-brand-content": "#003262",
        "admonition-font-size": "var(--font-size-normal)",
        "admonition-title-font-size": "var(--font-size-normal)",
        "code-font-size": "var(--font-size--small)",
    },
}


nbsphinx_thumbnails = utils.get_thumbnails("auto_examples")


def setup(app: Sphinx) -> None:
    DEFAULT_GALLERY_CONF["default_thumb_file"] = "docs/source/_static/img/logo.png"
    app.add_config_value("sphinx_gallery_conf", DEFAULT_GALLERY_CONF, "html")
