# type: ignore
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

from pathlib import Path

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))

# -- Project information -----------------------------------------------------

project = "moscot"
copyright = "2022, Theislab"
author = "foo"  # moscot.__author__
version = "0.0"  # moscot.__version__

github_org = "theislab"
github_repo = "moscot"
github_ref = "master"
github_nb_repo = "moscot_notebooks"
# _fetch_notebooks(repo_url=f"https://github.com/{github_org}/{github_nb_repo}")

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
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
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
source_suffix = [".rst", ".ipynb"]
master_doc = "index"
pygments_style = "sphinx"

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
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"
html_theme_options = {"navigation_depth": 4, "logo_only": True}
html_show_sphinx = False
html_show_sourcelink = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
