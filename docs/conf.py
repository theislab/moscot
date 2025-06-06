# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import sys
from datetime import datetime
from pathlib import Path

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import moscot

sys.path.insert(0, str(Path(__file__).parent / "extensions"))

# -- Project information -----------------------------------------------------

project = moscot.__name__
author = moscot.__author__
version = moscot.__version__
copyright = f"{datetime.now():%Y}, Theislab"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "myst_nb",
    "sphinx_design",  # for cards
    "sphinx_tippy",
    "typed_returns",
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "ott": ("https://ott-jax.readthedocs.io/en/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "anndata": ("https://anndata.readthedocs.io/en/latest/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/latest/", None),
    "squidpy": ("https://squidpy.readthedocs.io/en/latest/", None),
    "mudata": ("https://mudata.readthedocs.io/en/latest/", None),
}
master_doc = "index"
pygments_style = "tango"
pygments_dark_style = "monokai"

nitpicky = True
nitpick_ignore = [
    ("py:class", "numpy.float64"),
    # see: https://github.com/numpy/numpydoc/issues/275
    ("py:class", "None.  Remove all items from D."),
    ("py:class", "a set-like object providing a view on D's items"),
    ("py:class", "a set-like object providing a view on D's keys"),
    ("py:class", "v, remove specified key and return the corresponding value."),  # noqa: E501
    ("py:class", "None.  Update D from dict/iterable E and F."),
    ("py:class", "an object providing a view on D's values"),
    ("py:class", "a shallow copy of D"),
    # ignore these classes until ott-jax adds them to their docs
    ("py:class", "ott.initializers.quadratic.initializers.BaseQuadraticInitializer"),
    ("py:class", "ott.initializers.linear.initializers.SinkhornInitializer"),
]
# TODO(michalk8): remove once typing has been cleaned-up
nitpick_ignore_regex = [
    (r"py:class", r"moscot\..*(K|B|O)"),
    (r"py:class", r"numpy\._typing.*"),
    (r"py:class", r"moscot\..*Protocol.*"),
    (
        r"py:class",
        r"moscot.base.output.BaseSolverOutput",
    ),  # https://github.com/sphinx-doc/sphinx/issues/10974 means there is simply no way around this with generics
]


# bibliography
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "alpha"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}

# myst
nb_execution_mode = "off"
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
]
myst_heading_anchors = 3


# autodoc + napoleon
autosummary_generate = True
autodoc_member_order = "alphabetical"
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# spelling
spelling_lang = "en_US"
spelling_warning = True
spelling_word_list_filename = "spelling_wordlist.txt"
spelling_add_pypi_package_names = True
spelling_exclude_patterns = ["references.rst"]
# see: https://pyenchant.github.io/pyenchant/api/enchant.tokenize.html
spelling_filters = [
    "enchant.tokenize.URLFilter",
    "enchant.tokenize.EmailFilter",
    "enchant.tokenize.MentionFilter",
]

# hover
tippy_anchor_parent_selector = "div.content"
tippy_enable_mathjax = True
# no need because of sphinxcontrib-bibtex
tippy_enable_doitips = False
linkcheck_report_timeouts_as_broken = True
linkcheck_ignore = [
    # 403 Client Error
    r"https://doi.org/10.1126/science.aad0501",
    r"https://resources.aertslab.org/cistarget/tf_lists/",
    r"https://doi.org/10.1126/science.aax1971",
    r"https://doi.org/10.1093/nar/gkac235",
    r"https://www.science.org/doi/abs/10.1126/science.aax1971",
    r"https://doi.org/10.1101/2022.01.10.475692",
    r"https://www.biorxiv.org/content/10.1101/2023.04.14.536867v1",
    r"https://www.biorxiv.org/content/10.1101/2023.05.11.540374v2",
    r"https://www.biorxiv.org/content/early/2022/01/11/2022.01.10.475692",
    r"https://doi.org/10.1145/2516971.2516977",
    r"https://doi.org/10.3390/a13090212",
    r"https://www.mdpi.com/1999-4893/13/9/212",
]

exclude_patterns = ["_build", "**.ipynb_checkpoints", "notebooks/README.rst", "notebooks/CONTRIBUTING.rst"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css",
]

html_show_sphinx = False
html_show_sourcelink = False
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "img/light_mode_logo.png",
    "dark_logo": "img/dark_mode_logo.png",
    "light_css_variables": {
        "color-brand-primary": "#003262",
        "color-brand-content": "#003262",
        "admonition-font-size": "var(--font-size-normal)",
        "admonition-title-font-size": "var(--font-size-normal)",
        "code-font-size": "var(--font-size--small)",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/theislab/moscot",
            "html": "",
            "class": "fab fa-github",
        },
    ],
}
