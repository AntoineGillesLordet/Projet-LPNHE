# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../code/"))


# -- Project information -----------------------------------------------------

project = 'Uchuu X ZTF'
copyright = '2024, Antoine Gilles--Lordet'
author = 'Antoine Gilles--Lordet'

# -- Project information -----------------------------------------------------

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
templates_path = ["templates/"]

html_static_path = ["_static"]
exclude_patterns = [".build/*", "templates/*", ".ipynb_checkpoints/*"]

html_theme = "sphinx_rtd_theme"
napoleon_google_docstring = False
autodoc_typehints = "none"
