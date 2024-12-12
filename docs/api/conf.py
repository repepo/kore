# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../../bin'))

project = 'Kore'
copyright = '2024, Jorge Martinez'
author = 'Jorge Martinez'
release = '1.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.intersphinx',
    "sphinx.ext.extlinks"
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
add_module_names = False
toc_object_entries = True
# numfig = True



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    "navigation_with_keys": True,
    "announcement": "<em>These docs are a work-in-progress!</em>",
}


# -- Options for autodoc -------------------------------------------------
autodoc_typehints = "description"


# -- Options for TODOs -------------------------------------------------
todo_include_todos = True
