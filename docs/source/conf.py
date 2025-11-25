# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PowerGrid'
copyright = '2025, Zhenlin Wang'
author = 'Zhenlin Wang'
release = 'v2.0.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


extensions = [
    'myst_parser',
    'sphinxcontrib.mermaid',
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
]

myst_enable_extensions = [
    "colon_fence",  # allows ::: blocks
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "logo": {
        "text": "PowerGrid 2.0",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "primary_sidebar_end": [],
    "secondary_sidebar_items": ["page-toc"],
    "show_toc_level": 2,
    "navigation_depth": 4,
    "show_nav_level": 2,
    "collapse_navigation": False,
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/yourusername/powergrid",
            "icon": "fab fa-github-square",
        },
    ],
}
html_sidebars = {
    "**": ["search-field", "sidebar-nav-bs"]
}
html_static_path = ['_static']

# Custom CSS files
html_css_files = [
    'css/custom.css',
]
