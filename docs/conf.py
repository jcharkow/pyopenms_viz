"""
docs/conf
~~~~~~~~~
"""

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# Note this is based off the pyopenms-docs conf.py for consistency 

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import datetime
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------
project = 'pyopenms_viz'
copyright = f'{datetime.now().year}, OpenMS Team'
author = 'OpenMS Team'
version='0.0.1'
release=version

# if the variable is not set (e.g., when building locally and not on RTD)
rtd_branch = os.environ.get('READTHEDOCS_GIT_IDENTIFIER', '')
if not rtd_branch:
    release += 'local'

# if not built from release branch or tag
elif not rtd_branch.startswith('release') and not rtd_branch.startswith('Release'):
    release += 'dev'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.napoleon', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary', 'sphinx.ext.autosectionlabel', 'sphinx_copybutton', 'sphinx_gallery.gen_gallery']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True
autosectionlabel_prefix_document = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "switcher": {
        "json_url": "https://raw.githubusercontent.com/OpenMS/pyopenms-docs/master/docs/source/_static/switcher.json",
        "version_match": release
    },
    #"navbar_end": ["navbar-run-binder", "navbar-icon-links", "version-switcher"],
    "navbar_persistent": [], # default: ["search-button"] but we don't need it since we use the search bar in the sidebar
    "use_edit_page_button": True,
    "logo": {
        "alt_text": "pyOpenMS Documentation - Home",
    },
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/OpenMS/pyopenms_viz",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pyopenms-viz",
            "icon": "pypi", # defined in our custom.css
        },
        {
            "name": "OpenMS project",
            "url": "https://www.openms.de",
            "icon": "_static/OpenMS.svg",
            "type": "local",
            "attributes": {"target": "_blank"},
        },
    ],
    "show_nav_level": 1
}

html_context = {
    # "github_url": "https://github.com", # or your GitHub Enterprise site
    "github_user": "OpenMS",
    "github_repo": "pyopenms_viz",
    "github_version": "main",
    "doc_path": "docs/source/",
}

html_sidebars = {
    "**": ["search-field.html", "sidebar-nav-bs.html"]
}

html_favicon = 'img/pyOpenMSviz_logo_color.png'
html_logo = 'img/pyOpenMSviz_logo_color.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# adding custom icons probably only possible with newest pydata theme version
#html_js_files = ["piwik.js", "custom.js"] #, "custom_icon.js"]

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'pyopenms_viz_docs'

templates_path=['templates']

html_static_path = ['_static']

numfig = True

def setup(app):
    app.add_css_file("custom.css") 

sphinx_gallery_conf = {
        'examples_dirs': 'gallery_scripts',
        'gallery_dirs': 'gallery' }


# --- Always execute notebooks -------------
#nbsphinx_execute = 'always'
