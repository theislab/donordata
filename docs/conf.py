from datetime import datetime
from importlib.metadata import metadata

info = metadata("donordata")

project = "DonorData"
author = "Lucas Arnoldt, Jan Engelmann, Eva Holtkamp"
copyright = f"{datetime.now():%Y}, Theislab."
version = info["Version"]
release = info["Version"]
repository_url = "https://github.com/theislab/donordata"

templates_path = ["_templates"]
html_static_path = ["_static"]
nitpicky = False
needs_sphinx = "4.0"

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",
]

autosummary_generate = True
autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
myst_heading_anchors = 6
myst_enable_extensions = ["amsmath", "colon_fence", "deflist", "dollarmath", "html_image", "html_admonition", "linkify"]
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"

source_suffix = {".rst": "restructuredtext", ".ipynb": "myst-nb", ".myst": "myst-nb", ".md": "myst-nb"}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "mudata": ("https://mudata.scverse.org/stable/", None),
    "ehrdata": ("https://ehrdata.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/main/", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints", "tutorials/data"]

html_theme = "sphinx_book_theme"
html_title = project
html_logo = "_static/img/donordata_logo.svg"
html_favicon = "_static/img/donordata_logo_mark.svg"
ogp_social_cards = {"image": "_static/img/donordata_logo.png"}
html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
    "path_to_docs": "docs/",
    "navigation_with_keys": False,
}
pygments_style = "default"
