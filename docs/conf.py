from importlib.metadata import version as package_version

project = "learnergy"
copyright = "2020, Mateus Roder and Gustavo de Rosa"
author = "Mateus Roder and Gustavo de Rosa"
release = package_version("learnergy")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]
autosummary_generate = True
exclude_patterns = ["_build"]
html_theme = "alabaster"
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_member_order = "bysource"
