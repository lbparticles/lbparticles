import importlib.metadata

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "myst_nb",
    "IPython.sphinxext.ipython_console_highlighting",
]
master_doc = "index"

# General information about the project.
project = "LBParticles"
copyright = "2023 John Forbes & Contributors"

version = importlib.metadata.version('LBParticles')
release = importlib.metadata.version('LBParticles')

exclude_patterns = ["_build"]
html_theme = "sphinx_book_theme"
html_title = "LBParticles"
html_static_path = ["_static"]
html_show_sourcelink = False
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/LBParticles/LBParticles",
    "repository_branch": "main",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com/",
        "notebook_interface": "jupyterlab",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}
html_baseurl = "https://lbparticles.readthedocs.io/en/latest/"
nb_execution_mode = "force"
html_sourcelink_suffix = ""