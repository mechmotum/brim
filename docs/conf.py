"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""
import sys
from pathlib import Path

# Add source folder to path for autodoc
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path("../src").resolve()))

from process_tutorials import main as process_tutorials

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "SymBRiM"
copyright = "2023, Timo Stienstra"  # noqa: A001
author = "Timo Stienstra"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinxcontrib.bibtex",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "tutorials/exercises/*.ipynb"]

napoleon_numpy_docstring = True
napoleon_custom_sections = [("Explanation", "notes_style")]

# Enable nitpicky mode to find broken references
nitpicky = True

# Ignore external references that cannot be resolved without intersphinx
nitpick_ignore = [
    # SymPy references (external library)
    ("py:mod", "sympy"),
    ("py:mod", "sympy.physics.mechanics"),
    ("py:class", "sympy.physics.mechanics.system.System"),
    ("py:class", "sympy.physics.mechanics.kane.KanesMethod"),
    ("py:class", "sympy.core.expr.Expr"),
    ("py:class", "sympy.core.basic.Basic"),
    ("py:class", "sympy.core.symbol.Symbol"),
    ("py:class", "sympy.core.function.Function"),
    ("py:class", "sympy.physics.vector.frame.ReferenceFrame"),
    ("py:class", "sympy.physics.vector.point.Point"),
    ("py:class", "sympy.physics.vector.vector.Vector"),
    ("py:class", "sympy.physics.mechanics.rigidbody.RigidBody"),
    ("py:class", "sympy.physics.mechanics.loads.Force"),
    ("py:class", "sympy.physics.mechanics.loads.Torque"),
    ("py:class", "Expr"),
    ("py:class", "Symbol"),
    ("py:class", "Point"),
    ("py:class", "Vector"),
    ("py:class", "KanesMethod"),
    ("py:class", "System"),
    ("py:meth", "sympy.physics.mechanics.system.System.form_eoms"),
    ("py:meth", "sympy.physics.mechanics.system.System.validate_system"),
    # BicycleParameters references (external library)
    ("py:class", "Bicycle"),
    # SymMePlot references (external library)
    ("py:class", "symmeplot.matplotlib.plot_objects.PlotFrame"),
    ("py:class", "symmeplot.matplotlib.plot_base.MplPlotBase"),
    # Python standard library references
    ("py:class", "abc.ABCMeta"),
    ("py:class", "abc.abstractmethod"),
    ("py:class", "abc.ABC"),
    ("py:class", "optional"),  # Type hint that Sphinx doesn't recognize
    ("py:class", "function"),  # Generic function type
    # Internal references that may not be resolved due to autosummary issues
    ("py:class", "PlotModel"),  # From symbrim.utilities.plotting
    ("py:class", "LoadBase"),  # Generic load base class
    ("py:class", "T_position"),  # Type alias
    ("py:meth", "to_system"),  # Generic method reference
    # Private methods that are intentionally documented
    ("py:meth", "symbrim.core.base_classes.BrimBase._add_prefix"),
]

intersphinx_mapping = {
    "sympy": ("https://docs.sympy.org/dev/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "symmeplot": ("https://tjstienstra.github.io/symmeplot/", None),
    "py3": ("https://docs.python.org/3", None),
}

bibtex_bibfiles = ["references.bib"]

# Allow errors in notebooks (some tutorials require scipy which may not be installed)
nbsphinx_allow_errors = True

# Run process_tutorials.py to convert notebooks to create a zip file with exercises.
# Note: Disabled during build as it requires conda environment
# process_tutorials()

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

flavicon_html = "_static/favicon.ico"
