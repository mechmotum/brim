.. _contributing:

============
Contributing
============

SymBRiM has been designed to be extended by the community. If you have a bug or a
feature request, please open an issue on GitHub. If you would like to contribute, feel
free to fork the repository and submit a pull request. To establish an effective
development environment, we've taken inspiration from `Hypermodern Python`_ by Claudio
Jolowicz and `Setting up Python Projects`_ by Johannes Schmidt. This page provides a
brief overview of the various tools used in the development process. For the
installation of the development environment, please refer to the
:doc:`/guides/installation` page.

Linting
=======
Linting is used to maintain a consistent code style across contributors. We've adopted
the `ruff`_ linter, because it is fast and already supports quite a lot of rules.
Quite a lot of the rules have been incorporated, as they support the goal and they are
conveniently applied through a ``pre-commit`` hook. Type hinting is only used in
function signatures, as it also improves the readability and interface. No static type
checker, like ``mypy``, is used to enforce and check types due to it resulting in just a
lot of extra complications. To run ``ruff`` manually use the following (``--fix``
automatically fixes most issues): ::

    ruff . --fix

Testing
=======
The test suite uses `pytest`_ with a code coverage of 100%. This ensures that the code
is fully tested, which is important to check regressions and to ensure that the code
works as expected. If a feature can for some reason not be fully tested, then the lines
or functions can be ignored using ``# pragma: no cover``. To run the tests use: ::

    pytest .

When generating a coverage report locally, we recommend using: ::

    pytest --cov --cov-report html

Documentation
=============
The documentation is build using `sphinx`_. As SymBRiM is expected to expand quite a lot
we have chosen to use `sphinx.ext.autodoc`_ in combination with
`sphinx.ext.autosummary`_ to generate the entire API reference including its structure
from the source code. Therefore, it is important to document classes properly. To build
the documentation you can use: ::

    docs/make.bat html

.. _ruff: https://beta.ruff.rs
.. _pytest: https://docs.pytest.org
.. _sphinx: https://www.sphinx-doc.org
.. _sphinx.ext.autodoc: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
.. _sphinx.ext.autosummary: https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html
.. _Hypermodern Python: https://cjolowicz.github.io/posts/hypermodern-python-01-setup/
.. _Setting up Python Projects: https://johschmidt42.medium.com/setting-up-python-projects-part-i-408603868c08
