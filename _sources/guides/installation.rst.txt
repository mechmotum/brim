.. _installation:

============
Installation
============

Installation Users
==================

BRiM is currently not available on PyPI. Therefore, you'll need to install the
development version from GitHub using: ::

    pip install git+https://github.com/mechmotum/brim.git

The optional dependencies can be installed with: ::

    pip install git+https://github.com/moorepants/BicycleParameters.git
    pip install symmeplot

Installation Developers
=======================

BRiM uses `poetry`_ as package manager. To install BRiM after installing `poetry`_ and
cloning the repository, run: ::

    poetry install

BRiM offers dependency groups to assist developers:

- ``lint``: packages required for linting.
- ``test``: packages required for testing.
- ``docs``: packages required for building the documentation.

To install optional dependencies from a specific group, run: ::

    poetry install --with <group>

Some of the additional utilities also require extra packages. These can be installed
using: ::

    poetry install --extras parametrize
    poetry install --extras plotting

To install everything at once, run: ::

    poetry install --with lint,test,docs --all-extras

To quickly check code for linting errors, it is recommended to set up ``pre-commit``
hooks by executing: ::

    pre-commit install

.. _poetry: https://python-poetry.org
