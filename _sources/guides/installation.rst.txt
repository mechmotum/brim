.. _installation:

============
Installation
============

Installation Users
==================

SymBRiM is currently available on PyPI. ::

    pip install symbrim

The optional dependencies can be installed with: ::

    pip install bicycleparameters
    pip install symmeplot

Installation Developers
=======================

SymBRiM uses `uv`_ as package manager. To install SymBRiM after installing `uv`_
and cloning the repository, run: ::

    uv sync

SymBRiM offers dependency groups to assist developers, easiest is to just install
all of them: ::

    uv sync --all-extras

To quickly check code for linting errors, it is recommended to set up ``pre-commit``
hooks by executing: ::

    uvx pre-commit install

.. _uv: https://docs.astral.sh/uv/
