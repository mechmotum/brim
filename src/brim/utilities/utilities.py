"""Utilities for brim."""
from __future__ import annotations

import numpy as np
from sympy import Basic, Derivative, Dummy, Expr, lambdify
from sympy.core.random import random
from sympy.physics.mechanics import find_dynamicsymbols, msubs

__all__ = ["random_eval", "check_zero"]


def random_eval(expr: Expr, prec: int = 7, method: str = "lambdify") -> float:
    """Evaluate an expression with random values."""
    if not isinstance(expr, Basic):
        return expr
    free = tuple(expr.free_symbols.union(find_dynamicsymbols(expr)))
    if method == "lambdify":
        if any(isinstance(f, Derivative) for f in free):
            dummy_map = {f: Dummy() for f in free if isinstance(f, Derivative)}
            free = tuple(dummy_map.get(f, f) for f in free)
            expr = msubs(expr, dummy_map)
        return round(lambdify(free, expr, cse=True)(*(random() for _ in free)), prec)
    elif method == "evalf":
        return round(expr.evalf(prec, {s: random() for s in free}), prec)
    else:
        raise NotImplementedError(f"Method {method} not implemented.")


def check_zero(expr: Expr, n_evaluations: int = 10, atol: float = 1e-8) -> bool:
    """Check if an expression is zero based on random evaluations.

    Explanation
    -----------
    This function evaluates the given expression with randomly generated values for the
    free symbols and checks if the result is zero. To minimize the chances of false
    positives, the evaluation is performed multiple times. However, it should be noted
    that false negatives can still occur. Examples are when values are close to zero
    or functions like the Dirac function are used, which is likely to evaluate to zero.

    Parameters
    ----------
    expr : Expr
        The expression to be checked.
    n_evaluations : int, optional
        The number of evaluations to be performed. Default is 10.
    atol : float, optional
        The absolute tolerance used for comparison. Default is 1e-8.

    Returns
    -------
    bool
        Returns True if the expression evaluates to zero, and False otherwise.

    """
    if not isinstance(expr, Basic):
        return expr == 0
    free = tuple(expr.free_symbols.union(find_dynamicsymbols(expr)))
    if any(isinstance(f, Derivative) for f in free):
        dummy_map = {f: Dummy() for f in free if isinstance(f, Derivative)}
        free = tuple(dummy_map.get(f, f) for f in free)
        expr = msubs(expr, dummy_map)
    f = lambdify(free, expr, cse=True)
    # The comparison is to zero, so the relative tolerance is not used.
    rng = np.random.default_rng()
    return np.allclose(
        np.fromfunction(lambda i: f(*rng.random(len(free))), (n_evaluations,)),
        np.zeros(n_evaluations), 0, atol)
