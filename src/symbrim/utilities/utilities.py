"""Utilities for SymBRiM."""
from __future__ import annotations

import numpy as np
from sympy import Basic, Derivative, Dummy, Expr, lambdify
from sympy.core.random import random
from sympy.physics.mechanics import ReferenceFrame, Vector, find_dynamicsymbols, msubs

__all__ = ["random_eval", "check_zero", "express_single_component_towards"]


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
    if method == "evalf":
        return round(expr.evalf(prec, {s: random() for s in free}), prec)
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
        np.fromfunction(lambda _: f(*rng.random(len(free))), (n_evaluations,)),
        np.zeros(n_evaluations), 0, atol)


def express_single_component_towards(vector: Vector, frame: ReferenceFrame) -> Vector:
    """Express a vector closer to the frame of interest.

    This function can be used to simplify computations by expressing a single-component
    vector in a frame that is closer to the desired frame. A single-component vector is
    defined as a vector that has only one non-zero component in its representation. The
    function iteratively expresses the vector in intermediate frames until it reaches a
    frame that is closer to the desired frame or until it is no longer a
    single-component vector.

    Parameters
    ----------
    vector : Vector
        The single-component vector to express in a different frame.
    frame : ReferenceFrame
        The frame towards which to express the single-component vector.
    """
    if len(vector.args) != 1:
        return vector  # Not a basis vector
    flist = vector.args[0][1]._dict_list(frame, 0)
    out = vector
    for f in flist:
        new_out = out.express(f)
        n_zeros = 0
        for elem in new_out.args[0][0]:
            if elem == 0:
                n_zeros += 1
        if n_zeros < 2:
            break   # Not a basis vector anymore, so stop expressing
        out = new_out
    return out
