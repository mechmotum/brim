from __future__ import annotations

import pytest
from sympy import S, acos, cos, sqrt, symbols
from sympy.abc import a, b, c
from sympy.physics.mechanics import ReferenceFrame, dynamicsymbols

from symbrim.utilities.utilities import (
    check_zero,
    express_basis_vector_towards,
    random_eval,
)


class TestRandomEval:
    @pytest.mark.parametrize("method", ["lambdify", "evalf"])
    @pytest.mark.parametrize("expr", [
        dynamicsymbols("x"),
        sum(symbols("a:z")),
        dynamicsymbols("x").diff(),
    ])
    def test_non_zero(self, expr, method) -> None:
        assert float(random_eval(expr, method=method)) != 0.0

    @pytest.mark.parametrize("method", ["lambdify", "evalf"])
    @pytest.mark.parametrize("expr", [
        sqrt(dynamicsymbols("x") ** 2) - dynamicsymbols("x"),
    ])
    def test_zero(self, expr, method) -> None:
        assert float(random_eval(expr, method=method)) == 0.0

    @pytest.mark.parametrize("method", ["lambdify", "evalf"])
    @pytest.mark.parametrize("expr", [3, 3.3])
    def test_non_expression(self, expr, method) -> None:
        assert random_eval(expr, method=method) == expr

    def test_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            random_eval(symbols("a"), method="not_implemented")


class TestCheckZero:
    @pytest.mark.parametrize("expr", [
        acos(cos(a)) - a + sqrt(b**2) - b + sqrt(c**2) - c,
        sqrt(dynamicsymbols("x", 1)**2) - dynamicsymbols("x", 1),
        S.Zero,
        ])
    @pytest.mark.parametrize(("args", "kwargs"), [
        ((), {}),
        ((), {"n_evaluations": 100, "atol": 1e-10}),
    ])
    def test_is_zero(self, expr, args, kwargs) -> None:
        assert check_zero(expr, *args, **kwargs)

    @pytest.mark.parametrize("expr", [
        acos(cos(a)) - a + sqrt(b**2) - b + sqrt(c**2),
        sqrt(dynamicsymbols("x", 1)**2),
        ])
    @pytest.mark.parametrize(("args", "kwargs"), [
        ((), {}),
        ((), {"n_evaluations": 100, "atol": 1e-10}),
    ])
    def test_is_not_zero(self, expr, args, kwargs) -> None:
        assert not check_zero(expr, *args, **kwargs)

    def test_too_loose_tolerance(self) -> None:
        assert check_zero(acos(cos(a)) - a + 0.001, atol=1e-2)

    def test_non_expression(self) -> None:
        assert check_zero(0.0)
        assert not check_zero(3.3)

class TestExpressBasisVectorTowards:
    def test_express_basis_vector_towards(self) -> None:
        f1 = ReferenceFrame("f1")
        f2 = ReferenceFrame("f2")
        f3 = ReferenceFrame("f3")
        with pytest.raises(ValueError, match="No connecting orientation path found"):
            express_basis_vector_towards(f1.x, f3)
        f2.orient_axis(f1, f1.x, S.Pi / 2)
        f3.orient_axis(f2, f2.y, 0.5)
        assert express_basis_vector_towards(f1.x + f1.y, f2) == f1.x + f1.y
        assert express_basis_vector_towards(f1.x + f2.x, f3) == f1.x + f2.x
        assert express_basis_vector_towards(f1.x, f2) == f2.x
        assert express_basis_vector_towards(f2.x, f1) == f1.x
        assert express_basis_vector_towards(f1.y, f2) == -f2.z
        assert express_basis_vector_towards(f1.z, f2) == f2.y
        assert express_basis_vector_towards(f1.x, f3) == f2.x
        assert express_basis_vector_towards(f1.y, f3) == -f2.z
        assert express_basis_vector_towards(f1.z, f3) == f3.y
