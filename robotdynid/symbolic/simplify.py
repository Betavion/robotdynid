"""Targeted simplification helpers for large symbolic expressions."""

from __future__ import annotations

import sympy as sp


def simplify_scalar(expr: sp.Expr, *, trig: bool = False) -> sp.Expr:
    """Apply predictable local simplifications without using blanket simplify()."""
    result = sp.expand(expr)
    result = sp.cancel(result)
    if trig:
        result = sp.trigsimp(result)
    return result


def simplify_matrix_entries(matrix: sp.Matrix, *, trig: bool = False) -> sp.Matrix:
    """Apply targeted simplification entry-wise to a symbolic matrix."""
    mat = sp.Matrix(matrix)
    return mat.applyfunc(lambda expr: simplify_scalar(expr, trig=trig))
