"""A thin, predictable wrapper around SymPy CSE."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class CSEOutput:
    """Structured output of a common-subexpression elimination pass."""

    temporaries: tuple[tuple[sp.Symbol, sp.Expr], ...]
    reduced_expressions: tuple[sp.Expr, ...]


def apply_cse(expressions: list[sp.Expr] | tuple[sp.Expr, ...], *, symbol_prefix: str = "tmp") -> CSEOutput:
    """Run SymPy CSE with stable temporary naming."""
    temps, reduced = sp.cse(
        list(expressions),
        symbols=sp.numbered_symbols(symbol_prefix, real=True),
        order="none",
    )
    return CSEOutput(
        temporaries=tuple((symbol, expr) for symbol, expr in temps),
        reduced_expressions=tuple(reduced),
    )
