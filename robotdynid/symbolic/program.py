"""Internal intermediate representation for symbolic programs."""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp


def _chain_rule_derivatives(
    expr: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    variable_set: set[sp.Symbol],
    temporary_derivatives: dict[sp.Symbol, tuple[sp.Expr, ...]],
    zero_derivatives: tuple[sp.Expr, ...],
) -> tuple[sp.Expr, ...]:
    value = sp.sympify(expr)
    free_symbols = value.free_symbols
    dependency_symbols = tuple(symbol for symbol in free_symbols if symbol in temporary_derivatives)
    if not dependency_symbols and not free_symbols & variable_set:
        return zero_derivatives

    partials = {symbol: sp.diff(value, symbol) for symbol in dependency_symbols}
    row: list[sp.Expr] = []
    for variable_index, variable in enumerate(variables):
        derivative = sp.diff(value, variable)
        for symbol, partial in partials.items():
            if partial != 0:
                derivative += partial * temporary_derivatives[symbol][variable_index]
        row.append(derivative)
    return tuple(row)


@dataclass(frozen=True)
class SymbolicTemporary:
    """A named symbolic temporary."""

    symbol: sp.Symbol
    expr: sp.Expr


@dataclass(frozen=True)
class SymbolicBlock:
    """A semantically grouped chunk of symbolic temporaries and outputs."""

    name: str
    temporaries: tuple[SymbolicTemporary, ...]
    outputs: tuple[sp.Expr, ...]


@dataclass(frozen=True)
class SymbolicProgram:
    """A structured symbolic program composed of ordered blocks."""

    blocks: tuple[SymbolicBlock, ...]

    @property
    def temporary_count(self) -> int:
        return sum(len(block.temporaries) for block in self.blocks)

    @property
    def output_count(self) -> int:
        return sum(len(block.outputs) for block in self.blocks)

    @property
    def temporaries(self) -> tuple[SymbolicTemporary, ...]:
        return tuple(temp for block in self.blocks for temp in block.temporaries)

    def required_temporary_symbols(self, expressions: list[sp.Expr] | tuple[sp.Expr, ...]) -> set[sp.Symbol]:
        """Return program temporaries transitively needed by expressions."""
        temporary_by_symbol = {temp.symbol: temp for temp in self.temporaries}
        required: set[sp.Symbol] = set()
        pending: list[sp.Symbol] = []

        for expr in expressions:
            pending.extend(symbol for symbol in sp.sympify(expr).free_symbols if symbol in temporary_by_symbol)

        while pending:
            symbol = pending.pop()
            if symbol in required:
                continue
            required.add(symbol)
            pending.extend(
                dependency
                for dependency in temporary_by_symbol[symbol].expr.free_symbols
                if dependency in temporary_by_symbol and dependency not in required
            )
        return required

    def required_blocks(self, expressions: list[sp.Expr] | tuple[sp.Expr, ...]) -> tuple[SymbolicBlock, ...]:
        """Return program blocks pruned to temporaries needed by expressions."""
        required = self.required_temporary_symbols(expressions)
        blocks: list[SymbolicBlock] = []
        for block in self.blocks:
            temporaries = tuple(temp for temp in block.temporaries if temp.symbol in required)
            if temporaries:
                blocks.append(SymbolicBlock(name=block.name, temporaries=temporaries, outputs=tuple()))
        return tuple(blocks)

    def jacobian_matrix(self, expressions: sp.Matrix, variables: sp.Matrix) -> sp.Matrix:
        """Differentiate expressions while applying the chain rule through temporaries."""
        variable_list = tuple(sp.Matrix(variables))
        variable_set = set(variable_list)
        zero_derivatives = tuple(sp.Integer(0) for _ in variable_list)
        temporary_derivatives: dict[sp.Symbol, tuple[sp.Expr, ...]] = {}

        for temp in self.temporaries:
            temporary_derivatives[temp.symbol] = _chain_rule_derivatives(
                temp.expr,
                variable_list,
                variable_set,
                temporary_derivatives,
                zero_derivatives,
            )

        rows: list[list[sp.Expr]] = []
        for expr in sp.Matrix(expressions):
            rows.append(
                list(
                    _chain_rule_derivatives(
                        expr,
                        variable_list,
                        variable_set,
                        temporary_derivatives,
                        zero_derivatives,
                    )
                )
            )
        return sp.Matrix(rows)

    def resolve_expr(self, expr: sp.Expr) -> sp.Expr:
        """Expand program temporaries into a plain SymPy expression."""
        resolved = sp.sympify(expr)
        for temp in reversed(self.temporaries):
            resolved = resolved.xreplace({temp.symbol: temp.expr})
        return resolved

    def resolve_matrix(self, matrix: sp.Matrix) -> sp.Matrix:
        """Expand program temporaries into a plain SymPy matrix."""
        return sp.Matrix(matrix).applyfunc(self.resolve_expr)


@dataclass
class SymbolicBlockBuilder:
    """Mutable helper used while constructing one symbolic block."""

    name: str
    counter: list[int]
    symbol_prefix: str = "tmp"
    temporaries: list[SymbolicTemporary] = field(default_factory=list)

    def hoist_expr(self, expr: sp.Expr, *, prefix: str | None = None) -> sp.Expr:
        """Turn a non-trivial expression into a named temporary."""
        value = sp.sympify(expr)
        if value.is_Atom:
            return value
        symbol = sp.Symbol(f"{prefix or self.symbol_prefix}{self.counter[0]}", real=True)
        self.counter[0] += 1
        self.temporaries.append(SymbolicTemporary(symbol=symbol, expr=value))
        return symbol

    def hoist_matrix(self, matrix: sp.Matrix, *, prefix: str | None = None) -> sp.Matrix:
        """Hoist each non-trivial matrix entry into temporaries."""
        return sp.Matrix(matrix).applyfunc(lambda expr: self.hoist_expr(expr, prefix=prefix))

    def build(self, outputs: tuple[sp.Expr, ...] = tuple()) -> SymbolicBlock:
        return SymbolicBlock(name=self.name, temporaries=tuple(self.temporaries), outputs=tuple(outputs))


@dataclass
class SymbolicProgramBuilder:
    """Incremental builder for structured symbolic programs."""

    symbol_prefix: str = "tmp"
    _counter: list[int] = field(default_factory=lambda: [0])
    _blocks: list[SymbolicBlock] = field(default_factory=list)

    def begin_block(self, name: str) -> SymbolicBlockBuilder:
        return SymbolicBlockBuilder(name=name, counter=self._counter, symbol_prefix=self.symbol_prefix)

    def add_block(self, block: SymbolicBlock) -> None:
        self._blocks.append(block)

    def build(self) -> SymbolicProgram:
        return SymbolicProgram(blocks=tuple(self._blocks))
