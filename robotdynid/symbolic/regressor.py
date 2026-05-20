"""Standard symbolic regressor construction."""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from robotdynid.core.robot_model import RobotModel
from .program import SymbolicProgram
from .rne import InverseDynamicsBundle, build_inverse_dynamics
from .simplify import simplify_matrix_entries
from .symbols import SymbolicBuildOptions, SymbolicContext, build_symbolic_context


@dataclass(frozen=True)
class SymbolicRegressorBundle:
    """All symbolic artifacts needed by downstream identification/codegen."""

    context: SymbolicContext
    inverse_dynamics: InverseDynamicsBundle
    regressor_program: sp.Matrix
    linear_parameter_names: tuple[str, ...]
    program: SymbolicProgram
    _regressor: sp.Matrix | None = field(default=None, repr=False, compare=False)

    @property
    def regressor(self) -> sp.Matrix:
        """Expanded regressor, resolved lazily for compatibility."""
        if self._regressor is None:
            object.__setattr__(self, "_regressor", self.program.resolve_matrix(self.regressor_program))
        return self._regressor


def build_standard_regressor(
    robot: RobotModel,
    options: SymbolicBuildOptions = SymbolicBuildOptions(),
    *,
    simplify: bool = False,
) -> SymbolicRegressorBundle:
    """Build symbolic inverse dynamics and the standard linear regressor."""
    context = build_symbolic_context(robot, options)
    dynamics = build_inverse_dynamics(robot, context, options)
    regressor_program = dynamics.program.jacobian_matrix(dynamics.tau_total_program, sp.Matrix(context.linear_params))
    regressor = None
    if simplify:
        regressor = simplify_matrix_entries(dynamics.program.resolve_matrix(regressor_program), trig=True)
    return SymbolicRegressorBundle(
        context=context,
        inverse_dynamics=dynamics,
        regressor_program=regressor_program,
        linear_parameter_names=tuple(str(symbol) for symbol in context.linear_params),
        program=dynamics.program,
        _regressor=regressor,
    )
