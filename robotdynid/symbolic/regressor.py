"""Standard symbolic regressor construction."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from robotdynid.core.robot_model import RobotModel
from .rne import InverseDynamicsBundle, build_inverse_dynamics
from .simplify import simplify_matrix_entries
from .symbols import SymbolicBuildOptions, SymbolicContext, build_symbolic_context


@dataclass(frozen=True)
class SymbolicRegressorBundle:
    """All symbolic artifacts needed by downstream identification/codegen."""

    context: SymbolicContext
    inverse_dynamics: InverseDynamicsBundle
    regressor: sp.Matrix
    linear_parameter_names: tuple[str, ...]


def build_standard_regressor(
    robot: RobotModel,
    options: SymbolicBuildOptions = SymbolicBuildOptions(),
    *,
    simplify: bool = False,
) -> SymbolicRegressorBundle:
    """Build symbolic inverse dynamics and the standard linear regressor."""
    context = build_symbolic_context(robot, options)
    dynamics = build_inverse_dynamics(robot, context, options)
    regressor = dynamics.tau_total.jacobian(sp.Matrix(context.linear_params))
    if simplify:
        regressor = simplify_matrix_entries(regressor, trig=True)
    return SymbolicRegressorBundle(
        context=context,
        inverse_dynamics=dynamics,
        regressor=regressor,
        linear_parameter_names=tuple(str(symbol) for symbol in context.linear_params),
    )
