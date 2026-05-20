"""Base-parameter projection from the full symbolic regressor."""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from robotdynid.core.robot_model import BaseParamMetadata
from .program import SymbolicProgram
from .regressor import SymbolicRegressorBundle
from .symbols import SymbolicContext


@dataclass(frozen=True)
class BaseRegressorBundle:
    """Symbolic base-parameter regressor and associated parameter expressions."""

    context: SymbolicContext
    regressor_program: sp.Matrix
    program: SymbolicProgram
    base_parameter_names: tuple[str, ...]
    base_parameter_expressions: sp.Matrix
    joint_dynamics_parameters: tuple[sp.Symbol, ...]
    linear_parameter_names: tuple[str, ...]
    _regressor: sp.Matrix | None = field(default=None, repr=False, compare=False)

    @property
    def regressor(self) -> sp.Matrix:
        """Expanded base regressor, resolved lazily for compatibility."""
        if self._regressor is None:
            object.__setattr__(self, "_regressor", self.program.resolve_matrix(self.regressor_program))
        return self._regressor


def build_base_parameter_expressions(
    standard_params: tuple[sp.Symbol, ...],
    metadata: BaseParamMetadata,
) -> sp.Matrix:
    """Build beta = theta_keep + K * theta_dep from numerical metadata."""
    theta_keep = sp.Matrix([standard_params[index] for index in metadata.keep_indices])
    theta_dep = sp.Matrix([standard_params[index] for index in metadata.dependent_indices])
    dependency = sp.Matrix(metadata.dependency_matrix)
    if dependency.shape != (metadata.rank, len(metadata.dependent_indices)):
        raise ValueError("dependency_matrix shape does not match the selected base/dependent split.")
    if not metadata.dependent_indices:
        return theta_keep
    return theta_keep + dependency * theta_dep


def build_base_regressor(bundle: SymbolicRegressorBundle, metadata: BaseParamMetadata) -> BaseRegressorBundle:
    """Project a full symbolic regressor onto the selected base inertial columns."""
    standard_count = len(bundle.context.standard_params)
    full_regressor_program = bundle.regressor_program
    inertial_part_program = full_regressor_program[:, :standard_count]
    joint_dynamics_part_program = full_regressor_program[:, standard_count:]
    base_inertial_part_program = inertial_part_program[:, metadata.keep_indices]
    regressor_program = (
        sp.Matrix.hstack(base_inertial_part_program, joint_dynamics_part_program)
        if joint_dynamics_part_program.shape[1] > 0
        else base_inertial_part_program
    )
    base_param_expressions = build_base_parameter_expressions(bundle.context.standard_params, metadata)
    return BaseRegressorBundle(
        context=bundle.context,
        regressor_program=regressor_program,
        program=bundle.program,
        base_parameter_names=tuple(metadata.base_param_names),
        base_parameter_expressions=base_param_expressions,
        joint_dynamics_parameters=bundle.context.joint_dynamics_params,
        linear_parameter_names=tuple(metadata.base_param_names)
        + tuple(str(symbol) for symbol in bundle.context.joint_dynamics_params),
    )
