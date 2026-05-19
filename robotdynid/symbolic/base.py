"""Base-parameter projection from the full symbolic regressor."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from robotdynid.core.robot_model import BaseParamMetadata
from .regressor import SymbolicRegressorBundle
from .symbols import SymbolicContext


@dataclass(frozen=True)
class BaseRegressorBundle:
    """Symbolic base-parameter regressor and associated parameter expressions."""

    context: SymbolicContext
    regressor: sp.Matrix
    base_parameter_names: tuple[str, ...]
    base_parameter_expressions: sp.Matrix
    joint_dynamics_parameters: tuple[sp.Symbol, ...]
    linear_parameter_names: tuple[str, ...]


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
    return theta_keep + dependency * theta_dep


def build_base_regressor(bundle: SymbolicRegressorBundle, metadata: BaseParamMetadata) -> BaseRegressorBundle:
    """Project a full symbolic regressor onto the selected base inertial columns."""
    standard_count = len(bundle.context.standard_params)
    full_regressor = bundle.regressor
    inertial_part = full_regressor[:, :standard_count]
    joint_dynamics_part = full_regressor[:, standard_count:]
    base_inertial_part = inertial_part[:, metadata.keep_indices]
    regressor = (
        sp.Matrix.hstack(base_inertial_part, joint_dynamics_part)
        if joint_dynamics_part.shape[1] > 0
        else base_inertial_part
    )
    base_param_expressions = build_base_parameter_expressions(bundle.context.standard_params, metadata)
    return BaseRegressorBundle(
        context=bundle.context,
        regressor=regressor,
        base_parameter_names=tuple(metadata.base_param_names),
        base_parameter_expressions=base_param_expressions,
        joint_dynamics_parameters=bundle.context.joint_dynamics_params,
        linear_parameter_names=tuple(metadata.base_param_names)
        + tuple(str(symbol) for symbol in bundle.context.joint_dynamics_params),
    )
