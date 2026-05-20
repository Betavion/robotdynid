"""Numeric regressor evaluators backed by Pinocchio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotdynid.core.params import (
    generate_joint_dynamics_parameter_names,
    generate_standard_parameter_names,
)
from robotdynid.core.robot_model import BaseParamMetadata
from .pinocchio_backend import PinocchioModelBundle, compute_joint_torque_regressor


def _as_vector(values: np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.shape[0] != size:
        raise ValueError(f"{name} must have length {size}, got {array.shape[0]}.")
    return array


def _joint_dynamics_block(
    qd: np.ndarray,
    qdd: np.ndarray,
    stribeck_parameters: np.ndarray | None,
    enabled_groups: tuple[str, ...],
) -> np.ndarray:
    dof = qd.shape[0]
    columns: list[np.ndarray] = []

    if "ia" in enabled_groups:
        for index in range(dof):
            column = np.zeros((dof,), dtype=float)
            column[index] = qdd[index]
            columns.append(column)

    if "fv" in enabled_groups:
        for index in range(dof):
            column = np.zeros((dof,), dtype=float)
            column[index] = qd[index]
            columns.append(column)

    if "fc" in enabled_groups:
        for index in range(dof):
            column = np.zeros((dof,), dtype=float)
            column[index] = np.sign(qd[index])
            columns.append(column)

    if "fd" in enabled_groups:
        if stribeck_parameters is None:
            raise ValueError("stribeck_parameters must be provided when the fd group is enabled.")
        for index in range(dof):
            column = np.zeros((dof,), dtype=float)
            velocity = qd[index]
            column[index] = np.sign(velocity) * np.exp(-abs(velocity / stribeck_parameters[index]))
            columns.append(column)

    if "fo" in enabled_groups:
        for index in range(dof):
            column = np.zeros((dof,), dtype=float)
            column[index] = 1.0
            columns.append(column)

    return np.column_stack(columns) if columns else np.zeros((dof, 0), dtype=float)


@dataclass(frozen=True)
class PinocchioRegressorEvaluator:
    """Numeric evaluator that combines Pinocchio inertial regressor and joint-dynamics columns."""

    pinocchio_bundle: PinocchioModelBundle
    dof: int
    stribeck_parameter_size: int
    inertial_parameter_names: tuple[str, ...]
    joint_dynamics_parameter_names: tuple[str, ...]
    linear_parameter_names: tuple[str, ...]
    enabled_joint_dynamics_groups: tuple[str, ...]
    base_metadata: BaseParamMetadata | None = None

    def evaluate_regressor(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
        stribeck_parameters: np.ndarray | None = None,
    ) -> np.ndarray:
        q_arr = _as_vector(q, self.dof, "q")
        qd_arr = _as_vector(qd, self.dof, "qd")
        qdd_arr = _as_vector(qdd, self.dof, "qdd")
        stribeck_arr = (
            _as_vector(stribeck_parameters, self.stribeck_parameter_size, "stribeck_parameters")
            if self.stribeck_parameter_size > 0
            else None
        )

        inertial = compute_joint_torque_regressor(
            self.pinocchio_bundle,
            q=q_arr,
            v=qd_arr,
            a=qdd_arr,
            reorder_to_project=True,
        )
        if self.base_metadata is not None:
            inertial = inertial[:, self.base_metadata.keep_indices]

        joint_dynamics = _joint_dynamics_block(
            qd=qd_arr,
            qdd=qdd_arr,
            stribeck_parameters=stribeck_arr,
            enabled_groups=self.enabled_joint_dynamics_groups,
        )
        return np.hstack([inertial, joint_dynamics]) if joint_dynamics.shape[1] > 0 else inertial

    def predict_tau(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
        linear_parameters: np.ndarray,
        stribeck_parameters: np.ndarray | None = None,
    ) -> np.ndarray:
        regressor = self.evaluate_regressor(q, qd, qdd, stribeck_parameters=stribeck_parameters)
        parameters = _as_vector(linear_parameters, len(self.linear_parameter_names), "linear_parameters")
        return regressor @ parameters


def build_pinocchio_regressor_evaluator(
    pinocchio_bundle: PinocchioModelBundle,
    *,
    enabled_joint_dynamics_groups: tuple[str, ...] = ("fv", "fc", "fd"),
    base_metadata: BaseParamMetadata | None = None,
) -> PinocchioRegressorEvaluator:
    """Build a numeric evaluator for either the standard or base inertial regressor."""
    dof = pinocchio_bundle.model.nv
    if base_metadata is None:
        inertial_names = tuple(generate_standard_parameter_names(dof))
    else:
        inertial_names = tuple(base_metadata.base_param_names)
    joint_dynamics_names = tuple(
        generate_joint_dynamics_parameter_names(
            dof,
            enabled_groups=enabled_joint_dynamics_groups,
        )
    )
    stribeck_parameter_size = dof if "fd" in enabled_joint_dynamics_groups else 0
    return PinocchioRegressorEvaluator(
        pinocchio_bundle=pinocchio_bundle,
        dof=dof,
        stribeck_parameter_size=stribeck_parameter_size,
        inertial_parameter_names=inertial_names,
        joint_dynamics_parameter_names=joint_dynamics_names,
        linear_parameter_names=inertial_names + joint_dynamics_names,
        enabled_joint_dynamics_groups=enabled_joint_dynamics_groups,
        base_metadata=base_metadata,
    )
