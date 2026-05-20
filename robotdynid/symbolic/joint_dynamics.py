"""Symbolic joint-dynamics terms that remain linear in identification parameters."""

from __future__ import annotations

import sympy as sp

from .symbols import SymbolicBuildOptions, SymbolicContext


def build_joint_dynamics_regressor(
    context: SymbolicContext,
    options: SymbolicBuildOptions = SymbolicBuildOptions(),
) -> sp.Matrix:
    """Build the linear joint-dynamics regressor block in canonical group order."""
    dof = len(context.qd)
    columns: list[sp.Matrix] = []
    groups = options.enabled_joint_dynamics_groups
    stribeck_parameters = context.stribeck_parameters

    if "ia" in groups:
        for index in range(dof):
            column = sp.zeros(dof, 1)
            column[index, 0] = context.qdd[index]
            columns.append(column)

    if "fv" in groups:
        for index in range(dof):
            column = sp.zeros(dof, 1)
            column[index, 0] = context.qd[index]
            columns.append(column)

    if "fc" in groups:
        for index in range(dof):
            column = sp.zeros(dof, 1)
            column[index, 0] = sp.sign(context.qd[index])
            columns.append(column)

    if "fd" in groups:
        if len(stribeck_parameters) != dof:
            raise ValueError("Stribeck parameter symbols must be present when the fd group is enabled.")
        for index in range(dof):
            column = sp.zeros(dof, 1)
            velocity = context.qd[index]
            column[index, 0] = sp.sign(velocity) * sp.exp(-sp.Abs(velocity / stribeck_parameters[index]))
            columns.append(column)

    if "fo" in groups:
        for index in range(dof):
            column = sp.zeros(dof, 1)
            column[index, 0] = sp.Integer(1)
            columns.append(column)

    return sp.Matrix.hstack(*columns) if columns else sp.zeros(dof, 0)


def build_joint_dynamics_torque(
    context: SymbolicContext,
    options: SymbolicBuildOptions = SymbolicBuildOptions(),
) -> sp.Matrix:
    """Build the symbolic joint-dynamics torque vector."""
    if not context.joint_dynamics_params:
        return sp.zeros(len(context.qd), 1)

    regressor = build_joint_dynamics_regressor(context, options=options)
    params = sp.Matrix(context.joint_dynamics_params)
    return regressor * params
