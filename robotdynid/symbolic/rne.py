"""Symbolic recursive Newton-Euler dynamics for canonical serial robots."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from robotdynid.core.robot_model import RobotModel
from .joint_dynamics import build_joint_dynamics_torque
from .spatial_math import (
    axis_angle_rotation,
    force_cross_matrix,
    motion_cross_matrix,
    motion_transform_from_homogeneous,
    spatial_inertia_matrix,
)
from .symbols import SymbolicBuildOptions, SymbolicContext


@dataclass(frozen=True)
class InverseDynamicsBundle:
    """Symbolic inverse dynamics and reusable intermediate expressions."""

    tau_inertial: sp.Matrix
    tau_joint_dynamics: sp.Matrix
    tau_total: sp.Matrix


def _symbolic_link_inertia(context: SymbolicContext, link_index: int) -> sp.Matrix:
    base = 10 * link_index
    standard = context.standard_params
    inertia_origin = sp.Matrix(
        [
            [standard[base + 0], standard[base + 1], standard[base + 2]],
            [standard[base + 1], standard[base + 3], standard[base + 4]],
            [standard[base + 2], standard[base + 4], standard[base + 5]],
        ]
    )
    first_moment = sp.Matrix([[standard[base + 6]], [standard[base + 7]], [standard[base + 8]]])
    mass = standard[base + 9]
    return spatial_inertia_matrix(mass, first_moment, inertia_origin)


def _joint_spatial_transform(joint, q_value: sp.Expr) -> sp.Matrix:
    placement = joint.placement
    rotation0 = placement[:3, :3]
    translation0 = placement[:3, 3]
    axis = joint.axis

    if joint.joint_type in ("revolute", "continuous"):
        rotation = sp.simplify(rotation0 * axis_angle_rotation(axis, q_value))
        translation = translation0
    elif joint.joint_type == "prismatic":
        rotation = rotation0
        translation = sp.simplify(translation0 + rotation0 * axis * q_value)
    else:
        raise ValueError(f"Unsupported motion joint type: {joint.joint_type}")

    transform = sp.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = sp.Matrix(translation)
    return transform


def _joint_motion_subspace(joint) -> sp.Matrix:
    if joint.joint_type in ("revolute", "continuous"):
        return sp.Matrix.vstack(joint.axis, sp.zeros(3, 1))
    if joint.joint_type == "prismatic":
        return sp.Matrix.vstack(sp.zeros(3, 1), joint.axis)
    raise ValueError(f"Unsupported motion joint type: {joint.joint_type}")


def build_inverse_dynamics(
    robot: RobotModel,
    context: SymbolicContext,
    options: SymbolicBuildOptions = SymbolicBuildOptions(),
) -> InverseDynamicsBundle:
    """Build symbolic inverse dynamics for a serial fixed-base robot."""
    if robot.dof != len(robot.links):
        raise ValueError("RobotModel must contain one dynamic body per motion joint.")
    if robot.dof != len(context.q) or robot.dof != len(context.qd) or robot.dof != len(context.qdd):
        raise ValueError("SymbolicContext does not match robot DOF.")

    dof = robot.dof
    gravity = sp.Matrix([[0], [0], [0], [-robot.gravity[0]], [-robot.gravity[1]], [-robot.gravity[2]]])

    xup: list[sp.Matrix] = [sp.eye(6)] * dof
    subspaces: list[sp.Matrix] = [sp.zeros(6, 1)] * dof
    velocities: list[sp.Matrix] = [sp.zeros(6, 1)] * dof
    accelerations: list[sp.Matrix] = [sp.zeros(6, 1)] * dof
    forces: list[sp.Matrix] = [sp.zeros(6, 1)] * dof
    inertias: list[sp.Matrix] = [sp.zeros(6, 6)] * dof

    a_parent = gravity
    v_parent = sp.zeros(6, 1)

    for index, joint in enumerate(robot.joints):
        transform = _joint_spatial_transform(joint, context.q[index])
        xup[index] = motion_transform_from_homogeneous(transform)
        subspaces[index] = _joint_motion_subspace(joint)
        inertias[index] = _symbolic_link_inertia(context, index)

        v_joint = subspaces[index] * context.qd[index]
        velocities[index] = xup[index] * v_parent + v_joint
        accelerations[index] = (
            xup[index] * a_parent
            + subspaces[index] * context.qdd[index]
            + motion_cross_matrix(velocities[index]) * v_joint
        )
        forces[index] = (
            inertias[index] * accelerations[index]
            + force_cross_matrix(velocities[index]) * inertias[index] * velocities[index]
        )

        v_parent = velocities[index]
        a_parent = accelerations[index]

    tau_inertial = [sp.Integer(0)] * dof
    for index in range(dof - 1, -1, -1):
        tau_inertial[index] = (subspaces[index].T * forces[index])[0]
        if index > 0:
            forces[index - 1] = forces[index - 1] + xup[index].T * forces[index]

    tau_inertial_matrix = sp.Matrix(tau_inertial)
    tau_joint_dynamics = build_joint_dynamics_torque(context, options)
    return InverseDynamicsBundle(
        tau_inertial=tau_inertial_matrix,
        tau_joint_dynamics=tau_joint_dynamics,
        tau_total=tau_inertial_matrix + tau_joint_dynamics,
    )
