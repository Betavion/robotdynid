"""Symbolic recursive Newton-Euler dynamics for canonical serial robots."""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from robotdynid.core.robot_model import RobotModel
from .program import SymbolicBlock, SymbolicProgram, SymbolicProgramBuilder
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

    tau_inertial_program: sp.Matrix
    tau_total_program: sp.Matrix
    tau_joint_dynamics: sp.Matrix
    program: SymbolicProgram
    _tau_inertial: sp.Matrix | None = field(default=None, repr=False, compare=False)
    _tau_total: sp.Matrix | None = field(default=None, repr=False, compare=False)

    @property
    def tau_inertial(self) -> sp.Matrix:
        """Expanded inertial torque, resolved lazily for compatibility."""
        if self._tau_inertial is None:
            object.__setattr__(self, "_tau_inertial", self.program.resolve_matrix(self.tau_inertial_program))
        return self._tau_inertial

    @property
    def tau_total(self) -> sp.Matrix:
        """Expanded total torque, resolved lazily for compatibility."""
        if self._tau_total is None:
            object.__setattr__(self, "_tau_total", self.program.resolve_matrix(self.tau_total_program))
        return self._tau_total


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
        rotation = rotation0 * axis_angle_rotation(axis, q_value)
        translation = translation0
    elif joint.joint_type == "prismatic":
        rotation = rotation0
        translation = translation0 + rotation0 * axis * q_value
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
    program_builder = SymbolicProgramBuilder(symbol_prefix="k")

    for index, joint in enumerate(robot.joints):
        block = program_builder.begin_block(f"joint_{index + 1}_forward")
        transform = block.hoist_matrix(_joint_spatial_transform(joint, context.q[index]), prefix="t")
        xup[index] = block.hoist_matrix(motion_transform_from_homogeneous(transform), prefix="x")
        subspaces[index] = _joint_motion_subspace(joint)
        inertias[index] = _symbolic_link_inertia(context, index)

        v_joint = block.hoist_matrix(subspaces[index] * context.qd[index], prefix="vj")
        velocities[index] = block.hoist_matrix(xup[index] * v_parent + v_joint, prefix="v")
        velocity_cross = block.hoist_matrix(motion_cross_matrix(velocities[index]), prefix="crm")
        accelerations[index] = block.hoist_matrix(
            xup[index] * a_parent
            + subspaces[index] * context.qdd[index]
            + velocity_cross * v_joint,
            prefix="a",
        )
        force_cross = block.hoist_matrix(force_cross_matrix(velocities[index]), prefix="crf")
        inertia_acceleration = block.hoist_matrix(inertias[index] * accelerations[index], prefix="ia")
        inertia_velocity = block.hoist_matrix(inertias[index] * velocities[index], prefix="iv")
        forces[index] = (
            inertia_acceleration
            + force_cross * inertia_velocity
        )
        program_builder.add_block(block.build())

        v_parent = velocities[index]
        a_parent = accelerations[index]

    tau_inertial = [sp.Integer(0)] * dof
    for index in range(dof - 1, -1, -1):
        tau_inertial[index] = (subspaces[index].T * forces[index])[0]
        if index > 0:
            forces[index - 1] = forces[index - 1] + xup[index].T * forces[index]
        program_builder.add_block(
            SymbolicBlock(
                name=f"joint_{index + 1}_backward",
                temporaries=tuple(),
                outputs=(tau_inertial[index],),
            )
        )

    tau_inertial_program = sp.Matrix(tau_inertial)
    tau_joint_dynamics = build_joint_dynamics_torque(context, options)
    tau_total_program = tau_inertial_program + tau_joint_dynamics
    output_block = SymbolicBlock(
        name="tau_output",
        temporaries=tuple(),
        outputs=tuple(tau_inertial_program) + tuple(tau_joint_dynamics) + tuple(tau_total_program),
    )
    program_builder.add_block(output_block)
    program = program_builder.build()
    return InverseDynamicsBundle(
        tau_inertial_program=tau_inertial_program,
        tau_total_program=tau_total_program,
        tau_joint_dynamics=tau_joint_dynamics,
        program=program,
    )
