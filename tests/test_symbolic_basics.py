from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sympy as sp

from robotdynid import load_robot_from_urdf
from robotdynid.symbolic import (
    build_inverse_dynamics,
    SymbolicBuildOptions,
    build_joint_dynamics_regressor,
    build_joint_dynamics_torque,
    build_standard_regressor,
    build_symbolic_context,
    simplify_matrix_entries,
)
from robotdynid.symbolic.program import SymbolicProgramBuilder


URDF_TEXT = """\
<robot name="two_joint_symbolic">
  <link name="base_link"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>
  <link name="link1">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>
  <link name="link2">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
</robot>
"""


class SymbolicBasicsTests(unittest.TestCase):
    def _load_robot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            urdf_path = Path(tmpdir) / "robot.urdf"
            urdf_path.write_text(URDF_TEXT, encoding="utf-8")
            return load_robot_from_urdf(urdf_path)

    def test_build_symbolic_context(self) -> None:
        robot = self._load_robot()
        options = SymbolicBuildOptions(enabled_joint_dynamics_groups=("fv", "fc", "fd"))
        context = build_symbolic_context(robot, options)

        self.assertEqual(tuple(str(symbol) for symbol in context.q), ("q1", "q2"))
        self.assertEqual(tuple(str(symbol) for symbol in context.qd), ("qd1", "qd2"))
        self.assertEqual(tuple(str(symbol) for symbol in context.qdd), ("qdd1", "qdd2"))
        self.assertIsNone(context.q[0].is_positive)
        self.assertIsNone(context.qd[0].is_positive)
        self.assertIsNone(context.qdd[0].is_positive)
        self.assertEqual(tuple(str(symbol) for symbol in context.stribeck_parameters), ("stribeck1", "stribeck2"))
        self.assertTrue(context.stribeck_parameters[0].is_positive)
        self.assertEqual(tuple(str(symbol) for symbol in context.standard_params[:4]), ("I1xx", "I1xy", "I1xz", "I1yy"))
        self.assertEqual(tuple(str(symbol) for symbol in context.joint_dynamics_params), ("fv1", "fv2", "fc1", "fc2", "fd1", "fd2"))

    def test_joint_dynamics_regressor(self) -> None:
        robot = self._load_robot()
        options = SymbolicBuildOptions(enabled_joint_dynamics_groups=("fv", "fc", "fd"))
        context = build_symbolic_context(robot, options)
        regressor = build_joint_dynamics_regressor(context, options)
        self.assertEqual(regressor.shape, (2, 6))
        self.assertEqual(regressor[0, 0], context.qd[0])
        self.assertEqual(regressor[1, 1], context.qd[1])
        self.assertEqual(regressor[0, 2], sp.sign(context.qd[0]))
        self.assertEqual(regressor[1, 5], sp.sign(context.qd[1]) * sp.exp(-sp.Abs(context.qd[1] / context.stribeck_parameters[1])))

    def test_joint_dynamics_torque(self) -> None:
        robot = self._load_robot()
        options = SymbolicBuildOptions(enabled_joint_dynamics_groups=("fv", "fc"))
        context = build_symbolic_context(robot, options)
        torque = build_joint_dynamics_torque(context, options)
        self.assertEqual(torque.shape, (2, 1))
        self.assertEqual(
            torque[0],
            context.qd[0] * context.joint_dynamics_params[0] + sp.sign(context.qd[0]) * context.joint_dynamics_params[2],
        )

    def test_simplify_matrix_entries(self) -> None:
        q = sp.symbols("q", real=True)
        matrix = sp.Matrix([[sp.sin(q) ** 2 + sp.cos(q) ** 2, (q**2 - 1) / (q - 1)]])
        simplified = simplify_matrix_entries(matrix, trig=True)
        self.assertEqual(simplified[0, 0], 1)
        self.assertEqual(simplified[0, 1], q + 1)

    def test_symbolic_program_jacobian_matches_resolved_matrix(self) -> None:
        x, y, z = sp.symbols("x y z", real=True)
        builder = SymbolicProgramBuilder(symbol_prefix="t")
        block = builder.begin_block("synthetic")
        t0 = block.hoist_expr(x * y + sp.sin(z))
        t1 = block.hoist_expr(t0**2 + y)
        outputs = sp.Matrix([t1 + x * sp.cos(t0), t0 * y])
        builder.add_block(block.build(tuple(outputs)))
        program = builder.build()

        variables = sp.Matrix([x, y])
        actual = program.jacobian_matrix(outputs, variables)
        expected = program.resolve_matrix(outputs).jacobian(variables)

        difference = (program.resolve_matrix(actual) - expected).applyfunc(sp.simplify)
        self.assertEqual(difference, sp.zeros(2, 2))

    def test_inverse_dynamics_exposes_program(self) -> None:
        robot = self._load_robot()
        context = build_symbolic_context(robot, SymbolicBuildOptions(enabled_joint_dynamics_groups=tuple(), include_stribeck_parameters=False))
        bundle = build_inverse_dynamics(robot, context, SymbolicBuildOptions(enabled_joint_dynamics_groups=tuple(), include_stribeck_parameters=False))
        self.assertGreaterEqual(len(bundle.program.blocks), robot.dof + 1)
        self.assertGreater(bundle.program.output_count, 0)
        self.assertIsNone(bundle._tau_total)
        self.assertEqual(bundle.tau_total.shape, (robot.dof, 1))
        self.assertIsNotNone(bundle._tau_total)

    def test_standard_regressor_resolves_lazily(self) -> None:
        robot = self._load_robot()
        bundle = build_standard_regressor(
            robot,
            SymbolicBuildOptions(enabled_joint_dynamics_groups=tuple(), include_stribeck_parameters=False),
        )
        self.assertIsNone(bundle.inverse_dynamics._tau_total)
        self.assertIsNone(bundle._regressor)
        self.assertEqual(bundle.regressor_program.shape, bundle.regressor.shape)
        self.assertIsNotNone(bundle._regressor)

    def test_standard_regressor_program_eliminates_linear_parameters(self) -> None:
        robot = self._load_robot()
        bundle = build_standard_regressor(
            robot,
            SymbolicBuildOptions(enabled_joint_dynamics_groups=("fv", "fc"), include_stribeck_parameters=True),
        )
        linear_parameters = set(bundle.context.linear_params)
        self.assertFalse(set().union(*(expr.free_symbols for expr in bundle.regressor_program)) & linear_parameters)
