from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sympy as sp

from robotdynid import load_robot_from_urdf
from robotdynid.symbolic import (
    SymbolicBuildOptions,
    build_joint_dynamics_regressor,
    build_joint_dynamics_torque,
    build_symbolic_context,
    simplify_matrix_entries,
)


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
        self.assertEqual(tuple(str(symbol) for symbol in context.qds), ("qds1", "qds2"))
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
        self.assertEqual(regressor[1, 5], sp.sign(context.qd[1]) * sp.exp(-sp.Abs(context.qd[1] / context.qds[1])))

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
