from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from robotdynid.numeric import build_pinocchio_regressor_evaluator
from robotdynid.numeric.pinocchio_backend import build_pinocchio_model


URDF_TEXT = """\
<robot name="two_joint_numeric">
  <link name="base_link"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>
  <link name="link1">
    <inertial>
      <origin xyz="0.1 0 0" rpy="0 0 0"/>
      <mass value="1.5"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.25" iyz="0.0" izz="0.35"/>
    </inertial>
  </link>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0.5 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>
  <link name="link2">
    <inertial>
      <origin xyz="0.0 0.04 0.01" rpy="0 0 0"/>
      <mass value="1.2"/>
      <inertia ixx="0.1" ixy="0.01" ixz="0.0" iyy="0.18" iyz="0.02" izz="0.23"/>
    </inertial>
  </link>
</robot>
"""


class NumericRegressorEvaluatorTests(unittest.TestCase):
    def test_build_pinocchio_regressor_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            urdf_path = Path(tmpdir) / "robot.urdf"
            urdf_path.write_text(URDF_TEXT, encoding="utf-8")
            pin_bundle = build_pinocchio_model(urdf_path)

        evaluator = build_pinocchio_regressor_evaluator(
            pin_bundle,
            enabled_joint_dynamics_groups=("fv", "fc", "fd"),
        )
        q = np.array([0.2, -0.1], dtype=float)
        qd = np.array([0.4, 0.3], dtype=float)
        qdd = np.array([0.1, -0.2], dtype=float)
        qds = np.array([0.5, 0.6], dtype=float)
        regressor = evaluator.evaluate_regressor(q, qd, qdd, qds=qds)
        self.assertEqual(regressor.shape[0], 2)
        self.assertEqual(regressor.shape[1], len(evaluator.linear_parameter_names))
        self.assertEqual(len(evaluator.linear_parameter_names), 20 + 6)
