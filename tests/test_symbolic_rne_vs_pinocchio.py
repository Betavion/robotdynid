from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from robotdynid.core import extract_standard_parameter_values
from robotdynid.io import load_robot_from_urdf
from robotdynid.numeric.pinocchio_backend import build_pinocchio_model, compute_joint_torque_regressor
from robotdynid.symbolic import SymbolicBuildOptions, build_standard_regressor


URDF_TEXT = """\
<robot name="two_joint_rne">
  <link name="base_link"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>
  <link name="link1">
    <inertial>
      <origin xyz="0.1 0 0" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.3" iyz="0.0" izz="0.4"/>
    </inertial>
  </link>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0.6 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>
  <link name="link2">
    <inertial>
      <origin xyz="0 0.05 0.02" rpy="0 0 0"/>
      <mass value="1.5"/>
      <inertia ixx="0.1" ixy="0.01" ixz="0.0" iyy="0.2" iyz="0.02" izz="0.25"/>
    </inertial>
  </link>
</robot>
"""


class SymbolicRneVsPinocchioTests(unittest.TestCase):
    def _load_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            urdf_path = Path(tmpdir) / "robot.urdf"
            urdf_path.write_text(URDF_TEXT, encoding="utf-8")
            robot = load_robot_from_urdf(urdf_path)
            pin_bundle = build_pinocchio_model(urdf_path)
            return robot, pin_bundle

    def test_regressor_matches_pinocchio(self) -> None:
        robot, pin_bundle = self._load_models()
        bundle = build_standard_regressor(
            robot,
            SymbolicBuildOptions(enabled_joint_dynamics_groups=tuple(), include_qds=False),
        )

        q = np.array([0.3, -0.2], dtype=float)
        qd = np.array([0.5, -0.4], dtype=float)
        qdd = np.array([0.2, 0.1], dtype=float)
        theta_std = extract_standard_parameter_values(robot)

        substitutions = {}
        substitutions.update(zip(bundle.context.q, q))
        substitutions.update(zip(bundle.context.qd, qd))
        substitutions.update(zip(bundle.context.qdd, qdd))
        substitutions.update(zip(bundle.context.standard_params, theta_std))

        tau_symbolic = np.array(bundle.inverse_dynamics.tau_total.subs(substitutions), dtype=float).reshape(robot.dof)
        regressor_symbolic = np.array(bundle.regressor.subs(substitutions), dtype=float)

        regressor_pin = compute_joint_torque_regressor(
            pin_bundle,
            q=q,
            v=qd,
            a=qdd,
            reorder_to_project=True,
        )
        tau_pin = regressor_pin @ theta_std

        np.testing.assert_allclose(regressor_symbolic, regressor_pin, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(tau_symbolic, tau_pin, atol=1e-9, rtol=1e-9)
