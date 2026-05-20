from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import sympy as sp

from robotdynid.core import extract_standard_parameter_values
from robotdynid.io import load_robot_from_urdf
from robotdynid.numeric import BaseSelectionStrategy, select_base_parameters
from robotdynid.symbolic import (
    SymbolicBuildOptions,
    build_base_regressor,
    build_standard_regressor,
)


URDF_TEXT = """\
<robot name="two_joint_base">
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


class SymbolicBaseProjectionTests(unittest.TestCase):
    def _load_robot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            urdf_path = Path(tmpdir) / "robot.urdf"
            urdf_path.write_text(URDF_TEXT, encoding="utf-8")
            return load_robot_from_urdf(urdf_path)

    def test_base_projection_matches_full_regressor(self) -> None:
        robot = self._load_robot()
        bundle = build_standard_regressor(
            robot,
            SymbolicBuildOptions(enabled_joint_dynamics_groups=tuple(), include_stribeck_parameters=False),
        )
        theta_std = extract_standard_parameter_values(robot)

        sample_states = [
            (np.array([0.3, -0.2]), np.array([0.5, -0.4]), np.array([0.2, 0.1])),
            (np.array([-0.6, 0.4]), np.array([0.1, 0.3]), np.array([-0.5, 0.2])),
            (np.array([0.7, -0.1]), np.array([-0.2, 0.6]), np.array([0.4, -0.3])),
            (np.array([-0.2, 0.5]), np.array([0.7, -0.1]), np.array([0.1, 0.6])),
            (np.array([0.9, 0.2]), np.array([-0.4, -0.5]), np.array([0.3, -0.2])),
        ]

        stacked_rows = []
        for q, qd, qdd in sample_states:
            substitutions = {}
            substitutions.update(zip(bundle.context.q, q))
            substitutions.update(zip(bundle.context.qd, qd))
            substitutions.update(zip(bundle.context.qdd, qdd))
            numeric_regressor = np.array(bundle.regressor.subs(substitutions), dtype=float)
            stacked_rows.append(numeric_regressor)
        stacked = np.vstack(stacked_rows)

        metadata = select_base_parameters(
            stacked,
            standard_param_names=[str(symbol) for symbol in bundle.context.standard_params],
            strategy=BaseSelectionStrategy(scale_columns=True),
        )
        base_bundle = build_base_regressor(bundle, metadata)

        q, qd, qdd = sample_states[0]
        substitutions = {}
        substitutions.update(zip(bundle.context.q, q))
        substitutions.update(zip(bundle.context.qd, qd))
        substitutions.update(zip(bundle.context.qdd, qdd))
        substitutions.update(zip(bundle.context.standard_params, theta_std))

        full_tau = np.array(bundle.inverse_dynamics.tau_total.subs(substitutions), dtype=float).reshape(robot.dof)
        beta = np.array(base_bundle.base_parameter_expressions.subs(substitutions), dtype=float).reshape(metadata.rank)
        h_base = np.array(base_bundle.regressor.subs({**dict(zip(bundle.context.q, q)), **dict(zip(bundle.context.qd, qd)), **dict(zip(bundle.context.qdd, qdd))}), dtype=float)
        projected_tau = h_base @ beta
        np.testing.assert_allclose(projected_tau, full_tau, atol=1e-9, rtol=1e-9)
