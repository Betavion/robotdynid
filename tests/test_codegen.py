from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from robotdynid import load_robot_from_urdf
from robotdynid.numeric import BaseSelectionStrategy, select_base_parameters
from robotdynid.symbolic import (
    SymbolicBuildOptions,
    build_base_regressor,
    build_standard_regressor,
)
from robotdynid.codegen import generate_base_regressor_c_function, generate_prediction_c_function


URDF_TEXT = """\
<robot name="two_joint_codegen">
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


class CodegenTests(unittest.TestCase):
    def _build_base_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            urdf_path = Path(tmpdir) / "robot.urdf"
            urdf_path.write_text(URDF_TEXT, encoding="utf-8")
            robot = load_robot_from_urdf(urdf_path)

        standard_bundle = build_standard_regressor(
            robot,
            SymbolicBuildOptions(enabled_joint_dynamics_groups=("fv", "fc"), include_qds=True),
        )
        zero_subs = {**{q: 0.1 for q in standard_bundle.context.q}, **{qd: 0.2 for qd in standard_bundle.context.qd}, **{qdd: 0.3 for qdd in standard_bundle.context.qdd}, **{qds: 0.4 for qds in standard_bundle.context.qds}}
        standard_count = len(standard_bundle.context.standard_params)
        numeric_regressor = standard_bundle.regressor[:, :standard_count].subs(zero_subs)
        metadata = select_base_parameters(
            regressor_matrix=numeric_regressor,
            standard_param_names=[str(symbol) for symbol in standard_bundle.context.standard_params],
            strategy=BaseSelectionStrategy(scale_columns=True),
        )
        return build_base_regressor(standard_bundle, metadata)

    def test_generate_base_regressor_c_function(self) -> None:
        bundle = self._build_base_bundle()
        generated = generate_base_regressor_c_function(bundle)
        self.assertIn("void fill_H_bip_base", generated.source)
        self.assertIn("double *H", generated.source)
        self.assertIn("H[0] =", generated.source)

    def test_generate_prediction_c_function(self) -> None:
        bundle = self._build_base_bundle()
        generated = generate_prediction_c_function(bundle)
        self.assertIn("void predict_tau", generated.source)
        self.assertIn("const double *theta_lin", generated.source)
        self.assertIn("tau[0] =", generated.source)

    def test_generate_prediction_c_function_single_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            urdf_path = Path(tmpdir) / "robot.urdf"
            urdf_path.write_text(
                """\
<robot name="one_joint_codegen">
  <link name="base_link"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <link name="link1">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
</robot>
""",
                encoding="utf-8",
            )
            robot = load_robot_from_urdf(urdf_path)

        standard_bundle = build_standard_regressor(
            robot,
            SymbolicBuildOptions(enabled_joint_dynamics_groups=("fv",), include_qds=True),
        )
        zero_subs = {
            standard_bundle.context.q[0]: 0.2,
            standard_bundle.context.qd[0]: 0.3,
            standard_bundle.context.qdd[0]: 0.4,
            standard_bundle.context.qds[0]: 0.5,
        }
        standard_count = len(standard_bundle.context.standard_params)
        numeric_regressor = standard_bundle.regressor[:, :standard_count].subs(zero_subs)
        metadata = select_base_parameters(
            regressor_matrix=numeric_regressor,
            standard_param_names=[str(symbol) for symbol in standard_bundle.context.standard_params],
            strategy=BaseSelectionStrategy(scale_columns=True),
        )
        bundle = build_base_regressor(standard_bundle, metadata)
        generated = generate_prediction_c_function(bundle)
        self.assertIn("void predict_tau", generated.source)
