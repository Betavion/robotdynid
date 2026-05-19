from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from robotdynid.core import BaseParamMetadata, generate_joint_dynamics_parameter_names, generate_standard_parameter_names
from robotdynid.core.params import pinocchio_to_project_vector, project_to_pinocchio_vector
from robotdynid.io import load_robot_from_urdf
from robotdynid.numeric import BaseSelectionStrategy, select_base_parameters
from robotdynid.numeric.pinocchio_backend import build_pinocchio_model, compute_joint_torque_regressor


URDF_TEXT = """\
<robot name="two_joint_test">
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
      <origin xyz="0.1 0 0" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>

  <joint name="mount" type="fixed">
    <parent link="link1"/>
    <child link="link1_stub"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
  </joint>

  <link name="link1_stub">
    <inertial>
      <origin xyz="0.2 0 0" rpy="0 0 0"/>
      <mass value="3.0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>

  <joint name="joint2" type="revolute">
    <parent link="link1_stub"/>
    <child link="link2"/>
    <origin xyz="0.5 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>

  <link name="link2">
    <inertial>
      <origin xyz="0 0.1 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
</robot>
"""


class ParamsAndUrdfTests(unittest.TestCase):
    def test_standard_parameter_names(self) -> None:
        self.assertEqual(
            generate_standard_parameter_names(2),
            [
                "I1xx",
                "I1xy",
                "I1xz",
                "I1yy",
                "I1yz",
                "I1zz",
                "mx1",
                "my1",
                "mz1",
                "m1",
                "I2xx",
                "I2xy",
                "I2xz",
                "I2yy",
                "I2yz",
                "I2zz",
                "mx2",
                "my2",
                "mz2",
                "m2",
            ],
        )

    def test_joint_dynamics_parameter_names(self) -> None:
        self.assertEqual(
            generate_joint_dynamics_parameter_names(2, enabled_groups=("fv", "fc")),
            ["fv1", "fv2", "fc1", "fc2"],
        )

    def test_pinocchio_project_order_roundtrip(self) -> None:
        vector = np.arange(20.0)
        pinocchio_vector = project_to_pinocchio_vector(vector, body_count=2)
        project_vector = pinocchio_to_project_vector(pinocchio_vector, body_count=2)
        np.testing.assert_array_equal(project_vector, vector)

    def test_urdf_loader_collapses_fixed_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            urdf_path = Path(tmpdir) / "robot.urdf"
            urdf_path.write_text(URDF_TEXT, encoding="utf-8")
            robot = load_robot_from_urdf(urdf_path)

        self.assertEqual(robot.name, "two_joint_test")
        self.assertEqual(robot.base_frame, "base_link")
        self.assertEqual(robot.dof, 2)
        self.assertEqual(robot.joint_names, ["joint1", "joint2"])
        self.assertEqual(robot.link_names, ["link1", "link2"])
        self.assertEqual(robot.links[0].source_links, ("link1", "link1_stub"))
        self.assertEqual(robot.links[1].source_links, ("link2",))

        body1 = robot.links[0].inertia
        self.assertAlmostEqual(float(body1.mass), 5.0)
        self.assertAlmostEqual(float(body1.first_moment[0]), 3.8)
        self.assertAlmostEqual(float(body1.inertia_origin[1, 1]), 4.34)
        self.assertAlmostEqual(float(body1.inertia_origin[2, 2]), 4.34)

        joint2_translation = robot.joints[1].placement[:3, 3]
        self.assertAlmostEqual(float(joint2_translation[0]), 1.5)
        self.assertAlmostEqual(float(joint2_translation[1]), 0.0)
        self.assertAlmostEqual(float(joint2_translation[2]), 0.0)

    def test_pinocchio_backend_loads_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            urdf_path = Path(tmpdir) / "robot.urdf"
            urdf_path.write_text(URDF_TEXT, encoding="utf-8")
            bundle = build_pinocchio_model(urdf_path)
        self.assertEqual(bundle.model.name, "two_joint_test")
        regressor = compute_joint_torque_regressor(
            bundle,
            q=np.zeros(bundle.model.nq),
            v=np.zeros(bundle.model.nv),
            a=np.zeros(bundle.model.nv),
        )
        self.assertEqual(regressor.shape, (bundle.model.nv, 10 * (bundle.model.njoints - 1)))

    def test_select_base_parameters(self) -> None:
        regressor = np.array(
            [
                [1.0, 0.0, 1.0, 2.0],
                [0.0, 1.0, 1.0, 2.0],
                [1.0, 1.0, 2.0, 4.0],
            ]
        )
        metadata = select_base_parameters(
            regressor,
            standard_param_names=["p1", "p2", "p3", "p4"],
            strategy=BaseSelectionStrategy(scale_columns=True),
        )
        self.assertIsInstance(metadata, BaseParamMetadata)
        self.assertEqual(metadata.rank, 2)
        self.assertEqual(metadata.base_param_names, ["bip01", "bip02"])
        self.assertEqual(sorted(metadata.keep_indices + metadata.dependent_indices), [0, 1, 2, 3])

        keep = regressor[:, metadata.keep_indices]
        dep = regressor[:, metadata.dependent_indices]
        reconstructed = keep @ metadata.dependency_matrix
        np.testing.assert_allclose(reconstructed, dep, atol=1e-10)
