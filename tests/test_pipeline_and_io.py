from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robotdynid.assembly import build_base_identification_pipeline
from robotdynid.codegen import export_c_code_artifacts, generate_base_regressor_c_function
from robotdynid.core import BaseParamMetadata
from robotdynid.io import load_robot_from_urdf
from robotdynid.identify import (
    CsvDatasetConfig,
    MotionTorqueCsvDatasetConfig,
    load_identification_dataset_from_csv,
    load_identification_dataset_from_motion_and_torque_csv,
)
from robotdynid.symbolic import SymbolicBuildOptions


URDF_TEXT = """\
<robot name="two_joint_pipeline">
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


class PipelineAndIoTests(unittest.TestCase):
    def _write_robot(self, tmpdir: str) -> Path:
        urdf_path = Path(tmpdir) / "robot.urdf"
        urdf_path.write_text(URDF_TEXT, encoding="utf-8")
        return urdf_path

    def test_base_param_metadata_roundtrip(self) -> None:
        metadata = BaseParamMetadata(
            rank=2,
            keep_indices=[0, 3],
            dependent_indices=[1],
            dependency_matrix=np.array([[1.25], [-0.5]], dtype=float),
            standard_param_names=["I1xx", "I1xy", "I1xz", "I1yy"],
            base_param_names=["bip01", "bip02"],
            column_permutation=[0, 3, 1],
            qr_rank=2,
            svd_rank=2,
            tolerance=1e-9,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "base.json"
            metadata.to_json_file(path)
            restored = BaseParamMetadata.from_json_file(path)
        self.assertEqual(restored.rank, metadata.rank)
        self.assertEqual(restored.keep_indices, metadata.keep_indices)
        np.testing.assert_allclose(restored.dependency_matrix, metadata.dependency_matrix)

    def test_load_identification_dataset_from_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "pos1,pos2,vel1,vel2,acc1,acc2,torque1,torque2",
                        "0.0,0.1,1.0,-1.0,0.2,-0.2,2.0,-2.0",
                        "0.3,0.4,0.5,0.6,0.7,0.8,1.1,1.2",
                    ]
                ),
                encoding="utf-8",
            )
            dataset = load_identification_dataset_from_csv(
                csv_path,
                CsvDatasetConfig(dof=2),
            )
        self.assertEqual(dataset.q.shape, (2, 2))
        self.assertAlmostEqual(dataset.q[0, 0], 0.0)
        self.assertAlmostEqual(dataset.q[0, 1], 0.1)
        self.assertEqual(dataset.sample_weights.shape, (4,))

    def test_build_pipeline_and_export_codegen(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            urdf_path = self._write_robot(tmpdir)
            robot = load_robot_from_urdf(urdf_path)

            q = np.array([[0.2, -0.1], [0.3, 0.4], [-0.5, 0.2]], dtype=float)
            qd = np.array([[0.4, 0.3], [-0.2, 0.5], [0.6, -0.1]], dtype=float)
            qdd = np.array([[0.1, 0.2], [0.3, -0.2], [0.0, 0.4]], dtype=float)
            tau = np.zeros_like(q)

            from robotdynid.identify import IdentificationDataset

            selection_dataset = IdentificationDataset(q=q, qd=qd, qdd=qdd, tau=tau)
            pipeline = build_base_identification_pipeline(
                robot,
                selection_dataset,
                selection_stribeck_parameters=np.array([0.4, 0.5], dtype=float),
                symbolic_options=SymbolicBuildOptions(enabled_joint_dynamics_groups=("fv", "fc"), include_stribeck_parameters=True),
            )

            self.assertGreaterEqual(pipeline.base_metadata.rank, 1)
            generated = generate_base_regressor_c_function(pipeline.base_bundle)
            artifact_paths = export_c_code_artifacts(generated, pipeline.base_bundle, Path(tmpdir) / "generated")
            self.assertTrue(artifact_paths.source_path.exists())
            self.assertTrue(artifact_paths.header_path.exists())
            self.assertTrue(artifact_paths.metadata_path.exists())

            metadata = json.loads(artifact_paths.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["function_name"], "fill_H_bip_base")

    def test_load_identification_dataset_from_motion_and_torque_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            motion_path = Path(tmpdir) / "motion.csv"
            torque_path = Path(tmpdir) / "torque.csv"
            motion_path.write_text(
                "\n".join(
                    [
                        "timestamp,joint1_position,joint1_velocity,joint2_position,joint2_velocity",
                        "1.0,0.0,1.0,0.1,-1.0",
                        "1.1,0.2,1.5,0.0,-0.5",
                        "1.2,0.4,2.0,-0.1,0.0",
                    ]
                ),
                encoding="utf-8",
            )
            torque_path.write_text(
                "\n".join(
                    [
                        "timestamp,joint1_measure,joint2_measure",
                        "1.0,2.0,-2.0",
                        "1.1,2.5,-1.5",
                        "1.2,3.0,-1.0",
                    ]
                ),
                encoding="utf-8",
            )
            dataset = load_identification_dataset_from_motion_and_torque_csv(
                motion_path,
                torque_path,
                MotionTorqueCsvDatasetConfig(dof=2),
            )
        self.assertEqual(dataset.q.shape, (3, 2))
        self.assertEqual(dataset.qdd.shape, (3, 2))
        self.assertEqual(dataset.tau.shape, (3, 2))
