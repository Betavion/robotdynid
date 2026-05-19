from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robotdynid.workflow import IdentificationWorkflowConfig, run_identification_workflow


URDF_TEXT = """\
<robot name="two_joint_sia_app">
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

CSV_TEXT = """\
pos1,pos2,vel1,vel2,acc1,acc2,torque1,torque2
0.0,0.0,0.5,-0.5,0.1,-0.1,0.2,-0.2
0.1,-0.1,0.4,-0.4,0.0,0.0,0.1,-0.1
0.2,-0.2,0.3,-0.3,-0.1,0.1,0.0,0.0
0.3,-0.3,0.2,-0.2,0.2,-0.2,0.3,-0.3
"""


class IdentificationWorkflowTests(unittest.TestCase):
    def test_run_identification_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            urdf_path = root / "robot.urdf"
            csv_path = root / "data.csv"
            urdf_path.write_text(URDF_TEXT, encoding="utf-8")
            csv_path.write_text(CSV_TEXT, encoding="utf-8")

            payload = run_identification_workflow(
                IdentificationWorkflowConfig(
                    urdf_path=urdf_path,
                    csv_path=csv_path,
                    dof=2,
                    stride=1,
                    max_samples=4,
                    selection_samples=4,
                    selection_source="model",
                    position_offsets=(0.0, 0.0),
                    output_dir=root / "out",
                    save_prediction_plot=True,
                    qds_init=[0.2, 0.2],
                    max_iterations=2,
                    chunk_size=2,
                )
            )

            self.assertEqual(payload["sample_count"], 4)
            self.assertTrue((root / "out" / "identify_result.json").exists())
            self.assertTrue((root / "out" / "theta_lin.csv").exists())
            self.assertTrue((root / "out" / "qds_star.csv").exists())
            self.assertTrue((root / "out" / "base_metadata.json").exists())
            self.assertTrue((root / "out" / "prediction.png").exists())
