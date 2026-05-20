from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from robotdynid import load_robot_from_urdf
from robotdynid.identify import (
    AlternatingIdentifyConfig,
    IdentificationDataset,
    identify_with_stribeck,
    solve_linear_parameters,
    solve_linear_parameters_streaming,
)
from robotdynid.symbolic import (
    SymbolicBuildOptions,
    build_joint_dynamics_regressor,
    build_regressor_evaluator,
    build_symbolic_context,
)


URDF_TEXT = """\
<robot name="two_joint_identify">
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


class IdentificationTests(unittest.TestCase):
    def _make_robot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            urdf_path = Path(tmpdir) / "robot.urdf"
            urdf_path.write_text(URDF_TEXT, encoding="utf-8")
            return load_robot_from_urdf(urdf_path)

    def test_solve_linear_parameters(self) -> None:
        robot = self._make_robot()
        options = SymbolicBuildOptions(enabled_joint_dynamics_groups=("fv", "fc", "fd"), include_stribeck_parameters=True)
        context = build_symbolic_context(robot, options)
        regressor = build_joint_dynamics_regressor(context, options)
        evaluator = build_regressor_evaluator(
            context,
            regressor,
            tuple(str(symbol) for symbol in context.joint_dynamics_params),
        )

        sample_count = 16
        q = np.stack(
            [
                np.linspace(-0.8, 0.7, sample_count),
                np.linspace(0.6, -0.4, sample_count),
            ],
            axis=1,
        )
        qd = np.stack(
            [
                np.linspace(-1.0, 1.2, sample_count),
                np.linspace(0.9, -0.8, sample_count),
            ],
            axis=1,
        )
        qdd = np.stack(
            [
                np.linspace(0.4, -0.3, sample_count),
                np.linspace(-0.2, 0.5, sample_count),
            ],
            axis=1,
        )

        linear_parameters_true = np.array([0.3, -0.2, 0.4, 0.15, -0.08, 0.12], dtype=float)
        stribeck_parameters_true = np.array([0.35, 0.5], dtype=float)
        tau = np.vstack(
            [
                evaluator.predict_tau(
                    q[i],
                    qd[i],
                    qdd[i],
                    linear_parameters_true,
                    stribeck_parameters=stribeck_parameters_true,
                )
                for i in range(sample_count)
            ]
        )
        dataset = IdentificationDataset(q=q, qd=qd, qdd=qdd, tau=tau)
        result = solve_linear_parameters(dataset, evaluator, stribeck_parameters=stribeck_parameters_true)
        np.testing.assert_allclose(result.linear_parameters, linear_parameters_true, atol=1e-10, rtol=1e-10)

    def test_identify_with_stribeck(self) -> None:
        robot = self._make_robot()
        options = SymbolicBuildOptions(enabled_joint_dynamics_groups=("fv", "fc", "fd"), include_stribeck_parameters=True)
        context = build_symbolic_context(robot, options)
        regressor = build_joint_dynamics_regressor(context, options)
        evaluator = build_regressor_evaluator(
            context,
            regressor,
            tuple(str(symbol) for symbol in context.joint_dynamics_params),
        )

        sample_count = 32
        t = np.linspace(0.1, 1.9, sample_count)
        q = np.stack([0.7 * np.sin(t), 0.5 * np.cos(1.2 * t)], axis=1)
        qd = np.stack([0.7 * np.cos(t), -0.6 * np.sin(1.2 * t)], axis=1)
        qdd = np.stack([-0.7 * np.sin(t), -0.72 * np.cos(1.2 * t)], axis=1)

        linear_parameters_true = np.array([0.25, -0.1, 0.5, 0.18, -0.12, 0.22], dtype=float)
        stribeck_parameters_true = np.array([0.45, 0.3], dtype=float)
        tau = np.vstack(
            [
                evaluator.predict_tau(
                    q[i],
                    qd[i],
                    qdd[i],
                    linear_parameters_true,
                    stribeck_parameters=stribeck_parameters_true,
                )
                for i in range(sample_count)
            ]
        )
        dataset = IdentificationDataset(q=q, qd=qd, qdd=qdd, tau=tau)
        result = identify_with_stribeck(
            dataset,
            evaluator,
            AlternatingIdentifyConfig(
                stribeck_parameter_init=np.array([0.2, 0.6], dtype=float),
                max_iterations=12,
                optimizer_kwargs={"ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12},
            ),
        )

        self.assertEqual(result.linear_parameters.shape, linear_parameters_true.shape)
        self.assertEqual(result.stribeck_parameters.shape, stribeck_parameters_true.shape)
        self.assertLess(np.linalg.norm(result.linear_parameters - linear_parameters_true), 1e-5)
        self.assertLess(np.linalg.norm(result.stribeck_parameters - stribeck_parameters_true), 5e-3)

    def test_solve_linear_parameters_streaming(self) -> None:
        robot = self._make_robot()
        options = SymbolicBuildOptions(enabled_joint_dynamics_groups=("fv", "fc", "fd"), include_stribeck_parameters=True)
        context = build_symbolic_context(robot, options)
        regressor = build_joint_dynamics_regressor(context, options)
        evaluator = build_regressor_evaluator(
            context,
            regressor,
            tuple(str(symbol) for symbol in context.joint_dynamics_params),
        )

        sample_count = 20
        q = np.stack([np.linspace(-0.4, 0.5, sample_count), np.linspace(0.3, -0.2, sample_count)], axis=1)
        qd = np.stack([np.linspace(-1.1, 1.0, sample_count), np.linspace(0.8, -0.7, sample_count)], axis=1)
        qdd = np.stack([np.linspace(0.3, -0.4, sample_count), np.linspace(-0.2, 0.4, sample_count)], axis=1)
        linear_parameters_true = np.array([0.3, -0.2, 0.4, 0.15, -0.08, 0.12], dtype=float)
        stribeck_parameters_true = np.array([0.35, 0.5], dtype=float)
        tau = np.vstack(
            [
                evaluator.predict_tau(
                    q[i],
                    qd[i],
                    qdd[i],
                    linear_parameters_true,
                    stribeck_parameters=stribeck_parameters_true,
                )
                for i in range(sample_count)
            ]
        )
        dataset = IdentificationDataset(q=q, qd=qd, qdd=qdd, tau=tau)

        dense = solve_linear_parameters(dataset, evaluator, stribeck_parameters=stribeck_parameters_true)
        streamed = solve_linear_parameters_streaming(
            dataset,
            evaluator,
            stribeck_parameters=stribeck_parameters_true,
            chunk_size=7,
        )
        np.testing.assert_allclose(streamed.linear_parameters, dense.linear_parameters, atol=1e-10, rtol=1e-10)
