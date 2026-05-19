"""Canonical parameter ordering and mapping utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np

from .naming import base_parameter_name

if TYPE_CHECKING:
    from .robot_model import RobotModel

PROJECT_STANDARD_PARAMETER_LABELS: tuple[str, ...] = (
    "Ixx",
    "Ixy",
    "Ixz",
    "Iyy",
    "Iyz",
    "Izz",
    "mx",
    "my",
    "mz",
    "m",
)

PINOCCHIO_DYNAMIC_PARAMETER_LABELS: tuple[str, ...] = (
    "m",
    "mx",
    "my",
    "mz",
    "Ixx",
    "Ixy",
    "Iyy",
    "Ixz",
    "Iyz",
    "Izz",
)

_PROJECT_TO_PINOCCHIO_BLOCK = (9, 6, 7, 8, 0, 1, 3, 2, 4, 5)
_PINOCCHIO_TO_PROJECT_BLOCK = (4, 5, 7, 6, 8, 9, 1, 2, 3, 0)


def generate_standard_parameter_names(body_count: int) -> list[str]:
    """Generate standard inertial parameter names in project order."""
    if body_count < 0:
        raise ValueError("body_count must be non-negative.")
    names: list[str] = []
    for index in range(1, body_count + 1):
        names.extend(
            [
                f"I{index}xx",
                f"I{index}xy",
                f"I{index}xz",
                f"I{index}yy",
                f"I{index}yz",
                f"I{index}zz",
                f"mx{index}",
                f"my{index}",
                f"mz{index}",
                f"m{index}",
            ]
        )
    return names


def generate_joint_dynamics_parameter_names(
    joint_count: int,
    enabled_groups: Iterable[str] = ("fv", "fc", "fd"),
) -> list[str]:
    """Generate joint-dynamics parameter names in a stable block order."""
    if joint_count < 0:
        raise ValueError("joint_count must be non-negative.")
    groups = tuple(enabled_groups)
    supported = {"ia", "fv", "fc", "fd", "fo"}
    unsupported = set(groups) - supported
    if unsupported:
        raise ValueError(f"Unsupported joint-dynamics groups: {sorted(unsupported)}")

    names: list[str] = []
    for group in groups:
        for index in range(1, joint_count + 1):
            names.append(f"{group}{index}")
    return names


def base_parameter_names(count: int) -> list[str]:
    """Generate stable base-parameter public names."""
    if count < 0:
        raise ValueError("count must be non-negative.")
    return [base_parameter_name(index) for index in range(1, count + 1)]


def extract_standard_parameter_values(robot: "RobotModel") -> np.ndarray:
    """Extract numeric inertial parameters from a robot in project order."""
    values: list[float] = []
    for link in robot.links:
        inertia = link.inertia.inertia_origin
        first_moment = link.inertia.first_moment
        values.extend(
            [
                float(inertia[0, 0]),
                float(inertia[0, 1]),
                float(inertia[0, 2]),
                float(inertia[1, 1]),
                float(inertia[1, 2]),
                float(inertia[2, 2]),
                float(first_moment[0]),
                float(first_moment[1]),
                float(first_moment[2]),
                float(link.inertia.mass),
            ]
        )
    return np.asarray(values, dtype=float)


def _permute_block_vector(values: np.ndarray, block: tuple[int, ...], block_count: int) -> np.ndarray:
    array = np.asarray(values)
    expected = block_count * len(block)
    if array.ndim != 1:
        raise ValueError("Expected a flat vector.")
    if array.shape[0] != expected:
        raise ValueError(f"Expected vector length {expected}, got {array.shape[0]}.")

    output = np.empty_like(array)
    block_size = len(block)
    for body_index in range(block_count):
        start = body_index * block_size
        source = array[start : start + block_size]
        output[start : start + block_size] = source[list(block)]
    return output


def _permute_block_matrix_columns(
    matrix: np.ndarray,
    block: tuple[int, ...],
    block_count: int,
) -> np.ndarray:
    array = np.asarray(matrix)
    expected = block_count * len(block)
    if array.ndim != 2:
        raise ValueError("Expected a 2D matrix.")
    if array.shape[1] != expected:
        raise ValueError(f"Expected {expected} columns, got {array.shape[1]}.")

    block_size = len(block)
    reordered_blocks = []
    for body_index in range(block_count):
        start = body_index * block_size
        source = array[:, start : start + block_size]
        reordered_blocks.append(source[:, list(block)])
    return np.concatenate(reordered_blocks, axis=1) if reordered_blocks else array[:, :0]


def project_to_pinocchio_vector(values: np.ndarray, body_count: int) -> np.ndarray:
    """Reorder a flat project-order parameter vector into Pinocchio order."""
    return _permute_block_vector(values, _PROJECT_TO_PINOCCHIO_BLOCK, body_count)


def pinocchio_to_project_vector(values: np.ndarray, body_count: int) -> np.ndarray:
    """Reorder a flat Pinocchio-order parameter vector into project order."""
    return _permute_block_vector(values, _PINOCCHIO_TO_PROJECT_BLOCK, body_count)


def project_to_pinocchio_matrix(matrix: np.ndarray, body_count: int) -> np.ndarray:
    """Reorder project-order regressor columns into Pinocchio order."""
    return _permute_block_matrix_columns(matrix, _PROJECT_TO_PINOCCHIO_BLOCK, body_count)


def pinocchio_to_project_matrix(matrix: np.ndarray, body_count: int) -> np.ndarray:
    """Reorder Pinocchio-order regressor columns into project order."""
    return _permute_block_matrix_columns(matrix, _PINOCCHIO_TO_PROJECT_BLOCK, body_count)
