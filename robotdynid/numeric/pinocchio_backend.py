"""Pinocchio-backed numeric helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robotdynid.core.params import pinocchio_to_project_matrix, pinocchio_to_project_vector

import pinocchio as pin


@dataclass
class PinocchioModelBundle:
    """A loaded Pinocchio model and its associated data object."""

    model: Any
    data: Any
    urdf_path: Path

def build_pinocchio_model(urdf_path: str | Path) -> PinocchioModelBundle:
    """Load a Pinocchio model from URDF."""
    path = Path(urdf_path)
    model = pin.buildModelFromUrdf(str(path))
    data = model.createData()
    return PinocchioModelBundle(model=model, data=data, urdf_path=path)


def compute_joint_torque_regressor(
    bundle: PinocchioModelBundle,
    q: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    *,
    reorder_to_project: bool = False,
) -> np.ndarray:
    """Compute the Pinocchio torque regressor for one robot state."""
    regressor = np.asarray(
        pin.computeJointTorqueRegressor(bundle.model, bundle.data, q, v, a),
        dtype=float,
    )
    if reorder_to_project:
        body_count = regressor.shape[1] // 10
        regressor = pinocchio_to_project_matrix(regressor, body_count)
    return regressor


def extract_inertia_dynamic_parameters(
    bundle: PinocchioModelBundle,
    *,
    reorder_to_project: bool = False,
) -> np.ndarray:
    """Extract per-body dynamic parameters from a Pinocchio model."""
    values = [
        np.asarray(inertia.toDynamicParameters(), dtype=float)
        for inertia in bundle.model.inertias[1:]
    ]
    flat = np.concatenate(values, axis=0) if values else np.zeros((0,), dtype=float)
    if reorder_to_project:
        return pinocchio_to_project_vector(flat, len(values))
    return flat
