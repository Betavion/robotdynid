"""State-space sampling helpers for model-based base-parameter selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotdynid.identify import IdentificationDataset
from .pinocchio_backend import PinocchioModelBundle


@dataclass(frozen=True)
class StateSamplingConfig:
    """Configuration for random state sampling over a Pinocchio model."""

    sample_count: int = 800
    random_seed: int = 42
    velocity_scale: float = 0.5
    acceleration_scale: float = 0.5
    unbounded_position_limit: float = np.pi


def _finite_joint_bounds(lower: np.ndarray, upper: np.ndarray, fallback: float) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray(lower, dtype=float).copy()
    high = np.asarray(upper, dtype=float).copy()
    invalid = ~np.isfinite(low) | ~np.isfinite(high) | (high <= low) | (np.abs(low) > 1e10) | (np.abs(high) > 1e10)
    low[invalid] = -fallback
    high[invalid] = fallback
    return low, high


def sample_model_state_dataset(
    pinocchio_bundle: PinocchioModelBundle,
    config: StateSamplingConfig = StateSamplingConfig(),
) -> IdentificationDataset:
    """Generate a synthetic state-only dataset for base-parameter selection."""
    if config.sample_count < 1:
        raise ValueError("sample_count must be >= 1.")
    if config.velocity_scale <= 0 or config.acceleration_scale <= 0:
        raise ValueError("velocity_scale and acceleration_scale must be positive.")

    model = pinocchio_bundle.model
    dof = model.nv
    rng = np.random.default_rng(config.random_seed)

    lower_q, upper_q = _finite_joint_bounds(
        model.lowerPositionLimit[:dof],
        model.upperPositionLimit[:dof],
        config.unbounded_position_limit,
    )
    velocity_limit = np.asarray(model.velocityLimit[:dof], dtype=float)
    velocity_limit[~np.isfinite(velocity_limit) | (velocity_limit <= 0.0)] = 1.0
    acceleration_limit = config.acceleration_scale * velocity_limit

    q = rng.uniform(lower_q, upper_q, size=(config.sample_count, dof))
    qd = rng.uniform(-config.velocity_scale * velocity_limit, config.velocity_scale * velocity_limit, size=(config.sample_count, dof))
    qdd = rng.uniform(-acceleration_limit, acceleration_limit, size=(config.sample_count, dof))
    tau = np.zeros((config.sample_count, dof), dtype=float)
    return IdentificationDataset(q=q, qd=qd, qdd=qdd, tau=tau, sample_weights=None)
