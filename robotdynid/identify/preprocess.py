"""Reusable preprocessing helpers for identification datasets."""

from __future__ import annotations

import numpy as np


def weights_from_torque_std(tau: np.ndarray) -> np.ndarray:
    """Compute per-sample weights from inverse joint torque standard deviation."""
    std = tau.std(axis=0)
    std[std == 0.0] = 1.0
    return np.tile(1.0 / std, tau.shape[0])


def estimate_acceleration(timestamp: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Estimate joint acceleration from timestamped joint velocity samples."""
    if len(timestamp) < 2:
        return np.zeros_like(velocity)
    return np.gradient(velocity, timestamp, axis=0, edge_order=1)
