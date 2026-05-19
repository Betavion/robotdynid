"""Validated identification dataset containers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _ensure_2d(array: np.ndarray, columns: int, name: str) -> np.ndarray:
    data = np.asarray(array, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if data.shape[1] != columns:
        raise ValueError(f"{name} must have {columns} columns, got {data.shape[1]}.")
    return data


@dataclass(frozen=True)
class IdentificationDataset:
    """Stacked joint-space samples for linear/nonlinear parameter identification."""

    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    tau: np.ndarray
    sample_weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        q = np.asarray(self.q, dtype=float)
        if q.ndim != 2:
            raise ValueError("q must be a 2D array.")
        sample_count, dof = q.shape

        object.__setattr__(self, "q", q)
        object.__setattr__(self, "qd", _ensure_2d(self.qd, dof, "qd"))
        object.__setattr__(self, "qdd", _ensure_2d(self.qdd, dof, "qdd"))
        object.__setattr__(self, "tau", _ensure_2d(self.tau, dof, "tau"))

        if self.qd.shape[0] != sample_count or self.qdd.shape[0] != sample_count or self.tau.shape[0] != sample_count:
            raise ValueError("q, qd, qdd and tau must contain the same number of samples.")

        if self.sample_weights is not None:
            weights = np.asarray(self.sample_weights, dtype=float).reshape(-1)
            if weights.shape[0] != sample_count * dof:
                raise ValueError(
                    "sample_weights must have length sample_count * dof and be laid out in the same order as stacked tau."
                )
            object.__setattr__(self, "sample_weights", weights)

    @property
    def sample_count(self) -> int:
        return self.q.shape[0]

    @property
    def dof(self) -> int:
        return self.q.shape[1]
