"""CSV readers and mergers for identification inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import IdentificationDataset
from .preprocess import estimate_acceleration, weights_from_torque_std


@dataclass(frozen=True)
class CsvDatasetConfig:
    """Column naming and preprocessing configuration for one-file CSV loading."""

    dof: int
    position_prefix: str = "pos"
    velocity_prefix: str = "vel"
    acceleration_prefix: str = "acc"
    torque_prefix: str = "torque"
    position_offsets: tuple[float, ...] | None = None
    torque_weighting: str | None = "torque_std"
    stride: int = 1
    max_samples: int | None = None


@dataclass(frozen=True)
class MotionTorqueCsvDatasetConfig:
    """Configuration for datasets split across motion and torque CSV files."""

    dof: int
    timestamp_column: str = "timestamp"
    position_template: str = "joint{index}_position"
    velocity_template: str = "joint{index}_velocity"
    torque_template: str = "joint{index}_measure"
    position_offsets: tuple[float, ...] | None = None
    torque_weighting: str | None = "torque_std"
    stride: int = 1
    max_samples: int | None = None
    timestamp_tolerance: float = 1e-9


def _column_names(prefix: str, dof: int) -> list[str]:
    return [f"{prefix}{index}" for index in range(1, dof + 1)]


def _template_names(template: str, dof: int) -> list[str]:
    return [template.format(index=index) for index in range(1, dof + 1)]


def _apply_position_offsets(q: np.ndarray, offsets: tuple[float, ...] | None, dof: int) -> np.ndarray:
    if offsets is None:
        return q
    offset_array = np.asarray(offsets, dtype=float).reshape(1, -1)
    if offset_array.shape[1] != dof:
        raise ValueError(f"position_offsets must contain {dof} values.")
    return q + offset_array


def _weights_from_mode(mode: str | None, tau: np.ndarray) -> np.ndarray | None:
    if mode is None:
        return None
    if mode == "torque_std":
        return weights_from_torque_std(tau)
    raise ValueError(f"Unsupported torque_weighting mode: {mode}")


def _subsample_dataframe(dataframe: pd.DataFrame, stride: int, max_samples: int | None) -> pd.DataFrame:
    if stride < 1:
        raise ValueError("stride must be >= 1.")
    result = dataframe
    if stride > 1:
        result = result.iloc[::stride]
    if max_samples is not None:
        result = result.iloc[:max_samples]
    return result.reset_index(drop=True)


def load_identification_dataset_from_csv(
    csv_path: str | Path,
    config: CsvDatasetConfig,
) -> IdentificationDataset:
    """Load a CSV file into an IdentificationDataset."""
    dataframe = pd.read_csv(csv_path)
    required = (
        _column_names(config.position_prefix, config.dof)
        + _column_names(config.velocity_prefix, config.dof)
        + _column_names(config.acceleration_prefix, config.dof)
        + _column_names(config.torque_prefix, config.dof)
    )
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"CSV file is missing required columns: {missing}")

    dataframe = _subsample_dataframe(dataframe, config.stride, config.max_samples)
    q = dataframe[_column_names(config.position_prefix, config.dof)].to_numpy(dtype=float)
    qd = dataframe[_column_names(config.velocity_prefix, config.dof)].to_numpy(dtype=float)
    qdd = dataframe[_column_names(config.acceleration_prefix, config.dof)].to_numpy(dtype=float)
    tau = dataframe[_column_names(config.torque_prefix, config.dof)].to_numpy(dtype=float)

    return IdentificationDataset(
        q=_apply_position_offsets(q, config.position_offsets, config.dof),
        qd=qd,
        qdd=qdd,
        tau=tau,
        sample_weights=_weights_from_mode(config.torque_weighting, tau),
    )


def load_identification_dataset_from_motion_and_torque_csv(
    motion_csv_path: str | Path,
    torque_csv_path: str | Path,
    config: MotionTorqueCsvDatasetConfig,
) -> IdentificationDataset:
    """Load motion and torque CSV files into an IdentificationDataset."""
    motion = pd.read_csv(motion_csv_path)
    torque = pd.read_csv(torque_csv_path)

    required_motion = [config.timestamp_column] + _template_names(config.position_template, config.dof) + _template_names(
        config.velocity_template, config.dof
    )
    required_torque = [config.timestamp_column] + _template_names(config.torque_template, config.dof)
    missing_motion = [column for column in required_motion if column not in motion.columns]
    missing_torque = [column for column in required_torque if column not in torque.columns]
    if missing_motion:
        raise ValueError(f"Motion CSV is missing required columns: {missing_motion}")
    if missing_torque:
        raise ValueError(f"Torque CSV is missing required columns: {missing_torque}")

    motion = _subsample_dataframe(motion, config.stride, config.max_samples)
    torque = _subsample_dataframe(torque, config.stride, config.max_samples)

    motion_ts = motion[config.timestamp_column].to_numpy(dtype=float)
    torque_ts = torque[config.timestamp_column].to_numpy(dtype=float)
    if motion_ts.shape != torque_ts.shape or not np.allclose(motion_ts, torque_ts, atol=config.timestamp_tolerance, rtol=0.0):
        raise ValueError("Motion and torque CSV timestamps must align sample-by-sample within tolerance.")

    q = motion[_template_names(config.position_template, config.dof)].to_numpy(dtype=float)
    qd = motion[_template_names(config.velocity_template, config.dof)].to_numpy(dtype=float)
    qdd = estimate_acceleration(motion_ts, qd)
    tau = torque[_template_names(config.torque_template, config.dof)].to_numpy(dtype=float)

    return IdentificationDataset(
        q=_apply_position_offsets(q, config.position_offsets, config.dof),
        qd=qd,
        qdd=qdd,
        tau=tau,
        sample_weights=_weights_from_mode(config.torque_weighting, tau),
    )
