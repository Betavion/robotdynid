"""Identification pipelines built on top of linear symbolic regressors."""

from .alternating_stribeck import AlternatingIdentifyConfig, IdentificationResult, identify_with_stribeck
from .csv_readers import (
    CsvDatasetConfig,
    MotionTorqueCsvDatasetConfig,
    load_identification_dataset_from_csv,
    load_identification_dataset_from_motion_and_torque_csv,
)
from .dataset import IdentificationDataset
from .linear_least_squares import (
    LinearLeastSquaresResult,
    LinearRegularizationConfig,
    RobustLossConfig,
    solve_linear_parameters,
    solve_linear_parameters_streaming,
    stack_regression_problem,
)

__all__ = [
    "AlternatingIdentifyConfig",
    "CsvDatasetConfig",
    "IdentificationDataset",
    "IdentificationResult",
    "MotionTorqueCsvDatasetConfig",
    "load_identification_dataset_from_csv",
    "load_identification_dataset_from_motion_and_torque_csv",
    "LinearLeastSquaresResult",
    "LinearRegularizationConfig",
    "RobustLossConfig",
    "identify_with_stribeck",
    "solve_linear_parameters",
    "solve_linear_parameters_streaming",
    "stack_regression_problem",
]
