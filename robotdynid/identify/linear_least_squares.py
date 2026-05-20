"""Weighted linear least-squares utilities for symbolic regressors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotdynid.symbolic import LinearRegressorEvaluator
from .dataset import IdentificationDataset


@dataclass(frozen=True)
class LinearLeastSquaresResult:
    """Solution and diagnostics for one linear parameter solve."""

    linear_parameters: np.ndarray
    residual_vector: np.ndarray
    rmse: np.ndarray
    objective: float
    regressor_matrix: np.ndarray
    target_vector: np.ndarray


def _iter_sample_blocks(sample_count: int, chunk_size: int) -> range:
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")
    return range(0, sample_count, chunk_size)


def stack_regression_problem(
    dataset: IdentificationDataset,
    evaluator: LinearRegressorEvaluator,
    stribeck_parameters: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack H and tau across all samples into one linear least-squares problem."""
    rows: list[np.ndarray] = []
    for sample_index in range(dataset.sample_count):
        rows.append(
            evaluator.evaluate_regressor(
                dataset.q[sample_index],
                dataset.qd[sample_index],
                dataset.qdd[sample_index],
                stribeck_parameters=stribeck_parameters,
            )
        )
    regressor_matrix = np.vstack(rows)
    target_vector = dataset.tau.reshape(-1)
    return regressor_matrix, target_vector


def solve_linear_parameters(
    dataset: IdentificationDataset,
    evaluator: LinearRegressorEvaluator,
    stribeck_parameters: np.ndarray | None = None,
) -> LinearLeastSquaresResult:
    """Solve linear_parameters* = argmin ||W p - T||_2 with optional sample weights."""
    regressor_matrix, target_vector = stack_regression_problem(
        dataset,
        evaluator,
        stribeck_parameters=stribeck_parameters,
    )

    if dataset.sample_weights is not None:
        weighted_matrix = dataset.sample_weights[:, None] * regressor_matrix
        weighted_target = dataset.sample_weights * target_vector
    else:
        weighted_matrix = regressor_matrix
        weighted_target = target_vector

    linear_parameters, *_ = np.linalg.lstsq(weighted_matrix, weighted_target, rcond=None)
    residual_vector = target_vector - regressor_matrix @ linear_parameters
    rmse = np.sqrt(np.mean(residual_vector.reshape(dataset.sample_count, dataset.dof) ** 2, axis=0))
    objective = float(np.linalg.norm(weighted_target - weighted_matrix @ linear_parameters))
    return LinearLeastSquaresResult(
        linear_parameters=linear_parameters,
        residual_vector=residual_vector,
        rmse=rmse,
        objective=objective,
        regressor_matrix=regressor_matrix,
        target_vector=target_vector,
    )


def solve_linear_parameters_streaming(
    dataset: IdentificationDataset,
    evaluator: LinearRegressorEvaluator,
    stribeck_parameters: np.ndarray | None = None,
    *,
    chunk_size: int = 256,
) -> LinearLeastSquaresResult:
    """Solve linear parameters by accumulating normal equations over sample chunks."""
    param_count = len(evaluator.linear_parameter_names)
    ata = np.zeros((param_count, param_count), dtype=float)
    atb = np.zeros((param_count,), dtype=float)

    for start in _iter_sample_blocks(dataset.sample_count, chunk_size):
        stop = min(start + chunk_size, dataset.sample_count)
        regressor_chunk, target_chunk = stack_regression_problem(
            IdentificationDataset(
                q=dataset.q[start:stop],
                qd=dataset.qd[start:stop],
                qdd=dataset.qdd[start:stop],
                tau=dataset.tau[start:stop],
                sample_weights=(
                    dataset.sample_weights[start * dataset.dof : stop * dataset.dof]
                    if dataset.sample_weights is not None
                    else None
                ),
            ),
            evaluator,
            stribeck_parameters=stribeck_parameters,
        )
        if dataset.sample_weights is not None:
            weights = dataset.sample_weights[start * dataset.dof : stop * dataset.dof]
            weighted_matrix = weights[:, None] * regressor_chunk
            weighted_target = weights * target_chunk
        else:
            weighted_matrix = regressor_chunk
            weighted_target = target_chunk
        ata += weighted_matrix.T @ weighted_matrix
        atb += weighted_matrix.T @ weighted_target

    linear_parameters, *_ = np.linalg.lstsq(ata, atb, rcond=None)

    residuals: list[np.ndarray] = []
    squared_error_sum = np.zeros((dataset.dof,), dtype=float)
    for start in _iter_sample_blocks(dataset.sample_count, chunk_size):
        stop = min(start + chunk_size, dataset.sample_count)
        regressor_chunk, target_chunk = stack_regression_problem(
            IdentificationDataset(
                q=dataset.q[start:stop],
                qd=dataset.qd[start:stop],
                qdd=dataset.qdd[start:stop],
                tau=dataset.tau[start:stop],
                sample_weights=(
                    dataset.sample_weights[start * dataset.dof : stop * dataset.dof]
                    if dataset.sample_weights is not None
                    else None
                ),
            ),
            evaluator,
            stribeck_parameters=stribeck_parameters,
        )
        residual_chunk = target_chunk - regressor_chunk @ linear_parameters
        residuals.append(residual_chunk)
        squared_error_sum += np.sum(residual_chunk.reshape(stop - start, dataset.dof) ** 2, axis=0)

    residual_vector = np.concatenate(residuals, axis=0)
    if dataset.sample_weights is not None:
        weighted_residual = dataset.sample_weights * residual_vector
    else:
        weighted_residual = residual_vector
    rmse = np.sqrt(squared_error_sum / dataset.sample_count)
    objective = float(np.linalg.norm(weighted_residual))
    return LinearLeastSquaresResult(
        linear_parameters=linear_parameters,
        residual_vector=residual_vector,
        rmse=rmse,
        objective=objective,
        regressor_matrix=np.empty((0, param_count), dtype=float),
        target_vector=np.empty((0,), dtype=float),
    )
