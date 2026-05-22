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
    regularization_residual: np.ndarray | None = None
    robust_weights: np.ndarray | None = None
    effective_weights: np.ndarray | None = None
    iterations: int = 1


@dataclass(frozen=True)
class LinearRegularizationConfig:
    """Gaussian prior/Tikhonov regularization for linear parameters."""

    strength: float = 0.0
    prior: np.ndarray | None = None
    prior_std: np.ndarray | None = None
    covariance: np.ndarray | None = None


@dataclass(frozen=True)
class RobustLossConfig:
    """IRLS settings for robust linear least squares."""

    loss: str = "linear"
    f_scale: float = 1.0
    max_iterations: int = 1
    tolerance: float = 1e-6


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


def _base_sample_weights(dataset: IdentificationDataset) -> np.ndarray:
    if dataset.sample_weights is None:
        return np.ones(dataset.sample_count * dataset.dof, dtype=float)
    return np.asarray(dataset.sample_weights, dtype=float).reshape(-1)


def _regularization_terms(
    parameter_count: int,
    regularization: LinearRegularizationConfig | None,
) -> tuple[np.ndarray, np.ndarray]:
    if regularization is None:
        return np.zeros((0, parameter_count), dtype=float), np.zeros((0,), dtype=float)

    strength = float(regularization.strength)
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("regularization strength must be a finite non-negative value.")
    if strength == 0.0:
        return np.zeros((0, parameter_count), dtype=float), np.zeros((0,), dtype=float)

    prior = (
        np.zeros((parameter_count,), dtype=float)
        if regularization.prior is None
        else np.asarray(regularization.prior, dtype=float).reshape(-1)
    )
    if prior.shape[0] != parameter_count:
        raise ValueError(f"regularization prior must have length {parameter_count}, got {prior.shape[0]}.")

    if regularization.covariance is not None and regularization.prior_std is not None:
        raise ValueError("regularization covariance and prior_std are mutually exclusive.")

    if regularization.covariance is not None:
        covariance = np.asarray(regularization.covariance, dtype=float)
        if covariance.shape != (parameter_count, parameter_count):
            raise ValueError(
                "regularization covariance must have shape "
                f"({parameter_count}, {parameter_count}), got {covariance.shape}."
            )
        cholesky = np.linalg.cholesky(covariance)
        prior_precision_sqrt = np.linalg.solve(cholesky, np.eye(parameter_count, dtype=float))
    elif regularization.prior_std is not None:
        prior_std = np.asarray(regularization.prior_std, dtype=float).reshape(-1)
        if prior_std.shape[0] != parameter_count:
            raise ValueError(f"regularization prior_std must have length {parameter_count}, got {prior_std.shape[0]}.")
        if np.any(prior_std <= 0.0):
            raise ValueError("regularization prior_std entries must be positive.")
        prior_precision_sqrt = np.diag(1.0 / prior_std)
    else:
        prior_precision_sqrt = np.eye(parameter_count, dtype=float)

    matrix = np.sqrt(strength) * prior_precision_sqrt
    target = matrix @ prior
    return matrix, target


def _robust_loss_config(config: RobustLossConfig | None) -> RobustLossConfig:
    if config is None:
        return RobustLossConfig()
    loss = config.loss.strip().lower()
    supported = {"linear", "soft_l1", "huber", "cauchy", "arctan"}
    if loss not in supported:
        raise ValueError(f"Unsupported robust loss: {config.loss!r}. Supported values: {sorted(supported)}")
    if config.f_scale <= 0.0:
        raise ValueError("robust f_scale must be positive.")
    if config.max_iterations < 1:
        raise ValueError("robust max_iterations must be >= 1.")
    if config.tolerance < 0.0:
        raise ValueError("robust tolerance must be non-negative.")
    return RobustLossConfig(
        loss=loss,
        f_scale=float(config.f_scale),
        max_iterations=int(config.max_iterations),
        tolerance=float(config.tolerance),
    )


def _robust_weights(weighted_residual: np.ndarray, config: RobustLossConfig) -> np.ndarray:
    if config.loss == "linear":
        return np.ones_like(weighted_residual, dtype=float)
    z = np.square(weighted_residual / config.f_scale)
    if config.loss == "soft_l1":
        rho_prime = 1.0 / np.sqrt(1.0 + z)
    elif config.loss == "huber":
        rho_prime = np.where(z <= 1.0, 1.0, 1.0 / np.sqrt(np.maximum(z, np.finfo(float).tiny)))
    elif config.loss == "cauchy":
        rho_prime = 1.0 / (1.0 + z)
    elif config.loss == "arctan":
        rho_prime = 1.0 / (1.0 + z * z)
    else:
        raise ValueError(f"Unsupported robust loss: {config.loss!r}")
    return np.sqrt(np.maximum(rho_prime, np.finfo(float).tiny))


def _robust_cost(weighted_residual: np.ndarray, config: RobustLossConfig) -> float:
    z = np.square(weighted_residual / config.f_scale)
    if config.loss == "linear":
        rho = z
    elif config.loss == "soft_l1":
        rho = 2.0 * (np.sqrt(1.0 + z) - 1.0)
    elif config.loss == "huber":
        rho = np.where(z <= 1.0, z, 2.0 * np.sqrt(z) - 1.0)
    elif config.loss == "cauchy":
        rho = np.log1p(z)
    elif config.loss == "arctan":
        rho = np.arctan(z)
    else:
        raise ValueError(f"Unsupported robust loss: {config.loss!r}")
    return float(np.sum(config.f_scale**2 * rho))


def _solve_augmented_dense(
    regressor_matrix: np.ndarray,
    target_vector: np.ndarray,
    weights: np.ndarray,
    regularization_matrix: np.ndarray,
    regularization_target: np.ndarray,
) -> np.ndarray:
    weighted_matrix = weights[:, None] * regressor_matrix
    weighted_target = weights * target_vector
    if regularization_matrix.shape[0] > 0:
        weighted_matrix = np.vstack([weighted_matrix, regularization_matrix])
        weighted_target = np.concatenate([weighted_target, regularization_target])
    parameters, *_ = np.linalg.lstsq(weighted_matrix, weighted_target, rcond=None)
    return parameters


def _linear_result(
    dataset: IdentificationDataset,
    regressor_matrix: np.ndarray,
    target_vector: np.ndarray,
    linear_parameters: np.ndarray,
    base_weights: np.ndarray,
    robust_weights: np.ndarray,
    robust_config: RobustLossConfig,
    regularization_matrix: np.ndarray,
    regularization_target: np.ndarray,
    iterations: int,
) -> LinearLeastSquaresResult:
    residual_vector = target_vector - regressor_matrix @ linear_parameters
    weighted_residual = base_weights * residual_vector
    regularization_residual = (
        regularization_matrix @ linear_parameters - regularization_target
        if regularization_matrix.shape[0] > 0
        else None
    )
    regularization_cost = 0.0 if regularization_residual is None else float(regularization_residual @ regularization_residual)
    objective = float(np.sqrt(_robust_cost(weighted_residual, robust_config) + regularization_cost))
    rmse = np.sqrt(np.mean(residual_vector.reshape(dataset.sample_count, dataset.dof) ** 2, axis=0))
    return LinearLeastSquaresResult(
        linear_parameters=linear_parameters,
        residual_vector=residual_vector,
        rmse=rmse,
        objective=objective,
        regressor_matrix=regressor_matrix,
        target_vector=target_vector,
        regularization_residual=regularization_residual,
        robust_weights=None if robust_config.loss == "linear" else robust_weights,
        effective_weights=base_weights * robust_weights,
        iterations=iterations,
    )


def solve_linear_parameters(
    dataset: IdentificationDataset,
    evaluator: LinearRegressorEvaluator,
    stribeck_parameters: np.ndarray | None = None,
    regularization: LinearRegularizationConfig | None = None,
    robust_loss: RobustLossConfig | None = None,
) -> LinearLeastSquaresResult:
    """Solve linear_parameters* = argmin ||W p - T||_2 with optional sample weights."""
    regressor_matrix, target_vector = stack_regression_problem(
        dataset,
        evaluator,
        stribeck_parameters=stribeck_parameters,
    )
    robust_config = _robust_loss_config(robust_loss)
    regularization_matrix, regularization_target = _regularization_terms(
        regressor_matrix.shape[1],
        regularization,
    )
    base_weights = _base_sample_weights(dataset)
    robust_weights = np.ones_like(base_weights, dtype=float)

    iterations = robust_config.max_iterations if robust_config.loss != "linear" else 1
    linear_parameters = np.zeros((regressor_matrix.shape[1],), dtype=float)
    completed_iterations = 0
    for iteration in range(1, iterations + 1):
        completed_iterations = iteration
        linear_parameters = _solve_augmented_dense(
            regressor_matrix,
            target_vector,
            base_weights * robust_weights,
            regularization_matrix,
            regularization_target,
        )
        if robust_config.loss == "linear":
            break
        residual_vector = target_vector - regressor_matrix @ linear_parameters
        next_robust_weights = _robust_weights(base_weights * residual_vector, robust_config)
        if np.linalg.norm(next_robust_weights - robust_weights) <= robust_config.tolerance:
            robust_weights = next_robust_weights
            break
        robust_weights = next_robust_weights

    return _linear_result(
        dataset,
        regressor_matrix,
        target_vector,
        linear_parameters,
        base_weights,
        robust_weights,
        robust_config,
        regularization_matrix,
        regularization_target,
        completed_iterations,
    )


def solve_linear_parameters_streaming(
    dataset: IdentificationDataset,
    evaluator: LinearRegressorEvaluator,
    stribeck_parameters: np.ndarray | None = None,
    *,
    chunk_size: int = 256,
    regularization: LinearRegularizationConfig | None = None,
    robust_loss: RobustLossConfig | None = None,
) -> LinearLeastSquaresResult:
    """Solve linear parameters by accumulating normal equations over sample chunks."""
    param_count = len(evaluator.linear_parameter_names)
    robust_config = _robust_loss_config(robust_loss)
    regularization_matrix, regularization_target = _regularization_terms(param_count, regularization)
    base_weights = _base_sample_weights(dataset)
    robust_weights = np.ones_like(base_weights, dtype=float)

    def solve_with_weights(weights: np.ndarray) -> np.ndarray:
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
                    sample_weights=None,
                ),
                evaluator,
                stribeck_parameters=stribeck_parameters,
            )
            chunk_weights = weights[start * dataset.dof : stop * dataset.dof]
            weighted_matrix = chunk_weights[:, None] * regressor_chunk
            weighted_target = chunk_weights * target_chunk
            ata += weighted_matrix.T @ weighted_matrix
            atb += weighted_matrix.T @ weighted_target
        if regularization_matrix.shape[0] > 0:
            ata += regularization_matrix.T @ regularization_matrix
            atb += regularization_matrix.T @ regularization_target
        parameters, *_ = np.linalg.lstsq(ata, atb, rcond=None)
        return parameters

    def residual_for(parameters: np.ndarray) -> np.ndarray:
        residuals: list[np.ndarray] = []
        for start in _iter_sample_blocks(dataset.sample_count, chunk_size):
            stop = min(start + chunk_size, dataset.sample_count)
            regressor_chunk, target_chunk = stack_regression_problem(
                IdentificationDataset(
                    q=dataset.q[start:stop],
                    qd=dataset.qd[start:stop],
                    qdd=dataset.qdd[start:stop],
                    tau=dataset.tau[start:stop],
                    sample_weights=None,
                ),
                evaluator,
                stribeck_parameters=stribeck_parameters,
            )
            residuals.append(target_chunk - regressor_chunk @ parameters)
        return np.concatenate(residuals, axis=0)

    iterations = robust_config.max_iterations if robust_config.loss != "linear" else 1
    linear_parameters = np.zeros((param_count,), dtype=float)
    completed_iterations = 0
    for iteration in range(1, iterations + 1):
        completed_iterations = iteration
        linear_parameters = solve_with_weights(base_weights * robust_weights)
        if robust_config.loss == "linear":
            break
        next_robust_weights = _robust_weights(base_weights * residual_for(linear_parameters), robust_config)
        if np.linalg.norm(next_robust_weights - robust_weights) <= robust_config.tolerance:
            robust_weights = next_robust_weights
            break
        robust_weights = next_robust_weights

    residual_vector = residual_for(linear_parameters)
    squared_error_sum = np.zeros((dataset.dof,), dtype=float)
    for start in _iter_sample_blocks(dataset.sample_count, chunk_size):
        stop = min(start + chunk_size, dataset.sample_count)
        residual_chunk = residual_vector[(start * dataset.dof) : (stop * dataset.dof)]
        squared_error_sum += np.sum(residual_chunk.reshape(stop - start, dataset.dof) ** 2, axis=0)

    regularization_residual = (
        regularization_matrix @ linear_parameters - regularization_target
        if regularization_matrix.shape[0] > 0
        else None
    )
    regularization_cost = 0.0 if regularization_residual is None else float(regularization_residual @ regularization_residual)
    weighted_residual = base_weights * residual_vector
    rmse = np.sqrt(squared_error_sum / dataset.sample_count)
    objective = float(np.sqrt(_robust_cost(weighted_residual, robust_config) + regularization_cost))
    return LinearLeastSquaresResult(
        linear_parameters=linear_parameters,
        residual_vector=residual_vector,
        rmse=rmse,
        objective=objective,
        regressor_matrix=np.empty((0, param_count), dtype=float),
        target_vector=np.empty((0,), dtype=float),
        regularization_residual=regularization_residual,
        robust_weights=None if robust_config.loss == "linear" else robust_weights,
        effective_weights=base_weights * robust_weights,
        iterations=completed_iterations,
    )
