"""Alternating optimization for linear parameters and Stribeck qds."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

from robotdynid.symbolic import LinearRegressorEvaluator
from .dataset import IdentificationDataset
from .linear_least_squares import solve_linear_parameters, solve_linear_parameters_streaming


@dataclass(frozen=True)
class AlternatingIdentifyConfig:
    """Configuration for alternating Stribeck identification."""

    qds_init: np.ndarray
    qds_lower_bound: float = 1e-3
    qds_upper_bound: float = 10.0
    max_iterations: int = 15
    objective_tolerance: float = 1e-8
    qds_tolerance: float = 1e-6
    chunk_size: int | None = None
    optimizer_kwargs: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class IdentificationResult:
    """Final alternating-optimization result and diagnostics."""

    theta_lin: np.ndarray
    qds: np.ndarray
    converged: bool
    iterations: int
    objective_history: tuple[float, ...]
    rmse_history: tuple[np.ndarray, ...]
    linear_parameter_names: tuple[str, ...]


def identify_with_stribeck(
    dataset: IdentificationDataset,
    evaluator: LinearRegressorEvaluator,
    config: AlternatingIdentifyConfig,
) -> IdentificationResult:
    """Alternate between linear least squares and nonlinear qds updates."""
    qds = np.asarray(config.qds_init, dtype=float).reshape(-1)
    if qds.shape[0] != evaluator.qds_size:
        raise ValueError(f"qds_init must have length {evaluator.qds_size}, got {qds.shape[0]}.")

    lower = np.full_like(qds, config.qds_lower_bound, dtype=float)
    upper = np.full_like(qds, config.qds_upper_bound, dtype=float)

    objective_history: list[float] = []
    rmse_history: list[np.ndarray] = []
    converged = False
    theta_lin = np.zeros(len(evaluator.linear_parameter_names), dtype=float)

    use_streaming = config.chunk_size is not None and config.chunk_size > 0
    solver = (
        (lambda ds, ev, q: solve_linear_parameters_streaming(ds, ev, qds=q, chunk_size=config.chunk_size))
        if use_streaming
        else (lambda ds, ev, q: solve_linear_parameters(ds, ev, qds=q))
    )

    def weighted_residual(qds_candidate: np.ndarray) -> np.ndarray:
        candidate_result = solver(dataset, evaluator, qds_candidate)
        residual = candidate_result.residual_vector
        if dataset.sample_weights is not None:
            return dataset.sample_weights * residual
        return residual

    for iteration in range(1, config.max_iterations + 1):
        linear_result = solver(dataset, evaluator, qds)
        theta_lin = linear_result.theta_lin
        objective_history.append(linear_result.objective)
        rmse_history.append(linear_result.rmse)

        optimization = optimize.least_squares(
            weighted_residual,
            qds,
            bounds=(lower, upper),
            **config.optimizer_kwargs,
        )
        qds_next = optimization.x

        if np.linalg.norm(qds_next - qds) <= config.qds_tolerance:
            converged = True
            qds = qds_next
            break

        if len(objective_history) >= 2 and abs(objective_history[-1] - objective_history[-2]) <= config.objective_tolerance:
            converged = True
            qds = qds_next
            break

        qds = qds_next

    final_result = solver(dataset, evaluator, qds)
    objective_history.append(final_result.objective)
    rmse_history.append(final_result.rmse)
    return IdentificationResult(
        theta_lin=final_result.theta_lin,
        qds=qds,
        converged=converged,
        iterations=len(objective_history) - 1,
        objective_history=tuple(objective_history),
        rmse_history=tuple(rmse_history),
        linear_parameter_names=evaluator.linear_parameter_names,
    )
