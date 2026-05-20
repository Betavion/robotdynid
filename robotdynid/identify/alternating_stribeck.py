"""Alternating optimization for linear and Stribeck parameters."""

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

    stribeck_parameter_init: np.ndarray
    stribeck_parameter_lower_bound: float = 1e-3
    stribeck_parameter_upper_bound: float = 10.0
    max_iterations: int = 15
    objective_tolerance: float = 1e-8
    stribeck_parameter_tolerance: float = 1e-6
    chunk_size: int | None = None
    optimizer_kwargs: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class IdentificationResult:
    """Final alternating-optimization result and diagnostics."""

    linear_parameters: np.ndarray
    stribeck_parameters: np.ndarray
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
    """Alternate between linear least squares and nonlinear Stribeck updates."""
    stribeck_parameters = np.asarray(config.stribeck_parameter_init, dtype=float).reshape(-1)
    if stribeck_parameters.shape[0] != evaluator.stribeck_parameter_size:
        raise ValueError(
            "stribeck_parameter_init must have length "
            f"{evaluator.stribeck_parameter_size}, got {stribeck_parameters.shape[0]}."
        )

    lower = np.full_like(stribeck_parameters, config.stribeck_parameter_lower_bound, dtype=float)
    upper = np.full_like(stribeck_parameters, config.stribeck_parameter_upper_bound, dtype=float)

    objective_history: list[float] = []
    rmse_history: list[np.ndarray] = []
    converged = False

    use_streaming = config.chunk_size is not None and config.chunk_size > 0
    solver = (
        (
            lambda ds, ev, params: solve_linear_parameters_streaming(
                ds,
                ev,
                stribeck_parameters=params,
                chunk_size=config.chunk_size,
            )
        )
        if use_streaming
        else (lambda ds, ev, params: solve_linear_parameters(ds, ev, stribeck_parameters=params))
    )

    def weighted_residual(stribeck_candidate: np.ndarray) -> np.ndarray:
        candidate_result = solver(dataset, evaluator, stribeck_candidate)
        residual = candidate_result.residual_vector
        if dataset.sample_weights is not None:
            return dataset.sample_weights * residual
        return residual

    for iteration in range(1, config.max_iterations + 1):
        linear_result = solver(dataset, evaluator, stribeck_parameters)
        objective_history.append(linear_result.objective)
        rmse_history.append(linear_result.rmse)

        optimization = optimize.least_squares(
            weighted_residual,
            stribeck_parameters,
            bounds=(lower, upper),
            **config.optimizer_kwargs,
        )
        stribeck_next = optimization.x

        if np.linalg.norm(stribeck_next - stribeck_parameters) <= config.stribeck_parameter_tolerance:
            converged = True
            stribeck_parameters = stribeck_next
            break

        if len(objective_history) >= 2 and abs(objective_history[-1] - objective_history[-2]) <= config.objective_tolerance:
            converged = True
            stribeck_parameters = stribeck_next
            break

        stribeck_parameters = stribeck_next

    final_result = solver(dataset, evaluator, stribeck_parameters)
    objective_history.append(final_result.objective)
    rmse_history.append(final_result.rmse)
    return IdentificationResult(
        linear_parameters=final_result.linear_parameters,
        stribeck_parameters=stribeck_parameters,
        converged=converged,
        iterations=len(objective_history) - 1,
        objective_history=tuple(objective_history),
        rmse_history=tuple(rmse_history),
        linear_parameter_names=evaluator.linear_parameter_names,
    )
