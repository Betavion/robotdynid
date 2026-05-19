"""Numerical base-parameter selection via QR with column pivoting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg

from robotdynid.core.naming import base_parameter_name
from robotdynid.core.robot_model import BaseParamMetadata


@dataclass(frozen=True)
class BaseSelectionStrategy:
    """Numerical settings for base-parameter extraction."""

    scale_columns: bool = True
    rank_tol_factor: float = 100.0
    svd_tol_factor: float = 100.0
    verify_with_svd: bool = True


def _column_scales(regressor_matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(regressor_matrix, axis=0)
    norms[norms == 0.0] = 1.0
    return norms


def select_base_parameters(
    regressor_matrix: np.ndarray,
    standard_param_names: list[str],
    strategy: BaseSelectionStrategy = BaseSelectionStrategy(),
) -> BaseParamMetadata:
    """Select independent regressor columns and build base-parameter metadata."""
    matrix = np.asarray(regressor_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("regressor_matrix must be a 2D array.")
    if matrix.shape[1] != len(standard_param_names):
        raise ValueError("standard_param_names length must match the number of regressor columns.")
    if matrix.size == 0:
        raise ValueError("regressor_matrix must not be empty.")

    scales = _column_scales(matrix) if strategy.scale_columns else np.ones(matrix.shape[1], dtype=float)
    scaled = matrix / scales

    _, r_matrix, pivots = linalg.qr(scaled, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r_matrix))
    eps = np.finfo(float).eps
    leading = diag[0] if diag.size else 0.0
    tolerance = strategy.rank_tol_factor * max(matrix.shape) * eps * leading
    qr_rank = int(np.sum(diag > tolerance))

    svd_rank = qr_rank
    if strategy.verify_with_svd:
        singular_values = linalg.svd(scaled, compute_uv=False)
        leading_sv = singular_values[0] if singular_values.size else 0.0
        svd_tolerance = strategy.svd_tol_factor * max(matrix.shape) * eps * leading_sv
        svd_rank = int(np.sum(singular_values > svd_tolerance))
        if svd_rank != qr_rank:
            raise ValueError(
                f"QRCP rank ({qr_rank}) and SVD rank ({svd_rank}) disagree. "
                "The regressor data is numerically unstable for base selection."
            )

    keep_indices = list(map(int, pivots[:qr_rank]))
    dependent_indices = list(map(int, pivots[qr_rank:]))
    permutation = keep_indices + dependent_indices

    if dependent_indices:
        r11 = r_matrix[:qr_rank, :qr_rank]
        r12 = r_matrix[:qr_rank, qr_rank:]
        dependency_scaled = linalg.solve_triangular(r11, r12)
        keep_scales = np.diag(scales[keep_indices])
        dep_scales = np.diag(scales[dependent_indices])
        dependency_matrix = np.linalg.solve(keep_scales, dependency_scaled @ dep_scales)
    else:
        dependency_matrix = np.zeros((qr_rank, 0), dtype=float)

    return BaseParamMetadata(
        rank=qr_rank,
        keep_indices=keep_indices,
        dependent_indices=dependent_indices,
        dependency_matrix=dependency_matrix,
        standard_param_names=list(standard_param_names),
        base_param_names=[base_parameter_name(index) for index in range(1, qr_rank + 1)],
        column_permutation=permutation,
        qr_rank=qr_rank,
        svd_rank=svd_rank,
        tolerance=float(tolerance),
    )
