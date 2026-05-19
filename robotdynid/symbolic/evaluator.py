"""Numeric evaluators compiled from symbolic regressors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import sympy as sp

from .base import BaseRegressorBundle
from .regressor import SymbolicRegressorBundle
from .symbols import SymbolicContext


@dataclass(frozen=True)
class LinearRegressorEvaluator:
    """A compiled numeric evaluator for linear-in-parameters symbolic regressors."""

    dof: int
    q_size: int
    qd_size: int
    qdd_size: int
    qds_size: int
    linear_parameter_names: tuple[str, ...]
    _regressor_func: Callable[..., np.ndarray]

    def evaluate_regressor(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
        qds: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate the regressor matrix H(q, qd, qdd, qds)."""
        q_arr = _as_vector(q, self.q_size, "q")
        qd_arr = _as_vector(qd, self.qd_size, "qd")
        qdd_arr = _as_vector(qdd, self.qdd_size, "qdd")
        qds_arr = _as_optional_vector(qds, self.qds_size, "qds")
        values = (*q_arr, *qd_arr, *qdd_arr, *qds_arr)
        regressor = np.asarray(self._regressor_func(*values), dtype=float)
        return regressor.reshape(self.dof, len(self.linear_parameter_names))

    def predict_tau(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
        theta_lin: np.ndarray,
        qds: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate tau = H * theta_lin for one state."""
        regressor = self.evaluate_regressor(q, qd, qdd, qds=qds)
        theta = _as_vector(theta_lin, len(self.linear_parameter_names), "theta_lin")
        return regressor @ theta


def _as_vector(values: np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.shape[0] != size:
        raise ValueError(f"{name} must have length {size}, got {array.shape[0]}.")
    return array


def _as_optional_vector(values: np.ndarray | None, size: int, name: str) -> np.ndarray:
    if size == 0:
        return np.zeros((0,), dtype=float)
    if values is None:
        raise ValueError(f"{name} must be provided because the regressor depends on it.")
    return _as_vector(values, size, name)


def _build_lambdify_inputs(context: SymbolicContext) -> tuple[sp.Symbol, ...]:
    return context.q + context.qd + context.qdd + context.qds


def build_regressor_evaluator(
    context: SymbolicContext,
    regressor: sp.Matrix,
    linear_parameter_names: tuple[str, ...],
) -> LinearRegressorEvaluator:
    """Compile a symbolic regressor matrix into a numpy-based evaluator."""
    input_symbols = _build_lambdify_inputs(context)
    regressor_func = sp.lambdify(input_symbols, regressor, modules="numpy")
    return LinearRegressorEvaluator(
        dof=regressor.rows,
        q_size=len(context.q),
        qd_size=len(context.qd),
        qdd_size=len(context.qdd),
        qds_size=len(context.qds),
        linear_parameter_names=linear_parameter_names,
        _regressor_func=regressor_func,
    )


def build_standard_regressor_evaluator(bundle: SymbolicRegressorBundle) -> LinearRegressorEvaluator:
    """Compile a standard symbolic regressor bundle into a numeric evaluator."""
    return build_regressor_evaluator(bundle.context, bundle.regressor, bundle.linear_parameter_names)


def build_base_regressor_evaluator(bundle: BaseRegressorBundle) -> LinearRegressorEvaluator:
    """Compile a base symbolic regressor bundle into a numeric evaluator."""
    return build_regressor_evaluator(bundle.context, bundle.regressor, bundle.linear_parameter_names)
