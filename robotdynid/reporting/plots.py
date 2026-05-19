"""Prediction plotting helpers."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np

from robotdynid.identify import IdentificationDataset


def save_prediction_plot(
    output_dir: str | Path,
    *,
    dataset: IdentificationDataset,
    evaluator,
    theta_lin: np.ndarray,
    qds: np.ndarray,
    stride: int,
) -> None:
    """Save measured/predicted torque plots for one identification result."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    indices = np.arange(0, dataset.sample_count, max(stride, 1))
    predictions = np.vstack(
        [
            evaluator.predict_tau(dataset.q[index], dataset.qd[index], dataset.qdd[index], theta_lin, qds=qds)
            for index in indices
        ]
    )
    measured = dataset.tau[indices]
    residual = measured - predictions

    ncols = min(3, dataset.dof)
    nrows = math.ceil(dataset.dof / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)
    for joint_index in range(dataset.dof):
        axis = axes[joint_index]
        axis.plot(indices, measured[:, joint_index], label="measured")
        axis.plot(indices, predictions[:, joint_index], label="predicted", linestyle="--")
        axis.plot(indices, residual[:, joint_index], label="error", linestyle=":")
        axis.set_title(f"joint{joint_index + 1}")
        axis.grid(True)
        axis.legend(loc="lower right")
    for axis in axes[dataset.dof :]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path / "prediction.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
