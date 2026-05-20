"""Structured result payloads and artifact export."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from robotdynid.core.robot_model import BaseParamMetadata
from robotdynid.identify import IdentificationDataset, IdentificationResult


def build_result_payload(
    *,
    urdf_path: str,
    csv_source: str | dict[str, str],
    dataset: IdentificationDataset,
    selection_dataset: IdentificationDataset,
    selection_source: str,
    stride: int,
    max_samples: int | None,
    chunk_size: int | None,
    base_metadata: BaseParamMetadata,
    identification_result: IdentificationResult,
) -> dict[str, object]:
    """Build a serializable identification result payload."""
    return {
        "urdf": urdf_path,
        "csv": csv_source,
        "sample_count": dataset.sample_count,
        "selection_sample_count": selection_dataset.sample_count,
        "selection_source": selection_source,
        "base_rank": base_metadata.rank,
        "qds": identification_result.qds.tolist(),
        "theta_lin": identification_result.theta_lin.tolist(),
        "objective_history": list(identification_result.objective_history),
        "rmse_history": [rmse.tolist() for rmse in identification_result.rmse_history],
        "linear_parameter_names": list(identification_result.linear_parameter_names),
        "stride": stride,
        "max_samples": max_samples,
        "chunk_size": chunk_size,
    }


def save_identification_artifacts(
    output_dir: str | Path,
    payload: dict[str, object],
    base_metadata: BaseParamMetadata,
) -> None:
    """Write JSON and CSV artifacts for an identification run."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / "identify_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    base_metadata.to_json_file(output_path / "base_metadata.json")

    theta_names = payload["linear_parameter_names"]  # type: ignore[index]
    theta_values = payload["theta_lin"]  # type: ignore[index]
    theta_lines = ["name,value"] + [f"{name},{value}" for name, value in zip(theta_names, theta_values)]
    (output_path / "theta_lin.csv").write_text("\n".join(theta_lines) + "\n", encoding="utf-8")

    qds_lines = ["name,qds"] + [f"qds{index + 1},{value}" for index, value in enumerate(payload["qds"])]  # type: ignore[index]
    (output_path / "qds_star.csv").write_text("\n".join(qds_lines) + "\n", encoding="utf-8")
