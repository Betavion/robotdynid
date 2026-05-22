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
    optimization: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a serializable identification result payload."""
    return {
        "urdf": urdf_path,
        "csv": csv_source,
        "sample_count": dataset.sample_count,
        "selection_sample_count": selection_dataset.sample_count,
        "selection_source": selection_source,
        "base_rank": base_metadata.rank,
        "linear_parameters": identification_result.linear_parameters.tolist(),
        "stribeck_parameters": identification_result.stribeck_parameters.tolist(),
        "objective_history": list(identification_result.objective_history),
        "rmse_history": [rmse.tolist() for rmse in identification_result.rmse_history],
        "linear_parameter_names": list(identification_result.linear_parameter_names),
        "stribeck_parameter_names": [
            f"stribeck{index + 1}" for index in range(identification_result.stribeck_parameters.shape[0])
        ],
        "stride": stride,
        "max_samples": max_samples,
        "chunk_size": chunk_size,
        "optimization": optimization or {},
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

    linear_names = payload["linear_parameter_names"]  # type: ignore[index]
    linear_values = payload["linear_parameters"]  # type: ignore[index]
    linear_lines = ["name,value"] + [f"{name},{value}" for name, value in zip(linear_names, linear_values)]
    (output_path / "identified_linear_parameters.csv").write_text("\n".join(linear_lines) + "\n", encoding="utf-8")

    stribeck_names = payload["stribeck_parameter_names"]  # type: ignore[index]
    stribeck_values = payload["stribeck_parameters"]  # type: ignore[index]
    stribeck_lines = [
        "name,value",
        *[f"{name},{value}" for name, value in zip(stribeck_names, stribeck_values)],
    ]
    (output_path / "identified_stribeck_parameters.csv").write_text(
        "\n".join(stribeck_lines) + "\n",
        encoding="utf-8",
    )
