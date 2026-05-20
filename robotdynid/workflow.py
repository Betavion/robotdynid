"""Generic identification workflow for serial robot URDF models and motion data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .codegen import (
    CodegenConfig,
    export_c_code_artifacts,
    generate_base_regressor_c_function,
    generate_base_regressor_cpp_function,
    generate_prediction_c_function,
    generate_prediction_cpp_function,
)
from .identify import (
    AlternatingIdentifyConfig,
    CsvDatasetConfig,
    IdentificationDataset,
    MotionTorqueCsvDatasetConfig,
    identify_with_stribeck,
    load_identification_dataset_from_csv,
    load_identification_dataset_from_motion_and_torque_csv,
    stack_regression_problem,
)
from .numeric import (
    BaseSelectionStrategy,
    StateSamplingConfig,
    build_pinocchio_model,
    build_pinocchio_regressor_evaluator,
    sample_model_state_dataset,
    select_base_parameters,
)
from .reporting import build_result_payload, save_identification_artifacts, save_prediction_plot
from .io import load_robot_from_urdf
from .symbolic import SymbolicBuildOptions, build_base_regressor, build_standard_regressor


@dataclass(frozen=True)
class IdentificationWorkflowConfig:
    """Configuration for a complete URDF + motion-data identification run."""

    urdf_path: str | Path
    dof: int
    csv_path: str | Path | None = None
    motion_csv_path: str | Path | None = None
    torque_csv_path: str | Path | None = None
    stride: int = 100
    max_samples: int | None = 3000
    selection_samples: int = 800
    selection_source: str = "model"
    selection_random_seed: int = 42
    selection_velocity_scale: float = 0.5
    selection_acceleration_scale: float = 0.5
    position_offsets: tuple[float, ...] | None = None
    qds_init: np.ndarray | None = None
    max_iterations: int = 8
    chunk_size: int | None = None
    output_dir: str | Path | None = None
    prediction_plot_stride: int = 50
    save_prediction_plot: bool = True
    torque_weighting: str | None = "torque_std"
    optimizer_kwargs: dict[str, object] | None = None
    export_code: bool = False
    codegen_languages: tuple[str, ...] = ("c",)
    codegen_output_subdir: str = "codegen"
    codegen_helper_block_size: int = 64
    codegen_namespace: str = "robotdynid::generated"
    codegen_class_name: str = "RegressorKernel"


def _resolve_dataset(config: IdentificationWorkflowConfig) -> IdentificationDataset:
    if config.motion_csv_path is not None or config.torque_csv_path is not None:
        if config.motion_csv_path is None or config.torque_csv_path is None:
            raise ValueError("motion_csv_path and torque_csv_path must be provided together.")
        return load_identification_dataset_from_motion_and_torque_csv(
            config.motion_csv_path,
            config.torque_csv_path,
            MotionTorqueCsvDatasetConfig(
                dof=config.dof,
                position_offsets=config.position_offsets,
                torque_weighting=config.torque_weighting,
                stride=config.stride,
                max_samples=config.max_samples,
            ),
        )
    if config.csv_path is None:
        raise ValueError("Either csv_path or motion_csv_path/torque_csv_path must be provided.")
    return load_identification_dataset_from_csv(
        config.csv_path,
        CsvDatasetConfig(
            dof=config.dof,
            position_offsets=config.position_offsets,
            torque_weighting=config.torque_weighting,
            stride=config.stride,
            max_samples=config.max_samples,
        ),
    )


def _resolve_selection_dataset(
    dataset: IdentificationDataset,
    config: IdentificationWorkflowConfig,
    pin_bundle,
) -> IdentificationDataset:
    if config.selection_source == "model":
        return sample_model_state_dataset(
            pin_bundle,
            StateSamplingConfig(
                sample_count=config.selection_samples,
                random_seed=config.selection_random_seed,
                velocity_scale=config.selection_velocity_scale,
                acceleration_scale=config.selection_acceleration_scale,
            ),
        )
    if config.selection_source == "data":
        if dataset.sample_count <= config.selection_samples:
            return dataset
        indices = np.linspace(0, dataset.sample_count - 1, num=config.selection_samples, dtype=int)
        return IdentificationDataset(
            q=dataset.q[indices],
            qd=dataset.qd[indices],
            qdd=dataset.qdd[indices],
            tau=dataset.tau[indices],
            sample_weights=(
                dataset.sample_weights[
                    np.concatenate([np.arange(index * dataset.dof, (index + 1) * dataset.dof) for index in indices])
                ]
                if dataset.sample_weights is not None
                else None
            ),
        )
    raise ValueError("selection_source must be either 'model' or 'data'.")


def _resolve_qds_init(raw: np.ndarray | None, dof: int) -> np.ndarray:
    if raw is None:
        return np.full((dof,), 0.2, dtype=float)
    values = np.asarray(raw, dtype=float).reshape(-1)
    if values.shape[0] != dof:
        raise ValueError(f"qds_init must have length {dof}, got {values.shape[0]}.")
    return values


def _resolve_csv_source(config: IdentificationWorkflowConfig) -> str | dict[str, str]:
    if config.motion_csv_path is not None and config.torque_csv_path is not None:
        return {
            "motion_csv": str(config.motion_csv_path),
            "torque_csv": str(config.torque_csv_path),
        }
    return str(config.csv_path)


def _export_codegen_artifacts(
    config: IdentificationWorkflowConfig,
    *,
    base_metadata,
) -> dict[str, list[str]]:
    if config.output_dir is None or not config.export_code:
        return {}

    symbolic_robot = load_robot_from_urdf(config.urdf_path)
    symbolic_options = SymbolicBuildOptions(enabled_joint_dynamics_groups=("fv", "fc", "fd"), include_qds=True)
    standard_bundle = build_standard_regressor(symbolic_robot, symbolic_options, simplify=False)
    base_bundle = build_base_regressor(standard_bundle, base_metadata)

    outputs: dict[str, list[str]] = {}
    for language in config.codegen_languages:
        codegen_config = CodegenConfig(
            language=language,
            namespace=config.codegen_namespace,
            class_name=config.codegen_class_name,
            helper_block_size=config.codegen_helper_block_size,
        )
        language_dir = Path(config.output_dir) / config.codegen_output_subdir / language
        if language.lower() == "cpp":
            base_generated = generate_base_regressor_cpp_function(base_bundle, config=codegen_config)
            prediction_generated = generate_prediction_cpp_function(base_bundle, config=codegen_config)
        else:
            base_generated = generate_base_regressor_c_function(base_bundle, config=codegen_config)
            prediction_generated = generate_prediction_c_function(base_bundle, config=codegen_config)
        base_paths = export_c_code_artifacts(base_generated, base_bundle, language_dir)
        prediction_paths = export_c_code_artifacts(prediction_generated, base_bundle, language_dir)
        outputs[language] = [
            str(base_paths.source_path),
            str(base_paths.header_path),
            str(base_paths.metadata_path),
            str(prediction_paths.source_path),
            str(prediction_paths.header_path),
            str(prediction_paths.metadata_path),
        ]
    return outputs


def run_identification_workflow(config: IdentificationWorkflowConfig) -> dict[str, object]:
    """Run the complete identification workflow for a serial robot."""
    dataset = _resolve_dataset(config)
    pin_bundle = build_pinocchio_model(config.urdf_path)
    standard_evaluator = build_pinocchio_regressor_evaluator(
        pin_bundle,
        enabled_joint_dynamics_groups=tuple(),
    )
    selection_dataset = _resolve_selection_dataset(dataset, config, pin_bundle)
    inertial_regressor, _ = stack_regression_problem(selection_dataset, standard_evaluator)
    base_metadata = select_base_parameters(
        inertial_regressor,
        standard_param_names=list(standard_evaluator.linear_parameter_names),
        strategy=BaseSelectionStrategy(scale_columns=True),
    )

    identify_evaluator = build_pinocchio_regressor_evaluator(
        pin_bundle,
        enabled_joint_dynamics_groups=("fv", "fc", "fd"),
        base_metadata=base_metadata,
    )
    qds_init = _resolve_qds_init(config.qds_init, identify_evaluator.qds_size)
    result = identify_with_stribeck(
        dataset,
        identify_evaluator,
        AlternatingIdentifyConfig(
            qds_init=qds_init,
            max_iterations=config.max_iterations,
            chunk_size=config.chunk_size,
            optimizer_kwargs=config.optimizer_kwargs or {"ftol": 1e-9, "xtol": 1e-9, "gtol": 1e-9},
        ),
    )

    payload = build_result_payload(
        urdf_path=str(config.urdf_path),
        csv_source=_resolve_csv_source(config),
        dataset=dataset,
        selection_dataset=selection_dataset,
        selection_source=config.selection_source,
        stride=config.stride,
        max_samples=config.max_samples,
        chunk_size=config.chunk_size,
        used_legacy_mdh_offsets=False,
        base_metadata=base_metadata,
        identification_result=result,
    )

    if config.output_dir is not None:
        save_identification_artifacts(config.output_dir, payload, base_metadata)
        if config.save_prediction_plot:
            save_prediction_plot(
                config.output_dir,
                dataset=dataset,
                evaluator=identify_evaluator,
                theta_lin=result.theta_lin,
                qds=result.qds,
                stride=config.prediction_plot_stride,
            )
        codegen_outputs = _export_codegen_artifacts(config, base_metadata=base_metadata)
        if codegen_outputs:
            payload["codegen_outputs"] = codegen_outputs

    return payload
