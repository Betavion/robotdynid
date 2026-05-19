"""Assembly helpers for building symbolic/numeric base-identification pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotdynid.core.robot_model import BaseParamMetadata, RobotModel
from robotdynid.identify import IdentificationDataset, stack_regression_problem
from robotdynid.numeric import BaseSelectionStrategy, select_base_parameters
from robotdynid.symbolic import (
    BaseRegressorBundle,
    LinearRegressorEvaluator,
    SymbolicBuildOptions,
    SymbolicRegressorBundle,
    build_base_regressor,
    build_base_regressor_evaluator,
    build_standard_regressor,
    build_standard_regressor_evaluator,
)


@dataclass(frozen=True)
class BaseIdentificationPipeline:
    """Assembled symbolic/numeric artifacts required by identification and code generation."""

    robot: RobotModel
    standard_bundle: SymbolicRegressorBundle
    standard_evaluator: LinearRegressorEvaluator
    base_metadata: BaseParamMetadata
    base_bundle: BaseRegressorBundle
    base_evaluator: LinearRegressorEvaluator


def build_base_identification_pipeline(
    robot: RobotModel,
    selection_dataset: IdentificationDataset,
    *,
    selection_qds: np.ndarray | None = None,
    symbolic_options: SymbolicBuildOptions = SymbolicBuildOptions(),
    selection_strategy: BaseSelectionStrategy = BaseSelectionStrategy(),
) -> BaseIdentificationPipeline:
    """Assemble a symbolic regressor, numeric base selection and compiled evaluators."""
    standard_bundle = build_standard_regressor(robot, symbolic_options)
    standard_evaluator = build_standard_regressor_evaluator(standard_bundle)
    if standard_evaluator.qds_size > 0 and selection_qds is None:
        raise ValueError("selection_qds must be provided because the selected symbolic model depends on qds.")

    standard_param_count = len(standard_bundle.context.standard_params)
    regressor_matrix, _ = stack_regression_problem(selection_dataset, standard_evaluator, qds=selection_qds)
    inertial_regressor = regressor_matrix[:, :standard_param_count]
    base_metadata = select_base_parameters(
        inertial_regressor,
        standard_param_names=[str(symbol) for symbol in standard_bundle.context.standard_params],
        strategy=selection_strategy,
    )
    base_bundle = build_base_regressor(standard_bundle, base_metadata)
    base_evaluator = build_base_regressor_evaluator(base_bundle)
    return BaseIdentificationPipeline(
        robot=robot,
        standard_bundle=standard_bundle,
        standard_evaluator=standard_evaluator,
        base_metadata=base_metadata,
        base_bundle=base_bundle,
        base_evaluator=base_evaluator,
    )
