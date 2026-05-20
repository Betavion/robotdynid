"""Controlled C/C++ code generation for symbolic regressors and predictions."""

from .api_spec import CArtifactPaths, CFunctionSpec, CGeneratedCode, CodegenConfig
from .c_codegen import (
    export_c_code_artifacts,
    generate_base_regressor_c_function,
    generate_base_regressor_cpp_function,
    generate_prediction_c_function,
    generate_prediction_cpp_function,
)
from .cse_pipeline import CSEOutput, apply_cse

__all__ = [
    "CArtifactPaths",
    "CFunctionSpec",
    "CGeneratedCode",
    "CodegenConfig",
    "CSEOutput",
    "apply_cse",
    "export_c_code_artifacts",
    "generate_base_regressor_c_function",
    "generate_base_regressor_cpp_function",
    "generate_prediction_c_function",
    "generate_prediction_cpp_function",
]
