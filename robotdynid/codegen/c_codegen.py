"""Explicit C/C++ code generation for regressor and prediction kernels."""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp

from robotdynid.symbolic import BaseRegressorBundle

from .api_spec import CArtifactPaths, CFunctionSpec, CGeneratedCode, CodegenConfig
from .cse_pipeline import apply_cse


def _validate_language(language: str) -> str:
    lowered = language.lower()
    if lowered not in {"c", "cpp"}:
        raise ValueError("language must be either 'c' or 'cpp'.")
    return lowered


def _symbol_to_reference(symbol: sp.Symbol, bundle: BaseRegressorBundle, temp_index: dict[str, int]) -> str:
    name = str(symbol)
    if name in temp_index:
        return f"tmp[{temp_index[name]}]"

    q_names = {str(sym): f"q[{idx}]" for idx, sym in enumerate(bundle.context.q)}
    qd_names = {str(sym): f"qd[{idx}]" for idx, sym in enumerate(bundle.context.qd)}
    qdd_names = {str(sym): f"qdd[{idx}]" for idx, sym in enumerate(bundle.context.qdd)}
    stribeck_names = {
        str(sym): f"stribeck_parameters[{idx}]" for idx, sym in enumerate(bundle.context.stribeck_parameters)
    }
    parameter_names = {
        param_name: f"linear_parameters[{idx}]" for idx, param_name in enumerate(bundle.linear_parameter_names)
    }

    for mapping in (q_names, qd_names, qdd_names, stribeck_names, parameter_names):
        if name in mapping:
            return mapping[name]
    return name


def _ccode(expr: sp.Expr, bundle: BaseRegressorBundle, temp_index: dict[str, int]) -> str:
    replacements = {
        symbol: sp.Symbol(_symbol_to_reference(symbol, bundle, temp_index))
        for symbol in expr.free_symbols
    }
    return sp.ccode(expr.xreplace(replacements))


def _flatten_row_major(matrix: sp.Matrix) -> list[sp.Expr]:
    return [matrix[row, col] for row in range(matrix.rows) for col in range(matrix.cols)]


def _header_guard(stem: str, language: str) -> str:
    suffix = "HPP" if language == "cpp" else "H"
    return f"{stem.upper()}_{suffix}"


def _c_signature(spec: CFunctionSpec) -> str:
    if spec.argument_names == ("q", "qd", "qdd", "stribeck_parameters"):
        return (
            f"void {spec.name}(double *{spec.output_name}, const double *q, const double *qd, "
            "const double *qdd, const double *stribeck_parameters)"
        )
    return (
        f"void {spec.name}(double *{spec.output_name}, const double *q, const double *qd, "
        "const double *qdd, const double *stribeck_parameters, const double *linear_parameters)"
    )


def _cpp_method_name(spec: CFunctionSpec) -> str:
    return "FillBaseRegressor" if spec.output_name == "H" else "PredictTau"


def _helper_parameters(spec: CFunctionSpec) -> str:
    parameters = [
        "double *tmp",
        "const double *q",
        "const double *qd",
        "const double *qdd",
        "const double *stribeck_parameters",
    ]
    if "linear_parameters" in spec.argument_names:
        parameters.append("const double *linear_parameters")
    return ", ".join(parameters)


def _helper_arguments(spec: CFunctionSpec) -> str:
    arguments = ["tmp", "q", "qd", "qdd", "stribeck_parameters"]
    if "linear_parameters" in spec.argument_names:
        arguments.append("linear_parameters")
    return ", ".join(arguments)


def _cpp_declaration(spec: CFunctionSpec, config: CodegenConfig) -> str:
    method_name = _cpp_method_name(spec)
    if spec.output_name == "H":
        return (
            f"static void {method_name}(double *{spec.output_name}, const double *q, const double *qd, "
            "const double *qdd, const double *stribeck_parameters)"
        )
    return (
        f"static void {method_name}(double *{spec.output_name}, const double *q, const double *qd, "
        "const double *qdd, const double *stribeck_parameters, const double *linear_parameters)"
    )


def _emit_c_function(signature: str, body_lines: list[str]) -> str:
    return "\n".join([signature, "{"] + [f"  {line}" for line in body_lines] + ["}"])


def _emit_cpp_class_header(
    spec: CFunctionSpec,
    config: CodegenConfig,
    *,
    dof: int,
    columns: int,
    temporary_count: int,
    program_helper_count: int,
    helper_count: int,
) -> str:
    namespace_open = f"namespace {config.namespace} {{"
    namespace_close = "}  // namespace " + config.namespace
    method_decl = _cpp_declaration(spec, config) + ";"
    program_helper_lines = [
        f"  static void ComputeProgramBlock{idx}({_helper_parameters(spec)});"
        for idx in range(program_helper_count)
    ]
    helper_lines = [
        f"  static void ComputeBlock{idx}({_helper_parameters(spec)});"
        for idx in range(helper_count)
    ]
    return "\n".join(
        [
            "#pragma once",
            "",
            "#include <cstddef>",
            "",
            namespace_open,
            f"class {config.class_name} final {{",
            "public:",
            f"  static constexpr std::size_t kDof = {dof};",
            f"  static constexpr std::size_t kColumnCount = {columns};",
            f"  static constexpr std::size_t kTemporaryCount = {temporary_count};",
            f"  {method_decl}",
            "private:",
            *program_helper_lines,
            *helper_lines,
            "};",
            namespace_close,
            "",
        ]
    )


def _emit_c_header(spec: CFunctionSpec, language: str, stem: str) -> str:
    guard = _header_guard(stem, language)
    declaration = _c_signature(spec) + ";"
    return "\n".join(
        [
            f"#ifndef {guard}",
            f"#define {guard}",
            "",
            declaration,
            "",
            f"#endif /* {guard} */",
        ]
    )


def _temporary_blocks(
    temporaries: tuple[tuple[sp.Symbol, sp.Expr], ...],
    *,
    helper_block_size: int,
) -> list[tuple[int, tuple[tuple[sp.Symbol, sp.Expr], ...]]]:
    if helper_block_size < 1:
        raise ValueError("helper_block_size must be >= 1.")
    blocks = []
    for start in range(0, len(temporaries), helper_block_size):
        block = temporaries[start : start + helper_block_size]
        blocks.append((len(blocks), block))
    return blocks


def _emit_temporary_assignments(
    assignments: tuple[tuple[sp.Symbol, sp.Expr], ...],
    bundle: BaseRegressorBundle,
    temp_index: dict[str, int],
) -> list[str]:
    return [
        f"tmp[{temp_index[str(symbol)]}] = {_ccode(expr, bundle, temp_index)};"
        for symbol, expr in assignments
    ]


def _emit_c_helper(
    spec: CFunctionSpec,
    block_index: int,
    block: tuple[tuple[sp.Symbol, sp.Expr], ...],
    bundle: BaseRegressorBundle,
    temp_index: dict[str, int],
) -> str:
    helper_name = f"{spec.name}_block_{block_index}"
    signature = f"static inline void {helper_name}({_helper_parameters(spec)})"
    return _emit_c_function(signature, _emit_temporary_assignments(block, bundle, temp_index))


def _emit_program_c_helper(
    spec: CFunctionSpec,
    block_index: int,
    block,
    bundle: BaseRegressorBundle,
    temp_index: dict[str, int],
) -> str:
    helper_name = f"{spec.name}_program_block_{block_index}"
    signature = f"static inline void {helper_name}({_helper_parameters(spec)})"
    body = _emit_temporary_assignments(
        tuple((temp.symbol, temp.expr) for temp in block.temporaries),
        bundle,
        temp_index,
    )
    return _emit_c_function(signature, body)


def _emit_cpp_helper(
    spec: CFunctionSpec,
    config: CodegenConfig,
    block_index: int,
    block: tuple[tuple[sp.Symbol, sp.Expr], ...],
    bundle: BaseRegressorBundle,
    temp_index: dict[str, int],
) -> str:
    signature = (
        f"void {config.namespace}::{config.class_name}::ComputeBlock{block_index}"
        f"({_helper_parameters(spec)})"
    )
    return _emit_c_function(signature, _emit_temporary_assignments(block, bundle, temp_index))


def _emit_program_cpp_helper(
    spec: CFunctionSpec,
    config: CodegenConfig,
    block_index: int,
    block,
    bundle: BaseRegressorBundle,
    temp_index: dict[str, int],
) -> str:
    signature = (
        f"void {config.namespace}::{config.class_name}::ComputeProgramBlock{block_index}"
        f"({_helper_parameters(spec)})"
    )
    body = _emit_temporary_assignments(
        tuple((temp.symbol, temp.expr) for temp in block.temporaries),
        bundle,
        temp_index,
    )
    return _emit_c_function(signature, body)


def _emit_main_body(
    spec: CFunctionSpec,
    config: CodegenConfig,
    bundle: BaseRegressorBundle,
    flattened_outputs: list[sp.Expr],
    temp_index: dict[str, int],
    program_helper_count: int,
    helper_count: int,
    language: str,
) -> list[str]:
    lines: list[str] = []
    if temp_index:
        lines.append(f"double tmp[{len(temp_index)}] = {{0.0}};")
        helper_arguments = _helper_arguments(spec)
        for block_index in range(program_helper_count):
            if language == "cpp":
                lines.append(f"{config.class_name}::ComputeProgramBlock{block_index}({helper_arguments});")
            else:
                lines.append(f"{spec.name}_program_block_{block_index}({helper_arguments});")
        for block_index in range(helper_count):
            if language == "cpp":
                lines.append(f"{config.class_name}::ComputeBlock{block_index}({helper_arguments});")
            else:
                lines.append(f"{spec.name}_block_{block_index}({helper_arguments});")
    for index, expr in enumerate(flattened_outputs):
        lines.append(f"{spec.output_name}[{index}] = {_ccode(expr, bundle, temp_index)};")
    return lines


def _apply_fixed_stribeck_parameters(
    expressions: list[sp.Expr],
    bundle: BaseRegressorBundle,
    config: CodegenConfig,
) -> list[sp.Expr]:
    if config.fixed_stribeck_parameters is None:
        return expressions
    if len(config.fixed_stribeck_parameters) != len(bundle.context.stribeck_parameters):
        raise ValueError(f"fixed_stribeck_parameters must have length {len(bundle.context.stribeck_parameters)}.")
    substitutions = {
        symbol: sp.Float(value)
        for symbol, value in zip(bundle.context.stribeck_parameters, config.fixed_stribeck_parameters)
    }
    return [sp.sympify(expr).xreplace(substitutions) for expr in expressions]


def _build_generated_function(
    bundle: BaseRegressorBundle,
    spec: CFunctionSpec,
    output_expressions: list[sp.Expr],
    config: CodegenConfig,
) -> CGeneratedCode:
    language = _validate_language(config.language)
    output_expressions = _apply_fixed_stribeck_parameters(output_expressions, bundle, config)
    cse_output = apply_cse(output_expressions, symbol_prefix="aux")
    program_inputs = [expr for _, expr in cse_output.temporaries] + list(cse_output.reduced_expressions)
    program_blocks = list(enumerate(bundle.program.required_blocks(program_inputs)))
    program_temporaries = [temp for _, block in program_blocks for temp in block.temporaries]
    program_temp_index = {str(temp.symbol): idx for idx, temp in enumerate(program_temporaries)}

    cse_temp_index = {
        str(symbol): len(program_temporaries) + idx
        for idx, (symbol, _) in enumerate(cse_output.temporaries)
    }
    temp_index = program_temp_index | cse_temp_index
    blocks = _temporary_blocks(cse_output.temporaries, helper_block_size=config.helper_block_size)

    if language == "cpp":
        program_helper_defs = tuple(
            _emit_program_cpp_helper(spec, config, block_index, block, bundle, temp_index)
            for block_index, block in program_blocks
        )
        helper_defs = tuple(
            _emit_cpp_helper(spec, config, block_index, block, bundle, temp_index)
            for block_index, block in blocks
        )
        declaration = _emit_cpp_class_header(
            spec,
            config,
            dof=bundle.regressor_program.rows if spec.output_name == "H" else len(output_expressions),
            columns=bundle.regressor_program.cols,
            temporary_count=len(program_temporaries) + len(cse_output.temporaries),
            program_helper_count=len(program_blocks),
            helper_count=len(blocks),
        )
        main_signature = (
            f"void {config.namespace}::{config.class_name}::{_cpp_method_name(spec)}"
            + _cpp_declaration(spec, config)[len(f"static void {_cpp_method_name(spec)}") :]
        )
    else:
        program_helper_defs = tuple(
            _emit_program_c_helper(spec, block_index, block, bundle, temp_index)
            for block_index, block in program_blocks
        )
        helper_defs = tuple(
            _emit_c_helper(spec, block_index, block, bundle, temp_index)
            for block_index, block in blocks
        )
        declaration = _emit_c_header(spec, language, spec.name)
        main_signature = _c_signature(spec)

    body_lines = _emit_main_body(
        spec,
        config,
        bundle,
        list(cse_output.reduced_expressions),
        temp_index,
        len(program_blocks),
        len(blocks),
        language,
    )
    definition = _emit_c_function(main_signature, body_lines)
    return CGeneratedCode(
        language=language,
        function_spec=spec,
        declaration=declaration,
        definition=definition,
        helper_definitions=program_helper_defs + helper_defs,
    )


def generate_base_regressor_c_function(
    bundle: BaseRegressorBundle,
    *,
    function_name: str = "fill_H_bip_base",
    config: CodegenConfig = CodegenConfig(language="c"),
) -> CGeneratedCode:
    """Generate a row-major H(q, qd, qdd, stribeck_parameters) filling function."""
    spec = CFunctionSpec(
        name=function_name,
        output_name="H",
        argument_names=("q", "qd", "qdd", "stribeck_parameters"),
        docstring="Fill the row-major base regressor H(q, qd, qdd, stribeck_parameters).",
    )
    flattened = _flatten_row_major(bundle.regressor_program)
    return _build_generated_function(bundle, spec, flattened, config)


def generate_prediction_c_function(
    bundle: BaseRegressorBundle,
    *,
    function_name: str = "predict_tau",
    config: CodegenConfig = CodegenConfig(language="c"),
) -> CGeneratedCode:
    """Generate tau(q, qd, qdd, stribeck_parameters, linear_parameters) directly from the base regressor."""
    linear_symbols = sp.symbols(" ".join(bundle.linear_parameter_names), real=True)
    linear_symbols = linear_symbols if isinstance(linear_symbols, tuple) else (linear_symbols,)
    tau_expr = bundle.regressor_program * sp.Matrix(linear_symbols)
    spec = CFunctionSpec(
        name=function_name,
        output_name="tau",
        argument_names=("q", "qd", "qdd", "stribeck_parameters", "linear_parameters"),
        docstring="Predict tau directly from q, qd, qdd, stribeck_parameters and linear_parameters.",
    )
    flattened = [tau_expr[row, 0] for row in range(tau_expr.rows)]
    return _build_generated_function(bundle, spec, flattened, config)


def generate_base_regressor_cpp_function(
    bundle: BaseRegressorBundle,
    *,
    function_name: str = "fill_H_bip_base",
    config: CodegenConfig = CodegenConfig(language="cpp"),
) -> CGeneratedCode:
    """Generate a ROS2-friendly C++ kernel class for H(q, qd, qdd, stribeck_parameters)."""
    return generate_base_regressor_c_function(bundle, function_name=function_name, config=config)


def generate_prediction_cpp_function(
    bundle: BaseRegressorBundle,
    *,
    function_name: str = "predict_tau",
    config: CodegenConfig = CodegenConfig(language="cpp"),
) -> CGeneratedCode:
    """Generate a ROS2-friendly C++ kernel class for tau prediction."""
    return generate_prediction_c_function(bundle, function_name=function_name, config=config)


def export_c_code_artifacts(
    generated: CGeneratedCode,
    bundle: BaseRegressorBundle,
    output_dir: str | Path,
) -> CArtifactPaths:
    """Write generated source, header and metadata into output_dir."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = generated.function_spec.name
    source_suffix = ".cpp" if generated.language == "cpp" else ".c"
    header_suffix = ".hpp" if generated.language == "cpp" else ".h"
    source_path = directory / f"{stem}{source_suffix}"
    header_path = directory / f"{stem}{header_suffix}"
    metadata_path = directory / f"{stem}.json"

    source_lines: list[str] = []
    if generated.language == "cpp":
        source_lines.append(f'#include "{stem}{header_suffix}"')
        source_lines.append("#include <cmath>")
        source_lines.append("")
        source_lines.extend(
            [
                "using std::cos;",
                "using std::exp;",
                "using std::fabs;",
                "using std::pow;",
                "using std::sin;",
                "using std::sqrt;",
            ]
        )
        source_lines.append("")
    else:
        source_lines.append(f'#include "{stem}{header_suffix}"')
        source_lines.append("#include <math.h>")
        source_lines.append("")
    source_lines.extend(generated.helper_definitions)
    if generated.helper_definitions:
        source_lines.append("")
    source_lines.append(generated.definition)
    source_path.write_text("\n".join(source_lines) + "\n", encoding="utf-8")
    header_path.write_text(generated.declaration + "\n", encoding="utf-8")

    metadata = {
        "language": generated.language,
        "function_name": generated.function_spec.name,
        "output_name": generated.function_spec.output_name,
        "argument_names": list(generated.function_spec.argument_names),
        "dof": bundle.regressor_program.rows,
        "column_count": bundle.regressor_program.cols,
        "base_parameter_names": list(bundle.base_parameter_names),
        "linear_parameter_names": list(bundle.linear_parameter_names),
        "helper_count": len(generated.helper_definitions),
        "temporary_count": sum(helper.count("tmp[") for helper in generated.helper_definitions),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return CArtifactPaths(source_path=source_path, header_path=header_path, metadata_path=metadata_path)
