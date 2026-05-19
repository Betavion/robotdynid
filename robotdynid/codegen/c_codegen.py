"""Simple, explicit C code generation for regressor and prediction kernels."""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp

from robotdynid.symbolic import BaseRegressorBundle
from .api_spec import CArtifactPaths, CFunctionSpec, CGeneratedCode
from .cse_pipeline import apply_cse


def _symbol_to_c_reference(symbol: sp.Symbol, bundle: BaseRegressorBundle) -> str:
    name = str(symbol)
    q_names = {str(sym): f"q[{idx}]" for idx, sym in enumerate(bundle.context.q)}
    qd_names = {str(sym): f"qd[{idx}]" for idx, sym in enumerate(bundle.context.qd)}
    qdd_names = {str(sym): f"qdd[{idx}]" for idx, sym in enumerate(bundle.context.qdd)}
    qds_names = {str(sym): f"qds[{idx}]" for idx, sym in enumerate(bundle.context.qds)}
    theta_names = {param_name: f"theta_lin[{idx}]" for idx, param_name in enumerate(bundle.linear_parameter_names)}

    for mapping in (q_names, qd_names, qdd_names, qds_names, theta_names):
        if name in mapping:
            return mapping[name]
    return name


def _ccode(expr: sp.Expr, bundle: BaseRegressorBundle) -> str:
    replacements = {symbol: sp.Symbol(_symbol_to_c_reference(symbol, bundle)) for symbol in expr.free_symbols}
    return sp.ccode(expr.xreplace(replacements))


def _flatten_row_major(matrix: sp.Matrix) -> list[sp.Expr]:
    return [matrix[row, col] for row in range(matrix.rows) for col in range(matrix.cols)]


def _emit_c_function(function_name: str, body_lines: list[str], signature: str) -> str:
    lines = [signature, "{"] + [f"  {line}" for line in body_lines] + ["}"]
    return "\n".join(lines)


def _header_guard(stem: str) -> str:
    return f"{stem.upper()}_H"


def _emit_header(spec: CFunctionSpec, signature: str, stem: str) -> str:
    guard = _header_guard(stem)
    return "\n".join(
        [
            f"#ifndef {guard}",
            f"#define {guard}",
            "",
            signature + ";",
            "",
            f"#endif /* {guard} */",
        ]
    )


def _linear_parameter_symbols(bundle: BaseRegressorBundle) -> tuple[sp.Symbol, ...]:
    symbols = sp.symbols(" ".join(bundle.linear_parameter_names), real=True)
    return symbols if isinstance(symbols, tuple) else (symbols,)


def generate_base_regressor_c_function(
    bundle: BaseRegressorBundle,
    *,
    function_name: str = "fill_H_bip_base",
) -> CGeneratedCode:
    """Generate a row-major H(q, qd, qdd, qds) filling function."""
    flattened = _flatten_row_major(bundle.regressor)
    cse_output = apply_cse(flattened)
    body_lines: list[str] = []
    for symbol, expr in cse_output.temporaries:
        body_lines.append(f"const double {symbol} = {_ccode(expr, bundle)};")
    for index, expr in enumerate(cse_output.reduced_expressions):
        body_lines.append(f"H[{index}] = {_ccode(expr, bundle)};")
    spec = CFunctionSpec(
        name=function_name,
        output_name="H",
        argument_names=("q", "qd", "qdd", "qds"),
        docstring="Fill the row-major base regressor H(q, qd, qdd, qds).",
    )
    signature = f"void {function_name}(double *H, const double *q, const double *qd, const double *qdd, const double *qds)"
    return CGeneratedCode(function_spec=spec, source=_emit_c_function(function_name, body_lines, signature))


def generate_prediction_c_function(
    bundle: BaseRegressorBundle,
    *,
    function_name: str = "predict_tau",
) -> CGeneratedCode:
    """Generate tau(q, qd, qdd, qds, theta_lin) directly from the base regressor."""
    linear_symbols = _linear_parameter_symbols(bundle)
    theta_matrix = sp.Matrix(linear_symbols)
    tau_expr = bundle.regressor * theta_matrix
    flattened = [tau_expr[row, 0] for row in range(tau_expr.rows)]
    cse_output = apply_cse(flattened)
    body_lines: list[str] = []
    for symbol, expr in cse_output.temporaries:
        body_lines.append(f"const double {symbol} = {_ccode(expr, bundle)};")
    for index, expr in enumerate(cse_output.reduced_expressions):
        body_lines.append(f"tau[{index}] = {_ccode(expr, bundle)};")
    spec = CFunctionSpec(
        name=function_name,
        output_name="tau",
        argument_names=("q", "qd", "qdd", "qds", "theta_lin"),
        docstring="Predict tau directly from q, qd, qdd, qds and theta_lin.",
    )
    signature = (
        f"void {function_name}(double *tau, const double *q, const double *qd, "
        "const double *qdd, const double *qds, const double *theta_lin)"
    )
    return CGeneratedCode(function_spec=spec, source=_emit_c_function(function_name, body_lines, signature))


def export_c_code_artifacts(
    generated: CGeneratedCode,
    bundle: BaseRegressorBundle,
    output_dir: str | Path,
) -> CArtifactPaths:
    """Write generated C source, header and metadata into output_dir."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = generated.function_spec.name
    source_path = directory / f"{stem}.c"
    header_path = directory / f"{stem}.h"
    metadata_path = directory / f"{stem}.json"

    signature = generated.source.splitlines()[0]
    source_path.write_text(generated.source + "\n", encoding="utf-8")
    header_path.write_text(_emit_header(generated.function_spec, signature, stem) + "\n", encoding="utf-8")

    metadata = {
        "function_name": generated.function_spec.name,
        "output_name": generated.function_spec.output_name,
        "argument_names": list(generated.function_spec.argument_names),
        "dof": bundle.regressor.rows,
        "column_count": bundle.regressor.cols,
        "base_parameter_names": list(bundle.base_parameter_names),
        "linear_parameter_names": list(bundle.linear_parameter_names),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return CArtifactPaths(
        source_path=source_path,
        header_path=header_path,
        metadata_path=metadata_path,
    )
