"""Small data structures for generated code artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodegenConfig:
    """Configuration for C/C++ code generation."""

    language: str = "c"
    namespace: str = "robotdynid::generated"
    class_name: str = "RegressorKernel"
    helper_block_size: int = 64
    fixed_stribeck_parameters: tuple[float, ...] | None = None


@dataclass(frozen=True)
class CFunctionSpec:
    """Description of one generated C function."""

    name: str
    output_name: str
    argument_names: tuple[str, ...]
    docstring: str = ""


@dataclass(frozen=True)
class CGeneratedCode:
    """Generated function code in C or C++ form."""

    language: str
    function_spec: CFunctionSpec
    declaration: str
    definition: str
    helper_definitions: tuple[str, ...]


@dataclass(frozen=True)
class CArtifactPaths:
    """Filesystem locations for generated source, header and metadata."""

    source_path: Path
    header_path: Path
    metadata_path: Path
