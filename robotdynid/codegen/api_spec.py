"""Small data structures for generated C code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CFunctionSpec:
    """Description of one generated C function."""

    name: str
    output_name: str
    argument_names: tuple[str, ...]
    docstring: str = ""


@dataclass(frozen=True)
class CGeneratedCode:
    """C code generation result."""

    function_spec: CFunctionSpec
    source: str


@dataclass(frozen=True)
class CArtifactPaths:
    """Filesystem locations for generated C source, header and metadata."""

    source_path: Path
    header_path: Path
    metadata_path: Path
