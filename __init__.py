"""Repository-root import shim for local development and tests."""

from __future__ import annotations

from importlib import util
from pathlib import Path
import sys


_PACKAGE_DIR = Path(__file__).resolve().parent / "robotdynid"
_SPEC = util.spec_from_file_location(
    __name__,
    _PACKAGE_DIR / "__init__.py",
    submodule_search_locations=[str(_PACKAGE_DIR)],
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load robotdynid package from {_PACKAGE_DIR}")

_MODULE = sys.modules[__name__]
_MODULE.__file__ = str(_PACKAGE_DIR / "__init__.py")
_MODULE.__path__ = [str(_PACKAGE_DIR)]  # type: ignore[attr-defined]
_SPEC.loader.exec_module(_MODULE)
