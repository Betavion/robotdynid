"""Canonical robot data structures for the new URDF-first pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import sympy as sp

from .params import generate_standard_parameter_names


def _as_column(vec: sp.Matrix) -> sp.Matrix:
    if vec.shape == (3, 1):
        return vec
    if vec.shape == (1, 3):
        return vec.T
    raise ValueError(f"Expected a 3-vector, got shape {vec.shape}.")


@dataclass
class SpatialInertia:
    """Barycentric inertial data expressed about a reference-frame origin."""

    mass: sp.Expr
    first_moment: sp.Matrix
    inertia_origin: sp.Matrix
    reference_frame: str

    def __post_init__(self) -> None:
        self.first_moment = _as_column(sp.Matrix(self.first_moment))
        self.inertia_origin = sp.Matrix(self.inertia_origin)
        if self.inertia_origin.shape != (3, 3):
            raise ValueError("inertia_origin must be a 3x3 matrix.")

    @classmethod
    def zero(cls, reference_frame: str) -> "SpatialInertia":
        return cls(
            mass=sp.Integer(0),
            first_moment=sp.zeros(3, 1),
            inertia_origin=sp.zeros(3, 3),
            reference_frame=reference_frame,
        )

    @property
    def center_of_mass(self) -> sp.Matrix:
        if self.mass == 0:
            return sp.zeros(3, 1)
        return sp.simplify(self.first_moment / self.mass)

    def with_reference_frame(self, reference_frame: str) -> "SpatialInertia":
        return SpatialInertia(
            mass=self.mass,
            first_moment=self.first_moment.copy(),
            inertia_origin=self.inertia_origin.copy(),
            reference_frame=reference_frame,
        )

    def __add__(self, other: "SpatialInertia") -> "SpatialInertia":
        if self.reference_frame != other.reference_frame:
            raise ValueError("Can only add inertias expressed in the same reference frame.")
        return SpatialInertia(
            mass=sp.simplify(self.mass + other.mass),
            first_moment=self.first_moment + other.first_moment,
            inertia_origin=self.inertia_origin + other.inertia_origin,
            reference_frame=self.reference_frame,
        )


@dataclass
class LinkModel:
    """A dynamic body anchored at a motion-joint child frame."""

    name: str
    anchor_frame: str
    inertia: SpatialInertia
    source_links: tuple[str, ...] = field(default_factory=tuple)

    def append_source_link(self, link_name: str, inertia: SpatialInertia) -> None:
        self.inertia = self.inertia + inertia
        self.source_links = (*self.source_links, link_name)


@dataclass(frozen=True)
class JointModel:
    """A 1-DOF motion joint in canonical chain order."""

    name: str
    joint_type: str
    parent_frame: str
    child_frame: str
    axis: sp.Matrix
    placement: sp.Matrix
    raw_parent_link: str
    raw_child_link: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", _as_column(sp.Matrix(self.axis)))
        placement = sp.Matrix(self.placement)
        if placement.shape != (4, 4):
            raise ValueError("placement must be a 4x4 homogeneous transform.")
        object.__setattr__(self, "placement", placement)


@dataclass
class RobotModel:
    """Canonical serial robot model for symbolic and numeric backends."""

    name: str
    gravity: tuple[float, float, float]
    base_frame: str
    joints: list[JointModel]
    links: list[LinkModel]
    base_inertia: SpatialInertia | None = None

    @property
    def dof(self) -> int:
        return len(self.joints)

    @property
    def joint_names(self) -> list[str]:
        return [joint.name for joint in self.joints]

    @property
    def link_names(self) -> list[str]:
        return [link.name for link in self.links]

    @property
    def standard_parameter_names(self) -> list[str]:
        return generate_standard_parameter_names(len(self.links))


@dataclass
class BaseParamMetadata:
    """Selection result for numerical base-parameter extraction."""

    rank: int
    keep_indices: list[int]
    dependent_indices: list[int]
    dependency_matrix: np.ndarray
    standard_param_names: list[str]
    base_param_names: list[str]
    column_permutation: list[int]
    qr_rank: int
    svd_rank: int
    tolerance: float

    def to_dict(self) -> dict[str, object]:
        """Serialize base-parameter metadata into plain Python objects."""
        return {
            "rank": self.rank,
            "keep_indices": list(self.keep_indices),
            "dependent_indices": list(self.dependent_indices),
            "dependency_matrix": np.asarray(self.dependency_matrix, dtype=float).tolist(),
            "standard_param_names": list(self.standard_param_names),
            "base_param_names": list(self.base_param_names),
            "column_permutation": list(self.column_permutation),
            "qr_rank": self.qr_rank,
            "svd_rank": self.svd_rank,
            "tolerance": float(self.tolerance),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "BaseParamMetadata":
        """Deserialize metadata from a dictionary."""
        return cls(
            rank=int(payload["rank"]),
            keep_indices=[int(value) for value in payload["keep_indices"]],  # type: ignore[index]
            dependent_indices=[int(value) for value in payload["dependent_indices"]],  # type: ignore[index]
            dependency_matrix=np.asarray(payload["dependency_matrix"], dtype=float),  # type: ignore[arg-type]
            standard_param_names=[str(value) for value in payload["standard_param_names"]],  # type: ignore[index]
            base_param_names=[str(value) for value in payload["base_param_names"]],  # type: ignore[index]
            column_permutation=[int(value) for value in payload["column_permutation"]],  # type: ignore[index]
            qr_rank=int(payload["qr_rank"]),
            svd_rank=int(payload["svd_rank"]),
            tolerance=float(payload["tolerance"]),
        )

    def to_json_file(self, path: str | Path) -> None:
        """Write metadata to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json_file(cls, path: str | Path) -> "BaseParamMetadata":
        """Load metadata from a JSON file."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
