"""Rigid-body inertia conversion helpers used by the URDF loader."""

from __future__ import annotations

import sympy as sp

from robotdynid.core.robot_model import SpatialInertia


def scalar(value: str | float | int) -> sp.Expr:
    """Convert numeric XML content into a stable SymPy scalar."""
    if isinstance(value, str):
        return sp.nsimplify(value, rational=True)
    return sp.nsimplify(value, rational=True)


def vector3(values: tuple[str | float | int, str | float | int, str | float | int]) -> sp.Matrix:
    return sp.Matrix([[scalar(values[0])], [scalar(values[1])], [scalar(values[2])]])


def skew(vec: sp.Matrix) -> sp.Matrix:
    x, y, z = list(sp.Matrix(vec))
    return sp.Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def rpy_to_rotation(roll: sp.Expr, pitch: sp.Expr, yaw: sp.Expr) -> sp.Matrix:
    """URDF roll-pitch-yaw to rotation matrix."""
    cx, sx = sp.cos(roll), sp.sin(roll)
    cy, sy = sp.cos(pitch), sp.sin(pitch)
    cz, sz = sp.cos(yaw), sp.sin(yaw)
    rx = sp.Matrix([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = sp.Matrix([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = sp.Matrix([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return sp.simplify(rz * ry * rx)


def homogeneous_transform(rotation: sp.Matrix, translation: sp.Matrix) -> sp.Matrix:
    transform = sp.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = sp.Matrix(translation)
    return transform


def split_transform(transform: sp.Matrix) -> tuple[sp.Matrix, sp.Matrix]:
    matrix = sp.Matrix(transform)
    if matrix.shape != (4, 4):
        raise ValueError("Expected a 4x4 homogeneous transform.")
    return matrix[:3, :3], matrix[:3, 3]


def inertia_about_origin_from_com(mass: sp.Expr, com: sp.Matrix, inertia_com: sp.Matrix) -> sp.Matrix:
    identity = sp.eye(3)
    return sp.simplify(inertia_com + mass * (((com.T * com)[0]) * identity - com * com.T))


def inertia_about_com(inertia: SpatialInertia) -> sp.Matrix:
    if inertia.mass == 0:
        return sp.zeros(3, 3)
    com = inertia.center_of_mass
    identity = sp.eye(3)
    return sp.simplify(
        inertia.inertia_origin - inertia.mass * (((com.T * com)[0]) * identity - com * com.T)
    )


def transform_spatial_inertia(
    inertia: SpatialInertia,
    rotation: sp.Matrix,
    translation: sp.Matrix,
    target_frame: str,
) -> SpatialInertia:
    """Transform barycentric inertia from its current frame to a target frame."""
    if inertia.mass == 0:
        return SpatialInertia.zero(target_frame)

    com_source = inertia.center_of_mass
    com_target = sp.simplify(rotation * com_source + sp.Matrix(translation))
    inertia_com_target = sp.simplify(rotation * inertia_about_com(inertia) * rotation.T)
    inertia_origin_target = inertia_about_origin_from_com(inertia.mass, com_target, inertia_com_target)
    return SpatialInertia(
        mass=inertia.mass,
        first_moment=sp.simplify(inertia.mass * com_target),
        inertia_origin=inertia_origin_target,
        reference_frame=target_frame,
    )


def urdf_inertial_to_spatial_inertia(
    mass: sp.Expr,
    inertia_matrix_in_inertial: sp.Matrix,
    inertial_rotation_in_link: sp.Matrix,
    inertial_translation_in_link: sp.Matrix,
    link_frame: str,
) -> SpatialInertia:
    """Convert URDF inertial data into project barycentric parameters in link frame."""
    if mass == 0:
        return SpatialInertia.zero(link_frame)

    com_link = sp.Matrix(inertial_translation_in_link)
    inertia_com_link = sp.simplify(inertial_rotation_in_link * inertia_matrix_in_inertial * inertial_rotation_in_link.T)
    inertia_origin_link = inertia_about_origin_from_com(mass, com_link, inertia_com_link)
    return SpatialInertia(
        mass=mass,
        first_moment=sp.simplify(mass * com_link),
        inertia_origin=inertia_origin_link,
        reference_frame=link_frame,
    )
