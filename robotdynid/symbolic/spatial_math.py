"""Minimal spatial-algebra helpers for symbolic rigid-body dynamics."""

from __future__ import annotations

import sympy as sp


def skew3(vec: sp.Matrix) -> sp.Matrix:
    x, y, z = list(sp.Matrix(vec))
    return sp.Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def motion_cross_matrix(twist: sp.Matrix) -> sp.Matrix:
    """Spatial motion cross-product operator crm(v)."""
    vec = sp.Matrix(twist)
    omega = vec[:3, :]
    linear = vec[3:, :]
    omega_x = skew3(omega)
    linear_x = skew3(linear)
    upper = sp.Matrix.hstack(omega_x, sp.zeros(3, 3))
    lower = sp.Matrix.hstack(linear_x, omega_x)
    return sp.Matrix.vstack(upper, lower)


def force_cross_matrix(twist: sp.Matrix) -> sp.Matrix:
    """Spatial force cross-product operator crf(v)."""
    return -motion_cross_matrix(twist).T


def spatial_inertia_matrix(mass: sp.Expr, first_moment: sp.Matrix, inertia_origin: sp.Matrix) -> sp.Matrix:
    """Construct a 6x6 spatial inertia matrix from barycentric parameters."""
    first = sp.Matrix(first_moment)
    inertia = sp.Matrix(inertia_origin)
    upper = sp.Matrix.hstack(inertia, skew3(first))
    lower = sp.Matrix.hstack(skew3(first).T, mass * sp.eye(3))
    return sp.Matrix.vstack(upper, lower)


def motion_transform_from_homogeneous(transform: sp.Matrix) -> sp.Matrix:
    """Convert a homogeneous transform ^pT_c into a spatial motion transform Xup."""
    matrix = sp.Matrix(transform)
    rotation_pc = matrix[:3, :3]
    translation_pc = matrix[:3, 3]
    rotation_cp = rotation_pc.T
    upper = sp.Matrix.hstack(rotation_cp, sp.zeros(3, 3))
    lower = sp.Matrix.hstack(-rotation_cp * skew3(translation_pc), rotation_cp)
    return sp.Matrix.vstack(upper, lower)


def axis_angle_rotation(axis: sp.Matrix, angle: sp.Expr) -> sp.Matrix:
    """Rodrigues rotation formula for a unit URDF axis."""
    axis_vec = sp.Matrix(axis)
    norm_sq = sp.simplify((axis_vec.T * axis_vec)[0])
    if norm_sq == 0:
        raise ValueError("Joint axis must be non-zero.")
    unit_axis = axis_vec if norm_sq == 1 else axis_vec / sp.sqrt(norm_sq)
    axis_x = skew3(unit_axis)
    identity = sp.eye(3)
    return sp.simplify(
        sp.cos(angle) * identity
        + (1 - sp.cos(angle)) * (unit_axis * unit_axis.T)
        + sp.sin(angle) * axis_x
    )
