"""Canonical symbol creation for symbolic dynamics and identification."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from robotdynid.core.params import generate_joint_dynamics_parameter_names
from robotdynid.core.robot_model import RobotModel


@dataclass(frozen=True)
class SymbolicBuildOptions:
    """Configuration for symbolic model construction."""

    enabled_joint_dynamics_groups: tuple[str, ...] = ("fv", "fc", "fd")
    include_qds: bool = True
    positive_qds: bool = True


@dataclass(frozen=True)
class SymbolicContext:
    """All public symbols used by the symbolic pipeline."""

    q: tuple[sp.Symbol, ...]
    qd: tuple[sp.Symbol, ...]
    qdd: tuple[sp.Symbol, ...]
    qds: tuple[sp.Symbol, ...]
    standard_params: tuple[sp.Symbol, ...]
    joint_dynamics_params: tuple[sp.Symbol, ...]
    linear_params: tuple[sp.Symbol, ...]


def _named_symbols(prefix: str, count: int, *, positive: bool = False) -> tuple[sp.Symbol, ...]:
    if count < 0:
        raise ValueError("count must be non-negative.")
    return tuple(sp.symbols(f"{prefix}1:{count + 1}", real=True, positive=positive))


def build_symbolic_context(robot: RobotModel, options: SymbolicBuildOptions = SymbolicBuildOptions()) -> SymbolicContext:
    """Create a canonical, public symbol set for a robot model."""
    dof = robot.dof
    q = _named_symbols("q", dof)
    qd = _named_symbols("qd", dof)
    qdd = _named_symbols("qdd", dof)
    qds = _named_symbols("qds", dof, positive=options.positive_qds) if options.include_qds else tuple()
    standard_params = tuple(sp.symbols(" ".join(robot.standard_parameter_names), real=True))
    joint_dynamics_names = generate_joint_dynamics_parameter_names(
        dof,
        enabled_groups=options.enabled_joint_dynamics_groups,
    )
    if joint_dynamics_names:
        symbols = sp.symbols(" ".join(joint_dynamics_names), real=True)
        if isinstance(symbols, tuple):
            joint_dynamics_params = symbols
        else:
            joint_dynamics_params = (symbols,)
    else:
        joint_dynamics_params = tuple()
    linear_params = standard_params + joint_dynamics_params
    return SymbolicContext(
        q=q,
        qd=qd,
        qdd=qdd,
        qds=qds,
        standard_params=standard_params,
        joint_dynamics_params=joint_dynamics_params,
        linear_params=linear_params,
    )
