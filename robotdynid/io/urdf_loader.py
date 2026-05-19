"""URDF loader for the canonical serial robot model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import sympy as sp

from robotdynid.core.robot_model import JointModel, LinkModel, RobotModel, SpatialInertia
from robotdynid.io.inertia_mapping import (
    homogeneous_transform,
    rpy_to_rotation,
    scalar,
    split_transform,
    transform_spatial_inertia,
    urdf_inertial_to_spatial_inertia,
    vector3,
)

SUPPORTED_MOTION_JOINTS = {"revolute", "continuous", "prismatic"}
SUPPORTED_PASSIVE_JOINTS = {"fixed"}


@dataclass(frozen=True)
class _RawJoint:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    axis: sp.Matrix
    placement: sp.Matrix


def _parse_xyz_rpy(element: ET.Element | None) -> tuple[sp.Matrix, sp.Matrix]:
    if element is None:
        return sp.zeros(3, 1), sp.zeros(3, 1)

    xyz_text = element.attrib.get("xyz", "0 0 0")
    rpy_text = element.attrib.get("rpy", "0 0 0")
    xyz_values = tuple(part for part in xyz_text.split())
    rpy_values = tuple(part for part in rpy_text.split())
    if len(xyz_values) != 3 or len(rpy_values) != 3:
        raise ValueError("URDF origin xyz/rpy must each contain exactly three values.")
    return vector3(xyz_values), vector3(rpy_values)


def _parse_inertia_matrix(inertia_element: ET.Element) -> sp.Matrix:
    return sp.Matrix(
        [
            [scalar(inertia_element.attrib["ixx"]), scalar(inertia_element.attrib["ixy"]), scalar(inertia_element.attrib["ixz"])],
            [scalar(inertia_element.attrib["ixy"]), scalar(inertia_element.attrib["iyy"]), scalar(inertia_element.attrib["iyz"])],
            [scalar(inertia_element.attrib["ixz"]), scalar(inertia_element.attrib["iyz"]), scalar(inertia_element.attrib["izz"])],
        ]
    )


def _parse_link_inertia(link_element: ET.Element) -> SpatialInertia:
    name = link_element.attrib["name"]
    inertial = link_element.find("inertial")
    if inertial is None:
        return SpatialInertia.zero(name)

    mass_element = inertial.find("mass")
    inertia_element = inertial.find("inertia")
    if mass_element is None or inertia_element is None:
        raise ValueError(f"Link '{name}' has an incomplete inertial block.")

    origin_xyz, origin_rpy = _parse_xyz_rpy(inertial.find("origin"))
    rotation = rpy_to_rotation(origin_rpy[0], origin_rpy[1], origin_rpy[2])
    mass = scalar(mass_element.attrib["value"])
    inertia_matrix = _parse_inertia_matrix(inertia_element)
    return urdf_inertial_to_spatial_inertia(
        mass=mass,
        inertia_matrix_in_inertial=inertia_matrix,
        inertial_rotation_in_link=rotation,
        inertial_translation_in_link=origin_xyz,
        link_frame=name,
    )


def _parse_joint(joint_element: ET.Element) -> _RawJoint:
    name = joint_element.attrib["name"]
    joint_type = joint_element.attrib["type"]
    if joint_type not in SUPPORTED_MOTION_JOINTS | SUPPORTED_PASSIVE_JOINTS:
        raise ValueError(f"Unsupported joint type '{joint_type}' for joint '{name}'.")

    parent = joint_element.find("parent")
    child = joint_element.find("child")
    if parent is None or child is None:
        raise ValueError(f"Joint '{name}' is missing parent or child link information.")

    origin_xyz, origin_rpy = _parse_xyz_rpy(joint_element.find("origin"))
    rotation = rpy_to_rotation(origin_rpy[0], origin_rpy[1], origin_rpy[2])
    placement = homogeneous_transform(rotation, origin_xyz)

    if joint_type in SUPPORTED_MOTION_JOINTS:
        axis_element = joint_element.find("axis")
        axis_text = "1 0 0" if axis_element is None else axis_element.attrib.get("xyz", "1 0 0")
        axis_values = tuple(part for part in axis_text.split())
        if len(axis_values) != 3:
            raise ValueError(f"Joint '{name}' axis must contain exactly three values.")
        axis = vector3(axis_values)
    else:
        axis = sp.zeros(3, 1)

    return _RawJoint(
        name=name,
        joint_type=joint_type,
        parent_link=parent.attrib["link"],
        child_link=child.attrib["link"],
        axis=axis,
        placement=placement,
    )


def _ordered_serial_chain(raw_joints: Iterable[_RawJoint], link_names: set[str]) -> tuple[str, list[_RawJoint]]:
    incoming: dict[str, _RawJoint] = {}
    outgoing: dict[str, _RawJoint] = {}

    for joint in raw_joints:
        if joint.child_link in incoming:
            raise ValueError(f"Link '{joint.child_link}' has multiple parent joints; model is not a serial chain.")
        if joint.parent_link in outgoing:
            raise ValueError(f"Link '{joint.parent_link}' has multiple child joints; model is not a serial chain.")
        incoming[joint.child_link] = joint
        outgoing[joint.parent_link] = joint

    roots = sorted(link_names - set(incoming))
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root link, found {roots}.")
    root = roots[0]

    ordered: list[_RawJoint] = []
    current = root
    visited = set()
    while current in outgoing:
        joint = outgoing[current]
        if joint.name in visited:
            raise ValueError("Detected a cycle while traversing the URDF joint chain.")
        visited.add(joint.name)
        ordered.append(joint)
        current = joint.child_link

    if len(ordered) != len(list(raw_joints)):
        raise ValueError("The URDF graph is disconnected; expected a single serial chain.")
    return root, ordered


def _add_link_inertia_to_body(
    body: LinkModel,
    link_name: str,
    link_inertia: SpatialInertia,
    transform_from_body_anchor: sp.Matrix,
) -> None:
    rotation, translation = split_transform(transform_from_body_anchor)
    transformed = transform_spatial_inertia(
        inertia=link_inertia,
        rotation=rotation,
        translation=translation,
        target_frame=body.anchor_frame,
    )
    body.append_source_link(link_name, transformed)


def load_robot_from_urdf(
    urdf_path: str | Path,
    *,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
) -> RobotModel:
    """Load a serial open-chain URDF into the canonical RobotModel."""
    urdf_file = Path(urdf_path)
    root = ET.parse(urdf_file).getroot()
    if root.tag != "robot":
        raise ValueError("Expected the URDF root element to be <robot>.")

    name = root.attrib.get("name", urdf_file.stem)
    link_elements = root.findall("link")
    joint_elements = root.findall("joint")
    if not link_elements:
        raise ValueError("URDF does not contain any <link> elements.")

    link_inertias = {element.attrib["name"]: _parse_link_inertia(element) for element in link_elements}
    raw_joints = [_parse_joint(element) for element in joint_elements]
    link_names = set(link_inertias.keys())
    base_frame, ordered_joints = _ordered_serial_chain(raw_joints, link_names)

    base_body = LinkModel(
        name=base_frame,
        anchor_frame=base_frame,
        inertia=link_inertias[base_frame].with_reference_frame(base_frame),
        source_links=(base_frame,),
    )
    motion_joints: list[JointModel] = []
    dynamic_links: list[LinkModel] = []
    current_anchor_frame = base_frame
    transform_from_anchor_to_current_link = sp.eye(4)
    current_body: LinkModel = base_body

    for raw_joint in ordered_joints:
        transform_to_child = sp.simplify(transform_from_anchor_to_current_link * raw_joint.placement)

        if raw_joint.joint_type in SUPPORTED_MOTION_JOINTS:
            joint = JointModel(
                name=raw_joint.name,
                joint_type=raw_joint.joint_type,
                parent_frame=current_anchor_frame,
                child_frame=raw_joint.child_link,
                axis=raw_joint.axis,
                placement=transform_to_child,
                raw_parent_link=raw_joint.parent_link,
                raw_child_link=raw_joint.child_link,
            )
            motion_joints.append(joint)

            current_body = LinkModel(
                name=raw_joint.child_link,
                anchor_frame=raw_joint.child_link,
                inertia=link_inertias[raw_joint.child_link].with_reference_frame(raw_joint.child_link),
                source_links=(raw_joint.child_link,),
            )
            dynamic_links.append(current_body)
            current_anchor_frame = raw_joint.child_link
            transform_from_anchor_to_current_link = sp.eye(4)
            continue

        _add_link_inertia_to_body(
            body=current_body,
            link_name=raw_joint.child_link,
            link_inertia=link_inertias[raw_joint.child_link],
            transform_from_body_anchor=transform_to_child,
        )

        transform_from_anchor_to_current_link = transform_to_child

    return RobotModel(
        name=name,
        gravity=gravity,
        base_frame=base_frame,
        joints=motion_joints,
        links=dynamic_links,
        base_inertia=base_body.inertia,
    )
