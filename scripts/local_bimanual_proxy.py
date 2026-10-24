"""Build a local two-hand proxy from public O6 assets and retained robot-only TF.

No SDK/ROS connection. Right inertias, servo parameters and URDF joint stops
are hypotheses; matching a retained pose does not calibrate native collisions.
Assets are supplied separately and are not distributed with this script.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from replay_hand_dynamics import build_model, vector_text, wxyz
from rabokit_calibration_analysis import LEFT_ROOT, RIGHT_ROOT, world_transform


def measured_hinge_angle(zero, measured_xyzw, axis, tolerance=5e-5):
    """Use the declared signed URDF axis; reject rotation around another axis."""
    axis = np.asarray(axis, dtype=float)
    if (
        axis.shape != (3,)
        or not np.isfinite(axis).all()
        or abs(np.linalg.norm(axis) - 1) > 1e-8
    ):
        raise ValueError("Expected a finite unit hinge axis")
    relative = zero.inv() * Rotation.from_quat(measured_xyzw)
    rv = relative.as_rotvec()
    angle = float(rv @ axis)
    residual = float(np.linalg.norm(rv - angle * axis))
    if residual > tolerance:
        raise ValueError("Retained TF rotation does not match the declared hinge")
    return angle, residual


def build_bimanual_model(
    trial, left_assets, right_assets, target_mass=0.1, contact_time=0.02
):
    import mujoco

    if not np.isfinite(target_mass) or target_mass <= 0:
        raise ValueError("Expected a positive finite target mass hypothesis")
    if not np.isfinite(contact_time) or contact_time <= 0:
        raise ValueError("Expected a positive finite contact time hypothesis")
    assets = Path(right_assets)
    _, xml, frames, *_ = build_model(
        Path(trial), Path(left_assets), 0.1, 1, 20, 1, 1, contact_time=contact_time
    )
    urdf = ET.parse(assets / "linkerhand_o6_right.urdf").getroot()
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints = urdf.findall("joint")
    tree = ET.fromstring(xml)
    world = tree.find("worldbody")
    # Vary only the target; neighboring cubes keep the original 0.1 kg model.
    world.find("body[@name='target']/geom").set("mass", str(target_mass))
    asset = tree.find("asset")
    actuator = tree.find("actuator")
    contacts = tree.find("contact")
    prefix = "rcd72e2_link_"
    roots = [
        n
        for n, v in frames.items()
        if n.startswith(prefix) and not v["parent"].startswith(prefix)
    ]
    if len(roots) != 1:
        raise ValueError("Expected exactly one retained right-hand root")
    mapping = {"rh_hand_base_link": roots[0]}
    values = {}
    offset_errors = {}
    rotation_errors = {}
    sr, lp = world_transform(frames, LEFT_ROOT)
    _, rp = world_transform(frames, RIGHT_ROOT)
    sr = np.array(sr)
    origin = (np.array(lp) + rp) / 2
    right_rot, right_pos = world_transform(frames, roots[0])
    rotation = sr.T @ right_rot
    position = sr.T @ (np.array(right_pos) - origin)
    ET.SubElement(
        world,
        "body",
        name="right_root_reference",
        mocap="true",
        pos=vector_text(position),
        quat=vector_text(wxyz(rotation)),
    )
    base = ET.SubElement(
        world,
        "body",
        name="rh_hand_base_link",
        pos=vector_text(position),
        quat=vector_text(wxyz(rotation)),
    )
    ET.SubElement(base, "freejoint", name="right_hand_root_free")
    ET.SubElement(
        tree.find("equality"),
        "weld",
        body1="right_root_reference",
        body2="rh_hand_base_link",
        solref=".004 1",
    )
    bodies = {"rh_hand_base_link": base}
    for j in joints:
        name = j.get("name")
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        expected = np.fromstring(j.find("origin").get("xyz"), sep=" ")
        candidates = [n for n, v in frames.items() if v["parent"] == mapping[parent]]
        native = min(
            candidates,
            key=lambda n: np.linalg.norm(np.array(frames[n]["xyz"]) - expected),
        )
        error = float(np.linalg.norm(np.array(frames[native]["xyz"]) - expected))
        if error >= 0.0002:
            raise ValueError(
                "Public joint offset does not match the retained robot tree"
            )
        mapping[child] = native
        axis = np.fromstring(j.find("axis").get("xyz"), sep=" ")
        rpy = np.fromstring(j.find("origin").get("rpy"), sep=" ")
        zero = Rotation.from_euler("xyz", rpy)
        q, residual = measured_hinge_angle(zero, frames[native]["xyzw"], axis)
        values[name] = q
        offset_errors[name] = error
        rotation_errors[name] = residual
        body = ET.SubElement(
            bodies[parent],
            "body",
            name=child,
            pos=vector_text(frames[native]["xyz"]),
            quat=vector_text(wxyz(zero.as_matrix())),
        )
        bodies[child] = body
        limit = j.find("limit")
        ET.SubElement(
            body,
            "joint",
            name=name,
            type="hinge",
            axis=vector_text(axis),
            damping=".1",
            limited="true",
            range=vector_text([limit.get("lower"), limit.get("upper")]),
        )
        ET.SubElement(contacts, "exclude", body1=parent, body2=child)
        ET.SubElement(
            actuator,
            "position",
            name=name,
            joint=name,
            kp="20",
            kv="1",
            forcelimited="true",
            forcerange="-1 1",
        )
    for name, body in bodies.items():
        link = links[name]
        i = link.find("inertial")
        assert all(abs(float(v)) < 1e-12 for v in i.find("origin").get("rpy").split())
        tensor = i.find("inertia")
        ET.SubElement(
            body,
            "inertial",
            pos=i.find("origin").get("xyz"),
            mass=i.find("mass").get("value"),
            fullinertia=vector_text(
                [tensor.get(n) for n in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
            ),
        )
        file = (
            assets / Path(link.find("collision/geometry/mesh").get("filename")).name
        ).resolve()
        if not file.is_file():
            raise FileNotFoundError(file)
        ET.SubElement(asset, "mesh", name=name, file=str(file))
        ET.SubElement(body, "geom", name=name, type="mesh", mesh=name)
    xml = ET.tostring(tree, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for n, q in values.items():
        data.qpos[model.jnt_qposadr[model.joint(n).id]] = q
        data.ctrl[model.actuator(n).id] = q
    mujoco.mj_kinematics(model, data)
    fk = []
    for body, native in mapping.items():
        r, p = world_transform(frames, native)
        bid = model.body(body).id
        fk.append(
            dict(
                body=body,
                position_error_m=float(
                    np.linalg.norm(data.xpos[bid] - sr.T @ (np.array(p) - origin))
                ),
                rotation_error_rad=float(
                    Rotation.from_matrix(
                        data.xmat[bid].reshape(3, 3).T @ sr.T @ r
                    ).magnitude()
                ),
            )
        )
    report = dict(
        scope=__doc__,
        mapping=mapping,
        initial_joint_angles_rad=values,
        maximum_offset_difference_m=max(offset_errors.values()),
        maximum_non_hinge_residual_rad=max(rotation_errors.values()),
        maximum_fk_position_error_m=max(v["position_error_m"] for v in fk),
        maximum_fk_rotation_error_rad=max(v["rotation_error_rad"] for v in fk),
        fk=fk,
        right_dynamics_scope="Public inertias and URDF stops; kp20 kv1 force1 damping0.1 hypotheses, not native right dynamics calibration.",
        native_collision_mesh_identity_verified=False,
        target_mass_kg_assumed=target_mass,
        contact_time_s_assumed=contact_time,
    )
    return model, xml, values, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--left-assets", type=Path, required=True)
    parser.add_argument("--right-assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".xml").exists():
        parser.error("Use a new output path")
    model, xml, values, report = build_bimanual_model(
        args.trial, args.left_assets, args.right_assets
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with args.output.with_suffix(".xml").open("x", encoding="utf-8") as stream:
        stream.write(xml)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("mapping", "fk", "initial_joint_angles_rad")
            }
        )
    )


if __name__ == "__main__":
    main()
