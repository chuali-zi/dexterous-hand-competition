"""Bounded right-hand red-brick +X endpoint study before any dynamics.

Two retained right-arm numerical seeds only; no ROS or platform connection.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def pose(xyz, quat):
    value = np.eye(4)
    value[:3, 3] = xyz
    value[:3, :3] = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
    return value


def run(output, sdk_cache):
    started = time.monotonic()
    sys.path.insert(0, str(sdk_cache))
    from _robolib_core.arms import _lbot_kinematics as sdk
    sys.path.insert(0, str(Path('runs/mujoco-runtime').resolve()))
    import mujoco

    config = json.loads(Path('configs/bimanual-contact.json').read_text())
    source = json.loads(Path('runs/inward-transfer-77/source-candidate.json').read_text())
    target_xy = np.asarray(source['initial_object_positions']['neighbor_9'][:2])
    root = np.asarray(source['hand_root_position']).copy()
    root[1] = 2*target_xy[1]-root[1]
    quat = [0., 1., 0., 0.]
    mount = np.asarray(config['tool_to_hand']['right'])
    preset = np.asarray(sdk.real_joint_limits('right'))
    scene = np.asarray(config['joint_limits_sdk']['right'])
    bounds = np.column_stack((np.maximum(preset[:, 0], scene[:, 0]),
                              np.minimum(preset[:, 1], scene[:, 1])))
    seeds = {
        'native66.closed': json.loads(Path('runs/bimanual-contact-66/native-poses.json').read_text())['closed']['right_joints_sdk'],
        '74.opposition': json.loads(Path('runs/arm-feasible-74/opposition/asymmetric-contact-candidate.json').read_text())['arm_joints_sdk']['right'],
    }
    targets = {name.replace('lh_', 'rh_'): value for name, value in source['hand_targets'].items()}
    model_path = Path('runs/bimanual-contact-68/local-calibrated-contact.xml')
    model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    data = mujoco.MjData(model)
    corrections = {}
    for name, value in targets.items():
        joint = model.joint(name)
        lo, hi = model.jnt_range[joint.id]
        if not lo <= value <= hi:
            targets[name] = float(np.clip(value, lo, hi))
            corrections[name] = dict(requested=value, actual=targets[name], model_range=[float(lo), float(hi)])
        data.qpos[model.jnt_qposadr[joint.id]] = targets[name]
    address = int(model.jnt_qposadr[model.joint('right_hand_root_free').id])
    data.qpos[address:address+3], data.qpos[address+3:address+7] = root, quat
    mujoco.mj_forward(model, data)
    geometry = {}
    for name in ['rh_index_distal', 'rh_middle_distal', 'rh_thumb_distal', 'rh_ring_distal', 'rh_pinky_distal']:
        geom = model.geom(name)
        mesh = int(model.geom_dataid[geom.id])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices = model.mesh_vert[start:start+count] @ data.geom_xmat[geom.id].reshape(3, 3).T + data.geom_xpos[geom.id]
        geometry[name] = dict(minimum=vertices.min(axis=0).tolist(), maximum=vertices.max(axis=0).tolist())

    endpoints = {}
    for name, offset in [('contact', [0., 0., 0.]), ('high_staging', [-.020, 0., .080]),
                         ('push_end', [.080, 0., 0.]), ('withdrawn', [.060, 0., .025])]:
        goal = pose(root + offset, quat) @ np.linalg.inv(mount)
        attempts = []
        for seed_name, seed in seeds.items():
            def residual(q):
                fk = np.asarray(sdk.forward_kinematics('right', q.tolist()))
                return np.r_[(fk[:3, 3]-goal[:3, 3])/.001,
                             Rotation.from_matrix(goal[:3, :3].T @ fk[:3, :3]).as_rotvec()/.01]
            solve_start = time.monotonic()
            solved = least_squares(residual, np.clip(seed, bounds[:, 0], bounds[:, 1]),
                                   bounds=(bounds[:, 0], bounds[:, 1]), max_nfev=400,
                                   ftol=1e-12, xtol=1e-12, gtol=1e-10)
            error = residual(solved.x)
            position_error = float(np.linalg.norm(error[:3])*.001)
            orientation_error = float(np.linalg.norm(error[3:])*.01)
            margins = np.minimum(solved.x-bounds[:, 0], bounds[:, 1]-solved.x)
            accepted = position_error < 1e-5 and orientation_error < 1e-4
            row = dict(seed_source=seed_name, accepted=accepted, joints_sdk=solved.x.tolist(),
                       position_error_m=position_error, orientation_error_rad=orientation_error,
                       joint_margins_rad=margins.tolist(), minimum_joint_margin_rad=float(margins.min()),
                       nfev=int(solved.nfev), wall_seconds=time.monotonic()-solve_start)
            attempts.append(row)
            if accepted and margins.min() >= .020:
                break
        endpoints[name] = dict(root_position=(root+offset).tolist(), attempts=attempts,
                               accepted_with_margin=any(r['accepted'] and r['minimum_joint_margin_rad'] >= .020 for r in attempts))
    result = dict(candidate=dict(side='right', hand_root_position=root.tolist(), hand_root_quaternion_wxyz=quat,
                                 hand_targets=targets, target_body='neighbor_9', push_distance_m=.080,
                                 withdraw_height_m=.025, target_initial_position=source['initial_object_positions']['neighbor_9']),
                  model=str(model_path), finger_limit_corrections=corrections, static_distal_mesh_bounds=geometry,
                  right_sdk_preset_limits=preset.tolist(), right_intersection_limits=bounds.tolist(),
                  tool_to_hand_right=mount.tolist(), transform_rule='tool = hand @ inverse(tool_to_hand[right])',
                  endpoints=endpoints, all_endpoints_with_20mrad_margin=all(p['accepted_with_margin'] for p in endpoints.values()),
                  wall_seconds=time.monotonic()-started, dynamics_run=False, motion_commanded=False,
                  continuous_route_checked=False, whole_arm_collision_checked=False,
                  scope=__doc__)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/inward-transfer-77/right/endpoint-results.json'))
    parser.add_argument('--sdk-cache', type=Path, default=Path.home()/'.rabo_robolib/core-cp312.zip')
    args = parser.parse_args()
    report = run(args.output, args.sdk_cache)
    print(json.dumps({k: report[k] for k in ['candidate', 'finger_limit_corrections', 'endpoints',
                                           'all_endpoints_with_20mrad_margin', 'wall_seconds']}, indent=2))
    raise SystemExit(0 if report['all_endpoints_with_20mrad_margin'] else 1)
