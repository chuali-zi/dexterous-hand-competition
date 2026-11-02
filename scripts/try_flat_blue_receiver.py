"""Receive the blue brick on the previously measured flat open-hand pose.

Only the receiver geometry changes. Reuse the verified source, left push,
open fingers and common 20 mm lift; no object resets or physics changes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.optimize import least_squares

import try_blue_upper_feed as feed
import try_yellow_edge_acquisition as a

REFERENCE = Path('runs/guided-unload-115/align-level-support')


def prepare(edge_lift=False, align_pusher=False):
    old_command, plan = a.prepare(0.)
    aligned_left = None
    if align_pusher:
        a.restore(geometry=True)
        body = a.m.body(a.TARGET).id
        center = a.d.xpos[body].copy()
        matrix = Rotation.from_quat(a.d.xquat[body][[1, 2, 3, 0]]).as_matrix()
        normal = matrix[:, int(np.argmax(np.abs(matrix[0, :])))].copy()
        normal *= np.sign(normal[0])
        normal[2] = 0
        normal /= np.linalg.norm(normal)
        yaw = float(np.arctan2(normal[1], normal[0]))
        tangent = np.array([-normal[1], normal[0], 0])
        corners = a.p.CORNERS @ matrix.T + center
        start, startq = a.d.mocap_pos[a.p.t.LM].copy(), a.d.mocap_quat[a.p.t.LM].copy()
        leftq = (Rotation.from_euler('z', yaw) * Rotation.from_quat(a.p.t.QUAT[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]
        a.d.qpos[a.p.t.LA:a.p.t.LA + 3] = [0, 0, a.p.t.ROOT[2]]
        a.d.qpos[a.p.t.LA + 3:a.p.t.LA + 7] = leftq
        a.mj.mj_forward(a.m, a.d)
        vertices = np.concatenate([a.g.mesh_vertices('lh_' + finger + suffix)
                                   for finger in a.FACE_FINGERS for suffix in ('_distal', '_proximal')])
        leftroot = normal * ((corners @ normal).max() - (vertices @ normal).min() - .0005)
        leftroot += tangent * (center @ tangent - ((vertices @ tangent).min() + (vertices @ tangent).max()) / 2)
        leftroot[2] = a.p.t.ROOT[2]
        turn = Slerp([9., 12.], Rotation.from_quat(np.array([startq, leftq])[:, [1, 2, 3, 0]]))
        def aligned_left(t):
            if t < 9:
                return start, startq
            if t < 12:
                f = (t - 9) / 3
                return start * (1 - f) + (leftroot + [0, 0, .070]) * f, turn(t).as_quat()[[3, 0, 1, 2]]
            if t < 15:
                return leftroot + [0, 0, .070 * (15 - t) / 3], leftq
            travel = a.PUSH_DISTANCE * np.clip((t - 16) / 6, 0, 4 / 6 if edge_lift else 1)
            height = .020 * np.clip((t - (20 if edge_lift else 23)) / 2, 0, 1)
            return leftroot - travel * normal + [0, 0, height], leftq
        plan.update(pusher_aligned_to_actual_face=True, pusher_yaw_degrees=float(np.degrees(yaw)),
                    pusher_normal_world=normal.tolist(), left_contact_root_m=leftroot.tolist())
        a.restore()
    reference = np.load(REFERENCE / 'integration-state.npz', allow_pickle=False)
    if str(reference['mujoco_version']) != a.mj.__version__:
        raise ValueError('Reference runtime differs')
    probe = a.mj.MjData(a.m)
    a.mj.mj_setState(a.m, probe, reference['state'], int(reference['spec']))
    a.mj.mj_forward(a.m, probe)
    a.restore()
    goal = a.d.xpos[a.m.body(a.TARGET).id].copy()
    goal[0] -= a.PUSH_DISTANCE
    # Three extra millimeters clear the initial 1.914 mm index overlap seen in
    # the static transplant. This is an entry gap, not changed contact physics.
    goal[2] = a.g.FLOOR + .021
    root = goal + probe.mocap_pos[a.p.t.RM] - probe.xpos[a.m.body('neighbor_1').id]
    quat = probe.mocap_quat[a.p.t.RM].copy()
    endpoint = a.g.endpoint('right', root, quat)
    if not endpoint['accepted']:
        fixed_xy = endpoint
        seed = a.g.endpoint('right', root + [0, 0, .003], quat)
        if not seed['accepted']:
            raise ValueError('Neither the lowered nor reference flat endpoint is feasible')
        limits = a.CTX['right']['intersection'] + [.02, -.02]
        rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        def residual(q):
            h = a.forward(q)
            return np.r_[(h[:3, 3] - root) / [.01, .01, .00001],
                         Rotation.from_matrix(rotation.T @ h[:3, :3]).as_rotvec() / .0001,
                         (q - np.array(seed['joints'])) * .01]
        fitted = least_squares(residual, seed['joints'], bounds=(limits[:, 0], limits[:, 1]), max_nfev=200)
        actual = a.forward(fitted.x)
        offset = actual[:3, 3] - root
        angle = float(Rotation.from_matrix(rotation.T @ actual[:3, :3]).magnitude())
        accepted = bool(np.linalg.norm(offset[:2]) <= .010 and abs(offset[2]) < 1e-5 and angle < 1e-4
                        and np.all(fitted.x >= limits[:, 0]) and np.all(fitted.x <= limits[:, 1]))
        endpoint = dict(accepted=accepted, joints=fitted.x.tolist(), fixed_xy_endpoint=fixed_xy,
                        optimizer_converged=bool(fitted.success), evaluations=fitted.nfev,
                        permitted_horizontal_adjustment_m=.010, actual_root_offset_m=offset.tolist(),
                        angle_error_rad=angle, scope='Bounded placement proposal; full route checked separately')
        if not accepted:
            raise ValueError(f'Adjusted flat receiver rejected: {endpoint}')
    a.RIGHT_Q = np.array(endpoint['joints'])
    outside = a.RIGHT_Q.copy()
    outside[0] = -.5
    a.WAYPOINTS = [np.zeros(7), np.array([-.5, 0, 0, 0, 0, 0, 0]), outside, a.RIGHT_Q]
    catch = a.forward(a.RIGHT_Q)
    catch_quat = Rotation.from_matrix(catch[:3, :3]).as_quat()[[3, 0, 1, 2]]

    def command(t):
        poses, phase = old_command(t)
        if edge_lift and t >= 20:
            left, left_quat = old_command(20)[0]['left']
            poses['left'] = (left + [0, 0, .020 * np.clip((t - 20) / 2, 0, 1)], left_quat)
            phase = 'lift_from_edge' if t < 22 else 'settle_edge_hold' if t < 25 else 'hold'
        if aligned_left is not None:
            poses['left'] = aligned_left(t)
        if t < 9:
            index = min(int(t / 3), 2)
            fraction = (t - index * 3) / 3
            pose = a.forward(a.WAYPOINTS[index] * (1 - fraction) + a.WAYPOINTS[index + 1] * fraction)
            right = (pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]])
        else:
            right = (catch[:3, 3] + [0, 0, .020 * np.clip((t - (20 if edge_lift else 23)) / 2, 0, 1)], catch_quat)
        poses['right'] = right
        return poses, phase

    plan.pop('support_reference_error_m', None)
    plan.update(flat_receiver_reference=str(REFERENCE), receiver_endpoint=endpoint,
                nominal_received_center_m=goal.tolist(), right_joint_waypoints=[q.tolist() for q in a.WAYPOINTS],
                receiver_fingers_kept_open=True, right_curl_start_s=None, right_curl_schedule=None,
                duration_s=31., hold_start_s=28., diagnostic_only=True,
                seven_brick_source=str(feed.INITIAL))
    if edge_lift:
        a.DURATION = 28.
        plan.update(duration_s=28., hold_start_s=25., edge_lift=True, push_stop_s=20.,
                    original_full_push_distance_m=a.PUSH_DISTANCE, push_distance_m=a.PUSH_DISTANCE * 4 / 6,
                    common_lift_start_s=20., common_lift_end_s=22.)
    return command, plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('runs/blue-feed-117/right-return'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--edge-lift', action='store_true')
    parser.add_argument('--align-pusher', action='store_true')
    args = parser.parse_args()
    feed.configure(args.source)
    target = feed.g.RECORD['summary']['target']
    a.configure_continuation(args.source, target, edge_blue_receiver=True)
    a.DURATION, a.CUP_BEFORE_LIFT, a.CUP_START = 31., True, 32.
    a.DELAYED_PINKY = a.RECENTER_WIDE_FACE = False
    a.four_finger_face()
    command, plan = prepare(args.edge_lift, args.align_pusher)
    args.out.mkdir(parents=True, exist_ok=False)
    check = a.precheck(command)
    (args.out / 'plan.json').write_text(json.dumps(plan, allow_nan=False))
    (args.out / 'precheck.json').write_text(json.dumps(check, allow_nan=False))
    passed = check['passed']
    brief = dict(precheck_passed=passed, geometry_failures=len(check['geometry_failures']), left_stages=len(check['left_stages']))
    if passed and args.run:
        a.run(command, args.out, protected_payload='neighbor_5',
              additional_protected_payloads=('neighbor_1', 'neighbor_3', *feed.seven.release.feed.BASES))
        record = json.loads((args.out / 'result.json').read_text())
        summary = record['summary']
        summary.update(unintended_contacts=summary['unintended_hand_contacts'], diagnostic_only=True,
                       full_pyramid_verified=False, official_region_verified=False)
        (args.out / 'result.json').write_text(json.dumps(record, allow_nan=False))
        assessment = feed.seven.assess(args.out, held_payload=target)
        summary['local_seven_brick_final_structure_verified'] = assessment['local_seven_brick_final_structure_verified']
        summary['local_acquisition_verified'] &= summary['local_seven_brick_final_structure_verified']
        (args.out / 'result.json').write_text(json.dumps(record, allow_nan=False))
        (args.out / 'seven-brick-assessment.json').write_text(json.dumps(assessment, allow_nan=False))
        brief['dynamics'] = summary
        passed = summary['local_acquisition_verified']
    print(json.dumps(brief, indent=2))
    raise SystemExit(0 if passed else 1)
