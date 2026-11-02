"""Check and execute a proposed explicit joint route into the left top stop.

The optimizer output is only a proposal. Validate its seed, bounds, joint steps,
endpoint and actual FK hand geometry independently. No whole-arm collision or
native execution claim is made; object dynamics use the unchanged local model.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import try_third_yellow_release as release

g = release.g


def prepare(route):
    reference, plan = release.prepare_top_stop()
    ctx = g.p.t.routes.context()['left']
    seed = np.array(g.CHECK['stages']['left'][-1]['result']['final_joints_sdk'])
    np.testing.assert_allclose(route['seed'], seed, rtol=0, atol=1e-10)
    times = np.r_[0., route['times']]
    joints = np.vstack([seed, route['joints']])
    if times.shape != (29,) or joints.shape != (29, 7) or not np.isfinite(joints).all():
        raise ValueError('Expected the 28-point local top-stop proposal')
    np.testing.assert_allclose(times, np.linspace(0, 7, 29), rtol=0, atol=1e-12)
    limits = ctx['intersection'] + [.020, -.020]
    if np.any(joints < limits[:, 0] - 1e-10) or np.any(joints > limits[:, 1] + 1e-10):
        raise ValueError('Proposal exceeds retained joint limits')

    def joint_at(t):
        clock = t if t <= 1 else 1 + (t - 1) / 2
        return np.array([np.interp(clock, times, joints[:, i]) for i in range(7)])

    def command(t):
        poses, controls, _ = reference(0)
        fk = np.asarray(g.p.t.routes.sdk.forward_kinematics('left', joint_at(t).tolist())) @ ctx['mount']
        poses['left'] = (fk[:3, 3], Rotation.from_matrix(fk[:3, :3]).as_quat()[[3, 0, 1, 2]])
        return poses, controls, 'left_joint_reconfiguration' if t < 1 else 'left_joint_top_entry' if t < 13 else 'hold_top_stop'

    actual = command(13)[0]['left']
    desired = reference(7)[0]['left']
    position_error = float(np.linalg.norm(actual[0] - desired[0]))
    angle_error = float((Rotation.from_quat(actual[1][[1, 2, 3, 0]]).inv() *
                         Rotation.from_quat(desired[1][[1, 2, 3, 0]])).magnitude())
    if position_error > 1e-5 or angle_error > 1e-4:
        raise ValueError(f'Proposed endpoint misses top stop: {position_error}, {angle_error}')
    g.DURATION = 16.
    plan.update(duration_s=16., hold_start_s=13., explicit_left_joint_route=True,
                joint_route_times_s=[float(t if t <= 1 else 1 + (t - 1) * 2) for t in times],
                joint_route=joints.tolist(), optimizer_status=route['status'], optimizer_message=route['message'],
                original_proposal_max_position_error_m=route['maximum_position_error_m'],
                original_proposal_max_angle_error_rad=route['maximum_angle_error_rad'],
                endpoint_position_error_m=position_error, endpoint_angle_error_rad=angle_error)
    return command, plan, joint_at, limits


def precheck(command, joint_at, limits):
    # Retain the existing geometry check and right-arm path validation. The left
    # arm now follows the explicit joints rather than the greedy inverse solver.
    check = g.precheck(command)
    check['default_left_inverse_stages'] = check['stages']['left']
    stages = []
    previous = joint_at(0)
    for t in np.linspace(.25, g.DURATION, round(g.DURATION * 4)):
        q = joint_at(t)
        step = float(np.max(np.abs(q - previous)))
        margin = float(np.minimum(q - limits[:, 0], limits[:, 1] - q).min())
        stages.append(dict(t=float(t), result=dict(continuous=bool(step <= .15 and margin >= -1e-10),
            final_joints_sdk=q.tolist(), maximum_adjacent_joint_change_rad=step,
            margin_to_retained_bounds_rad=margin, method='explicit_bounded_joint_interpolation')))
        previous = q
    check['stages']['left'] = stages
    initial = check['initial_seed_errors']['left']
    check['passed'] = bool(not check['geometry_failures'] and initial['position_error_m'] < 1e-5
        and initial['angle_error_rad'] < 1e-4 and all(x['result']['continuous'] for x in stages)
        and len(check['stages']['right']) == round(g.DURATION * 4)
        and all(x['result']['continuous'] for x in check['stages']['right']))
    check['explicit_left_joint_route_verified'] = all(x['result']['continuous'] for x in stages)
    check['scope'] = 'Actual FK of linearly interpolated left joints; 20 Hz hand geometry; unchanged right SDK check; no full-arm collision bodies'
    return check


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('runs/guided-unload-115/align-level-support'))
    parser.add_argument('--route', type=Path, default=Path('runs/narrow-support-116/top-stop-global-route-with-hold.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    release.r.configure(args.source, target=release.feed.TARGET)
    command, plan, joint_at, limits = prepare(json.loads(args.route.read_text()))
    plan['joint_route_source'] = str(args.route)
    args.out.mkdir(parents=True, exist_ok=False)
    check = precheck(command, joint_at, limits)
    (args.out / 'plan.json').write_text(json.dumps(plan, allow_nan=False))
    (args.out / 'precheck.json').write_text(json.dumps(check, allow_nan=False))
    passed = check['passed']
    brief = dict(precheck_passed=passed, geometry_failures=len(check['geometry_failures']),
                 explicit_left_joint_route_verified=check['explicit_left_joint_route_verified'])
    if passed and args.run:
        g.run(command, plan, args.out, protected_payload='neighbor_5', additional_protected_payloads=(release.feed.SECOND, *release.feed.BASES))
        summary = release.evaluate(args.out)
        brief['dynamics'] = summary
        passed = summary['local_top_stop_verified']
    print(json.dumps(brief, indent=2))
    raise SystemExit(0 if passed else 1)
