#!/usr/bin/env python3
"""Compare retained near-static empty-hand effort with possible mesh self-contact.

Local only: no SDK, ROS or platform connection. Public inertia/convex meshes and
static balance are assumptions. A feasible friction-cone fit is nonunique and
cannot establish native contact identity, calibrated force, or a successful grip.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from rabokit_hand import HAND_JOINTS
from replay_hand_dynamics import build_model, read


def contact_balance_rays(model, data, contact, dofs):
    """Actuator torques opposing four mu=1 pyramidal rays on contact body 2."""
    import mujoco
    jacobians = []
    for geom in contact.geom:
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacobian, None, contact.pos,
                      int(model.geom_bodyid[geom]))
        jacobians.append(jacobian)
    relative = (jacobians[1] - jacobians[0])[:, dofs]
    normal, tangent1, tangent2 = contact.frame.reshape(3, 3)
    mu = float(contact.friction[0])
    rays = [normal + mu * (a*tangent1 + b*tangent2)
            for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    return np.column_stack([-relative.T @ ray for ray in rays])


def fit_contact_balance(columns, residual):
    """Fit feasible ray weights; zero residual alone is not contact evidence."""
    if columns.shape[1]:
        weights, error = nnls(columns, residual)
    else:
        weights, error = np.array([]), np.linalg.norm(residual)
    return {'nonnegative_ray_weights': weights.tolist(), 'fit_error_norm': float(error),
            'residual_norm': float(np.linalg.norm(residual)),
            'column_rank': int(np.linalg.matrix_rank(columns)) if columns.shape[1] else 0}


def audit(trial, assets, window=.5, maximum_speed=.001):
    import mujoco
    model, _, *_ = build_model(trial, assets, .1, 1., 20., 1., 1., empty=True)
    data = mujoco.MjData(model)
    trace = read(trial / 'empty_closed-command.json')
    samples = trace['joint_samples']
    end = samples[-1]['stamp'][0] + samples[-1]['stamp'][1]*1e-9
    recent = [s for s in samples
              if end-window <= s['stamp'][0]+s['stamp'][1]*1e-9 <= end
              and all(abs(s['velocity'][n]) <= maximum_speed for n in HAND_JOINTS)]
    if len(recent) < 3:
        raise ValueError('Fewer than three complete near-static empty-hand samples')
    dofs = [int(model.jnt_dofadr[model.joint(n).id]) for n in HAND_JOINTS]
    residuals = []
    positions = []
    for sample in recent:
        for n in HAND_JOINTS:
            data.qpos[model.jnt_qposadr[model.joint(n).id]] = sample['position'][n]
        data.qvel[:] = 0.
        mujoco.mj_forward(model, data)
        residuals.append([sample['effort'][n] - .1*sample['velocity'][n] - data.qfrc_bias[d]
                          for n, d in zip(HAND_JOINTS, dofs)])
        positions.append([sample['position'][n] for n in HAND_JOINTS])
    for n, value in zip(HAND_JOINTS, np.median(positions, axis=0)):
        data.qpos[model.jnt_qposadr[model.joint(n).id]] = value
    mujoco.mj_forward(model, data)
    contacts, columns = [], []
    for contact in data.contact:
        rays = contact_balance_rays(model, data, contact, dofs)
        columns.append(rays)
        contacts.append({'geom_pair': [model.geom(int(g)).name for g in contact.geom],
                         'signed_gap_m': float(contact.dist),
                         'affected_joints': [n for n, row in zip(HAND_JOINTS, rays)
                                             if np.linalg.norm(row) > 1e-10]})
    matrix = np.concatenate(columns, axis=1) if columns else np.zeros((len(dofs), 0))
    residual = np.median(residuals, axis=0)
    fit = fit_contact_balance(matrix, residual)
    return {'scope': __doc__, 'trial': str(trial), 'samples': len(recent),
            'window_seconds': window, 'maximum_joint_speed_rad_s': maximum_speed,
            'root_pose_source': 'retained empty_closed robot TF',
            'residual_definition': 'reported effort - 0.1 * velocity - public-model gravity',
            'median_residual_reported_units': dict(zip(HAND_JOINTS, residual.tolist())),
            'model_self_contacts_at_median_pose': contacts, 'friction_cone_fit': fit,
            'native_contact_identity_verified': False, 'physical_grasp_verified': False,
            'platform_motion_commanded': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', required=True, type=Path, action='append')
    parser.add_argument('--assets', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path')
    result = [audit(trial, args.assets) for trial in args.trial]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps([{'trial': r['trial'], 'samples': r['samples'],
                       'fit': r['friction_cone_fit'],
                       'contacts': r['model_self_contacts_at_median_pose']} for r in result]))


if __name__ == '__main__':
    main()
