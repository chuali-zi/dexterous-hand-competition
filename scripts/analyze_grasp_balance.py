#!/usr/bin/env python3
"""Offline point-contact wrench proxy, requiring numpy/scipy; no physical grasp verdict.

Positions, inward object-force normals and friction are hypotheses. Polyhedral
friction cones omit finite patches, force limits, compliance and native contacts.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


def balance_proxy(positions, normals, mu, rays=16):
    p, n = np.asarray(positions, dtype=float), np.asarray(normals, dtype=float)
    if (p.ndim != 2 or p.shape[1:] != (3,) or len(p) < 2 or n.shape != p.shape
            or not np.isfinite(p).all() or not np.isfinite(n).all()
            or not np.isfinite(mu) or mu < 0 or not isinstance(rays, int) or rays < 4):
        raise ValueError('Expected finite Nx3 points/normals, nonnegative friction and at least four rays')
    lengths = np.linalg.norm(n, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError('Contact normals must be nonzero')
    n = n / lengths[:, None]
    p = p - p.mean(axis=0)
    columns = []
    for point, normal in zip(p, n):
        reference = [0., 0., 1.] if abs(normal[2]) < .9 else [1., 0., 0.]
        tangent = np.cross(normal, reference)
        tangent /= np.linalg.norm(tangent)
        other = np.cross(normal, tangent)
        for angle in np.linspace(0, 2 * np.pi, rays, endpoint=False):
            force = normal + mu * (np.cos(angle) * tangent + np.sin(angle) * other)
            columns.append(np.r_[force, np.cross(point, force) / .025])
    wrench = np.array(columns).T
    count = wrench.shape[1]
    # A positive coefficient for every cone ray, with total weight one, places
    # zero inside their convex hull. Full wrench rank is checked separately.
    result = linprog(np.r_[np.zeros(count), -1.],
                     A_ub=np.column_stack((-np.eye(count), np.ones(count))), b_ub=np.zeros(count),
                     A_eq=np.column_stack((np.vstack((wrench, np.ones(count))), np.zeros(7))),
                     b_eq=np.r_[np.zeros(6), 1.], bounds=[(0, None)] * (count + 1), method='highs')
    margin = float(result.x[-1]) if result.success else 0.
    rank = int(np.linalg.matrix_rank(wrench))
    return {'assumed_friction_coefficient': float(mu), 'rays_per_contact': rays,
            'wrench_rank': rank, 'minimum_normalized_ray_weight': margin,
            'positive_internal_equilibrium': margin > 1e-8,
            'full_dimensional_positive_equilibrium_proxy': rank == 6 and margin > 1e-8,
            'solver_status': int(result.status), 'physical_grasp_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='JSON with pad_points_relative_source_m and pad_normals_sdk')
    parser.add_argument('--mu', type=float, action='append', help='Assumed friction, repeat to compare; default 0.4')
    parser.add_argument('--rays', type=int, default=16)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8'))
    result = {'scope': 'Hypothetical point contacts only; excludes native collision/force evidence',
              'source': str(args.input), 'results': [balance_proxy(data['pad_points_relative_source_m'],
                  data['pad_normals_sdk'], mu, args.rays) for mu in (args.mu or [.4])]}
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
