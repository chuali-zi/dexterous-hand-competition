#!/usr/bin/env python3
"""Offline point-contact wrench proxy, requiring numpy/scipy; no physical grasp verdict.

Positions, inward object-force normals and friction are hypotheses. The original
closure proxy omits force limits; gravity_support_proxy accepts explicit limits.
Both omit finite contact patches, compliance and native contact verification.
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


def gravity_support_proxy(
    positions, normals, center_of_mass, mass, mu, normal_force_limits, rays=16
):
    """Can bounded hypothetical point contacts balance gravity without a table?

    Coordinates are in metres in a common frame with +Z upward. Normals point
    INTO the object. Force limits are assumed contact-normal forces in newtons,
    not SDK normalized force settings or reported joint efforts. A feasible LP
    is only a static force-distribution hypothesis: no reachability, actuator/Jacobian limits,
    contact retention, finite-patch torsion or robustness is established. The
    polygonal cones are inscribed; their infeasibility need not imply that of
    exact circular cones, or of a different contact model.
    """

    p, n = np.asarray(positions, float), np.asarray(normals, float)
    com, limits = np.asarray(center_of_mass, float), np.asarray(
        normal_force_limits, float
    )
    if (
        p.ndim != 2
        or p.shape[1:] != (3,)
        or len(p) < 1
        or n.shape != p.shape
        or com.shape != (3,)
        or limits.shape != (len(p),)
        or not all(np.isfinite(v).all() for v in (p, n, com, limits))
        or np.any(limits < 0)
        or not np.isfinite(mass)
        or mass <= 0
        or not np.isfinite(mu)
        or mu < 0
        or not isinstance(rays, int)
        or rays < 4
    ):
        raise ValueError(
            "Expected finite contacts/COM, positive mass, nonnegative friction and force limits"
        )
    lengths = np.linalg.norm(n, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError("Contact normals must be nonzero")
    columns = []
    for point, normal in zip(p - com, n / lengths[:, None]):
        reference = [0.0, 0.0, 1.0] if abs(normal[2]) < 0.9 else [1.0, 0.0, 0.0]
        tangent = np.cross(normal, reference)
        tangent /= np.linalg.norm(tangent)
        other = np.cross(normal, tangent)
        for angle in np.linspace(0, 2 * np.pi, rays, endpoint=False):
            force = normal + mu * (np.cos(angle) * tangent + np.sin(angle) * other)
            columns.append(np.r_[force, np.cross(point, force)])
    wrench = np.asarray(columns).T
    normal_sum = np.kron(np.eye(len(p)), np.ones((1, rays)))
    required = np.array([0.0, 0.0, mass * 9.81, 0.0, 0.0, 0.0])
    # Scaling moment equations improves conditioning without moving the COM.
    scale = np.array([1.0, 1.0, 1.0, 40.0, 40.0, 40.0])
    result = linprog(
        np.ones(wrench.shape[1]),
        A_ub=normal_sum,
        b_ub=limits,
        A_eq=scale[:, None] * wrench,
        b_eq=scale * required,
        bounds=(0, None),
        method="highs",
    )
    return {
        "gravity_support_feasible": bool(result.success),
        "solver_status": int(result.status),
        "assumed_mass_kg": float(mass),
        "assumed_friction_coefficient": float(mu),
        "assumed_normal_force_limits_n": limits.tolist(),
        "minimum_total_normal_force_n": float(result.fun) if result.success else None,
        "normal_forces_n": (normal_sum @ result.x).tolist() if result.success else None,
        "force_residual_n": (
            float(np.linalg.norm((wrench @ result.x - required)[:3]))
            if result.success
            else None
        ),
        "torque_residual_nm": (
            float(np.linalg.norm((wrench @ result.x - required)[3:]))
            if result.success
            else None
        ),
        "physical_grasp_verified": False,
    }


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
