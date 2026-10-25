"""Offline underside opening queries; no robot commands or grasp verdicts."""

import numpy as np
from scipy.spatial.transform import Rotation


def bottom_gap_at_xy(position, quaternion_wxyz, table_z, points_xy, size=0.05):
    """Vertical gap below the original bottom face at each world XY query.

    The cube pose and size are hypotheses supplied by the caller. Return None
    outside the face's XY projection, or when that face is vertical/upturned:
    a high corner alone does not establish clearance along an insertion path.
    Negative gaps retain geometric table penetration. This does not account
    for the finger volume, nearby objects, compliance or a reachable entry.
    """
    p, q, xy = (
        np.asarray(v, dtype=float) for v in (position, quaternion_wxyz, points_xy)
    )
    if (
        p.shape != (3,)
        or q.shape != (4,)
        or xy.ndim != 2
        or xy.shape[1:] != (2,)
        or not all(np.isfinite(v).all() for v in (p, q, xy))
        or not np.isfinite(table_z)
        or not np.isfinite(size)
        or size <= 0
        or abs(np.linalg.norm(q) - 1) > 1e-6
    ):
        raise ValueError(
            "Expected finite cube pose, unit quaternion, Nx2 XY points and positive size"
        )
    rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    normal = rotation[:, 2]
    if normal[2] <= 1e-6:
        return [None] * len(xy)
    center = p - size / 2 * normal
    z = center[2] - (xy - center[:2]) @ normal[:2] / normal[2]
    world = np.column_stack((xy, z))
    local = (world - center) @ rotation
    inside = (np.abs(local[:, :2]) <= size / 2 + 1e-9).all(axis=1)
    return [
        float(height - table_z) if valid else None for height, valid in zip(z, inside)
    ]
