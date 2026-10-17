#!/usr/bin/env python3
"""Compare the left SDK tool point with measured wrist frames; no robot calls."""
import argparse
import json
import math
from pathlib import Path

IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
LEFT_ROOT = 'r412d23_link_c7c36cf32f'
RIGHT_ROOT = 'rbd03eb_link_13df66db66'
LEFT_WRIST = 'r412d23_link_c610da48be'
LEFT_HAND_BASE = 'r136d7b_link_2848c4d4a8'


def matvec(matrix, point):
    return tuple(sum(a * b for a, b in zip(row, point)) for row in matrix)


def transpose(matrix):
    return tuple(zip(*matrix))


def subtract(a, b):
    return tuple(x - y for x, y in zip(a, b))


def distance(a, b):
    return math.sqrt(sum(x * x for x in subtract(a, b)))


def compare_hand_frames(opened, closed, reopened):
    """Compare hand link origins relative to its own base, independent of arm drift."""
    names = sorted(name for name in opened if name.startswith('r136d7b_link_'))
    if len(names) < 6 or not all(name in view for view in (closed, reopened) for name in names):
        raise ValueError('Expected the same measured left-hand links in all three samples')

    def points(frames):
        rotation, origin = world_transform(frames, LEFT_HAND_BASE)
        return {name: matvec(transpose(rotation), subtract(world_transform(frames, name)[1], origin))
                for name in names}

    initial, bent, returned = map(points, (opened, closed, reopened))
    movement = {name: distance(initial[name], bent[name]) for name in names}
    residual = {name: distance(initial[name], returned[name]) for name in names}
    moving = sum(value > .02 for value in movement.values())
    return {'closing_displacement_m': movement, 'reopening_error_m': residual,
            'links_moved_over_20mm': moving, 'max_reopening_error_m': max(residual.values()),
            'sequence_verified': moving >= 4 and max(residual.values()) < .005,
            'physical_grasp_verified': False}


def quaternion_rotation(q):
    if len(q) != 4 or not all(math.isfinite(v) for v in q):
        raise ValueError('Expected a finite xyzw quaternion')
    norm = math.sqrt(sum(v * v for v in q))
    if norm < 1e-12:
        raise ValueError('Zero quaternion')
    x, y, z, w = (v / norm for v in q)
    return ((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
            (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
            (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)))


def compose(outer, inner):
    a, b = outer[0], inner[0]
    rotation = tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3))
                           for j in range(3)) for i in range(3))
    position = tuple(x + y for x, y in zip(outer[1], matvec(a, inner[1])))
    return rotation, position


def world_transform(frames, name, visited=()):
    if name == 'world':
        return IDENTITY, (0, 0, 0)
    if name in visited:
        raise ValueError('Cycle in robot transform chain')
    item = frames[name]
    return compose(world_transform(frames, item['parent'], (*visited, name)),
                   (quaternion_rotation(item['xyzw']), item['xyz']))


def project(camera_transform, intrinsics, point):
    optical = matvec(transpose(camera_transform[0]), subtract(point, camera_transform[1]))
    if optical[2] <= 0:
        raise ValueError('Point is behind the camera')
    return (intrinsics[0] * optical[0] / optical[2] + intrinsics[2],
            intrinsics[4] * optical[1] / optical[2] + intrinsics[5])


def analyse(root):
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    result = []
    offset_reference = same_point_origin = None
    for sample in summary['samples']:
        folder = root / sample['label']
        frames = json.loads((folder / 'robot-frames.json').read_text(encoding='utf-8'))['frames']
        camera_info = json.loads((folder / 'camera/camera.json').read_text(encoding='utf-8'))
        left, right = world_transform(frames, LEFT_ROOT), world_transform(frames, RIGHT_ROOT)
        # Candidate SDK origin: midpoint of the two measured arm root positions.
        origin = tuple((a + b) / 2 for a, b in zip(left[1], right[1]))
        wrist = world_transform(frames, LEFT_WRIST)
        sdk = sample['after']['pose'][0]
        tool_world = tuple(a + b for a, b in zip(origin, matvec(left[0], sdk)))
        offset = matvec(transpose(wrist[0]), subtract(tool_world, wrist[1]))
        if offset_reference is None:
            offset_reference = offset
            same_point_origin = subtract(wrist[1], matvec(left[0], sdk))
        predicted = tuple(a + b for a, b in zip(wrist[1], matvec(wrist[0], offset_reference)))
        camera = world_transform(frames, 'base_cam_optical')
        result.append({'label': sample['label'], 'sdk_origin_candidate_world_m': origin,
                       'sdk_point_offset_in_wrist_m': offset,
                       'fixed_offset_residual_m': distance(tool_world, predicted),
                       'same_point_residual_m': distance(
                           wrist[1], tuple(a + b for a, b in zip(same_point_origin, matvec(left[0], sdk)))),
                       'camera_world_m': camera[1],
                       'sdk_point_pixel_candidate': project(camera, camera_info['k'], tool_world),
                       'wrist_origin_pixel': project(camera, camera_info['k'], wrist[1])})
    return {'samples': result,
            'max_fixed_offset_residual_m': max(x['fixed_offset_residual_m'] for x in result),
            'max_same_point_residual_m': max(x['same_point_residual_m'] for x in result),
            'scope': 'candidate frame relationship for recorded left-wrist poses only',
            'extrinsics_validated': False, 'motion_target_available': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyse(args.capture), indent=2))


if __name__ == '__main__':
    main()
