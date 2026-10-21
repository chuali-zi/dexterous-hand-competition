#!/usr/bin/env python3
"""Offline stereo green-box fit with a fixed horizontal top; never commands motion.

Requires numpy, scipy and Pillow. The top height and 50 mm box model are explicit
assumptions. A small pixel residual or multistart spread is not physical accuracy.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

from rabokit_calibration_analysis import LEFT_ROOT, RIGHT_ROOT, world_transform


CAMERAS = ('base_cam', 'base_cam2')


def fit_box_xy(observations, project, prior):
    """Fit x/y/yaw from visible green/wood points; project supplies the fixed top."""
    if any(len(points) < 3 for view in observations for points in view):
        raise ValueError('Both views need visible green and wood; no fit available')

    def residual(q):
        rows = []
        for ci, (green, wood) in enumerate(observations):
            eq = ConvexHull(project(q, ci)).equations
            for points, sign in ((green, 1), (wood, -1)):
                distance = -np.max(points @ eq[:, :2].T + eq[:, 2], axis=1)
                rows.append(np.maximum(0, .5 - sign * distance) / np.sqrt(len(points)))
        return np.concatenate(rows)

    lower, upper = prior - [.035, .035, .7], prior + [.035, .035, .7]
    runs = []
    for dx, yaw in itertools.product((-.01, 0., .01), (-.25, 0., .25)):
        start = prior + [dx, 0., yaw]
        runs.append(least_squares(residual, start, bounds=(lower, upper),
                    x_scale=[.02, .02, .3], diff_step=1e-4, max_nfev=100))
    best = min(runs, key=lambda r: r.fun @ r.fun)
    cost = float(best.fun @ best.fun)
    near = np.array([r.x for r in runs if r.fun @ r.fun <= cost + .01])
    errors = []
    for ci, (green, wood) in enumerate(observations):
        eq = ConvexHull(project(best.x, ci)).equations
        gd = -np.max(green @ eq[:, :2].T + eq[:, 2], axis=1)
        wd = -np.max(wood @ eq[:, :2].T + eq[:, 2], axis=1)
        errors.append({'green_samples': len(green), 'wood_samples': len(wood),
                       'maximum_green_miss_px': float(max(0, -gd.min())),
                       'maximum_wood_intrusion_px': float(max(0, wd.max()))})
    return {'xy_yaw': best.x.tolist(), 'optimizer_converged': bool(best.success),
            'squared_cost': cost, 'near_best_multistart_min': near.min(axis=0).tolist(),
            'near_best_multistart_max': near.max(axis=0).tolist(),
            'multistart_range_is_confidence_interval': False, 'views': errors}


def visible_pixels(images, rois, threshold, occluders=None):
    observations = []
    for ci, image in enumerate(images):
        x0, y0, x1, y1 = rois[ci]
        im = image[y0:y1, x0:x1].astype(float)
        green = (im[:, :, 1] > threshold) & (im[:, :, 1] > 2 * im[:, :, 0]) & (im[:, :, 1] > 2 * im[:, :, 2])
        wood = (im[:, :, 0] > 60) & (im[:, :, 0] > 1.15 * im[:, :, 1]) & (im[:, :, 1] > 1.2 * im[:, :, 2])
        if occluders is not None:
            occ = occluders[ci][y0:y1, x0:x1].astype(float)
            visible = ~((occ[:, :, 0] < 45) & (occ[:, :, 1] < 55) & (occ[:, :, 2] < 85))
            green &= visible
            wood &= visible
        points = []
        for mask in (green, wood):
            y, x = np.nonzero(mask)
            points.append(np.column_stack((x + x0, y + y0))[::2])
        observations.append(points)
    return observations


def analyze(trial, reference, stage, top_z, side, threshold, rgb_folder=None):
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    grasp = read(trial / 'grasp.json')
    frames = read(trial / 'contact/robot-frames.json')['frames']
    left, right = (world_transform(frames, n) for n in (LEFT_ROOT, RIGHT_ROOT))
    sdk_r, sdk_p = np.array(left[0]), (np.array(left[1]) + right[1]) / 2
    source = np.array(grasp['source_surface']['colored_surface_point_sdk_m'])
    corners = np.array(list(itertools.product((-side / 2, side / 2), repeat=3)))
    cameras, sizes, intrinsics = [], [], []
    for cam in CAMERAS:
        info = read(trial / 'contact/camera' / cam / 'camera.json')
        rotation, position = world_transform(frames, info['frame_id'])
        cameras.append((np.array(rotation), np.array(position), np.array(info['k']).reshape(3, 3)))
        sizes.append((info['width'], info['height']))
        intrinsics.append(info)

    def project(q, ci):
        x, y, yaw = q
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        points = (corners @ rotation.T + [x, y, top_z - side / 2]) @ sdk_r.T + sdk_p
        cr, cp, k = cameras[ci]
        optical = (points - cp) @ cr
        if np.any(optical[:, 2] <= 0):
            raise ValueError('Box lies behind camera')
        q = optical @ k.T
        return q[:, :2] / q[:, 2, None]

    prior, rois = np.r_[source[:2], 0.], []
    for ci in range(2):
        uv = project(prior, ci)
        lo, hi = np.floor(uv.min(axis=0) - 15).astype(int), np.ceil(uv.max(axis=0) + 15).astype(int)
        rois.append((max(0, lo[0]), max(0, lo[1]), min(sizes[ci][0], hi[0]), min(sizes[ci][1], hi[1])))

    def images(folder, metadata):
        stamps = {(metadata[cam]['stamp']['sec'], metadata[cam]['stamp']['nanosec']) for cam in CAMERAS}
        if len(stamps) != 1:
            raise ValueError('Expected a synchronized stereo image pair')
        result = []
        for ci, cam in enumerate(CAMERAS):
            info = metadata[cam]
            if info['frame_id'] != intrinsics[ci]['frame_id'] or info['k'] != intrinsics[ci]['k']:
                raise ValueError('Camera identity or intrinsics differ from the reference model')
            with Image.open(folder / cam / 'frame.png') as image:
                if image.size != sizes[ci]:
                    raise ValueError('Image dimensions differ from recorded camera intrinsics')
                result.append(np.array(image.convert('RGB')))
        return result

    initial_metadata = read(reference / 'initial/stereo.json')['camera_messages']
    current_folder = rgb_folder if rgb_folder is not None else trial / stage / 'camera'
    current_metadata = {cam: read(current_folder / cam / 'camera.json') for cam in CAMERAS}
    initial, current = images(reference / 'initial', initial_metadata), images(current_folder, current_metadata)
    baseline = fit_box_xy(visible_pixels(initial, rois, threshold), project, prior)
    baseline_q = np.array(baseline['xy_yaw'])
    fitted = fit_box_xy(visible_pixels(current, rois, threshold), project, baseline_q)
    synthetic = fit_box_xy(visible_pixels(initial, rois, threshold, current), project, baseline_q)
    return {'scope': 'Offline fixed-horizontal-top box hypothesis; no motion or grasp decision',
            'stage': stage, 'rgb_folder': str(current_folder), 'reference': str(reference), 'top_z_sdk_m': top_z,
            'box_side_m': side, 'green_threshold': threshold, 'baseline': baseline,
            'occluded': fitted, 'synthetic_occlusion': synthetic,
            'estimated_xy_shift_m': (np.array(fitted['xy_yaw'])[:2] - baseline_q[:2]).tolist(),
            'synthetic_xy_error_m': float(np.linalg.norm(np.array(synthetic['xy_yaw'])[:2] - baseline_q[:2])),
            'physical_pose_accuracy_verified': False, 'official_score': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--reference', type=Path, help='Initial RGB directory source; defaults to trial')
    parser.add_argument('--stage', default='closed')
    parser.add_argument('--rgb-folder', type=Path, help='Saved synchronized frame folder containing base_cam/ and base_cam2/; defaults to the stage camera folder')
    parser.add_argument('--top-z', type=float, required=True, help='Assumed horizontal top height in SDK metres')
    parser.add_argument('--side', type=float, default=.05)
    parser.add_argument('--threshold', type=int, default=160)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.top_z) or not 0 < args.side < 1 or not 0 <= args.threshold <= 255:
        parser.error('Expected finite top height, box side in (0,1) metres and RGB threshold in [0,255]')
    result = analyze(args.trial, args.reference or args.trial, args.stage, args.top_z, args.side, args.threshold, args.rgb_folder)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'xy_shift_m': result['estimated_xy_shift_m'],
                      'synthetic_xy_error_m': result['synthetic_xy_error_m']}))


if __name__ == '__main__':
    main()
