#!/usr/bin/env python3
"""Compare fixed-height cube hypotheses using saved synchronized RGB and robot TF.

The same pixels and ROIs are used for every height. Scores compare image
constraints; they are not height confidence, native contact or grasp verdicts.
Requires numpy, scipy and Pillow. Never connects to the robot or simulator.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image

from analyze_occluded_brick import fit_box_xy, visible_pixels
from rabokit_calibration_analysis import LEFT_ROOT, RIGHT_ROOT, world_transform


def analyze(trial, stages, rises, cameras, side=.05, threshold=160):
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    source = np.array(read(trial / 'grasp.json')['source_surface']['colored_surface_point_sdk_m'])
    frames = read(trial / 'contact/robot-frames.json')['frames']
    left, right = (world_transform(frames, n) for n in (LEFT_ROOT, RIGHT_ROOT))
    sdk_r, sdk_p = np.array(left[0]), (np.array(left[1]) + right[1]) / 2
    corners = np.array(list(itertools.product((-side / 2, side / 2), repeat=3)))
    models, reference = [], []
    for name in cameras:
        info = read(trial / 'contact/camera' / name / 'camera.json')
        rotation, position = world_transform(frames, info['frame_id'])
        models.append((np.array(rotation), np.array(position), np.array(info['k']).reshape(3, 3)))
        reference.append(info)

    def projector(rise):
        def project(q, ci):
            x, y, yaw = q
            c, s = np.cos(yaw), np.sin(yaw)
            rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            world = (corners @ rotation.T + [x, y, source[2] + rise - side / 2]) @ sdk_r.T + sdk_p
            cr, cp, k = models[ci]
            optical = (world - cp) @ cr
            if np.any(optical[:, 2] <= 0):
                raise ValueError('Box lies behind camera')
            uv = optical @ k.T
            return uv[:, :2] / uv[:, 2, None]
        return project

    prior, rois = np.r_[source[:2], 0.], []
    for ci, info in enumerate(reference):
        uv = np.concatenate([projector(h)(prior, ci) for h in rises])
        lo = np.floor(uv.min(axis=0) - 20).astype(int)
        hi = np.ceil(uv.max(axis=0) + 20).astype(int)
        rois.append([int(max(0, lo[0])), int(max(0, lo[1])),
                     int(min(info['width'], hi[0])), int(min(info['height'], hi[1]))])

    rows = []
    for stage in stages:
        images, stamps = [], []
        for name, ref in zip(cameras, reference):
            folder = trial / stage / name if stage == 'initial' else trial / stage / 'camera' / name
            info = read(folder / 'camera.json')
            if info['frame_id'] != ref['frame_id'] or info['k'] != ref['k']:
                raise ValueError('Camera identity or intrinsics changed')
            stamps.append(info['stamp'])
            with Image.open(folder / 'frame.png') as image:
                image.load()
                if image.mode != 'RGB' or image.size != (ref['width'], ref['height']):
                    raise ValueError('Expected raw RGB with the reference dimensions')
                images.append(np.array(image))
        if any(stamp != stamps[0] for stamp in stamps):
            raise ValueError('Expected synchronized camera frames')
        observations = visible_pixels(images, rois, threshold)
        for rise in rises:
            rows.append({'stage': stage, 'stamp': stamps[0], 'assumed_rise_m': rise,
                         'fit': fit_box_xy(observations, projector(rise), prior)})
    return {'scope': 'Fixed horizontal cube hypotheses with shared pixel samples across heights; '
            'no native object pose, calibrated height accuracy, contact identity or grasp verdict.',
            'trial': str(trial), 'cameras': cameras, 'source_surface_sdk_m': source.tolist(),
            'box_side_m': side, 'green_threshold': threshold, 'rois': rois, 'rows': rows,
            'physical_pose_accuracy_verified': False, 'official_score': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--stage', action='append', help='Repeat for multiple stages; default closed and both lifts')
    parser.add_argument('--rise', type=float, action='append', help='Hypothesized rise in metres; default 0, 0.01, 0.02')
    parser.add_argument('--camera', action='append', choices=('base_cam', 'base_cam2', 'base_cam3'))
    parser.add_argument('--side', type=float, default=.05)
    parser.add_argument('--threshold', type=int, default=160)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rises = args.rise or [0., .01, .02]
    cameras = args.camera or ['base_cam', 'base_cam2', 'base_cam3']
    if (not np.isfinite(rises).all() or not np.isfinite(args.side) or not 0 < args.side < 1
            or not 0 <= args.threshold <= 255 or len(cameras) != len(set(cameras))):
        parser.error('Expected finite heights, side in (0,1) metres, RGB threshold in [0,255] and unique cameras')
    try:
        result = analyze(args.trial, args.stage or ['closed', 'lifted_000', 'lifted_001'],
                         rises, cameras, args.side, args.threshold)
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Saved {len(result["rows"])} image-constraint fits; no grasp verdict: {args.output}')


if __name__ == '__main__':
    main()
