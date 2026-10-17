#!/usr/bin/env python3
"""Capture synchronized RGB views and locate colored surfaces in SDK coordinates."""
import argparse
from itertools import product
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from rabokit_camera import capture_cameras, rgb_bytes, save_pair
from rabokit_bricks import detect
from rabokit_calibration_analysis import LEFT_ROOT, RIGHT_ROOT, matvec, subtract, transpose, world_transform
from hand_competition.closed_loop.geometry import Camera, distance, triangulate


def match_candidates(cameras, candidates, sdk_origin, sdk_rotation):
    matches, issues, used = [], [], set()
    for item in candidates[0]:
        options = []
        for index, other in enumerate(candidates[1]):
            if item['color'] != other['color']:
                continue
            try:
                point, residual = triangulate(cameras[0], item['centroid'],
                                               cameras[1], other['centroid'], .005)
                options.append((index, other, point, residual))
            except ValueError:
                pass
        if len(options) != 1 or options[0][0] in used:
            issues.append(f'{item["color"]} at {item["centroid"]}: missing/ambiguous stereo match')
            continue
        index, other, point, residual = options[0]
        used.add(index)
        uncertainty = 0.0
        for du1, dv1, du2, dv2 in product((-.5, .5), repeat=4):
            perturbed, _ = triangulate(
                cameras[0], (item['centroid'][0] + du1, item['centroid'][1] + dv1),
                cameras[1], (other['centroid'][0] + du2, other['centroid'][1] + dv2), .02)
            uncertainty = max(uncertainty, distance(point, perturbed))
        matches.append({'color': item['color'],
                        'colored_surface_point_sdk_m': matvec(transpose(sdk_rotation), subtract(point, sdk_origin)),
                        'pixels': [item['centroid'], other['centroid']],
                        'stereo_ray_residual_m': residual,
                        'half_pixel_perturbation_radius_m': uncertainty})
    if len(used) != len(candidates[1]):
        issues.append('Some second-view candidates have no unique first-view match')
    return matches, issues


def reconstruct(pairs, metadata, frames):
    cameras, candidates, component_issues = [], [], []
    for name in pairs:
        info = metadata[name]
        rotation, origin = world_transform(frames, info['frame_id'])
        k = [float(value) for value in info['k']]
        cameras.append(Camera(name, origin, transpose(rotation), k[0], k[4], k[2], k[5],
                              info['width'], info['height'], 'manual-mount-and-robot-frames'))
        image = pairs[name][0]
        detected = detect(image.width, image.height, rgb_bytes(image))
        candidates.append(detected['candidates'])
        if detected['rejected_components']:
            component_issues.append(f'{name}: rejected colored components remain')
    left = world_transform(frames, LEFT_ROOT)
    right = world_transform(frames, RIGHT_ROOT)
    sdk_origin = tuple((a + b) / 2 for a, b in zip(left[1], right[1]))
    matches, issues = match_candidates(cameras, candidates, sdk_origin, left[0])
    return {'matches': matches, 'issues': component_issues + issues,
            'candidate_counts': [len(view) for view in candidates],
            'sdk_origin_world_m': sdk_origin,
            'camera_origins_world_m': [camera.origin for camera in cameras],
            'baseline_m': distance(cameras[0].origin, cameras[1].origin),
            'camera_messages': metadata,
            'coordinate_basis': 'left SDK position reference; robot-root midpoint and manual camera mount',
            'metric_source': 'RGB triangulation; no object pose/TF input',
            'physical_grasp_verified': False, 'official_score': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=6)
    parser.add_argument('--capture-only', action='store_true', help='Save synchronized RGB without requiring visible brick matches')
    args = parser.parse_args()
    namespace = os.getenv('ROS2_NAMESPACE', '').strip('/')
    if not namespace:
        parser.error('Run in the Rabokit cloud simulation terminal')
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 30:
        parser.error('--seconds must be in (0, 30]')
    if args.output.exists():
        parser.error('--output must be a new directory')
    topics = {name: (f'/{namespace}/base/{name}/image_raw', f'/{namespace}/base/{name}/camera_info')
              for name in ('base_cam', 'base_cam2')}
    pairs, counts = capture_cameras(topics, args.seconds)
    metadata = {name: save_pair(*pairs[name], counts[name], args.seconds, *topics[name], args.output / name)
                for name in topics}
    if args.capture_only:
        (args.output / 'capture.json').write_text(json.dumps({'camera_messages': metadata,
            'metric_reconstruction_run': False, 'official_score': None}, indent=2) + '\n', encoding='utf-8')
        print('PASS: synchronized RGB capture (no object reconstruction)', flush=True)
        return
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('rabokit_robot_frames.py')),
                             '--seconds', '2', '--output', str(args.output / 'robot-frames.json')],
                            capture_output=True, text=True, timeout=15)
    (args.output / 'robot-frames.log').write_text(result.stdout + result.stderr, encoding='utf-8')
    result.check_returncode()
    frames = json.loads((args.output / 'robot-frames.json').read_text(encoding='utf-8'))['frames']
    report = reconstruct(pairs, metadata, frames)
    (args.output / 'stereo.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)
    if report['issues'] or not report['matches']:
        raise RuntimeError('Stereo observation is incomplete or ambiguous; see stereo.json')
    print(f'PASS: {len(report["matches"])} synchronized RGB stereo surface points')


if __name__ == '__main__':
    main()
