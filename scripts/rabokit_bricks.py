#!/usr/bin/env python3
"""Find colored brick candidates in a captured overhead RGB frame; no motion."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hand_competition.closed_loop.artifacts import read_ppm
from hand_competition.multi_block.perception import color_components


def saturated_color(r, g, b):
    # Bright primary colors separate the bricks from brown wood and dark lettering.
    if r > 180 and g > 160 and b < 80:
        return (255, 255, 0)
    if r > 180 and r > 2 * g and r > 2 * b:
        return (255, 0, 0)
    if g > 160 and g > 2 * r and g > 2 * b:
        return (0, 255, 0)
    if b > 180 and b > 2 * r and b > 2 * g:
        return (0, 0, 255)
    return (0, 0, 0)


def detect(width, height, rgb, minimum_area=80):
    mask = bytes(channel for i in range(0, len(rgb), 3)
                 for channel in saturated_color(*rgb[i:i + 3]))
    candidates, rejected = [], []
    for item in color_components(SimpleNamespace(width=width, height=height, rgb=mask)):
        if item.area < minimum_area:
            continue
        xmin, xmax, ymin, ymax = item.bounds
        box_width, box_height = xmax - xmin + 1, ymax - ymin + 1
        fill = item.area / (box_width * box_height)
        candidate = {**asdict(item), 'fill_ratio': fill}
        if item.touches_border or not 0.7 <= box_width / box_height <= 1.4 or fill < 0.7:
            rejected.append({**candidate, 'reason': 'clipped, merged or non-square component'})
        else:
            candidates.append(candidate)
    return {'candidates': candidates, 'rejected_components': rejected,
            'coordinates': 'image_pixels_only', 'metric_calibration_available': False,
            'official_score': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path, help='Directory containing frame.ppm and camera.json')
    args = parser.parse_args()
    width, height, rgb = read_ppm(args.capture / 'frame.ppm')
    report = detect(width, height, rgb)
    report['camera'] = json.loads((args.capture / 'camera.json').read_text(encoding='utf-8'))
    if (width, height) != (report['camera']['width'], report['camera']['height']):
        raise ValueError('Captured image and camera metadata dimensions differ')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
