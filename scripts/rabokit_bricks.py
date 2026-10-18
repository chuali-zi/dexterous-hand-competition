#!/usr/bin/env python3
"""Find colored brick candidates in a captured overhead RGB frame; no motion."""
import argparse
from dataclasses import asdict
import json
import math
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


def rotated_square(mask, width, component):
    """Fit a minimum-area box to a rejected component, using RGB pixels only."""
    xmin, xmax, ymin, ymax = component.bounds
    color = {'red': (255, 0, 0), 'green': (0, 255, 0),
             'blue': (0, 0, 255), 'yellow': (255, 255, 0)}[component.color]
    points = [(x, y) for x in range(xmin, xmax + 1) for y in range(ymin, ymax + 1)
              if tuple(mask[(y * width + x) * 3:(y * width + x) * 3 + 3]) == color]
    # A separate same-color component inside these bounds must not be folded in.
    if len(points) != component.area:
        return None

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower, upper = [], []
    for chain, ordered in ((lower, points), (upper, reversed(points))):
        for point in ordered:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return None
    boxes = []
    for a, b in zip(hull, hull[1:] + hull[:1]):
        angle = math.atan2(b[1] - a[1], b[0] - a[0])
        c, s = math.cos(angle), math.sin(angle)
        u = [x * c + y * s for x, y in hull]
        v = [-x * s + y * c for x, y in hull]
        # Pixel centers need their unit-pixel extent in the rotated axes.
        extent = abs(c) + abs(s)
        w, h = max(u) - min(u) + extent, max(v) - min(v) + extent
        boxes.append((w * h, w, h, angle))
    area, w, h, angle = min(boxes)
    fill = component.area / area
    if min(w, h) / max(w, h) < .85 or fill < .9:
        return None
    return {'width_px': w, 'height_px': h, 'angle_deg': math.degrees(angle),
            'fill_ratio': fill}


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
        axis_square = 0.7 <= box_width / box_height <= 1.4 and fill >= 0.7
        fit = None if item.touches_border or axis_square else rotated_square(mask, width, item)
        if fit is not None:
            candidate['rotated_square'] = fit
        if item.touches_border or not (axis_square or fit is not None):
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
