"""Check saved real RGB for elevation, tool following, and neighboring brick layout."""
from itertools import permutations
import json
import math
from pathlib import Path
from types import SimpleNamespace

from rabokit_stereo import reconstruct
from hand_competition.closed_loop.artifacts import read_ppm


def saved_scene(folder, colors=None):
    folder = Path(folder)
    metadata = json.loads((folder / 'camera/capture.json').read_text(encoding='utf-8'))['camera_messages']
    frames = json.loads((folder / 'robot-frames.json').read_text(encoding='utf-8'))['frames']
    pairs = {}
    for name in metadata:
        width, height, rgb = read_ppm(folder / 'camera' / name / 'frame.ppm')
        pairs[name] = (SimpleNamespace(width=width, height=height, encoding='rgb8',
                                      step=width * 3, data=rgb), None)
    return reconstruct(pairs, metadata, frames, colors=colors)


def target_observation(folder):
    # Only the selected color changes; all original component and stereo thresholds remain in force.
    observation = saved_scene(folder, colors={'green'})
    green = [item for item in observation['matches'] if item['color'] == 'green']
    if observation['issues'] or len(green) != 1:
        return {'valid': False, 'reason': 'Green hidden, clipped or ambiguous',
                'issues': observation['issues']}
    return {'valid': True, 'point': green[0]['colored_surface_point_sdk_m'],
            'half_pixel_radius_m': green[0]['half_pixel_perturbation_radius_m'],
            'scope': 'target only; neighboring layout checked after withdrawal'}


def pickup_evidence(source, first, second, tool_first, tool_second, offset, clearance):
    if not first['valid'] or not second['valid']:
        return {'physical_grasp_verified': False, 'reason': 'Target not visible in both elevated observations',
                'observations': [first, second]}
    points = [first['point'], second['point']]
    tools = [tool_first, tool_second]
    elevation = min(point[2] - source[2] for point in points)
    actual_delta = [b - a for a, b in zip(*points)]
    tool_delta = [b - a for a, b in zip(*tools)]
    following_error = math.dist(actual_delta, tool_delta)
    attachment_error = max(math.dist(point, (tool[0] + offset[0], tool[1] + offset[1], tool[2] - clearance))
                           for point, tool in zip(points, tools))
    verified = (elevation > .08 and math.dist(*tools) > .04 and
                following_error < .02 and attachment_error < .03)
    return {'physical_grasp_verified': verified, 'minimum_elevation_m': elevation,
            'tool_motion_m': math.dist(*tools), 'following_error_m': following_error,
            'attachment_error_m': attachment_error, 'observations': [first, second]}


def neighboring_layout(before, after, exclude_green=False):
    if before['issues'] or after['issues']:
        return {'consistent': False, 'reason': 'Scene observation incomplete or ambiguous'}
    colors = {item['color'] for item in before['matches'] + after['matches']}
    errors = {}
    for color in colors:
        if exclude_green and color == 'green':
            continue
        a = [item for item in before['matches'] if item['color'] == color]
        b = [item for item in after['matches'] if item['color'] == color]
        if len(a) != len(b):
            return {'consistent': False, 'reason': f'{color}: visible count changed'}
        if not a:
            continue

        def pixel_error(pair):
            return max(math.dist(x, y) for x, y in zip(pair[0]['pixels'], pair[1]['pixels']))

        assignment = min(permutations(b), key=lambda order: sum(pixel_error(pair) for pair in zip(a, order)))
        errors[color] = max(pixel_error(pair) for pair in zip(a, assignment))
    return {'consistent': all(value <= 1 for value in errors.values()),
            'maximum_pixel_displacement_by_color': errors,
            'scope': 'two-view visible colored arrangement at one-pixel tolerance'}
