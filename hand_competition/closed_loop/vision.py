"""RGB color components and calibrated stereo; no simulator pose is read."""
from collections import deque
from dataclasses import dataclass
from .geometry import triangulate


@dataclass(frozen=True)
class Detection:
    color: str
    centroid: tuple
    area: int


@dataclass(frozen=True)
class VisualEstimate:
    position_m: tuple
    color: str
    stereo_residual_m: float
    pixels: tuple


class VisionError(ValueError):
    pass


def classify(r, g, b):
    if r > 150 and g < 100 and b < 100: return 'red'
    if g > 140 and r < 100 and b < 100: return 'green'
    if b > 150 and r < 100 and g < 120: return 'blue'
    if r > 150 and g > 140 and b < 100: return 'yellow'
    return None


def components(frame, minimum_area=20):
    labels = [classify(*frame.rgb[i:i+3]) for i in range(0, len(frame.rgb), 3)]
    seen = bytearray(len(labels)); result = []; width = frame.width
    for index, color in enumerate(labels):
        if color is None or seen[index]: continue
        todo = deque([index]); seen[index] = 1; area = sx = sy = 0
        while todo:
            p = todo.popleft(); x = p % width; y = p // width
            area += 1; sx += x; sy += y
            neighbors = []
            if x: neighbors.append(p-1)
            if x+1 < width: neighbors.append(p+1)
            if y: neighbors.append(p-width)
            if y+1 < frame.height: neighbors.append(p+width)
            for q in neighbors:
                if not seen[q] and labels[q] == color:
                    seen[q] = 1; todo.append(q)
        if area >= minimum_area: result.append(Detection(color, (sx/area, sy/area), area))
    return result


def estimate_single_block(observation, cameras, max_residual_m=0.02):
    by_name = {camera.name: camera for camera in cameras}
    if len(observation.frames) != 2: raise VisionError('single-block baseline requires two calibrated RGB views')
    found = [components(frame) for frame in observation.frames]
    if any(len(items) != 1 for items in found):
        raise VisionError('missing/ambiguous colored block; no motion target produced')
    first, second = found[0][0], found[1][0]
    if first.color != second.color: raise VisionError('stereo color correspondence mismatch')
    try:
        point, residual = triangulate(by_name[observation.frames[0].camera_id], first.centroid,
                                      by_name[observation.frames[1].camera_id], second.centroid, max_residual_m)
    except ValueError as error: raise VisionError(str(error)) from error
    if not (-0.4 <= point[0] <= 0.4 and 0.05 <= point[1] <= 0.75 and -0.02 <= point[2] <= 0.35):
        raise VisionError('visual estimate outside experiment bounds')
    return VisualEstimate(point, first.color, residual, (first.centroid, second.centroid))
