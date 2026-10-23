"""Image containment can exclude the intended tilt; it cannot verify a grasp."""

import itertools
import math

from rabokit_calibration_analysis import (
    LEFT_ROOT,
    RIGHT_ROOT,
    matvec,
    transpose,
    world_transform,
)


def bound_evidence(projected, components, tolerance_px=5.0):
    """Use even rejected/non-square green fragments, without inventing corners."""
    visible = [c for c in components if c["area"] >= 10]
    if not visible:
        return dict(available=False, excluded=False, reason="no_usable_green_fragment")
    if not projected or not all(math.isfinite(v) for p in projected for v in p):
        raise ValueError("Expected finite projected tilt envelope")
    lo = [min(p[i] for p in projected) for i in (0, 1)]
    hi = [max(p[i] for p in projected) for i in (0, 1)]
    bounds = [
        min(c["bounds"][0] for c in visible),
        max(c["bounds"][1] for c in visible),
        min(c["bounds"][2] for c in visible),
        max(c["bounds"][3] for c in visible),
    ]
    leak = max(
        0.0, lo[0] - bounds[0], bounds[1] - hi[0], lo[1] - bounds[2], bounds[3] - hi[1]
    )
    return dict(
        available=True,
        excluded=leak > tolerance_px,
        maximum_bound_violation_px=leak,
        tolerance_px=tolerance_px,
        projected_bounds=[lo[0], hi[0], lo[1], hi[1]],
        observed_bounds=bounds,
    )


def check_tilt_visibility(frames, metadata, green_rgb, config):
    """Project the full 50 mm cube swept through 0..planned tilt, using camera TF.

    This is a conservative enclosure of the intended motion, not an estimate of
    the occluded brick's current pose. Two cameras must exclude it before the
    caller withdraws instead of proceeding with support insertion.
    """
    sr, lp = world_transform(frames, LEFT_ROOT)
    _, rp = world_transform(frames, RIGHT_ROOT)
    origin = [(a + b) / 2 for a, b in zip(lp, rp)]
    source = config["source_surface_sdk_m"]
    center = [source[0], source[1], source[2] - 0.025]
    pivot = config["tilt_pivot_sdk_m"]
    points = []
    for step in range(13):
        angle = config["tilt_angle_rad"] * step / 12
        c, s = math.cos(angle), math.sin(angle)
        for corner in itertools.product((-0.025, 0.025), repeat=3):
            x, y, z = [a + b - v for a, b, v in zip(center, corner, pivot)]
            sdk = [pivot[0] + c * x + s * z, pivot[1] + y, pivot[2] - s * x + c * z]
            points.append([a + b for a, b in zip(matvec(sr, sdk), origin)])
    cameras = {}
    for name, observation in green_rgb.items():
        info = metadata[name]
        cr, cp = world_transform(frames, info["frame_id"])
        projected = []
        for point in points:
            optical = matvec(transpose(cr), [a - b for a, b in zip(point, cp)])
            if optical[2] <= 0:
                raise ValueError("Tilt envelope is behind the camera")
            pixel = matvec([info["k"][i : i + 3] for i in (0, 3, 6)], optical)
            projected.append([pixel[0] / pixel[2], pixel[1] / pixel[2]])
        cameras[name] = bound_evidence(projected, observation["components"])
    available = sum(c["available"] for c in cameras.values())
    excluded = sum(c["excluded"] for c in cameras.values()) >= 2
    return dict(
        excluded=excluded,
        available_cameras=available,
        cameras=cameras,
        scope="Necessary RGB containment in the intended tilt region only; passing or missing green does not verify contact, support or grasp",
    )
