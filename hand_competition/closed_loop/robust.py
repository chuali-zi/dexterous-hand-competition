"""RGB location bundles; their mean precision never certifies stability.

Uncertainty is a heuristic under the documented bounded/stationary-noise model,
not a calibrated real-world confidence guarantee. Moving objects must not be
smoothed in this stationary estimator.
"""
from dataclasses import dataclass
import math
from .geometry import triangulate, distance


class BundleError(ValueError):
    pass


@dataclass(frozen=True)
class Sample:
    index: int
    sensor_time_s: float
    tip: tuple
    camera_ids: tuple
    pixels: tuple
    color: str


@dataclass(frozen=True)
class BundleEstimate:
    position_m: tuple
    uncertainty_xyz_m: tuple
    color: str
    sample_indices: tuple
    sensor_start_s: float
    sensor_end_s: float
    mean_pixels: tuple
    effective_samples: float

    @property
    def radius_m(self):
        return math.sqrt(sum(v*v for v in self.uncertainty_xyz_m))


def estimate_bundle(samples, cameras, config):
    if len(samples) < config.robust_min_samples:
        raise BundleError('insufficient_samples')
    if len({s.color for s in samples}) != 1:
        raise BundleError('identity_change')
    if any(samples[i].sensor_time_s-samples[i-1].sensor_time_s > config.max_observation_gap_s+1e-9
           for i in range(1, len(samples))):
        raise BundleError('sensor_gap')
    if any(distance(s.tip, samples[-1].tip) > config.robust_stationary_tip_m for s in samples):
        raise BundleError('robot_moved')
    names = [c.name for c in cameras]
    if len(names) != 2: raise BundleError('two_views_required')
    values = []
    for sample in samples:
        by_name = dict(zip(sample.camera_ids, sample.pixels))
        if set(by_name) != set(names): raise BundleError('camera_identity_change')
        values.append(tuple(v for name in names for v in by_name[name]))
    count = len(values)
    means = [sum(row[i] for row in values)/count for i in range(4)]
    if any(max(row[i] for row in values)-min(row[i] for row in values) > config.robust_pixel_span_px for i in range(4)):
        raise BundleError('nonstationary_pixels')
    # Scan chronological splits, including a one-frame tail. Averaging must not
    # erase a late step. Use segment variance with a conservative multiple-split
    # margin; constant image bias remains unobservable.
    for cut in range(2, count):
        for coordinate in range(4):
            a = [r[coordinate] for r in values[:cut]]
            b = [r[coordinate] for r in values[cut:]]
            ma, mb = sum(a)/len(a), sum(b)/len(b)
            va = sum((x-ma)**2 for x in a)/(len(a)-1)
            vb = sum((x-mb)**2 for x in b)/(len(b)-1) if len(b) > 1 else va
            scale = math.sqrt(max(va, vb)*(1/len(a)+1/len(b)))
            if abs(ma-mb) > config.robust_change_sigma*scale+config.robust_change_floor_px:
                raise BundleError('temporal_change')
    covariance = [[sum((row[i]-means[i])*(row[j]-means[j]) for row in values)/(count-1)
                   for j in range(4)] for i in range(4)]
    # Inflate rather than treating statistically significant serial correlation
    # as independent samples. Constant bias is not detectable by this check.
    rho = 0.0
    for i in range(4):
        denominator = sum((row[i]-means[i])**2 for row in values)
        if denominator > 1e-12:
            correlation = sum((values[k][i]-means[i])*(values[k-1][i]-means[i]) for k in range(1, count))/denominator
            if correlation > 2/math.sqrt(count): rho = max(rho, min(correlation, 0.9))
    effective = max(1.0, count*(1-rho)/(1+rho))
    pixels = (tuple(means[:2]), tuple(means[2:]))
    try:
        point, _ = triangulate(cameras[0], pixels[0], cameras[1], pixels[1], config.stereo_residual_m)
        columns = []
        for i in range(4):
            plus = means[:]; minus = means[:]
            plus[i] += 0.1; minus[i] -= 0.1
            high, _ = triangulate(cameras[0], plus[:2], cameras[1], plus[2:], config.stereo_residual_m)
            low, _ = triangulate(cameras[0], minus[:2], cameras[1], minus[2:], config.stereo_residual_m)
            columns.append(tuple((high[j]-low[j])/0.2 for j in range(3)))
    except ValueError as error: raise BundleError('invalid_mean_geometry') from error
    uncertainty = []
    for axis in range(3):
        variance = sum(columns[i][axis]*covariance[i][j]*columns[j][axis] for i in range(4) for j in range(4))/effective
        uncertainty.append(config.robust_sigma_factor*math.sqrt(max(0.0, variance)))
    return BundleEstimate(point, tuple(uncertainty), samples[-1].color,
                          tuple(s.index for s in samples), samples[0].sensor_time_s,
                          samples[-1].sensor_time_s, pixels, effective)


def bounded_placement(estimate, tip, config):
    point = estimate.position_m
    radius = estimate.radius_m+config.robust_absolute_floor_m
    return (config.build_xy[0] <= point[0] <= config.build_xy[1]
            and config.build_xy[2] <= point[1] <= config.build_xy[3]
            and distance(point, config.target)+radius <= config.goal_tolerance_m
            and abs(point[2]-config.table_center_height_m)+estimate.uncertainty_xyz_m[2]+config.robust_absolute_floor_m <= config.table_height_tolerance_m
            and distance(point, tip)-radius-config.robust_stationary_tip_m >= config.hand_separation_m)
