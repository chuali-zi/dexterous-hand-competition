"""Independent record verifier for stationary RGB bundles.

Does not import the controller or its estimator. It checks saved primary pixels,
stationarity assumptions, covariance propagation and conservative bounds.
"""
import math
from .geometry import triangulate, distance


def reconstruct(payload, samples, cameras, config):
    indices = payload['sample_indices']
    if (not isinstance(indices, list) or not config.robust_min_samples <= len(indices) <= config.robust_max_samples
            or any(type(i) is not int or not 0 <= i < len(samples) for i in indices)
            or indices != list(range(indices[0], indices[-1]+1)) or indices[-1] != len(samples)-1):
        raise ValueError('bundle requires consecutive fresh primary observations')
    selected = [samples[i] for i in indices]
    if any(s['pixels'] is None for s in selected): raise ValueError('bundle contains missing/ambiguous RGB')
    if len({s['color'] for s in selected}) != 1: raise ValueError('bundle identity change')
    if any(distance(s['tip'], selected[-1]['tip']) > config.robust_stationary_tip_m for s in selected):
        raise ValueError('bundle robot was moving')
    if any(b['sensor_time_s']-a['sensor_time_s'] > config.max_observation_gap_s+1e-9
           for a, b in zip(selected, selected[1:])): raise ValueError('bundle sensor gap')
    rows = []
    for s in selected:
        mapping = dict(zip(s['camera_ids'], s['pixels']))
        if set(mapping) != {c.name for c in cameras}: raise ValueError('bundle camera identity mismatch')
        rows.append([coordinate for camera in cameras for coordinate in mapping[camera.name]])
    n = len(rows)
    columns = list(zip(*rows)); means = [math.fsum(column)/n for column in columns]
    if any(max(c)-min(c) > config.robust_pixel_span_px for c in columns): raise ValueError('nonstationary bundle pixels')
    for cut in range(2, n):
        for column in columns:
            first, second = column[:cut], column[cut:]
            a, b = math.fsum(first)/len(first), math.fsum(second)/len(second)
            var_a = math.fsum((x-a)**2 for x in first)/(len(first)-1)
            var_b = math.fsum((x-b)**2 for x in second)/(len(second)-1) if len(second)>1 else var_a
            margin = config.robust_change_sigma*math.sqrt(max(var_a,var_b)*(1/len(first)+1/len(second)))+config.robust_change_floor_px
            if abs(a-b) > margin: raise ValueError('bundle hides temporal RGB change')
    centered = [[v-means[i] for i, v in enumerate(row)] for row in rows]
    covariance = [[math.fsum(r[i]*r[j] for r in centered)/(n-1) for j in range(4)] for i in range(4)]
    serial = 0.0
    for i in range(4):
        variance_sum = math.fsum(r[i]**2 for r in centered)
        if variance_sum > 1e-12:
            rho = math.fsum(a[i]*b[i] for a, b in zip(centered, centered[1:]))/variance_sum
            if rho > 2/math.sqrt(n): serial = max(serial, min(rho, .9))
    effective = max(1.0, n*(1-serial)/(1+serial))
    point, _ = triangulate(cameras[0], means[:2], cameras[1], means[2:], config.stereo_residual_m)
    derivatives = []
    for coordinate in range(4):
        high = list(means); low = list(means)
        high[coordinate] += .1; low[coordinate] -= .1
        p, _ = triangulate(cameras[0], high[:2], cameras[1], high[2:], config.stereo_residual_m)
        q, _ = triangulate(cameras[0], low[:2], cameras[1], low[2:], config.stereo_residual_m)
        derivatives.append([(p[j]-q[j])/.2 for j in range(3)])
    uncertainty = [config.robust_sigma_factor*math.sqrt(max(0.0, math.fsum(
        derivatives[i][axis]*covariance[i][j]*derivatives[j][axis]
        for i in range(4) for j in range(4))/effective)) for axis in range(3)]
    expected = {'sample_indices': indices, 'position_m': point, 'uncertainty_xyz_m': uncertainty,
                'color': selected[-1]['color'], 'sensor_start_s': selected[0]['sensor_time_s'],
                'sensor_end_s': selected[-1]['sensor_time_s'], 'mean_pixels': (means[:2], means[2:]),
                'effective_samples': effective}
    def equal(a, b):
        if isinstance(a, (list, tuple)):
            return isinstance(b, (list, tuple)) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        if isinstance(a, str): return a == b
        return type(b) in (int, float) and math.isfinite(b) and abs(a-b) <= 1e-8
    if any(not equal(value, payload.get(key)) for key, value in expected.items()):
        raise ValueError('bundle statistics differ from independently reconstructed RGB')
    expected['tip'] = selected[-1]['tip']
    expected['radius_m'] = math.sqrt(math.fsum(v*v for v in uncertainty))
    if expected['radius_m'] > config.grasp_distance_m/2: raise ValueError('bundle exceeds uncertainty budget')
    return expected


def held(bundle, config):
    return distance(bundle['position_m'], bundle['tip'])+bundle['radius_m']+config.robust_absolute_floor_m <= config.grasp_distance_m


def placement(bundle, config):
    p = bundle['position_m']; radius = bundle['radius_m']+config.robust_absolute_floor_m
    return (bundle['radius_m'] <= config.robust_max_uncertainty_m and config.build_xy[0] <= p[0] <= config.build_xy[1] and config.build_xy[2] <= p[1] <= config.build_xy[3]
            and distance(p, config.target)+radius <= config.goal_tolerance_m
            and abs(p[2]-config.table_center_height_m)+bundle['uncertainty_xyz_m'][2]+config.robust_absolute_floor_m <= config.table_height_tolerance_m
            and distance(p, bundle['tip'])-radius-config.robust_stationary_tip_m >= config.hand_separation_m)
