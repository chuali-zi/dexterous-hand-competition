"""Project a recorded slide onto the visible table strip in retained real RGB.

The horizontal table plane and RGB-derived brick dimensions are inherited
assumptions. Image top/bottom are never interpreted as physical table edges.
No simulation or platform connection is made.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from rabokit_calibration_analysis import LEFT_ROOT, RIGHT_ROOT, world_transform


def check(capture, trace, config, out):
    frames = json.loads((capture / 'robot-frames.json').read_text())['frames']
    info = json.loads((capture / 'base_cam/camera.json').read_text())
    image = np.asarray(Image.open(capture / 'base_cam/frame.png').convert('RGB'))
    if image.shape[:2] != (info['height'], info['width']):
        raise ValueError('RGB dimensions do not match camera metadata')
    sr, lp = world_transform(frames, LEFT_ROOT)
    _, rp = world_transform(frames, RIGHT_ROOT)
    sr, origin = np.asarray(sr), (np.asarray(lp) + rp) / 2
    cr, cp = world_transform(frames, info['frame_id'])
    cr, cp, k = np.asarray(cr), np.asarray(cp), np.asarray(info['k']).reshape(3, 3)
    source = np.asarray(config['source_surface_sdk_m'])
    floor = source[2] - .050
    initial_xy = np.asarray(trace['rows'][0]['target_position'][:2])

    def project(points):
        optical = (np.asarray(points) @ sr.T + origin - cp) @ cr
        if np.any(optical[:, 2] <= 0):
            raise ValueError('Projection behind camera')
        homogeneous = optical @ k.T
        return homogeneous[:, :2] / homogeneous[:, 2, None]

    def backproject(u, v):
        ray = sr.T @ cr @ np.linalg.solve(k, [u, v, 1.])
        camera = sr.T @ (cp - origin)
        return camera + ray * (floor - camera[2]) / ray[2]

    # A bounded visible strip surrounding this experiment, not the whole table.
    v_lo, v_hi = 180, 340
    rgb = image.astype(float)
    r, g, b = rgb.transpose(2, 0, 1)
    wood = (r > 70) & (r > g + 20) & (g > b + 15)
    edges = []
    for y in range(v_lo, v_hi + 1):
        x = np.flatnonzero(wood[y])
        if not len(x) or x[0] < 4 or x[-1] >= info['width'] - 4:
            raise ValueError('Visible interior table boundary is not established')
        edges.append([int(x[0]), int(x[-1])])
    edge_left = float(np.median(np.asarray(edges)[:, 0]))
    edge_right = float(np.median(np.asarray(edges)[:, 1]))
    # Four pixels of edge-classification margin; not a calibrated full error bound.
    u_lo = float(np.max(np.asarray(edges)[:, 0]) + 4)
    u_hi = float(np.min(np.asarray(edges)[:, 1]) - 4)
    corners = np.array(list(itertools.product([-.025, .025], repeat=3)))
    traces, first_outside, first_center_outside = [], None, None
    for row in trace['rows']:
        rot = Rotation.from_quat(np.asarray(row['target_quaternion_wxyz'])[[1, 2, 3, 0]])
        bottom = rot.apply(corners) + row['target_position']
        # All eight corners are required when a brick tilts: the upper corners
        # can project beyond its bottom face. This is the whole-body XY footprint.
        bottom[:, 2] = floor
        uv = project(bottom)
        center = project([[row['target_position'][0], row['target_position'][1], floor]])[0]
        inside = bool(np.all((uv[:, 0] >= u_lo) & (uv[:, 0] <= u_hi)
                             & (uv[:, 1] >= v_lo) & (uv[:, 1] <= v_hi)))
        outside_left = bool(np.min(uv[:, 0]) < edge_left)
        if outside_left and first_outside is None:
            first_outside = row['seconds']
        if center[0] < edge_left and first_center_outside is None:
            first_center_outside = row['seconds']
        traces.append(dict(seconds=row['seconds'], footprint_pixels=uv.tolist(),
                           center_on_table_pixels=center.tolist(), visible_strip_contains_footprint=inside,
                           beyond_visible_left_edge=outside_left))
    left_x = float(backproject(edge_left, 260)[0])
    inset_x = float(backproject(u_lo, 260)[0])
    final_x = trace['rows'][-1]['target_position'][0]
    # This rectangle was selected in the retained RGB between the supply blocks
    # and printed text. Its boundaries are deliberately internal analysis bounds.
    blank = [200, 270, 30, 445]
    bx0, bx1, by0, by1 = blank
    blank_fraction = float(np.mean(wood[by0:by1+1, bx0:bx1+1]))
    if blank_fraction != 1.0:
        raise ValueError('Chosen blank rectangle no longer matches retained wood-only pixels')
    blank_xyz = np.array([backproject(u, v) for u, v in itertools.product([bx0, bx1], [by0, by1])])
    blank_min, blank_max = blank_xyz.min(axis=0), blank_xyz.max(axis=0)
    stereo = json.loads((capture / 'stereo.json').read_text())
    red = max((m for m in stereo['matches'] if m['color'] == 'red'),
              key=lambda m: tuple(m['pixels'][0]))
    red_surface = np.array(red['colored_surface_point_sdk_m'])
    red_center = red_surface - [0, 0, .025]
    green = next(m for m in stereo['matches'] if m['color'] == 'green')
    green_surface = np.array(green['colored_surface_point_sdk_m'])
    above = min((m for m in stereo['matches']
                 if abs(m['colored_surface_point_sdk_m'][0]-red_surface[0]) < .025
                 and m['colored_surface_point_sdk_m'][1] > red_surface[1]),
                key=lambda m: m['colored_surface_point_sdk_m'][1])
    # Round-trip and known same-source reprojection catch transform/axis mistakes;
    # they do not measure independent calibration accuracy.
    sample_uv = np.array([[22, 180], [474, 340], [200, 30], [270, 445]], dtype=float)
    roundtrip_error = float(np.max(np.abs(project([backproject(*uv) for uv in sample_uv])-sample_uv)))
    red_reprojection_error = float(np.linalg.norm(project([red_surface])[0]-red['pixels'][0]))
    delta = project([source, source+[.010, 0, 0], source+[0, .010, 0]])
    summary = dict(recorded_rgb=str(capture / 'base_cam/frame.png'), table_plane_sdk_z_m=float(floor),
                   visible_left_edge_u=edge_left, visible_right_edge_u=edge_right,
                   inset_visible_strip_pixels=[u_lo, u_hi, v_lo, v_hi],
                   left_edge_sdk_x_m=left_x, inset_left_sdk_x_m=inset_x,
                   target_body=trace.get('candidate', {}).get('target_body', 'target'),
                   axis_aligned_50mm_brick_max_minus_x_mm=1000*(initial_xy[0]-.025-left_x),
                   four_pixel_inset_max_minus_x_mm=1000*(initial_xy[0]-.025-inset_x),
                   first_footprint_beyond_left_edge_s=first_outside,
                   first_center_beyond_left_edge_s=first_center_outside,
                   final_center_beyond_left_edge_mm=1000*(left_x-final_x),
                   entire_recorded_footprint_inside_visible_strip=all(x['visible_strip_contains_footprint'] for x in traces),
                   recorded_slide_supported_by_visible_strip=all(x['visible_strip_contains_footprint'] for x in traces),
                   final_footprint_inside_blank_rectangle=bool(np.all(
                       (np.array(traces[-1]['footprint_pixels'])[:, 0] >= bx0) &
                       (np.array(traces[-1]['footprint_pixels'])[:, 0] <= bx1) &
                       (np.array(traces[-1]['footprint_pixels'])[:, 1] >= by0) &
                       (np.array(traces[-1]['footprint_pixels'])[:, 1] <= by1))),
                   sdk_axis_image_directions={'plus_x': 'right', 'plus_y': 'up',
                                              'plus_x_10mm_pixel_delta_at_source': (delta[1]-delta[0]).tolist(),
                                              'plus_y_10mm_pixel_delta_at_source': (delta[2]-delta[0]).tolist()},
                   visible_blank_rectangle={'pixels_u0_u1_v0_v1': blank,
                                            'wood_pixel_fraction': blank_fraction,
                                            'sdk_min_m': blank_min.tolist(), 'sdk_max_m': blank_max.tolist(),
                                            'axis_aligned_50mm_center_sdk_min_m': (blank_min+[.025, .025, .025]).tolist(),
                                            'axis_aligned_50mm_center_sdk_max_m': (blank_max+[-.025, -.025, .025]).tolist(),
                                            'meaning': 'Observed blank table strip between supply and printed text; no construction-zone identity or calibration error bound.'},
                   outer_right_red={'source': str(capture/'stereo.json'), 'pixels': red['pixels'][0],
                                    'rgb_surface_sdk_m': red_surface.tolist(),
                                    'assumed_50mm_center_sdk_m': red_center.tolist(),
                                    'half_pixel_depth_perturbation_radius_m': red['half_pixel_perturbation_radius_m'],
                                    'minus_x_gap_to_green_surface_faces_mm': float(1000*(red_surface[0]-green_surface[0]-.050)),
                                    'plus_y_gap_to_blue_surface_faces_mm': float(1000*(above['colored_surface_point_sdk_m'][1]-red_surface[1]-.050)),
                                    'contact_side_for_plus_x_push': '-X face, approached through the inter-column gap or from image-down (-Y); whole-hand fit is unverified.',
                                    'visible_clear_sides': ['+X toward the blank table strip', '-Y toward lower image rows'],
                                    'caution': 'Center Z inherits 50mm height; red/green surface depths differ ~2mm. Retained initial arrangement, not current live state.'},
                   projection_checks={'pixel_roundtrip_max_error': roundtrip_error,
                                      'red_same_source_reprojection_error_pixels': red_reprojection_error},
                   scope='Conditional geometric check from retained RGB, fixed camera/robot TF and assumed table plane. Not a new native trial, table-mesh measurement or construction-region certification.',
                   limitations=['Vertical strip bounds are analysis limits inside the image, not table edges.',
                                'Four-pixel inset does not include all calibration/brick-pose uncertainty.',
                                'Neighbor occupancy, hand/arm collision, opposite table extent and destination region are separate.'])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'table-support.json').write_text(json.dumps(dict(summary=summary, rows=traces), indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(image)
    ax.add_patch(Rectangle((u_lo, v_lo), u_hi-u_lo, v_hi-v_lo, fill=False, edgecolor='cyan', linewidth=1.3,
                           label='Visible analysis strip (inset)'))
    ax.add_patch(Rectangle((bx0, by0), bx1-bx0, by1-by0, fill=False, edgecolor='magenta', linewidth=1.8,
                           label='Observed wood-only blank rectangle'))
    ax.plot(red['pixels'][0][0], red['pixels'][0][1], marker='x', color='white', markersize=10,
            label='Outer right red RGB center')
    ax.axvline(edge_left, color='yellow', linestyle='--', linewidth=1.5, label='Observed wood/background boundary')
    for t, color, label in [(0, 'lime', 'Initial footprint'), (6, 'orange', 'Mid-push footprint'), (14, 'red', 'Final footprint')]:
        row = min(traces, key=lambda r: abs(r['seconds']-t))
        uv = np.array(row['footprint_pixels'])
        uv = uv[ConvexHull(uv).vertices]
        ax.add_patch(Polygon(uv, closed=True, fill=False, edgecolor=color, linewidth=2.5, label=label))
    centers = np.array([x['center_on_table_pixels'] for x in traces])
    ax.plot(centers[:, 0], centers[:, 1], color='white', linewidth=1.3)
    ax.set_xlim(-45, info['width']-.5)
    ax.set_ylim(info['height']-.5, -.5)
    supported = summary['recorded_slide_supported_by_visible_strip']
    ax.set_title('Recorded RGB + table-plane projection: ' +
                 ('footprint remains in visible table strip' if supported else 'footprint exits visible table strip'))
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'table-footprint-projection.png', dpi=150)
    plt.close(fig)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, default=Path('runs/brick-index-depth-45/initial'))
    parser.add_argument('--trace', type=Path, default=Path('runs/simple-transfer-76/slide/center-two-low-withdraw.json'))
    parser.add_argument('--config', type=Path, default=Path('configs/bimanual-contact.json'))
    parser.add_argument('--out', type=Path, default=Path('runs/simple-transfer-76/table-support'))
    args = parser.parse_args()
    result = check(args.capture, json.loads(args.trace.read_text()), json.loads(args.config.read_text()), args.out)
    print(json.dumps(result, indent=2))
