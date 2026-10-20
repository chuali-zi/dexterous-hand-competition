#!/usr/bin/env python3
"""Check the current pad pickup with the local SDK and a retained grasp.json; no robot connection."""
import argparse
import json
import math
from pathlib import Path
import time

from rabokit_grasp import high_pass_through_plan, pad_tool_geometry, preshape_fingers_plan
from rabokit_hand import targets
from rabokit_joint_path import centered_inverse, pickup_inspection_plan, seeded_approach_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path, help='Retained grasp.json supplying start and source_surface')
    parser.add_argument('--seed-record', type=Path, help='Original achieved contact reference; defaults to record')
    parser.add_argument('--candidate', type=Path, help='Offline study JSON; omit to check the current controller recipe')
    parser.add_argument('--output', type=Path, required=True, help='New JSON file for the numeric plan or rejection')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output must be a new file')
    reference = json.loads(args.record.read_text(encoding='utf-8'))
    seed_record = json.loads(args.seed_record.read_text(encoding='utf-8')) if args.seed_record else reference
    import rabo_robolib
    from _robolib_core.arms import _lbot_kinematics as k

    started = time.monotonic()
    numeric_checks = {}
    forward = lambda joints: k.forward_kinematics('left', joints)
    inverse = lambda target, seed: centered_inverse(target, seed, forward, k.REAL_JOINT_LIMITS_LEFT, numeric_checks)
    clearance, offset_y, offset_x, rotation = pad_tool_geometry()
    candidate = None
    if args.candidate:
        import numpy as np
        candidate = json.loads(args.candidate.read_text(encoding='utf-8'))
        clearance, offset_y, offset_x = (candidate[key] for key in (
            'tool_clearance_m', 'tool_offset_y_m', 'tool_offset_x_m'))
        roll, pitch, yaw = candidate['tool_pose'][1]
        cx, sx, cy, sy, cz, sz = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
        rotation = np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                             [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx], [-sy, cy*sx, cy*cx]])
    surface = reference['source_surface']
    point = surface['colored_surface_point_sdk_m']
    result = {'scope': 'Retained RGB and robot states; numerical SDK route only, no whole-body collision check',
              'sdk_public_version': getattr(rabo_robolib, '__version__', None),
              'record': str(args.record), 'seed_record': str(args.seed_record or args.record),
              'robot_initialized': False, 'motion_commanded': False,
              'physical_grasp_verified': False, 'numeric_seed_is_achieved_pose': False,
              'tool_clearance_m': clearance, 'tool_offset_y_m': offset_y, 'tool_offset_x_m': offset_x,
              'tool_rotation': rotation.tolist(), 'candidate': candidate, 'passed': False}
    try:
        events, checks = seeded_approach_plan(reference['start'], point,
            surface['half_pixel_perturbation_radius_m'], clearance, seed_record, 'pickup', inverse, forward,
            tool_offset_y=offset_y, tool_offset_x=offset_x, tool_rotation=rotation)
        events, extra = pickup_inspection_plan(events, point, inverse, forward,
                                               tool_offset_y=offset_y, tool_offset_x=offset_x)
        checks.extend(extra)
        events = preshape_fingers_plan(events, limited_pinch=True, prepare_thumb_high=True,
                                        compare_empty_closure=True, pad_grasp=True)
        events = high_pass_through_plan(events, reference['start'], point[2])
        previous, seen, maximum = reference['start']['joints'], {}, 0.
        for event in events:
            joints = seen[event['reverse_of']] if 'reverse_of' in event else event.get('joints', previous)
            if (len(joints) != len(k.REAL_JOINT_LIMITS_LEFT) or
                    any(not low <= value <= high for value, (low, high) in zip(joints, k.REAL_JOINT_LIMITS_LEFT))):
                raise ValueError('Planned joint target exceeds SDK joint limits')
            maximum = max(maximum, max(abs(a - b) for a, b in zip(joints, previous)))
            if maximum > .15:
                raise ValueError('Planned joint step exceeds 0.15 rad')
            previous, seen[event['name']] = joints, joints
            if 'hand' in event:
                pose = event['hand']
                values = targets(pose, event.get('thumb_yaw_rad'), pad_grasp=True)
                if candidate and pose != 'open':
                    fraction = (0. if pose in ('pinch_preshape_without_pinky', 'pinch_high_thumb_ready_without_pinky')
                                else candidate['preparation_fraction'] if pose in (
                                    'pinch_finger_ready_without_pinky', 'pinch_contact_ready_without_pinky')
                                else candidate['opposition_fraction'] if pose in (
                                    'pinch_high_thumb_opposed_without_pinky', 'pinch_thumb_opposed_without_pinky') else 1.)
                    values = {name: value + fraction * (candidate['goals'][name] - value)
                              for name, value in candidate['high_goals'].items()}
                event['native_hand_targets_rad'] = values
        result.update(passed=True, events=events, checks=checks, event_count=len(events),
                      maximum_joint_step_rad=maximum,
                      pass_through_joint_count=sum(bool(e.get('pass_through')) for e in events))
    except ValueError as error:
        result['error'] = str(error)
    result.update(numeric_inverse_checks=numeric_checks, wall_seconds=time.monotonic() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key not in (
        'events', 'checks', 'fk_continuation_fallbacks', 'candidate')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
