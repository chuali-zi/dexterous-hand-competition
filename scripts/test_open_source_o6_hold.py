"""CPU cross-engine experiment for the public O6 support-withdrawal recipe.

Uses our URDF-derived convex meshes, not the author's USD/PhysX model.
This tests holding a prepositioned cube, not picking it up from a table.
"""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np


def build_model(source, mass, native_drive=False):
    tree = ET.parse(source)
    root = tree.getroot()
    root.find('option').set('timestep', str(1 / 240))
    world = root.find('worldbody')
    for item in list(world):
        if item.get('name') != 'rh_hand_base_link':
            world.remove(item)
    hand = world.find('body')
    hand.set('pos', '0 0 .22')
    hand.set('quat', '1 0 0 0')
    hand.remove(hand.find('freejoint'))
    root.remove(root.find('equality'))
    for contact in list(root.find('contact')):
        if not native_drive or not contact.get('body1', '').startswith('rh_'):
            root.find('contact').remove(contact)
    bodies = list(hand.iter('body'))
    # Upstream explicitly disables self collisions. Exclude all hand body pairs.
    if not native_drive:
        for i, a in enumerate(bodies):
            for b in bodies[i + 1:]:
                ET.SubElement(root.find('contact'), 'exclude', body1=a.get('name'), body2=b.get('name'))
    for actuator in list(root.find('actuator')):
        if not actuator.get('name').startswith('rh_'):
            root.find('actuator').remove(actuator)
        elif not native_drive:
            actuator.set('kp', '3')
            actuator.set('kv', '.12')
            actuator.set('forcerange', '-.5 .5')
    ET.SubElement(world, 'geom', name='ground', type='plane', size='1 1 .01', pos='0 0 0')
    cube = ET.SubElement(world, 'body', name='target', pos='.043 0 .315')
    ET.SubElement(cube, 'freejoint', name='target_free')
    ET.SubElement(cube, 'geom', name='cube', type='box', size='.025 .025 .025', mass=str(mass))
    support = ET.SubElement(world, 'body', name='support', mocap='true', pos='.043 0 .281')
    ET.SubElement(support, 'geom', name='support_geom', type='box', size='.016 .016 .009')
    return ET.tostring(root, encoding='unicode')


def evaluate(rows):
    hold = [r for r in rows if 4 <= r['t'] <= 6.1]
    if len(hold) < 2 or not any(r['t'] >= 8.5 for r in rows):
        return {'pass': False, 'reason': 'incomplete hold or release observation'}
    p = np.array([r['cube'] for r in hold])
    drift = float(np.max(np.linalg.norm(p - p[0], axis=1)))
    released = [r['cube'][2] for r in rows if r['t'] >= 8.5]
    checks = {
        'hold_at_least_2s': hold[-1]['t'] - hold[0]['t'] >= 2,
        'near_initial_height': bool(p[:, 2].min() > .29),
        'drift_under_10mm': drift < .01,
        'no_support_contact': not any(r['support_contact'] for r in hold),
        'finger_contact': all(r['finger_contact'] for r in hold),
        'falls_after_opening': bool(p[0, 2] - min(released) > .08),
    }
    return {'pass': all(checks.values()), 'checks': checks,
            'hold_min_z_m': float(p[:, 2].min()), 'hold_drift_m': drift,
            'release_drop_m': float(p[0, 2] - min(released))}


def run(source, out, mass, native_drive, close):
    import mujoco
    out.mkdir(parents=True, exist_ok=True)
    xml = build_model(source, mass, native_drive)
    (out / 'model.xml').write_text(xml, encoding='utf-8')
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    target = model.body('target').id
    support = model.body('support').id
    closed = []
    for i in range(model.nu):
        name = model.actuator(i).name
        closed.append(1.2 if 'yaw' in name else .5 if 'thumb_cmc_pitch' in name
                      else .93 if 'thumb_ip' in name else 1.157 if 'dip' in name else 1.3)
    rows = []
    for step in range(2161):
        t = step / 240
        amount = np.clip((t - 1) / 2, 0, 1) * (1 - np.clip(t - 7, 0, 1)) if close else 0
        amount = amount * amount * (3 - 2 * amount)
        data.ctrl[:] = np.array(closed) * amount
        retract = np.clip(t - 3, 0, 1)
        retract = retract * retract * (3 - 2 * retract)
        data.mocap_pos[0] = [.043, 0, .281] + np.array([.16, 0, -.13]) * retract
        mujoco.mj_forward(model, data)
        if step % 8 == 0:
            pairs = [set(model.geom_bodyid[[c.geom1, c.geom2]]) for c in data.contact]
            rows.append({'t': t, 'cube': data.xpos[target].tolist(),
                         'support_contact': {target, support} in pairs,
                         'finger_contact': any(target in pair and any(
                             model.body(int(b)).name.startswith('rh_') and b != model.body('rh_hand_base_link').id
                             for b in pair) for pair in pairs),
                         'joint_positions': {model.actuator(i).name: float(data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]])
                                             for i in range(model.nu)}})
        mujoco.mj_step(model, data)
    result = evaluate(rows)
    result.update(mass_kg=mass, native_drive=native_drive, close=close,
                  scope='URDF/MuJoCo cross-engine hold test; no table pickup, no original USD',
                  friction=1.0, contact_solref_s=.0111322812997299,
                  caveats='Convex mesh contact and native passive damping retained; not identical PhysX material/solver.')
    (out / 'trajectory.json').write_text(json.dumps(rows), encoding='utf-8')
    (out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=Path('runs/mujoco-runtime'))
    parser.add_argument('--source', type=Path, default=Path('runs/bimanual-contact-68/local-calibrated-contact.xml'))
    parser.add_argument('--out', type=Path, default=Path('runs/open-source-75/o6-physics'))
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime.resolve()))
    for name, mass, native, close in [('demo_50g', .05, False, True),
                                       ('current_100g', .1, True, True),
                                       ('open_control', .05, False, False)]:
        print(name, json.dumps(run(args.source, args.out / name, mass, native, close)), flush=True)
