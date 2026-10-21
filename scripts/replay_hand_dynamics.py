#!/usr/bin/env python3
"""Local reduced-hand MuJoCo replay from retained robot/RGB records, never ROS/SDK.

Public inertias, convex mesh collision, cube mass/friction and servo gains are
hypotheses. Local object states are diagnostic outputs, not platform truth or
verified training labels. No arm dynamics or policy training is included.
Requires mujoco, numpy and scipy; public O6 URDF/STLs are supplied separately.
"""
import argparse
from bisect import bisect_right
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from rabokit_calibration_analysis import LEFT_HAND_BASE, LEFT_ROOT, RIGHT_ROOT, world_transform
from rabokit_hand import HAND_JOINTS, quaternion_product, target_quaternion


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def vector_text(values):
    return ' '.join(format(float(v), '.17g') for v in values)


def wxyz(matrix):
    q = Rotation.from_matrix(matrix).as_quat()
    return q[[3, 0, 1, 2]]


def joint_values(frames):
    values = {}
    for name, (child, _, origin, axis) in HAND_JOINTS.items():
        zero = target_quaternion(origin, axis, 0.)
        actual = np.asarray(frames['r136d7b_link_' + child]['xyzw'], dtype=float)
        if actual.shape != (4,) or not np.isfinite(actual).all() or np.linalg.norm(actual) < 1e-9:
            raise ValueError(f'Invalid hand quaternion: {name}')
        relative = np.asarray(quaternion_product(tuple(-v for v in zero[:3]) + (zero[3],),
                                                 actual / np.linalg.norm(actual)))
        if relative[3] < 0:
            relative = -relative
        if max(abs(relative[i]) for i in range(3) if i != axis) > 1e-5:
            raise ValueError(f'Unexpected non-hinge rotation: {name}')
        values[name] = 2 * math.atan2(relative[axis], relative[3])
    return values


def build_model(trial, assets, mass, friction, kp, kv, force, empty=False, contact_time=.02):
    """Use measured parent offsets/hinge axes; keep public inertias as explicit guesses."""
    import mujoco
    stage = 'empty_closed' if empty else 'contact'
    frames = read(trial / stage / 'robot-frames.json')['frames']
    sr, lp = world_transform(frames, LEFT_ROOT)
    _, rp = world_transform(frames, RIGHT_ROOT)
    sr, sdk_origin = np.array(sr), (np.array(lp) + rp) / 2
    br, bp = world_transform(frames, LEFT_HAND_BASE)
    base_rotation, base_position = sr.T @ br, sr.T @ (np.array(bp) - sdk_origin)
    urdf = ET.parse(assets / 'linkerhand_o6_left.urdf').getroot()
    links = {x.get('name'): x for x in urdf.findall('link')}
    joints = {x.get('name'): x for x in urdf.findall('joint')}
    if set(joints) != set(HAND_JOINTS):
        raise ValueError('Expected the public twelve-link, eleven-joint O6 model')
    xml = ET.Element('mujoco', model='retained_hand_proxy')
    ET.SubElement(xml, 'compiler', angle='radian', autolimits='false', balanceinertia='true')
    ET.SubElement(xml, 'option', timestep='.001', gravity='0 0 -9.81', integrator='implicitfast')
    default = ET.SubElement(xml, 'default')
    ET.SubElement(default, 'geom', friction=vector_text([friction, .005, .0001]), condim='3',
                  solref=vector_text([contact_time, 1.]))
    asset, world, actuators, contacts = (ET.SubElement(xml, n) for n in ('asset', 'worldbody', 'actuator', 'contact'))
    root = ET.SubElement(world, 'body', name='lh_hand_base_link', mocap='true',
                         pos=vector_text(base_position), quat=vector_text(wxyz(base_rotation)))
    bodies = {'lh_hand_base_link': root}
    native = {'lh_hand_base_link': LEFT_HAND_BASE}
    for name, (child, _, origin, axis) in HAND_JOINTS.items():
        definition = joints[name]
        parent = definition.find('parent').get('link')
        link = definition.find('child').get('link')
        native[link] = 'r136d7b_link_' + child
        # Under pinned MuJoCo 3.3.7, a mocap root is world-welded; preserve adjacent-link filtering.
        ET.SubElement(contacts, 'exclude', body1=parent, body2=link)
        zero = target_quaternion(origin, axis, 0.)
        body = ET.SubElement(bodies[parent], 'body', name=link,
            pos=vector_text(frames[native[link]]['xyz']), quat=vector_text([zero[3], *zero[:3]]))
        bodies[link] = body
        # Native topics command eleven independent hinges, unlike the public mimic hints.
        ET.SubElement(body, 'joint', name=name, type='hinge', axis=vector_text(np.eye(3)[axis]), limited='false')
        ET.SubElement(actuators, 'position', name=name, joint=name, kp=str(kp), kv=str(kv),
                      forcelimited='true', forcerange=vector_text([-force, force]))
    for name, body in bodies.items():
        link = links[name]
        inertial = link.find('inertial')
        if any(abs(float(v)) > 1e-12 for v in inertial.find('origin').get('rpy').split()):
            raise ValueError('This importer expects the public zero-RPY inertia frames')
        tensor = inertial.find('inertia')
        ET.SubElement(body, 'inertial', pos=inertial.find('origin').get('xyz'),
            mass=inertial.find('mass').get('value'),
            fullinertia=vector_text([tensor.get(n) for n in ('ixx','iyy','izz','ixy','ixz','iyz')]))
        mesh = (assets / Path(link.find('collision/geometry/mesh').get('filename')).name).resolve()
        if not mesh.is_file():
            raise FileNotFoundError(mesh)
        ET.SubElement(asset, 'mesh', name=name, file=str(mesh))
        ET.SubElement(body, 'geom', name=name, type='mesh', mesh=name)
    if not empty:
        source = np.array(read(trial / 'grasp.json')['source_surface']['colored_surface_point_sdk_m'])
        ET.SubElement(world, 'geom', name='table', type='plane', pos=vector_text(source-[0,0,.05]), size='1 1 .01')
        for i, item in enumerate(read(trial / 'initial/stereo.json')['matches']):
            position = np.array(item['colored_surface_point_sdk_m']) - [0, 0, .025]
            if item['color'] == 'green':
                body = ET.SubElement(world, 'body', name='target', pos=vector_text(position))
                ET.SubElement(body, 'freejoint', name='target_free')
                ET.SubElement(body, 'geom', name='target', type='box', size='.025 .025 .025', mass=str(mass))
            else:
                ET.SubElement(world, 'geom', name=f'neighbor_{i}', type='box',
                              pos=vector_text(position), size='.025 .025 .025')
    model_xml = ET.tostring(xml, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(model_xml)
    return model, model_xml, frames, native, sr, sdk_origin


def replay(trial, assets, mass=.1, friction=.6, kp=20., kv=.02, force=1., empty=False, contact_time=.02):
    import mujoco
    started = time.monotonic()
    model, xml, frames, native, sr, sdk_origin = build_model(trial, assets, mass, friction, kp, kv, force, empty, contact_time)
    data = mujoco.MjData(model)
    indices = {n:int(model.jnt_qposadr[model.joint(n).id]) for n in HAND_JOINTS}
    for n, value in joint_values(frames).items():
        data.qpos[indices[n]] = value
    mujoco.mj_forward(model, data)
    errors = {}
    for name, frame in native.items():
        rotation, position = world_transform(frames, frame)
        body = model.body(name).id
        errors[name] = {'position_m':float(np.linalg.norm(data.xpos[body]-sr.T@(np.array(position)-sdk_origin))),
                       'rotation_rad':float(Rotation.from_matrix(data.xmat[body].reshape(3,3).T @ sr.T @ rotation).magnitude())}
    if max(e['position_m'] for e in errors.values()) > 1e-6 or max(e['rotation_rad'] for e in errors.values()) > 1e-5:
        raise ValueError(f'Proxy hinge mapping disagrees with retained robot TF: {errors}')
    stages = ('empty_closed',) if empty else ('thumb_ready','thumb_opposed','closed')
    traces = [read(trial / (stage+'-command.json')) for stage in stages]
    commands = [c for trace in traces for c in trace['issued_commands']]
    times = [c['robot_tf_reference_stamp_sec'] for c in commands]
    if any(b < a for a,b in zip(times,times[1:])):
        raise ValueError('Published TF reference times must be ordered')
    first = traces[0]['joint_samples'][0]
    origin = first['stamp'][0] + first['stamp'][1] * 1e-9
    initial = first['position']
    for name in HAND_JOINTS:
        data.qpos[indices[name]] = initial[name]
    for i,name in enumerate(HAND_JOINTS):
        data.ctrl[model.actuator(name).id] = initial[name]
    mujoco.mj_forward(model,data)
    base_position = data.mocap_pos[0].copy()
    end_command = traces[-1]['joint_samples'][-1]['stamp']
    hold_end = end_command[0] + end_command[1]*1e-9 - origin + 6.
    duration = hold_end + (0. if empty else 4.)
    target_id = model.body('target').id if not empty else None
    initial_target = data.xpos[target_id].copy() if not empty else None
    rows, next_sample, before_lift = [], 0., None
    while data.time < duration:
        index = bisect_right(times, origin + data.time) - 1
        targets = commands[index]['targets_rad'] if index >= 0 else initial
        for name in HAND_JOINTS:
            data.ctrl[model.actuator(name).id] = targets[name]
        rise = 0. if empty else .02 * np.clip((data.time-hold_end)/2.,0.,1.)
        data.mocap_pos[0] = base_position + [0,0,rise]
        mujoco.mj_step(model,data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError('Nonfinite reduced-model state')
        if data.time >= next_sample:
            contacts = []
            for contact in data.contact:
                pair = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) for i in contact.geom]
                if 'target' in pair:
                    contacts.append(pair)
            rows.append({'seconds':float(data.time),'hand_position_rad':{n:float(data.qpos[i]) for n,i in indices.items()},
                'actuator_force':{n:float(data.actuator_force[model.actuator(n).id]) for n in HAND_JOINTS},
                'tool_rise_m':float(rise), 'local_target_position':data.xpos[target_id].tolist() if not empty else None,
                'local_target_quaternion_wxyz':data.xquat[target_id].tolist() if not empty else None,
                'local_target_contacts':contacts})
            next_sample = data.time + .02
        if not empty and before_lift is None and data.time >= hold_end:
            before_lift = []
            for i, contact in enumerate(data.contact):
                pair = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in contact.geom]
                if 'target' in pair:
                    local_force = np.zeros(6)
                    mujoco.mj_contactForce(model,data,i,local_force)
                    before_lift.append({'geom_pair':pair,'position_sdk_m':contact.pos.tolist(),
                        'frame_first_axis':contact.frame[:3].tolist(),'force_in_contact_frame':local_force.tolist()})
    return {'scope':__doc__, 'mujoco_version':mujoco.__version__, 'trial':str(trial),
            'parameters':{'mass_kg':mass,'friction':friction,'kp':kp,'kv':kv,'force_limit':force,'empty':empty,'contact_time':contact_time},
            'platform_physics_calibrated':False,'physical_grasp_verified':False,'platform_motion_commanded':False,
            'command_timing_scope':'Robot TF references at publication, not measured servo application times; post-command hold/lift is synthetic.',
            'kinematic_errors':errors,'initial_local_target_position':initial_target.tolist() if not empty else None,
            'local_contacts_before_synthetic_lift':before_lift,'rows':rows,'wall_seconds':time.monotonic()-started},xml


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    for name,default in (('mass',.1),('friction',.6),('kp',20.),('kv',.02),('force',1.),('contact-time',.02)):
        parser.add_argument('--'+name,type=float,default=default)
    parser.add_argument('--empty',action='store_true')
    args=parser.parse_args()
    if (args.output.suffix.lower() != '.json' or args.output.exists() or args.output.with_suffix('.xml').exists()
            or any(not math.isfinite(v) or v<=0 for v in (args.mass,args.friction,args.kp,args.kv,args.force,args.contact_time))):
        parser.error('Expected a new .json output path and finite positive model parameters')
    result,xml=replay(args.trial,args.assets,args.mass,args.friction,args.kp,args.kv,args.force,args.empty,args.contact_time)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with args.output.with_suffix('.xml').open('x',encoding='utf-8') as stream:
        stream.write(xml)
    print(json.dumps({'output':str(args.output),'samples':len(result['rows']),'wall_seconds':result['wall_seconds'],
                      'platform_physics_calibrated':False,'physical_grasp_verified':False}))


if __name__=='__main__':
    main()
