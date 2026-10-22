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


def closure_targets(targets, delta, seconds_after_command):
    """Synthetic two-second correction after the retained command has completed."""
    result = dict(targets)
    fraction = min(1., max(0., seconds_after_command / 2.))
    for name, value in delta.items():
        result[name] += fraction * value
    return result


def parse_delta(value):
    name, separator, amount = value.partition('=')
    try:
        amount = float(amount)
    except ValueError:
        raise argparse.ArgumentTypeError('Expected native_joint=finite_radians') from None
    if not separator or name not in HAND_JOINTS or not math.isfinite(amount):
        raise argparse.ArgumentTypeError('Expected native_joint=finite_radians')
    return name, amount


def cube_bottom_z(position, quaternion_wxyz):
    """Lowest corner of the model's oriented 50 mm cube, not its center height."""
    q = np.asarray(quaternion_wxyz)
    rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    return float(position[2] - .025 * np.abs(rotation[2]).sum())


def add_hand_root(xml, world, position, quaternion, root_mode):
    """A dynamic free root welded to a moving reference supplies contact velocity."""
    attributes = {'pos': vector_text(position), 'quat': vector_text(quaternion)}
    if root_mode == 'mocap':
        return ET.SubElement(world, 'body', name='lh_hand_base_link', mocap='true', **attributes)
    if root_mode != 'weld':
        raise ValueError('Unknown root mode')
    ET.SubElement(world, 'body', name='root_reference', mocap='true', **attributes)
    root = ET.SubElement(world, 'body', name='lh_hand_base_link', **attributes)
    ET.SubElement(root, 'freejoint', name='hand_root_free')
    equality = ET.SubElement(xml, 'equality')
    ET.SubElement(equality, 'weld', body1='root_reference', body2='lh_hand_base_link', solref='.004 1')
    return root


def build_model(trial, assets, mass, friction, kp, kv, force, empty=False, contact_time=.02,
                timestep=.002, root_mode='weld', cone='pyramidal', impratio=1.):
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
    ET.SubElement(xml, 'option', timestep=str(timestep), gravity='0 0 -9.81', integrator='implicitfast',
                  cone=cone, impratio=str(impratio))
    default = ET.SubElement(xml, 'default')
    ET.SubElement(default, 'geom', friction=vector_text([friction, .005, .0001]), condim='3',
                  solref=vector_text([contact_time, 1.]))
    asset, world, actuators, contacts = (ET.SubElement(xml, n) for n in ('asset', 'worldbody', 'actuator', 'contact'))
    root = add_hand_root(xml, world, base_position, wxyz(base_rotation), root_mode)
    bodies = {'lh_hand_base_link': root}
    native = {'lh_hand_base_link': LEFT_HAND_BASE}
    for name, (child, _, origin, axis) in HAND_JOINTS.items():
        definition = joints[name]
        parent = definition.find('parent').get('link')
        link = definition.find('child').get('link')
        native[link] = 'r136d7b_link_' + child
        # Keep adjacent filtering explicit in both dynamic-root and legacy world-welded mocap modes.
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


def replay(trial, assets, mass=.1, friction=.6, kp=20., kv=1., force=1., empty=False, contact_time=.02,
           hold_seconds=6., timestep=.002, closure_delta=None, root_mode='weld', settle_z=0., settle_y=0.,
           cone='pyramidal', impratio=1.):
    import mujoco
    started = time.monotonic()
    model, xml, frames, native, sr, sdk_origin = build_model(trial, assets, mass, friction, kp, kv, force,
        empty=empty, contact_time=contact_time, timestep=timestep, root_mode=root_mode, cone=cone, impratio=impratio)
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
    root_id = model.body('lh_hand_base_link').id
    root_adr = int(model.jnt_qposadr[model.joint('hand_root_free').id]) if root_mode == 'weld' else None
    end_command = traces[-1]['joint_samples'][-1]['stamp']
    command_end = end_command[0] + end_command[1]*1e-9 - origin
    hold_end = command_end + hold_seconds
    duration = hold_end + (0. if empty else 4.)
    target_id = model.body('target').id if not empty else None
    target_adr = int(model.jnt_qposadr[model.joint('target_free').id]) if not empty else None
    table_z = float(model.geom_pos[model.geom('table').id,2]) if not empty else None
    initial_target = data.xpos[target_id].copy() if not empty else None
    rows, next_sample, before_lift = [], 0., None
    other_contacts = {}
    while data.time < duration:
        index = bisect_right(times, origin + data.time) - 1
        targets = commands[index]['targets_rad'] if index >= 0 else initial
        if closure_delta:
            targets = closure_targets(targets, closure_delta, data.time-command_end)
        for name in HAND_JOINTS:
            data.ctrl[model.actuator(name).id] = targets[name]
        rise = 0. if empty else .02 * np.clip((data.time-hold_end)/2.,0.,1.)
        settle = settle_z * np.clip((data.time-command_end)/2.,0.,1.)
        settle_side = settle_y * np.clip((data.time-command_end)/2.,0.,1.)
        data.mocap_pos[0] = base_position + [0,settle_side,settle+rise]
        mujoco.mj_step(model,data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError('Nonfinite reduced-model state')
        for contact in data.contact:
            pair = tuple(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(i)) for i in contact.geom)
            if 'target' not in pair and any(n.startswith('lh_') for n in pair):
                other_contacts[pair] = min(other_contacts.get(pair, 0.), float(contact.dist))
        if data.time >= next_sample:
            root_velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,root_id,root_velocity,0)
            root_position = data.qpos[root_adr:root_adr+3] if root_adr is not None else data.mocap_pos[0]
            root_quaternion = data.qpos[root_adr+3:root_adr+7] if root_adr is not None else data.mocap_quat[0]
            contacts = []
            for contact in data.contact:
                pair = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) for i in contact.geom]
                if 'target' in pair:
                    contacts.append(pair)
            rows.append({'seconds':float(data.time),'hand_position_rad':{n:float(data.qpos[i]) for n,i in indices.items()},
                'actuator_force':{n:float(data.actuator_force[model.actuator(n).id]) for n in HAND_JOINTS},
                # Free-root qpos is the post-step body pose; cached xpos/xquat still refer to the prior step.
                'tool_rise_m':float(rise), 'local_target_position':data.qpos[target_adr:target_adr+3].tolist() if not empty else None,
                'synthetic_root_settle_z_m':float(settle),
                'synthetic_root_settle_y_m':float(settle_side),
                'local_target_quaternion_wxyz':data.qpos[target_adr+3:target_adr+7].tolist() if not empty else None,
                'local_target_bottom_clearance_m':cube_bottom_z(data.qpos[target_adr:target_adr+3],
                    data.qpos[target_adr+3:target_adr+7])-table_z if not empty else None,
                'local_root_linear_velocity_m_s':root_velocity[3:].tolist(),
                'local_root_tracking_error_m':float(np.linalg.norm(root_position-data.mocap_pos[0])),
                'local_root_tracking_rotation_rad':2*math.acos(min(1.,abs(float(root_quaternion @ data.mocap_quat[0])))),
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
            'parameters':{'mass_kg':mass,'friction':friction,'kp':kp,'kv':kv,'force_limit':force,'empty':empty,
                          'contact_time':contact_time,'hold_seconds':hold_seconds,'timestep':timestep,
                          'root_mode':root_mode,'cone':cone,'impratio':impratio},
            'root_drive_scope':'Dynamic free body with 0.004 s weld to mocap reference; not native arm dynamics.' if root_mode == 'weld'
                               else 'Legacy direct mocap: root motion has no contact velocity; lift retention is not meaningful.',
            'synthetic_closure_delta_rad':dict(closure_delta or {}),
            'synthetic_closure_start_seconds':command_end if closure_delta else None,
            'synthetic_closure_ramp_seconds':2. if closure_delta else None,
            'synthetic_root_settle_z_m':settle_z,
            'synthetic_root_settle_y_m':settle_y,
            'synthetic_root_settle_start_seconds':command_end if settle_z or settle_y else None,
            'local_non_target_hand_contacts':[{'geom_pair':list(p),'minimum_signed_gap_m':d} for p,d in other_contacts.items()],
            'reference_origin_stamp_sec':origin,'synthetic_lift_start_seconds':None if empty else hold_end,
            'platform_physics_calibrated':False,'physical_grasp_verified':False,'platform_motion_commanded':False,
            'command_timing_scope':'Robot TF references at publication, not measured servo application times; post-command hold/lift is synthetic.',
            'sample_timing_scope':'Joint and free-target qpos at row seconds; cached root velocity/contact/actuator fields at the beginning of the last step.',
            'kinematic_errors':errors,'initial_local_target_position':initial_target.tolist() if not empty else None,
            'local_contacts_before_synthetic_lift':before_lift,'rows':rows,'wall_seconds':time.monotonic()-started},xml


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    for name,default in (('mass',.1),('friction',.6),('kp',20.),('kv',1.),('force',1.),('contact-time',.02),('hold-seconds',6.),('timestep',.002)):
        parser.add_argument('--'+name,type=float,default=default)
    parser.add_argument('--empty',action='store_true')
    parser.add_argument('--root-mode',choices=('weld','mocap'),default='weld',
                        help='Dynamic welded root by default; direct mocap is only a legacy diagnostic')
    parser.add_argument('--settle-z',type=float,default=0.,
                        help='Local-only root Z offset in metres, ramped after the saved command')
    parser.add_argument('--settle-y',type=float,default=0.,
                        help='Local-only root Y offset in metres, ramped after the saved command')
    parser.add_argument('--cone',choices=('pyramidal','elliptic'),default='pyramidal',
                        help='Local solver condition; does not edit the Rabokit scene')
    parser.add_argument('--impratio',type=float,default=1.,help='Local friction impedance ratio')
    parser.add_argument('--closure-delta',type=parse_delta,action='append',default=[],metavar='JOINT=RAD',
                        help='Local-only correction ramp after the saved command, repeat for selected joints')
    args=parser.parse_args()
    if (args.output.suffix.lower() != '.json' or args.output.exists() or args.output.with_suffix('.xml').exists()
            or any(not math.isfinite(v) or v<=0 for v in (args.mass,args.friction,args.kp,args.kv,args.force,args.contact_time,args.hold_seconds,args.timestep,args.impratio))):
        parser.error('Expected a new .json output path and finite positive model parameters')
    if not all(math.isfinite(v) for v in (args.settle_z,args.settle_y)):
        parser.error('Synthetic root offset must be finite')
    if (args.closure_delta or args.settle_z or args.settle_y) and (args.empty or args.hold_seconds < 2.):
        parser.error('Synthetic correction requires a contact replay with at least two seconds hold')
    result,xml=replay(args.trial,args.assets,mass=args.mass,friction=args.friction,kp=args.kp,kv=args.kv,
        force=args.force,empty=args.empty,contact_time=args.contact_time,hold_seconds=args.hold_seconds,
        timestep=args.timestep,closure_delta=dict(args.closure_delta),root_mode=args.root_mode,
        settle_z=args.settle_z,settle_y=args.settle_y,cone=args.cone,impratio=args.impratio)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with args.output.with_suffix('.xml').open('x',encoding='utf-8') as stream:
        stream.write(xml)
    print(json.dumps({'output':str(args.output),'samples':len(result['rows']),'wall_seconds':result['wall_seconds'],
                      'platform_physics_calibrated':False,'physical_grasp_verified':False}))


if __name__=='__main__':
    main()
