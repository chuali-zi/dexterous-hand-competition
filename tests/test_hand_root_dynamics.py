"""Optional engine regression for the moving root, using analytic jaws, no assets."""
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from replay_hand_dynamics import add_hand_hinge, add_hand_root


@unittest.skipIf(mujoco is None, 'Optional MuJoCo runtime not installed')
class MovingRootTests(unittest.TestCase):
    def test_saved_joint_stop_and_passive_damping_affect_the_engine(self):
        final = {}
        for limited in (True, False):
            xml = ET.Element('mujoco')
            ET.SubElement(xml, 'compiler', angle='radian')
            ET.SubElement(xml, 'option', timestep='.002', gravity='0 0 0', integrator='implicitfast')
            body = ET.SubElement(ET.SubElement(xml, 'worldbody'), 'body')
            add_hand_hinge(body, 'lh_thumb_ip', 1, .1 if limited else 0., limited)
            ET.SubElement(body, 'geom', type='sphere', pos='.05 0 0', size='.01', mass='.1')
            ET.SubElement(ET.SubElement(xml, 'actuator'), 'position', joint='lh_thumb_ip', kp='20', kv='1',
                          forcelimited='true', forcerange='-.01 .01')
            model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
            data = mujoco.MjData(model)
            data.qpos[0], data.qvel[0] = .5, .2
            mujoco.mj_forward(model, data)
            self.assertAlmostEqual(float(data.qfrc_passive[0]), -.02 if limited else 0., places=12)
            # Light loading isolates the stop; a high torque can penetrate a soft limit substantially.
            data.qpos[0], data.qvel[0] = 1.1, 0.
            data.ctrl[0] = 1.35
            while data.time < 3.:
                mujoco.mj_step(model, data)
            final[limited] = float(data.qpos[0])
        self.assertLess(final[True], 1.20)  # MuJoCo joint stops are soft constraints.
        self.assertGreater(final[False], 1.33)

    def test_dynamic_root_supplies_velocity_and_transports_a_friction_grasp(self):
        results = {}
        for mode in ['mocap', 'weld']:
            xml = ET.Element('mujoco')
            ET.SubElement(xml, 'option', timestep='.002', integrator='implicitfast')
            default = ET.SubElement(xml, 'default')
            ET.SubElement(default, 'geom', friction='1 .005 .0001', solref='.01 1')
            world = ET.SubElement(xml, 'worldbody')
            root = add_hand_root(xml, world, [0, 0, 1.], [1., 0, 0, 0], mode)
            for x in [-.0335, .0335]:
                ET.SubElement(root, 'geom', type='box', pos=f'{x} 0 0', size='.01 .04 .04', mass='1')
            cube = ET.SubElement(world, 'body', name='cube', pos='0 0 1')
            ET.SubElement(cube, 'freejoint', name='cube_free')
            ET.SubElement(cube, 'geom', type='box', size='.025 .025 .025', mass='.02')
            model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
            data = mujoco.MjData(model)
            cube_adr = model.jnt_qposadr[model.joint('cube_free').id]
            velocity = np.zeros(6)
            middle_velocity = None
            while data.time < 3.:
                rise = .02 * np.clip((data.time-.5)/2., 0., 1.)
                data.mocap_pos[0, 2] = 1. + rise
                mujoco.mj_step(model, data)
                if middle_velocity is None and data.time >= 1.5:
                    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                            model.body('lh_hand_base_link').id, velocity, 0)
                    middle_velocity = float(velocity[5])
            results[mode] = (middle_velocity, float(data.qpos[cube_adr+2]-1.))
        self.assertAlmostEqual(results['mocap'][0], 0., places=12)
        self.assertAlmostEqual(results['weld'][0], .01, delta=.001)
        self.assertLess(results['mocap'][1], .005)
        self.assertGreater(results['weld'][1], .015)


if __name__ == '__main__':
    unittest.main()
